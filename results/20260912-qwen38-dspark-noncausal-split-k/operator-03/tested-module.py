"""Opt-in DSpark C1/Q7 BF16 noncausal attention via native Split-K decode.

Experimental, pinned to vllm-xpu-kernels 0.1.14.1. No custom kernel and no
acceptance/model changes. Valid packed C1 metadata is a caller precondition.
Every query gets the same full parent KV length, including zero-length dummy
metadata; never apply the causal helper's position-dependent decrement.
"""
import functools
import importlib.metadata
import inspect
import os
import sys

import torch

ENABLE_ENV = 'B70_DSPARK_NONCAUSAL_SPLIT_K'


def eligible(a):
    """Host-only guards; never inspect tensor values or synchronize the GPU."""
    q, k, v = a['q'], a['k'], a['v']
    if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
        return False
    if q.shape != (7, 32, 128) or q.dtype != torch.bfloat16 or q.device.type != 'xpu':
        return False
    if k.ndim != 4 or k.shape[1:] != (1664, 8, 128) or v.shape != k.shape:
        return False
    if any(t.dtype != q.dtype or t.device != q.device or t.stride(-1) != 1 for t in (q, k, v)):
        return False
    if a['max_seqlen_q'] != 7 or a['causal'] or a['fa_version'] != 2:
        return False
    if a['window_size'] is not None and tuple(a['window_size']) != (-1, -1):
        return False
    if any(a[key] is not None for key in ('cu_seqlens_k', 'q_v', 'alibi_slopes',
            'scheduler_metadata', 'q_descale', 's_aux', 'num_splits_kv', 'host_kv_lens')):
        return False
    if any(a[key] for key in ('dropout_p', 'softcap', 'return_attn_probs',
                              'return_softmax_lse', 'num_splits', 'deterministic')):
        return False
    cu, used, table = a['cu_seqlens_q'], a['seqused_k'], a['block_table']
    if not all(isinstance(t, torch.Tensor) and t.device == q.device and t.dtype == torch.int32
               for t in (cu, used, table)):
        return False
    if cu.shape != (2,) or used.shape != (1,) or table.ndim != 2 or table.shape[0] != 1:
        return False
    if not 0 < a['max_seqlen_k'] <= table.shape[1] * 1664:
        return False
    out = a['out']
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype or
                            out.device != q.device or not out.is_contiguous()):
        return False
    return True


def make_wrapper(native):
    signature = inspect.signature(native)

    @functools.wraps(native)
    def forward(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not eligible(values):
            return native(*args, **kwargs)
        forward.dispatches += 1
        if forward.dispatches == 1:
            print('[B70_DSPARK_NONCAUSAL_SPLIT_K] eligible C1/Q7 BF16 dispatch; full KV length per query', flush=True)
        expanded = dict(values)
        # Native decode does not write output for an empty KV row. Initialize
        # every replay so artificial empty/dummy states cannot expose stale data.
        expanded['out'] = (torch.zeros_like(values['q'], memory_format=torch.contiguous_format)
                           if values['out'] is None else values['out'].zero_())
        expanded.update(
            max_seqlen_q=1,
            cu_seqlens_q=torch.arange(8, dtype=torch.int32, device=values['q'].device),
            seqused_k=values['seqused_k'].repeat_interleave(7),
            block_table=values['block_table'].repeat_interleave(7, dim=0),
            is_mix_batch=False,
        )
        # Calls the original public interface's native Q=1 decode route. The
        # repeated metadata operations are captured and rerun on graph replay.
        # Unsupported kernel errors are not swallowed by this wrapper.
        return native(**expanded)

    forward.dispatches = 0
    forward._b70_noncausal_split_k = True
    return forward


def install():
    if os.environ.get(ENABLE_ENV) != '1':
        return False
    version = importlib.metadata.version('vllm-xpu-kernels')
    if version != '0.1.14.1':
        raise RuntimeError(f'Unqualified XPU kernels version: {version}')
    from vllm_xpu_kernels import flash_attn_interface as fa
    native = fa.flash_attn_varlen_func
    if getattr(native, '_b70_noncausal_split_k', False):
        return True
    # _xpu_ops imports this function by value. Update that known binding if
    # already imported; later imports naturally see the new interface binding.
    xpu_ops = sys.modules.get('vllm._xpu_ops')
    if xpu_ops is not None and xpu_ops.flash_attn_varlen_func is not native:
        raise RuntimeError('Unexpected vllm._xpu_ops attention binding')
    wrapped = make_wrapper(native)
    fa.flash_attn_varlen_func = wrapped
    if xpu_ops is not None:
        xpu_ops.flash_attn_varlen_func = wrapped
    print('[B70_DSPARK_NONCAUSAL_SPLIT_K] installed opt-in native route', flush=True)
    return True


def install_worker_profile(worker_class):
    """Compatibility with the existing disposable worker-import shim."""
    return None
