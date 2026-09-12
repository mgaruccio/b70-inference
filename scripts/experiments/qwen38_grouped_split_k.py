"""Opt-in, C1/MTP4 grouped-query Split-K prototype; not a production launcher.

Patch the *pre-expansion* native helper, never the five pseudo-sequences it
creates. Only B70_GROUPED_SPLIT_K=1 enables install(). The two explicit split
choices (16/32) and stage choices (1/2) are operator-test knobs, not a tuner.
No tensor contents are inspected on the host, and no workspaces are cached.

Research basis: triton-lang.org/main/python-api/generated/triton.language.dot.html
and intel/intel-xpu-backend-for-triton's hardware-reference.md: FP16 DPAS with
FP32 accumulation, not FP8 dot/dot_scaled. Installed FP8 conversion support must
be qualified; eligible compilation/launch failures deliberately propagate.
"""

import functools
import math
import os

ENABLE_ENV = "B70_GROUPED_SPLIT_K"
SPLITS_ENV = "B70_GROUPED_SPLIT_K_SPLITS"
STAGES_ENV = "B70_GROUPED_SPLIT_K_STAGES"
_KERNELS = None  # JIT function handles only; never tensors or capture workspaces.


def _nonoverlapping(tensor):
    """Host metadata only; accept pitched/transpose/split views, reject overlap."""
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size <= 1:
            continue
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


def unsupported_reason(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                       max_seqlen_k, k_descale, v_descale, softmax_scale,
                       s_aux, window_size, softcap):
    """Return a host-known rejection reason, without item(), CPU copies or sync.

    The native caller already restricts this helper to uniform causal paged
    attention without LSE, alibi, q_v or scheduler auxiliaries. Its remaining
    sink/window/softcap arguments are checked here. Malformed *device contents*
    are handled in the kernel, not by a graph-breaking host predicate.
    """
    import torch

    tensors = (q, k, v, cu_seqlens_q, seqused_k, block_table)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return "missing tensor"
    if q.device.type != "xpu" or any(t.device != q.device for t in tensors):
        return "device"
    if q.shape != (5, 24, 256) or q.dtype != torch.float16:
        return "C1 MTP4 query shape/dtype"
    if (k.ndim != 4 or k.shape[0] < 1 or k.shape[1:] != (1664, 4, 256)
            or v.shape != k.shape or k.dtype != torch.float8_e4m3fn
            or v.dtype != torch.float8_e4m3fn):
        return "KV shape/dtype"
    if any(t.stride(-1) != 1 or not _nonoverlapping(t) for t in (q, k, v)):
        return "Q/K/V strides"
    if (cu_seqlens_q.shape != (2,) or seqused_k.shape != (1,)
            or block_table.ndim != 2 or block_table.shape[0] != 1
            or block_table.shape[1] < 1):
        return "C1 metadata shape"
    if any(t.dtype != torch.int32 or not _nonoverlapping(t)
           for t in (cu_seqlens_q, seqused_k, block_table)):
        return "metadata dtype/strides"
    if (type(max_seqlen_k) is not int or not 1 <= max_seqlen_k < 2**31
            or block_table.shape[1] * 1664 >= 2**31):
        return "host KV bound"
    for descale in (k_descale, v_descale):
        if (not isinstance(descale, torch.Tensor) or descale.device != q.device
                or descale.dtype != torch.float32 or descale.numel() < 1
                or descale.untyped_storage().nbytes() != 4
                or descale.storage_offset() != 0
                or any(size > 1 and stride != 0
                       for size, stride in zip(descale.shape, descale.stride()))):
            return "singleton-storage FP32 descales required"
    if out is not None and (
            not isinstance(out, torch.Tensor) or out.shape != q.shape
            or out.dtype != q.dtype or out.device != q.device
            or out.stride(-1) != 1 or not _nonoverlapping(out)):
        return "output shape/dtype/device/strides"
    if (s_aux is not None or type(window_size) not in (tuple, list)
            or len(window_size) != 2
            or any(type(x) is not int or x != -1 for x in window_size)):
        return "sink/window"
    if type(softcap) not in (int, float) or softcap != 0:
        return "softcap"
    if (type(softmax_scale) not in (int, float)
            or not math.isfinite(softmax_scale) or softmax_scale <= 0):
        return "softmax scale"
    return None


def _kernels():
    # Lazy imports keep the disabled shim dependency-free. Globals let Triton's
    # JIT resolve tl consistently when inspecting these nested definitions.
    global _KERNELS, triton, tl
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def partial(Q, K, V, CU, USED, TABLE, KS, VS, PART, LSE,
                Q0: tl.constexpr, Q1: tl.constexpr,
                K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
                CU0: tl.constexpr, BT1: tl.constexpr,
                PAGES: tl.constexpr, CAPACITY: tl.constexpr,
                SCALE: tl.constexpr, SPLITS: tl.constexpr,
                BN: tl.constexpr):
        head = tl.program_id(0)
        split = tl.program_id(1)
        row = tl.arange(0, 32)
        d = tl.arange(0, 256)
        n = tl.arange(0, BN)
        # Cumulative lengths are device inputs too. A dummy graph row is zero,
        # not a request to read another sequence's KV. Widen before subtracting
        # so even int32-min dummy lengths cannot overflow into a huge KV read.
        meta_ok = (tl.load(CU) == 0) & (tl.load(CU + CU0) == 5)
        used = tl.load(USED).to(tl.int64)
        live = tl.minimum(tl.maximum(used, 1), CAPACITY)
        live = tl.where(meta_ok, live, 0)
        limit = tl.minimum(tl.maximum(used - 4 + row // 6, 1), CAPACITY)
        row_ok = (row < 30) & meta_ok
        q = tl.load(Q + (row // 6)[:, None] * Q0
                    + (head * 6 + row % 6)[:, None] * Q1 + d[None, :],
                    mask=row_ok[:, None], other=0)
        ks = tl.load(KS)
        vs = tl.load(VS)
        tiles_per_split = tl.cdiv(tl.cdiv(live, BN), SPLITS)
        begin = split * tiles_per_split * BN
        end = tl.minimum(begin + tiles_per_split * BN, live)
        m = tl.full((32,), -float("inf"), tl.float32)
        z = tl.zeros((32,), tl.float32)
        acc = tl.zeros((32, 256), tl.float32)
        for start in range(begin, end, BN):
            pos = start + n
            page = tl.load(TABLE + (pos // 1664) * BT1,
                           mask=pos < live, other=-1)
            valid = (pos < live) & (page >= 0) & (page < PAGES)
            safe_page = tl.where(valid, page, 0).to(tl.int64)
            # HND serving views have K1/V1=512, K2/V2=1664*512,
            # and V's pointer is already offset by 256. Do not assume NHD.
            kt = tl.load(K + safe_page[None, :] * K0
                         + (pos % 1664)[None, :] * K1 + head * K2
                         + d[:, None], mask=valid[None, :], other=0)
            vt = tl.load(V + safe_page[:, None] * V0
                         + (pos % 1664)[:, None] * V1 + head * V2
                         + d[None, :], mask=valid[:, None], other=0)
            # Each descaled FP16 tile is shared by all 5*6 query rows.
            kt = (kt.to(tl.float16).to(tl.float32) * ks).to(tl.float16)
            vt = (vt.to(tl.float16).to(tl.float32) * vs).to(tl.float16)
            score = tl.dot(q, kt) * SCALE  # FP16 DPAS, FP32 accumulator.
            mask = row_ok[:, None] & valid[None, :] & (pos[None, :] < limit[:, None])
            score = tl.where(mask, score, -float("inf"))
            next_m = tl.maximum(m, tl.max(score, 1))
            origin = tl.where(next_m == -float("inf"), 0.0, next_m)
            alpha = tl.exp(m - origin)
            prob = tl.exp(score - origin[:, None])
            z = z * alpha + tl.sum(prob, 1)
            acc = acc * alpha[:, None] + tl.dot(prob.to(tl.float16), vt)
            m = next_m
        denom = tl.where(z > 0, z, 1.0)
        base = (head * SPLITS + split) * 32 + row
        tl.store(PART + base[:, None] * 256 + d[None, :], acc / denom[:, None])
        # Empty splits and padded rows have exactly zero partial output/-inf
        # LSE; no -inf - -inf or 0/0 reaches the reduction.
        tl.store(LSE + base, m + tl.log(denom))

    @triton.jit
    def reduce(PART, LSE, OUT, O0: tl.constexpr, O1: tl.constexpr,
               SPLITS: tl.constexpr):
        head = tl.program_id(0)
        row = tl.program_id(1)  # Only the 30 real rows are launched.
        split = tl.arange(0, SPLITS)
        d = tl.arange(0, 256)
        base = (head * SPLITS + split) * 32 + row
        lse = tl.load(LSE + base)
        m = tl.max(lse, 0)
        origin = tl.where(m == -float("inf"), 0.0, m)
        weights = tl.exp(lse - origin)
        denom = tl.sum(weights, 0)
        parts = tl.load(PART + base[:, None] * 256 + d[None, :])
        y = tl.sum(parts * weights[:, None], 0) / tl.where(denom > 0, denom, 1.0)
        tl.store(OUT + (row // 6) * O0 + (head * 6 + row % 6) * O1 + d, y)

    _KERNELS = partial, reduce
    return _KERNELS


def _launch(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
            max_seqlen_k, k_descale, v_descale, softmax_scale,
            *, num_splits, num_stages):
    import torch

    partial, reduce = _kernels()
    if out is None:
        out = torch.empty_like(q)
    # Per-call allocations: graph capture's private pool retains their addresses
    # for replay. No shape-keyed/global mutable workspace can alias live graphs.
    parts = torch.empty((4, num_splits, 32, 256), dtype=torch.float32, device=q.device)
    lse = torch.empty((4, num_splits, 32), dtype=torch.float32, device=q.device)
    partial[(4, num_splits)](
        q, k, v, cu_seqlens_q, seqused_k, block_table, k_descale, v_descale,
        parts, lse, *q.stride()[:2], *k.stride()[:3], *v.stride()[:3],
        cu_seqlens_q.stride(0), block_table.stride(1), k.shape[0],
        min(max_seqlen_k, block_table.shape[1] * 1664), float(softmax_scale),
        num_splits, 32, num_warps=4, num_stages=num_stages,
    )
    reduce[(4, 30)](parts, lse, out, *out.stride()[:2], num_splits,
                    num_warps=4, num_stages=1)
    return out


def _make_wrapper(native, num_splits, num_stages):
    first_dispatch = True

    @functools.wraps(native)
    def grouped(q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                max_seqlen_k, k_descale, v_descale, softmax_scale,
                s_aux, window_size, softcap):
        nonlocal first_dispatch
        args = (q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                max_seqlen_k, k_descale, v_descale, softmax_scale,
                s_aux, window_size, softcap)
        if unsupported_reason(*args) is not None:
            return native(*args)
        if first_dispatch:
            first_dispatch = False
            print(f"[B70_GROUPED_SPLIT_K] eligible pre-expansion q5 C1 dispatch "
                  f"splits={num_splits} stages={num_stages}", flush=True)
        # Do not catch errors here: a supported path must not silently become a
        # native baseline when its Triton compiler or launch fails.
        return _launch(*args[:11], num_splits=num_splits, num_stages=num_stages)

    grouped._b70_grouped_split_k = (num_splits, num_stages)
    return grouped


def install():
    """Install only the native helper hook, only on an explicit opt-in."""
    if os.environ.get(ENABLE_ENV) != "1":
        return False
    splits = int(os.environ.get(SPLITS_ENV, "16"))
    stages = int(os.environ.get(STAGES_ENV, "1"))
    if splits not in (16, 32) or stages not in (1, 2):
        raise ValueError("grouped Split-K requires splits in {16,32}, stages in {1,2}")
    from vllm_xpu_kernels import flash_attn_interface as fa

    native = fa._spec_decode_varlen_fwd
    installed = getattr(native, "_b70_grouped_split_k", None)
    if installed is not None:
        if installed != (splits, stages):
            raise RuntimeError("grouped Split-K already installed with different options")
        return True
    fa._spec_decode_varlen_fwd = _make_wrapper(native, splits, stages)
    print(f"[B70_GROUPED_SPLIT_K] installed opt-in pre-expansion helper "
          f"splits={splits} stages={stages}", flush=True)
    return True


def install_worker_profile(worker_class):
    """Compatibility with the disposable serving worker shim; no profiling."""
    return None
