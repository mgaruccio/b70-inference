# SPDX-License-Identifier: Apache-2.0
"""Drafter-only int8 lm_head for DFlash 2 candidate selection on XPU.

The drafter shares the target's bf16 lm_head (vocab x hidden, ~615 MB) and runs a full
pass of it every draft step only to pick its top-k candidates. This keeps a per-row
symmetric int8 copy (~307 MB + scales) used *only* for drafting; the target's logits and
verification are untouched, so speculative decoding stays lossless.
Kill switch: FLINT_XPU_DRAFT_HEAD_INT8=0."""

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("FLINT_XPU_DRAFT_HEAD_INT8", "1") != "0"
BM, BN, BK, NWARPS = 8, 64, 64, 4


@triton.jit
def _i8_head_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = (pn * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + rm[:, None] * K + k0 + rk[None, :], mask=rm[:, None] < M, other=0.0)
        b = tl.load(w_ptr + rn[None, :] * K + k0 + rk[:, None])
        acc += tl.dot(a, b.to(tl.bfloat16))
    acc = acc * tl.load(s_ptr + rn)[None, :]
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rm[:, None] < M)


def quantize_rows(w: torch.Tensor, chunk: int = 16384):
    """bf16 [N, K] -> (int8 [N, K], fp32 [N]) per-row symmetric."""
    n, k = w.shape
    q = torch.empty((n, k), dtype=torch.int8, device=w.device)
    s = torch.empty((n,), dtype=torch.float32, device=w.device)
    for i in range(0, n, chunk):
        blk = w[i:i + chunk].float()
        sc = blk.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        q[i:i + chunk] = torch.round(blk / sc[:, None]).clamp_(-127, 127).to(torch.int8)
        s[i:i + chunk] = sc
    return q, s


def i8_head(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = q.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and n % BN == 0 and k % BK == 0
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m:
        _i8_head_kernel[(n // BN, triton.cdiv(m, BM))](x, q, s, out, M=m, N=n, K=k, BM=BM,
                                                       BN=BN, BK=BK, num_warps=NWARPS)
    return out


class _Int8HeadMethod:
    def __init__(self, q, s):
        self.q, self.s = q, s

    def apply(self, layer, x, bias=None):
        assert bias is None
        return i8_head(x.contiguous(), self.q, self.s)


class Int8DraftHead:
    """Proxy for a ParallelLMHead whose quant_method computes int8 logits; every other
    attribute (shard_indices, tp_size, ...) is the wrapped head's."""

    def __init__(self, head, q, s):
        self._head = head
        self.quant_method = _Int8HeadMethod(q, s)

    def __getattr__(self, name):
        return getattr(self._head, name)


# Off by default: no clear gain (spec4: greedy +1%, service sampling -2% at k=3).
LINEAR_ENABLED = os.environ.get("FLINT_XPU_DRAFT_LINEAR_INT8", "0") == "1"
LINEAR_MAX_M = 64  # larger batches (context prefill) keep the bf16 GEMM


class _Int8LinearMethod:
    """Weight-only int8 apply for an unquantized drafter linear (small M); falls back to
    the original bf16 method otherwise."""

    def __init__(self, orig):
        self.orig = orig

    def apply(self, layer, x, bias=None):
        x2 = x.reshape(-1, x.shape[-1])
        if bias is not None or x2.shape[0] > LINEAR_MAX_M or x.dtype != torch.bfloat16:
            return self.orig.apply(layer, x, bias)
        y = i8_head(x2.contiguous(), layer._xpu_i8_q, layer._xpu_i8_s)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    def __getattr__(self, name):
        return getattr(self.orig, name)


def quantize_drafter_linears(model) -> int:
    """Give every unquantized bf16 2-D linear of the drafter an int8 weight copy and
    the int8 apply method. Returns the number of layers converted."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    n = 0
    for name, m in model.named_modules():
        qm = getattr(m, "quant_method", None)
        w = getattr(m, "weight", None)
        if (not isinstance(qm, UnquantizedLinearMethod) or w is None or w.dim() != 2
                or w.dtype != torch.bfloat16 or w.shape[0] % BN or w.shape[1] % BK
                or getattr(m, "bias", None) is not None):
            continue
        m._xpu_i8_q, m._xpu_i8_s = quantize_rows(w.detach())
        m.quant_method = _Int8LinearMethod(qm)
        n += 1
    return n


def get_draft_head(owner, head):
    """Cached int8 proxy for ``head`` (built outside graph capture), or ``head``."""
    if not ENABLED or not torch.xpu.is_available():
        return head
    proxy = getattr(owner, "_xpu_int8_draft_head", None)
    if proxy is not None and proxy._head is head:
        return proxy
    if torch.xpu.is_current_stream_capturing():
        return head
    w = getattr(head, "weight", None)
    if (w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[0] % BN
            or w.shape[1] % BK or getattr(head, "tp_size", 1) != 1):
        return head
    q, s = quantize_rows(w.detach())
    proxy = Int8DraftHead(head, q, s)
    owner._xpu_int8_draft_head = proxy
    print(f"flint xpu: int8 draft lm_head built {tuple(q.shape)}", flush=True)
    if LINEAR_ENABLED:
        drafter = getattr(owner, "model", None)
        if drafter is not None:
            n = quantize_drafter_linears(drafter)
            print(f"flint xpu: int8 drafter linears: {n}", flush=True)
    return proxy
