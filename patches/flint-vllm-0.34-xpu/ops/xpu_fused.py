"""Portable Triton kernels for Flint on XPU (no PTX inline asm / CUDA libdevice).

Numerics match the eager XPU bring-up path (fp32 compute, one bf16 rounding per
output) and are per-row, so results are deterministic and batch-invariant:

- ``rms_norm``: ``bf16(x * rsqrt(mean(x^2) + eps) * w)``.
- ``fused_add_rms_norm``: ``s = x + residual`` in fp32, returns
  ``(bf16(norm(s)), bf16(s))`` with the norm from the unrounded ``s``.
- ``silu_and_mul``: ``bf16(silu(g) * u)`` in fp32.
- ``qk_norm_rope``: per-head fp32 RMSNorm of q and k, rounded to bf16, then
  (sliding layers) fp32 NeoX RoPE from the fp32 cos/sin cache, rounded to
  bf16; v is copied out contiguous. One program per (token, head).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, x_stride, eps,
                     H: tl.constexpr, HAS_RESIDUAL: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    s = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    if HAS_RESIDUAL:
        s += tl.load(res_ptr + row * H + offs).to(tl.float32)
        tl.store(res_out_ptr + row * H + offs, s.to(tl.bfloat16))
    rs = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / H + eps)
    tl.store(out_ptr + row * H + offs, (s * rs * w).to(tl.bfloat16))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1 and h & (h - 1) == 0
    out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    if m:
        _rms_norm_kernel[(m,)](x, x, weight, out, out, x.stride(0), eps, H=h,
                               HAS_RESIDUAL=False, num_warps=4)
    return out


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, eps: float):
    """Out of place: returns ``(norm(x + residual), bf16(x + residual))``.

    Inputs are never written: callers' activations may alias layer workspaces."""
    m, h = x.shape
    assert x.stride(-1) == 1 and residual.is_contiguous() and residual.shape == x.shape
    assert x.dtype == residual.dtype == torch.bfloat16 and h & (h - 1) == 0
    out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    res_out = torch.empty((m, h), dtype=x.dtype, device=x.device)
    if m:
        _rms_norm_kernel[(m,)](x, residual, weight, out, res_out, x.stride(0), eps, H=h,
                               HAS_RESIDUAL=True, num_warps=4)
    return out, res_out


@triton.jit
def _silu_and_mul_kernel(x_ptr, out_ptr, x_stride, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BD + tl.arange(0, BD)
    g = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    u = tl.load(x_ptr + row * x_stride + D + offs).to(tl.float32)
    silu = g / (1.0 + tl.exp(-g))
    tl.store(out_ptr + row * D + offs, (silu * u).to(tl.bfloat16))


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    m, d2 = x.shape
    d = d2 // 2
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1
    bd = min(d, 1024)
    assert d % bd == 0
    out = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if m:
        _silu_and_mul_kernel[(m, d // bd)](x, out, x.stride(0), D=d, BD=bd, num_warps=4)
    return out


@triton.jit
def _qk_norm_rope_kernel(qkv_ptr, pos_ptr, qw_ptr, kw_ptr, cs_ptr,
                         q_ptr, k_ptr, v_ptr, qkv_stride, eps,
                         NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                         HAS_ROPE: tl.constexpr):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    HALF: tl.constexpr = HD // 2
    base = qkv_ptr + tok * qkv_stride
    if head < NQ + NKV:
        is_q = head < NQ
        src = base + head * HD
        lo = tl.arange(0, HALF)
        a = tl.load(src + lo).to(tl.float32)
        b = tl.load(src + HALF + lo).to(tl.float32)
        if is_q:
            wa = tl.load(qw_ptr + lo).to(tl.float32)
            wb = tl.load(qw_ptr + HALF + lo).to(tl.float32)
        else:
            wa = tl.load(kw_ptr + lo).to(tl.float32)
            wb = tl.load(kw_ptr + HALF + lo).to(tl.float32)
        rs = 1.0 / tl.sqrt((tl.sum(a * a, axis=0) + tl.sum(b * b, axis=0)) / HD + eps)
        a = (a * rs * wa).to(tl.bfloat16).to(tl.float32)
        b = (b * rs * wb).to(tl.bfloat16).to(tl.float32)
        if HAS_ROPE:
            pos = tl.load(pos_ptr + tok).to(tl.int64)
            cos = tl.load(cs_ptr + pos * HD + lo)
            sin = tl.load(cs_ptr + pos * HD + HALF + lo)
            a, b = a * cos - b * sin, b * cos + a * sin
        if is_q:
            dst = q_ptr + tok * (NQ * HD) + head * HD
        else:
            dst = k_ptr + tok * (NKV * HD) + (head - NQ) * HD
        tl.store(dst + lo, a.to(tl.bfloat16))
        tl.store(dst + HALF + lo, b.to(tl.bfloat16))
    else:
        vh = head - NQ - NKV
        offs = tl.arange(0, HD)
        v = tl.load(base + (NQ + NKV) * HD + vh * HD + offs)
        tl.store(v_ptr + tok * (NKV * HD) + vh * HD + offs, v)


def qk_norm_rope(qkv: torch.Tensor, positions: torch.Tensor, q_weight: torch.Tensor,
                 k_weight: torch.Tensor, eps: float, cos_sin: torch.Tensor | None,
                 num_heads: int, num_kv_heads: int, head_dim: int):
    """``(q, k, v)`` as contiguous bf16 [M, heads*head_dim] from fused ``qkv``."""
    m = qkv.shape[0]
    assert qkv.dtype == torch.bfloat16 and qkv.stride(-1) == 1
    assert qkv.shape[1] == (num_heads + 2 * num_kv_heads) * head_dim
    if cos_sin is not None:
        assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous()
        assert cos_sin.shape[1] == head_dim
    q = torch.empty((m, num_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    k = torch.empty((m, num_kv_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((m, num_kv_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
    if m:
        _qk_norm_rope_kernel[(m, num_heads + 2 * num_kv_heads)](
            qkv, positions, q_weight, k_weight,
            cos_sin if cos_sin is not None else qkv,
            q, k, v, qkv.stride(0), eps,
            NQ=num_heads, NKV=num_kv_heads, HD=head_dim,
            HAS_ROPE=cos_sin is not None, num_warps=1)
    return q, k, v
