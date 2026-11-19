"""Deterministic, batch-invariant Flint router GEMM for XPU.

The reference router computes ``F.linear(x.float(), w.float())``. Flint's router
weights are bf16 in the checkpoint and the router input is the bf16 output of
the post-attention RMSNorm, so bf16 x bf16 products accumulated in fp32 match
the reference. Fixed tiles and a fixed K loop make every row independent of
the batch size and of run-to-run scheduling (oneDNN's fp32 GEMM is neither).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _router_kernel(x_ptr, w_ptr, o_ptr, M,
                   N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :],
                    mask=rm[:, None] < M, other=0.0)
        b = tl.load(w_ptr + rn[None, :] * K + (k0 + rk)[:, None])
        acc += tl.dot(a, b, out_dtype=tl.float32)
    tl.store(o_ptr + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)


def flint_router_logits(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp32 router logits ``x @ w.T`` for bf16 ``x`` [M, K] and ``w`` [N, K]."""
    assert x.dtype == w.dtype == torch.bfloat16 and w.is_contiguous()
    # The tested kernel indexes rows with stride K. (Passing the row stride as an
    # extra runtime argument faulted on the B70 with the XPU Triton backend.)
    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]
    BM, BN, BK = 16, 16, 64
    assert K % BK == 0 and N % BN == 0
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    if M:
        _router_kernel[(triton.cdiv(M, BM), N // BN)](
            x, w, out, M, N, K, BM, BN, BK)
    return out
