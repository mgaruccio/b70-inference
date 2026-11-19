"""Small-batch W4A16 MoE for Flint on XPU (decode and speculative verify).

Reads the int4 expert weights in the layout ``XpuFusedMoe`` rewrites them to:
``w13`` int8 [E, 2I, H/2] and ``w2`` int8 [E, H, I/2], two signed 4-bit
two's-complement nibbles per byte (low nibble = lower K index), with bf16
group scales [E, N, K/64]. Dequant is ``s4 * scale``.

Two kernels replace remap -> grouped GEMM1 -> act -> grouped GEMM2 -> gather:

1. ``_moe_gate_up_act``: per (token, slot) and N block, fp32 gate/up GEMV with
   the group scale applied to fp32 partial sums; ``silu(g) * u`` kept in fp32.
2. ``_moe_down_sum``: per token and H block, fp32 down GEMV for each of the
   top-k experts, summed in slot order with the fp32 routing weights.

Every row is computed independently with a fixed reduction order, so results
are deterministic and identical for any batch size handled here.
"""

import torch

from vllm.triton_utils import tl, triton

GROUP = 64


@triton.jit
def _dot_s4_group(p_ptr, xe, xo, BN: tl.constexpr, ROW_BYTES: tl.constexpr,
                  rows, kb0):
    # rows: [BN] row indices; kb0: byte offset of the 64-wide K group.
    offs = tl.arange(0, 32)
    pk = tl.load(p_ptr + rows[:, None] * ROW_BYTES + kb0 + offs[None, :])
    lo = ((pk << 4).to(tl.int8) >> 4).to(tl.float32)
    hi = (pk >> 4).to(tl.float32)
    return tl.sum(lo * xe[None, :] + hi * xo[None, :], axis=1)


@triton.jit
def _moe_gate_up_act(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                     H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                     BN: tl.constexpr):
    pid = tl.program_id(0)           # token * TOPK + slot
    nb = tl.program_id(1)
    tok = pid // TOPK
    e = tl.load(ids_ptr + pid).to(tl.int64)
    G: tl.constexpr = H // 64
    ROW_BYTES: tl.constexpr = H // 2
    n = nb * BN + tl.arange(0, BN)
    wbase = w13_ptr + e * (2 * I * ROW_BYTES)
    sbase = s13_ptr + e * (2 * I * G)
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    xo2 = tl.arange(0, 32) * 2
    for g in range(G):
        xe = tl.load(x_ptr + tok * H + g * 64 + xo2).to(tl.float32)
        xo = tl.load(x_ptr + tok * H + g * 64 + xo2 + 1).to(tl.float32)
        pg = _dot_s4_group(wbase, xe, xo, BN, ROW_BYTES, n, g * 32)
        pu = _dot_s4_group(wbase, xe, xo, BN, ROW_BYTES, n + I, g * 32)
        acc_g += pg * tl.load(sbase + n * G + g).to(tl.float32)
        acc_u += pu * tl.load(sbase + (n + I) * G + g).to(tl.float32)
    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    tl.store(act_ptr + pid * I + n, act)


@triton.jit
def _moe_down_sum(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, out_ptr,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                  BN: tl.constexpr):
    tok = tl.program_id(0)
    nb = tl.program_id(1)
    G: tl.constexpr = I // 64
    ROW_BYTES: tl.constexpr = I // 2
    n = nb * BN + tl.arange(0, BN)
    xo2 = tl.arange(0, 32) * 2
    acc = tl.zeros((BN,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        slot = tok * TOPK + j
        e = tl.load(ids_ptr + slot).to(tl.int64)
        w = tl.load(wts_ptr + slot)
        wbase = w2_ptr + e * (H * ROW_BYTES)
        sbase = s2_ptr + e * (H * G)
        part = tl.zeros((BN,), dtype=tl.float32)
        for g in range(G):
            ae = tl.load(act_ptr + slot * I + g * 64 + xo2)
            ao = tl.load(act_ptr + slot * I + g * 64 + xo2 + 1)
            p = _dot_s4_group(wbase, ae, ao, BN, ROW_BYTES, n, g * 32)
            part += p * tl.load(sbase + n * G + g).to(tl.float32)
        acc += w * part
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


def moe_decode(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
               w13: torch.Tensor, s13: torch.Tensor, w2: torch.Tensor,
               s2: torch.Tensor) -> torch.Tensor:
    """Routed-expert output (bf16 [M, H]) for small ``M``."""
    M, H = x.shape
    E, two_i, hb = w13.shape
    I = two_i // 2
    TOPK = topk_ids.shape[1]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and hb * 2 == H
    assert w13.dtype == w2.dtype == torch.int8 and w13.is_contiguous() and w2.is_contiguous()
    assert w2.shape == (E, H, I // 2) and s13.shape == (E, 2 * I, H // GROUP)
    assert s2.shape == (E, H, I // GROUP) and s13.is_contiguous() and s2.is_contiguous()
    ids = topk_ids.to(torch.int32).contiguous()
    wts = topk_weights.to(torch.float32).contiguous()
    act = torch.empty((M * TOPK, I), dtype=torch.float32, device=x.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    if M:
        BN1, BN2 = 16, 32
        _moe_gate_up_act[(M * TOPK, I // BN1)](x, ids, w13, s13, act, H=H, I=I,
                                               TOPK=TOPK, BN=BN1, num_warps=2)
        _moe_down_sum[(M, H // BN2)](act, ids, wts, w2, s2, out, H=H, I=I,
                                     TOPK=TOPK, BN=BN2, num_warps=2)
    return out


@triton.jit
def _dot_s4_groups(p_ptr, xe, xo, ROW_BYTES: tl.constexpr, rows, kb0,
                   BN: tl.constexpr, GPB: tl.constexpr):
    # GPB consecutive 64-wide groups: returns [BN, GPB] fp32 partial dots.
    offs = tl.arange(0, 32 * GPB)
    pk = tl.load(p_ptr + rows[:, None] * ROW_BYTES + kb0 + offs[None, :])
    lo = ((pk << 4).to(tl.int8) >> 4).to(tl.float32)
    hi = (pk >> 4).to(tl.float32)
    prod = lo * xe[None, :] + hi * xo[None, :]
    return tl.sum(tl.reshape(prod, (BN, GPB, 32)), axis=2)


@triton.jit
def _moe_gate_up_act_v2(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                        BN: tl.constexpr, GPB: tl.constexpr):
    pid = tl.program_id(0)
    nb = tl.program_id(1)
    tok = pid // TOPK
    e = tl.load(ids_ptr + pid).to(tl.int64)
    G: tl.constexpr = H // 64
    ROW_BYTES: tl.constexpr = H // 2
    n = nb * BN + tl.arange(0, BN)
    wbase = w13_ptr + e * (2 * I * ROW_BYTES)
    sbase = s13_ptr + e * (2 * I * G)
    gi = tl.arange(0, GPB)
    xo2 = tl.arange(0, 32 * GPB) * 2
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    for g0 in range(0, G, GPB):
        xe = tl.load(x_ptr + tok * H + g0 * 64 + xo2).to(tl.float32)
        xo = tl.load(x_ptr + tok * H + g0 * 64 + xo2 + 1).to(tl.float32)
        pg = _dot_s4_groups(wbase, xe, xo, ROW_BYTES, n, g0 * 32, BN, GPB)
        pu = _dot_s4_groups(wbase, xe, xo, ROW_BYTES, n + I, g0 * 32, BN, GPB)
        sg = tl.load(sbase + n[:, None] * G + g0 + gi[None, :]).to(tl.float32)
        su = tl.load(sbase + (n + I)[:, None] * G + g0 + gi[None, :]).to(tl.float32)
        acc_g += tl.sum(pg * sg, axis=1)
        acc_u += tl.sum(pu * su, axis=1)
    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    tl.store(act_ptr + pid * I + n, act)


@triton.jit
def _moe_down_sum_v2(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, out_ptr,
                     H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                     BN: tl.constexpr, GPB: tl.constexpr):
    tok = tl.program_id(0)
    nb = tl.program_id(1)
    G: tl.constexpr = I // 64
    ROW_BYTES: tl.constexpr = I // 2
    n = nb * BN + tl.arange(0, BN)
    gi = tl.arange(0, GPB)
    xo2 = tl.arange(0, 32 * GPB) * 2
    acc = tl.zeros((BN,), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        slot = tok * TOPK + j
        e = tl.load(ids_ptr + slot).to(tl.int64)
        w = tl.load(wts_ptr + slot)
        wbase = w2_ptr + e * (H * ROW_BYTES)
        sbase = s2_ptr + e * (H * G)
        part = tl.zeros((BN,), dtype=tl.float32)
        for g0 in range(0, G, GPB):
            ae = tl.load(act_ptr + slot * I + g0 * 64 + xo2)
            ao = tl.load(act_ptr + slot * I + g0 * 64 + xo2 + 1)
            p = _dot_s4_groups(wbase, ae, ao, ROW_BYTES, n, g0 * 32, BN, GPB)
            sc = tl.load(sbase + n[:, None] * G + g0 + gi[None, :]).to(tl.float32)
            part += tl.sum(p * sc, axis=1)
        acc += w * part
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


def moe_decode_v2(x, topk_ids, topk_weights, w13, s13, w2, s2,
                  BN1=16, GPB1=16, W1=4, BN2=16, GPB2=8, W2=2):
    """Tuned M=1 path on the B70 (graph replay: gate_up ~27.5 us, down ~13.2 us)."""
    M, H = x.shape
    E, two_i, hb = w13.shape
    I = two_i // 2
    TOPK = topk_ids.shape[1]
    ids = topk_ids.to(torch.int32).contiguous()
    wts = topk_weights.to(torch.float32).contiguous()
    act = torch.empty((M * TOPK, I), dtype=torch.float32, device=x.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    if M:
        _moe_gate_up_act_v2[(M * TOPK, I // BN1)](x, ids, w13, s13, act, H=H, I=I, TOPK=TOPK,
                                                  BN=BN1, GPB=GPB1, num_warps=W1)
        _moe_down_sum_v2[(M, H // BN2)](act, ids, wts, w2, s2, out, H=H, I=I, TOPK=TOPK,
                                        BN=BN2, GPB=GPB2, num_warps=W2)
    return out
