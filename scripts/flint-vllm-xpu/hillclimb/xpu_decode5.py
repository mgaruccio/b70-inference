"""GEMV core v4: unsigned-nibble unpack ((w >> 4i) & 15), -8 offset folded per word,
per-lane [BN, NW] fp32 accumulation with one cross-lane reduction at the end.

Word layout everywhere: nibble i of int32 word j holds k = 8j + i.
GPTQ uint4b8: u = nibble (value u - 8). Flint MoE signed nibbles: u = nibble ^ 8.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _wdot(pk, x2, xw, XOR8: tl.constexpr):
    """pk int32 [BN, NW]; x2 fp32 [NW, 8] (x[8j + i]); xw fp32 [NW] = sum_i x2.
    Returns fp32 [BN, NW]: sum_i (u_i - 8) x[8j + i]."""
    if XOR8:
        pk = pk ^ -2004318072
    sh = 4 * tl.arange(0, 8)
    u = ((pk[:, :, None] >> sh[None, None, :]) & 15).to(tl.float32)
    return tl.sum(u * x2[None, :, :], axis=2) - 8.0 * xw[None, :]


@triton.jit
def _xtile(x_ptr, k0, NW: tl.constexpr):
    wo = tl.arange(0, NW)
    i8 = tl.arange(0, 8)
    x2 = tl.load(x_ptr + k0 + wo[:, None] * 8 + i8[None, :]).to(tl.float32)
    return x2, tl.sum(x2, axis=1)


@triton.jit
def _gptq_gemv4(x_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, NW: tl.constexpr, SILU_MUL: tl.constexpr):
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = K // 8
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    acc = tl.zeros((BN, NW), dtype=tl.float32)
    if SILU_MUL:
        HN: tl.constexpr = N // 2
        acc_u = tl.zeros((BN, NW), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        x2, xw = _xtile(x_ptr + tok * K, w0 * 8, NW)
        g = (w0 + wo) // 8
        pk = tl.load(w_ptr + n[:, None] * RW + w0 + wo[None, :])
        sc = tl.load(s_ptr + g[None, :] * N + n[:, None]).to(tl.float32)
        acc += _wdot(pk, x2, xw, False) * sc
        if SILU_MUL:
            pu = tl.load(w_ptr + (n + HN)[:, None] * RW + w0 + wo[None, :])
            su = tl.load(s_ptr + g[None, :] * N + (n + HN)[:, None]).to(tl.float32)
            acc_u += _wdot(pu, x2, xw, False) * su
    y = tl.sum(acc, axis=1)
    if SILU_MUL:
        yu = tl.sum(acc_u, axis=1)
        tl.store(o_ptr + tok * (N // 2) + n, (y / (1.0 + tl.exp(-y)) * yu).to(tl.bfloat16))
    else:
        tl.store(o_ptr + tok * N + n, y.to(tl.bfloat16))


@triton.jit
def _moe_gate_up4(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                  BN: tl.constexpr, NW: tl.constexpr):
    nb = tl.program_id(0)
    pid = tl.program_id(1)
    tok = pid // TOPK
    e = tl.load(ids_ptr + pid).to(tl.int64)
    RW: tl.constexpr = H // 8
    G: tl.constexpr = H // 64
    n = nb * BN + tl.arange(0, BN)
    wbase = w13_ptr + e * (2 * I * RW)
    sbase = s13_ptr + e * (2 * I * G)
    wo = tl.arange(0, NW)
    acc_g = tl.zeros((BN, NW), dtype=tl.float32)
    acc_u = tl.zeros((BN, NW), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        x2, xw = _xtile(x_ptr + tok * H, w0 * 8, NW)
        g = (w0 + wo) // 8
        pg = tl.load(wbase + n[:, None] * RW + w0 + wo[None, :])
        pu = tl.load(wbase + (n + I)[:, None] * RW + w0 + wo[None, :])
        sg = tl.load(sbase + n[:, None] * G + g[None, :]).to(tl.float32)
        su = tl.load(sbase + (n + I)[:, None] * G + g[None, :]).to(tl.float32)
        acc_g += _wdot(pg, x2, xw, True) * sg
        acc_u += _wdot(pu, x2, xw, True) * su
    yg = tl.sum(acc_g, axis=1)
    yu = tl.sum(acc_u, axis=1)
    tl.store(act_ptr + pid * I + n, yg / (1.0 + tl.exp(-yg)) * yu)


@triton.jit
def _moe_down4(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
               H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
               BN: tl.constexpr, ADD_SHARED: tl.constexpr):
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, RW)
    g = wo // 8
    acc = tl.zeros((BN, RW), dtype=tl.float32)
    for jj in tl.static_range(TOPK):
        slot = tok * TOPK + jj
        e = tl.load(ids_ptr + slot).to(tl.int64)
        w = tl.load(wts_ptr + slot)
        pk = tl.load(w2_ptr + e * (H * RW) + n[:, None] * RW + wo[None, :])
        sc = tl.load(s2_ptr + e * (H * G) + n[:, None] * G + g[None, :]).to(tl.float32)
        x2, xw = _xtile(act_ptr + slot * I, 0, RW)
        acc += _wdot(pk, x2, xw, True) * (sc * w)
    y = tl.sum(acc, axis=1)
    if ADD_SHARED:
        y += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, y.to(tl.bfloat16))
