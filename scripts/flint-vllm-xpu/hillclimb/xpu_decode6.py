"""MoE GEMV v5: inputs in nibble-major ("permuted") order so x loads are contiguous
and the inner loop is an FMA chain.

For a K-long input with RW = K / 8 words per row, the permuted vector is
xp[i * RW + j] = x[8 j + i] (nibble i of word j).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _permute_x(x_ptr, xp_ptr, K: tl.constexpr):
    tok = tl.program_id(0)
    RW: tl.constexpr = K // 8
    j = tl.arange(0, RW)
    i8 = tl.arange(0, 8)
    v = tl.load(x_ptr + tok * K + j[:, None] * 8 + i8[None, :])
    tl.store(xp_ptr + tok * K + i8[None, :] * RW + j[:, None], v)


@triton.jit
def _nib(pk, i: tl.constexpr):
    return ((pk << (28 - 4 * i)) >> 28).to(tl.float32)


@triton.jit
def _moe_gate_up5(xp_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                  BN: tl.constexpr, NW: tl.constexpr, PERM_OUT: tl.constexpr):
    nb = tl.program_id(0)
    pid = tl.program_id(1)
    tok = pid // TOPK
    e = tl.load(ids_ptr + pid).to(tl.int64)
    RW: tl.constexpr = H // 8
    G: tl.constexpr = H // 64
    GI: tl.constexpr = NW // 8
    n = nb * BN + tl.arange(0, BN)
    wbase = w13_ptr + e * (2 * I * RW)
    sbase = s13_ptr + e * (2 * I * G)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    xb = xp_ptr + tok * H
    for w0 in range(0, RW, NW):
        pg = tl.load(wbase + n[:, None] * RW + w0 + wo[None, :])
        pu = tl.load(wbase + (n + I)[:, None] * RW + w0 + wo[None, :])
        sg = tl.load(sbase + n[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        su = tl.load(sbase + (n + I)[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        x0 = tl.load(xb + w0 + wo).to(tl.float32)
        tg = _nib(pg, 0) * x0[None, :]
        tu = _nib(pu, 0) * x0[None, :]
        for i in tl.static_range(1, 8):
            xi = tl.load(xb + i * RW + w0 + wo).to(tl.float32)
            tg += _nib(pg, i) * xi[None, :]
            tu += _nib(pu, i) * xi[None, :]
        acc_g += tl.sum(tl.sum(tl.reshape(tg, (BN, GI, 8)), axis=2) * sg, axis=1)
        acc_u += tl.sum(tl.sum(tl.reshape(tu, (BN, GI, 8)), axis=2) * su, axis=1)
    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    if PERM_OUT:
        RWI: tl.constexpr = I // 8
        tl.store(act_ptr + pid * I + (n % 8) * RWI + n // 8, act)
    else:
        tl.store(act_ptr + pid * I + n, act)


@triton.jit
def _moe_down5(actp_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
               H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
               BN: tl.constexpr, ADD_SHARED: tl.constexpr):
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, RW)
    gi = tl.arange(0, G)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for jj in tl.static_range(TOPK):
        slot = tok * TOPK + jj
        e = tl.load(ids_ptr + slot).to(tl.int64)
        w = tl.load(wts_ptr + slot)
        pk = tl.load(w2_ptr + e * (H * RW) + n[:, None] * RW + wo[None, :])
        sc = tl.load(s2_ptr + e * (H * G) + n[:, None] * G + gi[None, :]).to(tl.float32)
        ab = actp_ptr + slot * I
        t = _nib(pk, 0) * tl.load(ab + wo)[None, :]
        for i in tl.static_range(1, 8):
            t += _nib(pk, i) * tl.load(ab + i * RW + wo)[None, :]
        acc += w * tl.sum(tl.sum(tl.reshape(t, (BN, G, 8)), axis=2) * sc, axis=1)
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))
