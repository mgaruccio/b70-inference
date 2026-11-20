"""Tensor-descriptor (2D block load) GEMV variants: GPTQ dense and MoE gate_up (+shared)."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _nib(pk, i: tl.constexpr):
    return ((pk << (28 - 4 * i)) >> 28).to(tl.float32)


@triton.jit
def _gptq_gemv_bp(xp_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                  BN: tl.constexpr, NW: tl.constexpr):
    """y = x @ W^T, GPTQ uint4b8 W [N, K/8] int32, scales [K/64, N]; x permuted
    nibble-major (xp[i * K/8 + j] = x[8j + i]); weight tile via 2D block loads."""
    nb = tl.program_id(0)
    RW: tl.constexpr = K // 8
    GI: tl.constexpr = NW // 8
    n0 = nb * BN
    n = n0 + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    wd = tl.make_tensor_descriptor(w_ptr, shape=[N, RW], strides=[RW, 1], block_shape=[BN, NW])
    acc = tl.zeros((BN,), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        pk = wd.load([n0, w0]) ^ -2004318072
        sc = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + n[:, None]).to(tl.float32)
        t = _nib(pk, 0) * tl.load(xp_ptr + w0 + wo).to(tl.float32)[None, :]
        for i in tl.static_range(1, 8):
            t += _nib(pk, i) * tl.load(xp_ptr + i * RW + w0 + wo).to(tl.float32)[None, :]
        acc += tl.sum(tl.sum(tl.reshape(t, (BN, GI, 8)), axis=2) * sc, axis=1)
    tl.store(o_ptr + n, acc.to(tl.bfloat16))


@triton.jit
def _gptq_gemv_pp(xp_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                  BN: tl.constexpr, NW: tl.constexpr):
    """Same as _gptq_gemv_bp with plain pointer loads (control)."""
    nb = tl.program_id(0)
    RW: tl.constexpr = K // 8
    GI: tl.constexpr = NW // 8
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        pk = tl.load(w_ptr + n[:, None] * RW + w0 + wo[None, :]) ^ -2004318072
        sc = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + n[:, None]).to(tl.float32)
        t = _nib(pk, 0) * tl.load(xp_ptr + w0 + wo).to(tl.float32)[None, :]
        for i in tl.static_range(1, 8):
            t += _nib(pk, i) * tl.load(xp_ptr + i * RW + w0 + wo).to(tl.float32)[None, :]
        acc += tl.sum(tl.sum(tl.reshape(t, (BN, GI, 8)), axis=2) * sc, axis=1)
    tl.store(o_ptr + n, acc.to(tl.bfloat16))


@triton.jit
def _moe_gate_up_bp(xp_ptr, ids_ptr, w13_ptr, s13_ptr, w13s_ptr, s13s_ptr, act_ptr,
                    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                    NSH: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr):
    nb = tl.program_id(0)
    pid = tl.program_id(1)
    TA: tl.constexpr = TOPK + NSH
    tok = pid // TA
    slot = pid % TA
    RW: tl.constexpr = H // 8
    G: tl.constexpr = H // 64
    GI: tl.constexpr = NW // 8
    if slot < TOPK:
        e = tl.load(ids_ptr + tok * TOPK + slot).to(tl.int64)
        wbase = w13_ptr + e * (2 * I * RW)
        sbase = s13_ptr + e * (2 * I * G)
    else:
        p = (slot - TOPK).to(tl.int64)
        wbase = w13s_ptr + p * (2 * I * RW)
        sbase = s13s_ptr + p * (2 * I * G)
    n0 = nb * BN
    n = n0 + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    wd = tl.make_tensor_descriptor(wbase, shape=[2 * I, RW], strides=[RW, 1], block_shape=[BN, NW])
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    xb = xp_ptr + tok * H
    for w0 in range(0, RW, NW):
        pg = wd.load([n0, w0])
        pu = wd.load([n0 + I, w0])
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
    tl.store(act_ptr + pid * I + n, act)
