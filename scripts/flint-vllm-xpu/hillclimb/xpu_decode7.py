"""MoE down (+ shared pseudo-experts) variants for the B70."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _nib(pk, i: tl.constexpr):
    return ((pk << (28 - 4 * i)) >> 28).to(tl.float32)


@triton.jit
def _moe_down_g2(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, w2s_ptr, s2s_ptr, out_ptr,
                 H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, NSH: tl.constexpr):
    """One output row per program; all TOPK + NSH expert rows fetched as one [TA, RW]
    gathered tile (row r = slot r), then a fixed-order reduction."""
    n = tl.program_id(0).to(tl.int64)
    tok = tl.program_id(1)
    TA: tl.constexpr = TOPK + NSH
    T2: tl.constexpr = 8
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    r = tl.arange(0, T2)
    routed = r < TOPK
    valid = r < TA
    e = tl.load(ids_ptr + tok * TOPK + r, mask=routed, other=0).to(tl.int64)
    w = tl.where(routed, tl.load(wts_ptr + tok * TOPK + r, mask=routed, other=0.0),
                 tl.where(valid, 1.0, 0.0))
    p = tl.where(routed, 0, r - TOPK).to(tl.int64)
    wo = tl.arange(0, RW)
    gi = tl.arange(0, G)
    wrow = tl.where(routed[:, None], w2_ptr + (e * H + n)[:, None] * RW,
                    w2s_ptr + (p * H + n)[:, None] * RW)
    srow = tl.where(routed[:, None], s2_ptr + (e * H + n)[:, None] * G,
                    s2s_ptr + (p * H + n)[:, None] * G)
    pk = tl.load(wrow + wo[None, :], mask=valid[:, None], other=0)
    sc = tl.load(srow + gi[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
    ab = act_ptr + (tok * TA + r)[:, None] * I + wo[None, :] * 8
    t = _nib(pk, 0) * tl.load(ab, mask=valid[:, None], other=0.0)
    for i in tl.static_range(1, 8):
        t += _nib(pk, i) * tl.load(ab + i, mask=valid[:, None], other=0.0)
    per = tl.sum(tl.sum(tl.reshape(t, (T2, G, 8)), axis=2) * sc, axis=1)   # [T2]
    y = tl.sum(per * w, axis=0)
    tl.store(out_ptr + tok * H + n, y.to(tl.bfloat16))


@triton.jit
def _down_one(acc, act_ptr, wbase, sbase, w, n, I: tl.constexpr, BN: tl.constexpr):
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    wo = tl.arange(0, RW)
    gi = tl.arange(0, G)
    pk = tl.load(wbase + n[:, None] * RW + wo[None, :])
    sc = tl.load(sbase + n[:, None] * G + gi[None, :]).to(tl.float32)
    part = _nib(pk, 0) * tl.load(act_ptr + wo * 8)
    for i in tl.static_range(1, 8):
        part += _nib(pk, i) * tl.load(act_ptr + wo * 8 + i)
    return acc + w * tl.sum(tl.sum(tl.reshape(part, (BN, G, 8)), axis=2) * sc, axis=1)


@triton.jit
def _moe_down_pre(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, w2s_ptr, s2s_ptr, out_ptr,
                  H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                  NSH: tl.constexpr, BN: tl.constexpr):
    """Current loop structure, but all ids / weights loaded up front as one vector."""
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    TA: tl.constexpr = TOPK + NSH
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    r = tl.arange(0, 8)
    ev = tl.load(ids_ptr + tok * TOPK + r, mask=r < TOPK, other=0).to(tl.int64)
    wv = tl.load(wts_ptr + tok * TOPK + r, mask=r < TOPK, other=0.0)
    n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for jj in tl.static_range(TOPK):
        e = tl.sum(tl.where(r == jj, ev, 0), axis=0)
        w = tl.sum(tl.where(r == jj, wv, 0.0), axis=0)
        acc = _down_one(acc, act_ptr + (tok * TA + jj) * I, w2_ptr + e * (H * RW),
                        s2_ptr + e * (H * G), w, n, I, BN)
    for p in tl.static_range(NSH):
        acc = _down_one(acc, act_ptr + (tok * TA + TOPK + p) * I, w2s_ptr + p * (H * RW),
                        s2s_ptr + p * (H * G), 1.0, n, I, BN)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))
