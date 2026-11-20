"""Word-load W4A16 decode GEMV kernels (int32 = 8 nibbles; nibble i <-> k = 8j + i).

The MoE layout (int8 [E, N, K/2], low nibble = lower k) viewed as int32 [E, N, K/8]
has nibble i of word j at k = 8j + i, the same order as GPTQ int32 packing, so one
core serves both. ``XOR8`` converts GPTQ uint4b8 (u - 8) to signed nibbles.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _nib(pk, i: tl.constexpr):
    return ((pk << (28 - 4 * i)) >> 28).to(tl.float32)


@triton.jit
def _gemv_part(pk, x_ptr, k0, NW: tl.constexpr):
    """pk: int32 [..., NW] words covering k0 .. k0 + 8 NW; returns fp32 [..., NW]."""
    wo = tl.arange(0, NW)
    part = _nib(pk, 0) * tl.load(x_ptr + k0 + wo * 8).to(tl.float32)
    for i in tl.static_range(1, 8):
        part += _nib(pk, i) * tl.load(x_ptr + k0 + wo * 8 + i).to(tl.float32)
    return part


@triton.jit
def _moe_gate_up_w(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                   H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                   BN: tl.constexpr, NW: tl.constexpr):
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
    for w0 in range(0, RW, NW):
        pg = tl.load(wbase + n[:, None] * RW + w0 + wo[None, :])
        pu = tl.load(wbase + (n + I)[:, None] * RW + w0 + wo[None, :])
        sg = tl.load(sbase + n[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        su = tl.load(sbase + (n + I)[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        xg = _gemv_part(pg, x_ptr + tok * H, w0 * 8, NW)
        xu = _gemv_part(pu, x_ptr + tok * H, w0 * 8, NW)
        acc_g += tl.sum(tl.sum(tl.reshape(xg, (BN, GI, 8)), axis=2) * sg, axis=1)
        acc_u += tl.sum(tl.sum(tl.reshape(xu, (BN, GI, 8)), axis=2) * su, axis=1)
    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    tl.store(act_ptr + pid * I + n, act)


@triton.jit
def _moe_down_w(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
                H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                BN: tl.constexpr, ADD_SHARED: tl.constexpr):
    """All TOPK experts in one program: one gathered [TK2, BN, I/8] word load."""
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    TK2: tl.constexpr = 8
    j = tl.arange(0, TK2)
    jm = j < TOPK
    e = tl.load(ids_ptr + tok * TOPK + j, mask=jm, other=0).to(tl.int64)
    wt = tl.load(wts_ptr + tok * TOPK + j, mask=jm, other=0.0)
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, RW)
    gi = tl.arange(0, G)
    m3 = jm[:, None, None]
    pk = tl.load(w2_ptr + e[:, None, None] * (H * RW) + n[None, :, None] * RW + wo[None, None, :],
                 mask=m3, other=0)
    sc = tl.load(s2_ptr + e[:, None, None] * (H * G) + n[None, :, None] * G + gi[None, None, :],
                 mask=m3, other=0.0).to(tl.float32)
    abase = act_ptr + (tok * TOPK + j)[:, None] * I
    part = _nib(pk, 0) * tl.load(abase + wo[None, :] * 8, mask=jm[:, None], other=0.0)[:, None, :]
    for i in tl.static_range(1, 8):
        part += _nib(pk, i) * tl.load(abase + wo[None, :] * 8 + i, mask=jm[:, None],
                                      other=0.0)[:, None, :]
    gsum = tl.sum(tl.reshape(part, (TK2, BN, G, 8)), axis=3)
    per = tl.sum(gsum * sc, axis=2)                     # [TK2, BN]
    acc = tl.sum(per * wt[:, None], axis=0)
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


@triton.jit
def _moe_down_wl(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
                 H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                 BN: tl.constexpr, ADD_SHARED: tl.constexpr):
    """Per-expert loop variant (static unroll) with word loads."""
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
        part = _gemv_part(pk, act_ptr + slot * I, 0, RW)
        acc += w * tl.sum(tl.sum(tl.reshape(part, (BN, G, 8)), axis=2) * sc, axis=1)
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


@triton.jit
def _router_topk1(p_ptr, logit_ptr, ids_ptr, wts_ptr,
                  N: tl.constexpr, KS: tl.constexpr, TOPK: tl.constexpr,
                  TK2: tl.constexpr):
    """Same semantics as xpu_decode3._router_topk (launch with num_warps=1)."""
    row = tl.program_id(0)
    rn = tl.arange(0, N)
    lg = tl.load(p_ptr + row * KS * N + rn)
    for ks in tl.static_range(1, KS):
        lg += tl.load(p_ptr + (row * KS + ks) * N + rn)
    tl.store(logit_ptr + row * N + rn, lg)
    key = tl.where(lg == lg, lg, float('-inf'))
    taken = rn < 0
    j8 = tl.arange(0, TK2)
    ids = tl.zeros((TK2,), dtype=tl.int32)
    sc = tl.full((TK2,), float('-inf'), dtype=tl.float32)
    for j in tl.static_range(TOPK):
        kk = tl.where(taken, float('-inf'), key)
        mx = tl.max(kk, axis=0)
        idx = tl.min(tl.where((kk == mx) & (~taken), rn, N), axis=0)
        taken = taken | (rn == idx)
        ids = tl.where(j8 == j, idx, ids)
        sc = tl.where(j8 == j, mx, sc)
    s = tl.where(j8 < TOPK, 1.0 / (1.0 + tl.exp(-sc)), 0.0)
    tot = tl.sum(s, axis=0)
    w = tl.where(tot > 0, s / tot, 0.0)
    tl.store(ids_ptr + row * TOPK + j8, ids, mask=j8 < TOPK)
    tl.store(wts_ptr + row * TOPK + j8, w, mask=j8 < TOPK)


@triton.jit
def _gptq_gemv(x_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
               BN: tl.constexpr, NW: tl.constexpr, SILU_MUL: tl.constexpr):
    """y = x @ W^T for GPTQ sym uint4b8 W stored [N, K/8] int32 (nibble i of word j
    is k = 8j + i) with scales [K/64, N]. SILU_MUL: rows [0, N/2) are gate and
    [N/2, N) up; writes silu(g) * u (bf16, N/2 wide)."""
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = K // 8
    GI: tl.constexpr = NW // 8
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    acc = tl.zeros((BN,), dtype=tl.float32)
    if SILU_MUL:
        HN: tl.constexpr = N // 2
        acc_u = tl.zeros((BN,), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        pk = tl.load(w_ptr + n[:, None] * RW + w0 + wo[None, :]) ^ -2004318072
        sc = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + n[:, None]).to(tl.float32)
        part = _gemv_part(pk, x_ptr + tok * K, w0 * 8, NW)
        acc += tl.sum(tl.sum(tl.reshape(part, (BN, GI, 8)), axis=2) * sc, axis=1)
        if SILU_MUL:
            pu = tl.load(w_ptr + (n + HN)[:, None] * RW + w0 + wo[None, :]) ^ -2004318072
            su = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + (n + HN)[:, None]).to(tl.float32)
            partu = _gemv_part(pu, x_ptr + tok * K, w0 * 8, NW)
            acc_u += tl.sum(tl.sum(tl.reshape(partu, (BN, GI, 8)), axis=2) * su, axis=1)
    if SILU_MUL:
        y = acc / (1.0 + tl.exp(-acc)) * acc_u
        tl.store(o_ptr + tok * (N // 2) + n, y.to(tl.bfloat16))
    else:
        tl.store(o_ptr + tok * N + n, acc.to(tl.bfloat16))


def gptq_gemv(x, qw, sc, silu_mul=False, BN=4, NW=64, W=1):
    """x bf16 [M, K]; qw int32 [N, K/8] contiguous; sc [K/64, N] contiguous."""
    M, K = x.shape
    N = qw.shape[0]
    assert qw.dtype == torch.int32 and qw.is_contiguous() and qw.shape[1] * 8 == K
    assert sc.shape == (K // 64, N) and sc.is_contiguous()
    x = x.contiguous()
    rows = N // 2 if silu_mul else N
    out = torch.empty(M, rows, device=x.device, dtype=torch.bfloat16)
    NW = min(NW, K // 8)
    _gptq_gemv[(rows // BN, M)](x, qw, sc, out, N=N, K=K, BN=BN, NW=NW, SILU_MUL=silu_mul,
                                num_warps=W)
    return out


@triton.jit
def _dot3(pk, x_ptr, k0, BN: tl.constexpr, NW: tl.constexpr, XOR8: tl.constexpr):
    """pk int32 [BN, NW] words for k0 .. k0 + 8 NW; returns fp32 [BN, NW // 8] group dots
    (64-wide groups). x is read once as a contiguous [NW, 8] tile."""
    i8 = tl.arange(0, 8)
    wo = tl.arange(0, NW)
    x2 = tl.load(x_ptr + k0 + wo[:, None] * 8 + i8[None, :]).to(tl.float32)
    if XOR8:
        pk = pk ^ -2004318072
    nib = ((pk[:, :, None] << (28 - 4 * i8)[None, None, :]) >> 28).to(tl.float32)
    prod = nib * x2[None, :, :]
    return tl.sum(tl.reshape(prod, (BN, NW // 8, 64)), axis=2)


@triton.jit
def _moe_gate_up_3(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                   H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                   BN: tl.constexpr, NW: tl.constexpr):
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
    for w0 in range(0, RW, NW):
        pg = tl.load(wbase + n[:, None] * RW + w0 + wo[None, :])
        pu = tl.load(wbase + (n + I)[:, None] * RW + w0 + wo[None, :])
        sg = tl.load(sbase + n[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        su = tl.load(sbase + (n + I)[:, None] * G + w0 // 8 + gi[None, :]).to(tl.float32)
        acc_g += tl.sum(_dot3(pg, x_ptr + tok * H, w0 * 8, BN, NW, False) * sg, axis=1)
        acc_u += tl.sum(_dot3(pu, x_ptr + tok * H, w0 * 8, BN, NW, False) * su, axis=1)
    act = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
    tl.store(act_ptr + pid * I + n, act)


@triton.jit
def _moe_down_3(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
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
        acc += w * tl.sum(_dot3(pk, act_ptr + slot * I, 0, BN, RW, False) * sc, axis=1)
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


@triton.jit
def _gptq_gemv3(x_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                BN: tl.constexpr, NW: tl.constexpr, SILU_MUL: tl.constexpr):
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = K // 8
    GI: tl.constexpr = NW // 8
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    acc = tl.zeros((BN,), dtype=tl.float32)
    if SILU_MUL:
        HN: tl.constexpr = N // 2
        acc_u = tl.zeros((BN,), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        pk = tl.load(w_ptr + n[:, None] * RW + w0 + wo[None, :])
        sc = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + n[:, None]).to(tl.float32)
        acc += tl.sum(_dot3(pk, x_ptr + tok * K, w0 * 8, BN, NW, True) * sc, axis=1)
        if SILU_MUL:
            pu = tl.load(w_ptr + (n + HN)[:, None] * RW + w0 + wo[None, :])
            su = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + (n + HN)[:, None]).to(tl.float32)
            acc_u += tl.sum(_dot3(pu, x_ptr + tok * K, w0 * 8, BN, NW, True) * su, axis=1)
    if SILU_MUL:
        y = acc / (1.0 + tl.exp(-acc)) * acc_u
        tl.store(o_ptr + tok * (N // 2) + n, y.to(tl.bfloat16))
    else:
        tl.store(o_ptr + tok * N + n, acc.to(tl.bfloat16))
