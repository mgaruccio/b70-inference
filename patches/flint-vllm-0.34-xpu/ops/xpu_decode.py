"""Flint XPU decode kernels (router + top-k, W4A16 MoE GEMVs) for the B70.

Router: ``_router_partial`` writes fp32 partial logits [M, KS, N] of bf16 x bf16
products. Every row runs the same code (fixed [BN, K/KS] tile, fixed reduction),
looping over at most RB rows per program, so logits are bitwise independent of
M. ``_router_topk`` sums the KS partials in fixed order, writes fp32 logits and
selects the top-k by logit (ties -> lower expert id; non-finite logits rank
last), weights sigmoid(l) / sum(sigmoid(l)). Ids are always valid and distinct.

MoE: the expert layout ``XpuFusedMoe`` produces (int8 [E, N, K/2], low nibble =
lower k, signed 4-bit) viewed as int32 [E, N, K/8] has nibble i of word j at
k = 8j + i. bf16 group scales [E, N, K/64]. fp32 accumulation with the group
scale applied to each 64-wide partial sum. ``_moe_down`` sums the top-k experts
in slot order with the fp32 routing weights and adds the shared-expert output in
fp32 before rounding to bf16. Each token is computed independently.
"""

import torch

from vllm.triton_utils import tl, triton

ROUTER_BN, ROUTER_KS, ROUTER_RB, ROUTER_W = 8, 4, 16, 4


@triton.jit
def _router_partial(x_ptr, w_ptr, p_ptr, M,
                    N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                    KS: tl.constexpr, RB: tl.constexpr):
    pn = tl.program_id(0)
    ks = tl.program_id(1)
    pm = tl.program_id(2)
    KC: tl.constexpr = K // KS
    rn = pn * BN + tl.arange(0, BN)
    rk = ks * KC + tl.arange(0, KC)
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
    m0 = pm * RB
    for m in range(m0, tl.minimum(m0 + RB, M)):
        x = tl.load(x_ptr + m * K + rk).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(p_ptr + (m * KS + ks) * N + rn, acc)


@triton.jit
def _router_topk(p_ptr, logit_ptr, ids_ptr, wts_ptr,
                 N: tl.constexpr, KS: tl.constexpr, TOPK: tl.constexpr,
                 TK2: tl.constexpr):
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


def router(x: torch.Tensor, w: torch.Tensor, topk: int):
    """(fp32 logits [M, N], int32 ids [M, topk], fp32 weights [M, topk]) for bf16
    ``x`` [M, K] and router weight ``w`` [N, K]."""
    assert x.dtype == w.dtype == torch.bfloat16 and w.is_contiguous()
    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]
    BN, KS, RB = ROUTER_BN, ROUTER_KS, ROUTER_RB
    assert K % KS == 0 and N % BN == 0 and N & (N - 1) == 0
    part = torch.empty(M, KS, N, device=x.device, dtype=torch.float32)
    logits = torch.empty(M, N, device=x.device, dtype=torch.float32)
    ids = torch.empty(M, topk, device=x.device, dtype=torch.int32)
    wts = torch.empty(M, topk, device=x.device, dtype=torch.float32)
    if M:
        _router_partial[(N // BN, KS, triton.cdiv(M, RB))](
            x, w, part, M, N=N, K=K, BN=BN, KS=KS, RB=RB, num_warps=ROUTER_W)
        _router_topk[(M,)](part, logits, ids, wts, N=N, KS=KS, TOPK=topk,
                           TK2=triton.next_power_of_2(topk), num_warps=1)
    return logits, ids, wts


@triton.jit
def _dot3(pk, x_ptr, k0, BN: tl.constexpr, NW: tl.constexpr, XOR8: tl.constexpr):
    """pk int32 [BN, NW] (k0 .. k0 + 8 NW) -> fp32 [BN, NW // 8] 64-wide group dots.
    x is read as one contiguous [NW, 8] tile."""
    i8 = tl.arange(0, 8)
    wo = tl.arange(0, NW)
    x2 = tl.load(x_ptr + k0 + wo[:, None] * 8 + i8[None, :]).to(tl.float32)
    if XOR8:
        pk = pk ^ -2004318072
    nib = ((pk[:, :, None] << (28 - 4 * i8)[None, None, :]) >> 28).to(tl.float32)
    return tl.sum(tl.reshape(nib * x2[None, :, :], (BN, NW // 8, 64)), axis=2)


@triton.jit
def _nib(pk, i: tl.constexpr):
    return ((pk << (28 - 4 * i)) >> 28).to(tl.float32)


@triton.jit
def _moe_gate_up(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                 H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                 BN: tl.constexpr, NW: tl.constexpr):
    nb = tl.program_id(0)
    pid = tl.program_id(1)            # token * TOPK + slot
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
def _moe_down(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
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
        abase = act_ptr + slot * I
        part = _nib(pk, 0) * tl.load(abase + wo * 8)
        for i in tl.static_range(1, 8):
            part += _nib(pk, i) * tl.load(abase + wo * 8 + i)
        acc += w * tl.sum(tl.sum(tl.reshape(part, (BN, G, 8)), axis=2) * sc, axis=1)
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


MOE_GATE_BN, MOE_GATE_NW, MOE_GATE_W, MOE_GATE_WS = 2, 64, 1, 16
MOE_DOWN_BN, MOE_DOWN_W = 1, 1


def moe_decode(x, ids, wts, w13, s13, w2, s2, shared=None):
    """Routed (+ shared, if given) expert output, bf16 [M, H]. ``ids`` int32 and
    ``wts`` fp32 [M, topk]; ``w13``/``w2`` int8 expert weights (XpuFusedMoe layout)."""
    M, H = x.shape
    E, two_i, hb = w13.shape
    I = two_i // 2
    TOPK = ids.shape[1]
    assert x.dtype == torch.bfloat16 and hb * 2 == H and w2.shape == (E, H, I // 2)
    assert w13.dtype == w2.dtype == torch.int8 and w13.is_contiguous() and w2.is_contiguous()
    assert s13.shape == (E, 2 * I, H // 64) and s2.shape == (E, H, I // 64)
    assert s13.is_contiguous() and s2.is_contiguous()
    assert ids.dtype == torch.int32 and wts.dtype == torch.float32
    x = x.contiguous()
    act = torch.empty((M * TOPK, I), dtype=torch.float32, device=x.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    if shared is not None:
        assert shared.shape == (M, H) and shared.dtype == torch.bfloat16
        shared = shared.contiguous()
    if M:
        _moe_gate_up[(I // MOE_GATE_BN, M * TOPK)](
            x, ids, w13.view(torch.int32), s13, act, H=H, I=I, TOPK=TOPK,
            BN=MOE_GATE_BN, NW=MOE_GATE_NW, num_warps=MOE_GATE_W, warp_size=MOE_GATE_WS)
        _moe_down[(H // MOE_DOWN_BN, M)](
            act, ids, wts, w2.view(torch.int32), s2, shared if shared is not None else out,
            out, H=H, I=I, TOPK=TOPK, BN=MOE_DOWN_BN, ADD_SHARED=shared is not None,
            num_warps=MOE_DOWN_W)
    return out


# ---------------------------------------------------------------------------
# Round 2: shared expert as NSH pseudo-experts inside the MoE kernels, and a
# gate_up GEMV that reads x in nibble-major order (contiguous loads, FMA chain).
#
# Permuted input: for K = H and RW = H / 8 words per row, xp[i * RW + j] = x[8 j + i].
# Shared expert (GPTQ uint4b8, intermediate IS = NSH * I) is repacked at load time
# into the routed layout as int32 words with signed nibbles (u ^ 8 == (u - 8) & 15):
#   w13s [NSH, 2I, H/8], s13s [NSH, 2I, H/64], w2s [NSH, H, I/8], s2s [NSH, H, I/64].
# Pseudo-expert p covers shared rows/columns [p I, (p + 1) I) and has weight 1.

_XOR8 = -2004318072  # 0x88888888 as int32


def pack_shared_expert(gu_qw, gu_sc, dn_qw, dn_sc, I):
    """Repack a GPTQ (uint4b8, group 64) shared MLP into NSH = IS / I pseudo-experts.
    gu_qw int32 [2 IS, H/8] (gate rows, then up rows), gu_sc [H/64, 2 IS];
    dn_qw int32 [H, IS/8], dn_sc [IS/64, H]."""
    two_is, hw = gu_qw.shape
    IS = two_is // 2
    H = hw * 8
    P = IS // I
    assert IS % I == 0 and I % 64 == 0 and dn_qw.shape == (H, IS // 8)
    assert gu_sc.shape == (H // 64, two_is) and dn_sc.shape == (IS // 64, H)
    g = gu_qw[:IS].reshape(P, I, hw)
    u = gu_qw[IS:].reshape(P, I, hw)
    w13 = (torch.cat([g, u], 1) ^ _XOR8).contiguous()
    st = gu_sc.t()
    s13 = torch.cat([st[:IS].reshape(P, I, H // 64), st[IS:].reshape(P, I, H // 64)], 1)
    w2 = (dn_qw.reshape(H, P, I // 8).permute(1, 0, 2) ^ _XOR8).contiguous()
    s2 = dn_sc.t().reshape(H, P, I // 64).permute(1, 0, 2).contiguous()
    return w13, s13.to(torch.bfloat16).contiguous(), w2, s2.to(torch.bfloat16)


@triton.jit
def _permute_x(x_ptr, xp_ptr, K: tl.constexpr):
    tok = tl.program_id(0)
    RW: tl.constexpr = K // 8
    j = tl.arange(0, RW)
    i8 = tl.arange(0, 8)
    v = tl.load(x_ptr + tok * K + j[:, None] * 8 + i8[None, :])
    tl.store(xp_ptr + tok * K + i8[None, :] * RW + j[:, None], v)


@triton.jit
def _moe_gate_up_sh(xp_ptr, ids_ptr, w13_ptr, s13_ptr, w13s_ptr, s13s_ptr, act_ptr,
                    H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                    NSH: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr):
    nb = tl.program_id(0)
    pid = tl.program_id(1)            # token * (TOPK + NSH) + slot
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
    n = nb * BN + tl.arange(0, BN)
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
    tl.store(act_ptr + pid * I + n, act)


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
def _moe_down_sh(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, w2s_ptr, s2s_ptr, out_ptr,
                 H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                 NSH: tl.constexpr, BN: tl.constexpr):
    """Routed experts in slot order (fp32 routing weights), then the NSH shared
    pseudo-experts (weight 1), all in fp32; one bf16 rounding at the end."""
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    TA: tl.constexpr = TOPK + NSH
    RW: tl.constexpr = I // 8
    G: tl.constexpr = I // 64
    n = nb * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for jj in tl.static_range(TOPK):
        e = tl.load(ids_ptr + tok * TOPK + jj).to(tl.int64)
        w = tl.load(wts_ptr + tok * TOPK + jj)
        acc = _down_one(acc, act_ptr + (tok * TA + jj) * I, w2_ptr + e * (H * RW),
                        s2_ptr + e * (H * G), w, n, I, BN)
    for p in tl.static_range(NSH):
        acc = _down_one(acc, act_ptr + (tok * TA + TOPK + p) * I, w2s_ptr + p * (H * RW),
                        s2s_ptr + p * (H * G), 1.0, n, I, BN)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


SH_GATE_BN, SH_GATE_NW, SH_GATE_W, SH_GATE_WS = 1, 64, 1, 16
SH_DOWN_BN, SH_DOWN_W = 1, 1


def moe_decode_sh(x, ids, wts, w13, s13, w2, s2, sh, xp=None):
    """Routed + shared expert output, bf16 [M, H]. ``sh`` = pack_shared_expert(...);
    ``xp`` optionally the nibble-major permuted ``x`` (else computed here)."""
    M, H = x.shape
    E, two_i, hb = w13.shape
    I = two_i // 2
    TOPK = ids.shape[1]
    w13s, s13s, w2s, s2s = sh
    NSH = w13s.shape[0]
    assert x.dtype == torch.bfloat16 and hb * 2 == H and w2.shape == (E, H, I // 2)
    assert w13.dtype == w2.dtype == torch.int8 and w13.is_contiguous() and w2.is_contiguous()
    assert s13.shape == (E, 2 * I, H // 64) and s2.shape == (E, H, I // 64)
    assert s13.is_contiguous() and s2.is_contiguous()
    assert w13s.shape == (NSH, 2 * I, H // 8) and w2s.shape == (NSH, H, I // 8)
    assert s13s.shape == (NSH, 2 * I, H // 64) and s2s.shape == (NSH, H, I // 64)
    assert ids.dtype == torch.int32 and wts.dtype == torch.float32
    x = x.contiguous()
    perm = xp is None
    if perm:
        xp = torch.empty_like(x)
    else:
        assert xp.shape == x.shape and xp.dtype == x.dtype and xp.is_contiguous()
    act = torch.empty((M * (TOPK + NSH), I), dtype=torch.float32, device=x.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    if M:
        if perm:
            _permute_x[(M,)](x, xp, K=H, num_warps=4)
        _moe_gate_up_sh[(I // SH_GATE_BN, M * (TOPK + NSH))](
            xp, ids, w13.view(torch.int32), s13, w13s, s13s, act, H=H, I=I, TOPK=TOPK,
            NSH=NSH, BN=SH_GATE_BN, NW=SH_GATE_NW, num_warps=SH_GATE_W, warp_size=SH_GATE_WS)
        _moe_down_sh[(H // SH_DOWN_BN, M)](
            act, ids, wts, w2.view(torch.int32), s2, w2s, s2s, out, H=H, I=I, TOPK=TOPK,
            NSH=NSH, BN=SH_DOWN_BN, num_warps=SH_DOWN_W)
    return out


# ---------------------------------------------------------------------------
# Round 3: RMSNorm that also (or only) writes the nibble-major permuted output,
# and a GPTQ (uint4b8, group 64) GEMV reading the permuted input (used for qkv).


@triton.jit
def _rms_norm_perm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, outp_ptr, res_out_ptr, x_stride, eps,
                          H: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
                          WRITE_NORMAL: tl.constexpr):
    """Same arithmetic, in the same order, as ``xpu_fused._rms_norm_kernel``; the
    permuted copy stores the identical values at xp[(k % 8) * H / 8 + k / 8]."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    s = tl.load(x_ptr + row * x_stride + offs).to(tl.float32)
    if HAS_RESIDUAL:
        s += tl.load(res_ptr + row * H + offs).to(tl.float32)
        tl.store(res_out_ptr + row * H + offs, s.to(tl.bfloat16))
    rs = 1.0 / tl.sqrt(tl.sum(s * s, axis=0) / H + eps)
    y = (s * rs * w).to(tl.bfloat16)
    if WRITE_NORMAL:
        tl.store(out_ptr + row * H + offs, y)
    RW: tl.constexpr = H // 8
    tl.store(outp_ptr + row * H + (offs % 8) * RW + offs // 8, y)


def rms_norm_perm(x, residual, weight, eps, write_normal):
    """Returns (normal output or None, permuted output, bf16(x + residual) or None)."""
    m, h = x.shape
    assert x.dtype == torch.bfloat16 and x.stride(-1) == 1 and h & (h - 1) == 0 and h % 64 == 0
    if residual is not None:
        assert residual.is_contiguous() and residual.shape == x.shape
        assert residual.dtype == torch.bfloat16
    outp = torch.empty((m, h), dtype=x.dtype, device=x.device)
    out = torch.empty((m, h), dtype=x.dtype, device=x.device) if write_normal else None
    res_out = torch.empty((m, h), dtype=x.dtype, device=x.device) if residual is not None else None
    if m:
        _rms_norm_perm_kernel[(m,)](
            x, x if residual is None else residual, weight, outp if out is None else out, outp,
            outp if res_out is None else res_out, x.stride(0), eps, H=h,
            HAS_RESIDUAL=residual is not None, WRITE_NORMAL=write_normal, num_warps=4)
    return out, outp, res_out


@triton.jit
def _gptq_gemv_perm(xp_ptr, w_ptr, s_ptr, o_ptr, N: tl.constexpr, K: tl.constexpr,
                    BN: tl.constexpr, NW: tl.constexpr):
    """y = x @ W^T for GPTQ uint4b8 W [N, K/8] int32 (nibble i of word j is k = 8j + i),
    scales [K/64, N]; x given nibble-major permuted; fp32 accumulation."""
    nb = tl.program_id(0)
    tok = tl.program_id(1)
    RW: tl.constexpr = K // 8
    GI: tl.constexpr = NW // 8
    n = nb * BN + tl.arange(0, BN)
    wo = tl.arange(0, NW)
    gi = tl.arange(0, GI)
    xb = xp_ptr + tok * K
    acc = tl.zeros((BN,), dtype=tl.float32)
    for w0 in range(0, RW, NW):
        pk = tl.load(w_ptr + n[:, None] * RW + w0 + wo[None, :]) ^ -2004318072
        sc = tl.load(s_ptr + (w0 // 8 + gi)[None, :] * N + n[:, None]).to(tl.float32)
        t = _nib(pk, 0) * tl.load(xb + w0 + wo).to(tl.float32)[None, :]
        for i in tl.static_range(1, 8):
            t += _nib(pk, i) * tl.load(xb + i * RW + w0 + wo).to(tl.float32)[None, :]
        acc += tl.sum(tl.sum(tl.reshape(t, (BN, GI, 8)), axis=2) * sc, axis=1)
    tl.store(o_ptr + tok * N + n, acc.to(tl.bfloat16))


GPTQ_BN, GPTQ_NW, GPTQ_W, GPTQ_WS = 4, 64, 1, 16


def gptq_ok(layer) -> bool:
    """True if ``layer`` holds sym GPTQ int4 (zero point 8, group 64) in the layout
    the XPU W4A16 kernel leaves: qweight [K/8, N] int32 view of a contiguous
    [N, K/8], scales [K/64, N]. Uses .item(): call outside graph capture."""
    q = getattr(layer, "qweight", None)
    sc = getattr(layer, "scales", None)
    zp = getattr(layer, "qzeros", None)
    if q is None or sc is None or zp is None or q.dtype != torch.int32 or q.dim() != 2:
        return False
    k8, n = q.shape
    return (q.t().is_contiguous() and sc.is_contiguous() and tuple(sc.shape) == (k8 // 8, n)
            and zp.numel() == 1 and int(zp.item()) == 8 and n % GPTQ_BN == 0
            and k8 % GPTQ_NW == 0)


def gptq_linear_perm(xp, layer):
    """bf16 [M, N] = x @ W^T with ``xp`` the permuted bf16 [M, K] input."""
    q = layer.qweight.t()
    M, K = xp.shape
    N = q.shape[0]
    assert q.shape[1] * 8 == K and xp.is_contiguous()
    out = torch.empty((M, N), dtype=torch.bfloat16, device=xp.device)
    if M:
        _gptq_gemv_perm[(N // GPTQ_BN, M)](xp, q, layer.scales, out, N=N, K=K, BN=GPTQ_BN,
                                           NW=GPTQ_NW, num_warps=GPTQ_W, warp_size=GPTQ_WS)
    return out
