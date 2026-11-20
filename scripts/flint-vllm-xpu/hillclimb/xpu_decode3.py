"""Flint XPU decode kernels, round 3 (router GEMV + top-k, MoE down + shared add).

Router: ``_router_partial`` computes fp32 partial logits [M, KS, N] of
bf16 x bf16 products; each row is computed by identical code (a fixed tile and
reduction) regardless of M, so results are batch-invariant. ``_router_topk``
sums the KS partials in fixed order, writes fp32 logits, and selects the top-k
by logit (ties -> lower expert id; non-finite logits rank last), with weights
sigmoid(l) / sum(sigmoid(l)). Ids are always valid and distinct.
"""

import torch

from vllm.triton_utils import tl, triton


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
    for i in range(RB):
        m = pm * RB + i
        ok = m < M
        x = tl.load(x_ptr + m * K + rk, mask=(rk < K) & ok, other=0.0).to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        tl.store(p_ptr + (m * KS + ks) * N + rn, acc, mask=(rn < N) & ok)


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


ROUTER_CFG = dict(BN=8, KS=4, RB=1, W=4)


def router_topk(x, w, topk, BN=None, KS=None, RB=None, W=None):
    """Returns (fp32 logits [M, N], int32 ids [M, topk], fp32 weights [M, topk])."""
    BN = BN or ROUTER_CFG['BN']; KS = KS or ROUTER_CFG['KS']
    RB = RB or ROUTER_CFG['RB']; W = W or ROUTER_CFG['W']
    assert x.dtype == w.dtype == torch.bfloat16 and w.is_contiguous()
    x = x.contiguous()
    M, K = x.shape
    N = w.shape[0]
    assert K % KS == 0 and N % BN == 0
    part = torch.empty(M, KS, N, device=x.device, dtype=torch.float32)
    logits = torch.empty(M, N, device=x.device, dtype=torch.float32)
    ids = torch.empty(M, topk, device=x.device, dtype=torch.int32)
    wts = torch.empty(M, topk, device=x.device, dtype=torch.float32)
    if M:
        _router_partial[(N // BN, KS, triton.cdiv(M, RB))](
            x, w, part, M, N=N, K=K, BN=BN, KS=KS, RB=RB, num_warps=W)
        _router_topk[(M,)](part, logits, ids, wts, N=N, KS=KS, TOPK=topk,
                           TK2=triton.next_power_of_2(topk), num_warps=4)
    return logits, ids, wts


@triton.jit
def _dot_s4_groups(p_ptr, xe, xo, ROW_BYTES: tl.constexpr, rows, kb0,
                   BN: tl.constexpr, GPB: tl.constexpr):
    offs = tl.arange(0, 32 * GPB)
    pk = tl.load(p_ptr + rows[:, None] * ROW_BYTES + kb0 + offs[None, :])
    lo = ((pk << 4).to(tl.int8) >> 4).to(tl.float32)
    hi = (pk >> 4).to(tl.float32)
    prod = lo * xe[None, :] + hi * xo[None, :]
    return tl.sum(tl.reshape(prod, (BN, GPB, 32)), axis=2)


@triton.jit
def _moe_down_sum_v3(act_ptr, ids_ptr, wts_ptr, w2_ptr, s2_ptr, sh_ptr, out_ptr,
                     H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                     BN: tl.constexpr, GPB: tl.constexpr, ADD_SHARED: tl.constexpr):
    """Like v2, plus the shared-expert output added in fp32 before rounding."""
    tok = tl.program_id(1)
    nb = tl.program_id(0)
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
    if ADD_SHARED:
        acc += tl.load(sh_ptr + tok * H + n).to(tl.float32)
    tl.store(out_ptr + tok * H + n, acc.to(tl.bfloat16))


@triton.jit
def _moe_gate_up_act_v3(x_ptr, ids_ptr, w13_ptr, s13_ptr, act_ptr,
                        H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr,
                        BN: tl.constexpr, GPB: tl.constexpr):
    """v2 with the N block on axis 0 and the (token, slot) on axis 1."""
    nb = tl.program_id(0)
    pid = tl.program_id(1)
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


MOE_CFG = dict(BN1=16, GPB1=16, W1=4, BN2=16, GPB2=8, W2=2)


def moe_decode_v3(x, ids, wts, w13, s13, w2, s2, shared=None, cfg=None):
    """Routed (+ optional shared) expert output, bf16 [M, H]; ids int32, wts fp32."""
    c = dict(MOE_CFG, **(cfg or {}))
    M, H = x.shape
    I = w13.shape[1] // 2
    TOPK = ids.shape[1]
    act = torch.empty((M * TOPK, I), dtype=torch.float32, device=x.device)
    out = torch.empty((M, H), dtype=torch.bfloat16, device=x.device)
    if M:
        _moe_gate_up_act_v3[(I // c['BN1'], M * TOPK)](
            x, ids, w13, s13, act, H=H, I=I, TOPK=TOPK, BN=c['BN1'], GPB=c['GPB1'],
            num_warps=c['W1'])
        _moe_down_sum_v3[(H // c['BN2'], M)](
            act, ids, wts, w2, s2, shared if shared is not None else out, out,
            H=H, I=I, TOPK=TOPK, BN=c['BN2'], GPB=c['GPB2'],
            ADD_SHARED=shared is not None, num_warps=c['W2'])
    return out
