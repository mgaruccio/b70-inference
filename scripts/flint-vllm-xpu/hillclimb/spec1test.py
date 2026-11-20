"""Multi-row (M<=8, spec verify) decode kernels: batch invariance vs M=1, identity with the
dec5 M=1 kernels, and cold graph timing per layer for M = 1, 2, 4, 8."""
import faulthandler, json, pathlib, sys, types
faulthandler.dump_traceback_later(1500, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode as xd
import xpu_decode_dec5 as x5
root = pathlib.Path(sys.argv[1]); dev = 'xpu'; torch.manual_seed(0)
R = {}
def log(k, v):
    R[k] = v; print(k, v, flush=True)
def graph_us(fns, iters=10):
    for f in fns[:2]: f()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for f in fns: f()
    for _ in range(20): g.replay()
    torch.xpu.synchronize()
    res = []
    for _ in range(5):
        s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        s.record()
        for _ in range(iters): g.replay()
        e.record(); torch.xpu.synchronize()
        res.append(s.elapsed_time(e) / (iters * len(fns)) * 1000)
    return round(sorted(res)[2], 2)

H, E, TOPK, I, NSH = 2048, 256, 6, 512, 2
IS = NSH * I
NT = 8  # distinct weight tables (cycled over 47 layers => cold)
def mk_table():
    w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
    w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
    s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
    s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
    gq = torch.randint(-2**31, 2**31 - 1, (2 * IS, H // 8), dtype=torch.int32, device=dev)
    gs = (torch.rand(H // 64, 2 * IS, device=dev) * 0.01).to(torch.bfloat16)
    dq = torch.randint(-2**31, 2**31 - 1, (H, IS // 8), dtype=torch.int32, device=dev)
    ds = (torch.rand(IS // 64, H, device=dev) * 0.01).to(torch.bfloat16)
    return w13, s13, w2, s2, xd.pack_shared_expert(gq, gs, dq, ds, I)
tabs = [mk_table() for _ in range(NT)]
rw = (torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16)

# ---- correctness ----
w13, s13, w2, s2, pk = tabs[0]
def check(x, tag):
    _, ids, wts = xd.router(x, rw, TOPK)
    M = x.shape[0]
    full = xd.moe_decode_sh(x, ids, wts, w13, s13, w2, s2, pk)
    rows = [xd.moe_decode_sh(x[t:t + 1], ids[t:t + 1], wts[t:t + 1], w13, s13, w2, s2, pk) for t in range(M)]
    old = [x5.moe_decode_sh(x[t:t + 1], ids[t:t + 1], wts[t:t + 1], w13, s13, w2, s2, pk) for t in range(M)]
    uniq = len(set(ids.flatten().tolist()))
    return dict(M=M, uniq=uniq, inv=all(torch.equal(full[t:t + 1], rows[t]) for t in range(M)),
                same_as_dec5=all(torch.equal(rows[t], old[t]) for t in range(M)),
                det=torch.equal(xd.moe_decode_sh(x, ids, wts, w13, s13, w2, s2, pk), full))
res = []
for M in (2, 3, 4, 5, 8):
    xr = (torch.randn(M, H, device=dev) * 0.3).to(torch.bfloat16)
    base = torch.randn(1, H, device=dev) * 0.3
    xc = (base + 0.05 * torch.randn(M, H, device=dev)).to(torch.bfloat16)  # shared experts across rows
    res.append(check(xr, 'rand')); res.append(check(xc, 'corr'))
log('moe_checks', res)
moe_ok = all(r['inv'] and r['same_as_dec5'] and r['det'] for r in res) and any(r['uniq'] < 6 * r['M'] for r in res)
N2 = 5120
qw = torch.randint(-2**31, 2**31 - 1, (N2, H // 8), dtype=torch.int32, device=dev)
qs = (torch.rand(H // 64, N2, device=dev) * 0.01).to(torch.bfloat16)
lay = types.SimpleNamespace(qweight=qw.t(), scales=qs, qzeros=torch.tensor([8], dtype=torch.int8, device=dev))
g_ok = True
for M in (2, 3, 8):
    xp = (torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16)
    y = xd.gptq_linear_perm(xp, lay)
    g_ok &= all(torch.equal(y[t:t + 1], xd.gptq_linear_perm(xp[t:t + 1].contiguous(), lay)) for t in range(M))
    g_ok &= all(torch.equal(y[t:t + 1], x5.gptq_linear_perm(xp[t:t + 1].contiguous(), lay)) for t in range(M))
log('gptq_inv_and_same_as_dec5', g_ok)
R['PASS'] = bool(moe_ok and g_ok)
log('PASS', R['PASS'])

# ---- timing (cold, graph, 47 MoE layers / 49 qkv layers) ----
L = 47
def routing(M, mode, l):
    g = torch.Generator(device='cpu').manual_seed(1000 + l)
    if mode == 'same':
        r = torch.randperm(E, generator=g)[:TOPK].repeat(M, 1)
    elif mode == 'half':  # each token keeps 3 experts of the previous token
        rows = [torch.randperm(E, generator=g)[:TOPK]]
        for _ in range(M - 1):
            prev = rows[-1][:3]
            cand = [e for e in torch.randperm(E, generator=g).tolist() if e not in prev.tolist()][:3]
            rows.append(torch.cat([prev, torch.tensor(cand)]))
        r = torch.stack(rows)
    else:
        r = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(M)])
    w = torch.rand(M, TOPK, generator=g)
    return r.to(torch.int32).to(dev), (w / w.sum(-1, keepdim=True)).float().to(dev)
T = {}
for mode in ('rand', 'half', 'same'):
    for M in (1, 2, 4, 8):
        if mode != 'rand' and M == 1:
            continue
        xs = [(torch.randn(M, H, device=dev) * 0.3).to(torch.bfloat16) for _ in range(L)]
        rt = [routing(M, mode, l) for l in range(L)]
        uniq = sum(len(set(r[0].flatten().tolist())) for r in rt) / L
        fns = [(lambda l=l: xd.moe_decode_sh(xs[l], rt[l][0], rt[l][1], *tabs[l % NT][:4], tabs[l % NT][4]))
               for l in range(L)]
        k = f'moe_{mode}_M{M}'
        T[k] = dict(uniq=round(uniq, 1), grouped=graph_us(fns))
        if M > 1:
            xd.MULTI_GROUPED_DOWN = False
            T[k]['gu_only'] = graph_us(fns)
            xd.MULTI_GROUPED = False
            T[k]['tokf'] = graph_us(fns)
            xd.MULTI_GROUPED = xd.MULTI_GROUPED_DOWN = True
        print(k, T[k], flush=True)
qws = [(torch.randint(-2**31, 2**31 - 1, (N2, H // 8), dtype=torch.int32, device=dev),
        (torch.rand(H // 64, N2, device=dev) * 0.01).to(torch.bfloat16)) for _ in range(49)]
lays = [types.SimpleNamespace(qweight=q.t(), scales=s, qzeros=None) for q, s in qws]
for M in (1, 2, 4, 8):
    xps = [(torch.randn(M, H, device=dev) * 0.5).to(torch.bfloat16) for _ in range(49)]
    fns = [(lambda l=l: xd.gptq_linear_perm(xps[l], lays[l])) for l in range(49)]
    k = f'qkv_M{M}'
    T[k] = dict(grouped=graph_us(fns))
    if M > 1:
        xd.GPTQ_GROUPED = False
        T[k]['tokf'] = graph_us(fns)
        xd.GPTQ_GROUPED = True
    print(k, T[k], flush=True)
R['timing_us_per_layer'] = T
(root / 'spec1test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
