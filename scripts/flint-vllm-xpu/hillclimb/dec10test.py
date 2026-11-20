"""2D block-load GEMV variants (cold, graph)."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode as xd
import xpu_decode8 as d8
try:
    import triton
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="xpu"))
except Exception as ex:
    print("set_allocator", ex)
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
def sweep(name, mk, cfgs):
    out = {}
    for c in cfgs:
        k = '_'.join(f'{a}{b}' for a, b in c.items())
        try:
            out[k] = graph_us(mk(**c))
        except Exception as ex:
            out[k] = repr(ex)[:160]
        print(name, k, out[k], flush=True)
    R[name] = out


L = 49
def gref(x, qw, sc):
    q = qw.cpu().to(torch.int64); N, K = q.shape[0], q.shape[1] * 8
    nib = torch.stack([(q >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).double() - 8
    return x.cpu().double() @ (nib * sc.cpu().double().t().repeat_interleave(64, 1)).t()
cfgs = [dict(BN=b, NW=n, W=w, WS=s) for b, n, w, s in
        [(2, 64, 1, 16), (4, 32, 1, 16), (4, 64, 1, 16), (8, 16, 1, 16), (8, 32, 1, 16), (8, 32, 2, 32),
         (16, 16, 1, 16), (16, 16, 2, 16), (16, 32, 2, 32), (32, 16, 2, 16), (32, 16, 4, 32), (8, 64, 2, 16)]]
for name, N, K in [('qkv', 5120, 2048), ('o', 2048, 4096)]:
    qws = [torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev) for _ in range(L)]
    scs = [(torch.rand(K // 64, N, device=dev) * 0.01).to(torch.bfloat16) for _ in range(L)]
    x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
    xp = torch.empty_like(x); xd._permute_x[(1,)](x, xp, K=K, num_warps=4)
    o = torch.empty(1, N, device=dev, dtype=torch.bfloat16)
    ref = gref(x, qws[0], scs[0])[0]
    for kn, kern in (('bp', d8._gptq_gemv_bp), ('pp', d8._gptq_gemv_pp)):
        kern[(N // 8, )](xp, qws[0], scs[0], o, N=N, K=K, BN=8, NW=32, num_warps=1)
        log(f'{name}_{kn}_relerr', ((o[0].double().cpu() - ref).abs().max() / ref.abs().max()).item())
        sweep(f'{name}_{kn}', lambda BN, NW, W, WS, kern=kern: [(lambda l=l: kern[(N // BN,)](
            xp, qws[l], scs[l], o, N=N, K=K, BN=BN, NW=NW, num_warps=W, warp_size=WS)) for l in range(L)], cfgs)
    del qws, scs
H, E, TOPK, I, NSH = 2048, 256, 6, 512, 2
L = 47
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
w13w = w13.view(torch.int32)
IS = NSH * I
packs = []
for l in range(L):
    gq = torch.randint(-2**31, 2**31 - 1, (2 * IS, H // 8), dtype=torch.int32, device=dev)
    gs = (torch.rand(H // 64, 2 * IS, device=dev) * 0.01).to(torch.bfloat16)
    dq = torch.randint(-2**31, 2**31 - 1, (H, IS // 8), dtype=torch.int32, device=dev)
    ds = (torch.rand(IS // 64, H, device=dev) * 0.01).to(torch.bfloat16)
    packs.append(xd.pack_shared_expert(gq, gs, dq, ds, I))
idss = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
xs = [(torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16) for _ in range(L)]
xps = [torch.empty_like(x) for x in xs]
for l in range(L): xd._permute_x[(1,)](xs[l], xps[l], K=H, num_warps=4)
acts = [torch.empty(TOPK + NSH, I, device=dev) for _ in range(L)]
def cur(l, BN=1, NW=64, W=1, WS=16):
    xd._moe_gate_up_sh[(I // BN, TOPK + NSH)](xps[l], idss[l], w13w, s13, packs[l][0], packs[l][1], acts[l], H=H, I=I, TOPK=TOPK, NSH=NSH, BN=BN, NW=NW, num_warps=W, warp_size=WS)
def bp(l, BN, NW, W, WS):
    d8._moe_gate_up_bp[(I // BN, TOPK + NSH)](xps[l], idss[l], w13w, s13, packs[l][0], packs[l][1], acts[l], H=H, I=I, TOPK=TOPK, NSH=NSH, BN=BN, NW=NW, num_warps=W, warp_size=WS)
cur(0); ref = acts[0].clone(); acts[0].zero_(); bp(0, 8, 32, 1, 16)
log('gate_bp_relerr_vs_cur', ((acts[0] - ref).abs().max() / ref.abs().max()).item())
log('gate_cur', [graph_us([(lambda l=l: cur(l)) for l in range(L)]) for _ in range(3)])
sweep('gate_bp', lambda BN, NW, W, WS: [(lambda l=l: bp(l, BN, NW, W, WS)) for l in range(L)], cfgs)
(root / 'dec10test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
