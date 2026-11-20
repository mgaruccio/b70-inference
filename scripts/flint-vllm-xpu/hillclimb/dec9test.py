"""MoE down+shared variants (cold, graph)."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode as xd
import xpu_decode7 as d7
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


H, E, TOPK, I, L, NSH = 2048, 256, 6, 512, 47, 2
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
w2w = w2.view(torch.int32)
IS = NSH * I
packs = []
for l in range(L):
    gq = torch.randint(-2**31, 2**31 - 1, (2 * IS, H // 8), dtype=torch.int32, device=dev)
    gs = (torch.rand(H // 64, 2 * IS, device=dev) * 0.01).to(torch.bfloat16)
    dq = torch.randint(-2**31, 2**31 - 1, (H, IS // 8), dtype=torch.int32, device=dev)
    ds = (torch.rand(IS // 64, H, device=dev) * 0.01).to(torch.bfloat16)
    packs.append(xd.pack_shared_expert(gq, gs, dq, ds, I))
idss = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
wtss = []
for _ in range(L):
    w = torch.rand(1, TOPK, device=dev); wtss.append((w / w.sum(-1, keepdim=True)).float())
acts = [torch.randn(TOPK + NSH, I, device=dev) * 0.1 for _ in range(L)]
outs = [torch.empty(1, H, dtype=torch.bfloat16, device=dev) for _ in range(L)]
def cur(l, BN=1, W=1, WS=32):
    xd._moe_down_sh[(H // BN, 1)](acts[l], idss[l], wtss[l], w2w, s2, packs[l][2], packs[l][3], outs[l], H=H, I=I, TOPK=TOPK, NSH=NSH, BN=BN, num_warps=W, warp_size=WS)
def g2(l, W=1, WS=32):
    d7._moe_down_g2[(H, 1)](acts[l], idss[l], wtss[l], w2w, s2, packs[l][2], packs[l][3], outs[l], H=H, I=I, TOPK=TOPK, NSH=NSH, num_warps=W, warp_size=WS)
def pre(l, BN=1, W=1, WS=32):
    d7._moe_down_pre[(H // BN, 1)](acts[l], idss[l], wtss[l], w2w, s2, packs[l][2], packs[l][3], outs[l], H=H, I=I, TOPK=TOPK, NSH=NSH, BN=BN, num_warps=W, warp_size=WS)
cur(0); ref = outs[0].float().clone()
for nm, f in (('g2', g2), ('pre', pre)):
    outs[0].zero_(); f(0)
    log(nm + '_relerr_vs_cur', ((outs[0].float() - ref).abs().max() / ref.abs().max()).item())
log('cur', [graph_us([(lambda l=l: cur(l)) for l in range(L)]) for _ in range(2)])
sweep('cur_sweep', lambda BN, W, WS: [(lambda l=l: cur(l, BN, W, WS)) for l in range(L)],
      [dict(BN=b, W=w, WS=s) for b, w, s in [(1, 1, 16), (2, 1, 16), (2, 1, 32), (1, 2, 32)]])
sweep('g2', lambda W, WS: [(lambda l=l: g2(l, W, WS)) for l in range(L)],
      [dict(W=w, WS=s) for w, s in [(1, 32), (1, 16), (2, 32), (2, 16), (4, 32), (4, 16)]])
sweep('pre', lambda BN, W, WS: [(lambda l=l: pre(l, BN, W, WS)) for l in range(L)],
      [dict(BN=b, W=w, WS=s) for b, w, s in [(1, 1, 32), (1, 1, 16), (2, 1, 32), (2, 1, 16), (1, 2, 32)]])
(root / 'dec9test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
