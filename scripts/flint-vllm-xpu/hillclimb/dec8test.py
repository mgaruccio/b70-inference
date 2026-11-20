"""MoE GEMV v5 (permuted inputs): accuracy (fp64) and cold-weight graph timing, MoE + GPTQ shapes."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode as xd
import xpu_decode4 as d4
import xpu_decode6 as d6
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

# ---------------- MoE
H, E, TOPK, I, L = 2048, 256, 6, 512, 47
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
w13w = w13.view(torch.int32); w2w = w2.view(torch.int32)
xs = [(torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16) for _ in range(L)]
idss = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
wtss = []
for _ in range(L):
    w = torch.rand(1, TOPK, device=dev); wtss.append((w / w.sum(-1, keepdim=True)).float())
shared = torch.randn(1, H, device=dev).to(torch.bfloat16)
def deq(q, s):
    lo = ((q.to(torch.int16) << 12) >> 12).double(); hi = (q.to(torch.int16) >> 4).double()
    return torch.stack([lo, hi], -1).flatten(-2) * s.double().repeat_interleave(64, -1)
e0 = idss[0][0].long().cpu()
W13 = deq(w13[e0].cpu(), s13[e0].cpu()); W2 = deq(w2[e0].cpu(), s2[e0].cpu())
gu = torch.einsum('eik,k->ei', W13, xs[0][0].double().cpu())
a_ref = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
acts = [torch.empty(TOPK, I, dtype=torch.float32, device=dev) for _ in range(L)]
out = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
xps = [torch.empty_like(x) for x in xs]
for l in range(L): d6._permute_x[(1,)](xs[l], xps[l], K=H, num_warps=4)
d6._moe_gate_up5[(I // 2, TOPK)](xps[0], idss[0], w13w, s13, acts[0], H=H, I=I, TOPK=TOPK, BN=2, NW=64, PERM_OUT=False, num_warps=1)
log('gate5_relerr', ((acts[0].double().cpu() - a_ref).abs().max() / a_ref.abs().max()).item())
refo = (torch.einsum('ehi,ei->eh', W2, acts[0].double().cpu()) * wtss[0][0].double().cpu()[:, None]).sum(0) + shared[0].double().cpu()
actps = [torch.empty_like(a) for a in acts]
d6._moe_gate_up5[(I // 2, TOPK)](xps[0], idss[0], w13w, s13, actps[0], H=H, I=I, TOPK=TOPK, BN=2, NW=64, PERM_OUT=True, num_warps=1)
log('perm_out_ok', torch.equal(actps[0].view(TOPK, 8, I // 8).transpose(1, 2).reshape(TOPK, I), acts[0]))
d6._moe_down5[(H, 1)](actps[0], idss[0], wtss[0], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=1, ADD_SHARED=True, num_warps=1)
log('down5_relerr', ((out[0].double().cpu() - refo).abs().max() / refo.abs().max()).item())
for l in range(L):
    xd.moe_decode(xs[l], idss[l], wtss[l], w13, s13, w2, s2, shared=shared)
log('cur_moe_full', graph_us([(lambda l=l: xd.moe_decode(xs[l], idss[l], wtss[l], w13, s13, w2, s2, shared=shared)) for l in range(L)]))
log('cur_gate', graph_us([(lambda l=l: xd._moe_gate_up[(I // 2, TOPK)](xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=2, NW=64, num_warps=1, warp_size=16)) for l in range(L)]))
log('cur_down', graph_us([(lambda l=l: xd._moe_down[(H, 1)](acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=1, ADD_SHARED=True, num_warps=1)) for l in range(L)]))
log('permute_us', graph_us([(lambda l=l: d6._permute_x[(1,)](xs[l], xps[l], K=H, num_warps=4)) for l in range(L)]))
log('permute_us_w1', graph_us([(lambda l=l: d6._permute_x[(1,)](xs[l], xps[l], K=H, num_warps=1)) for l in range(L)]))
sweep('gate5', lambda BN, NW, W, WS: [(lambda l=l: d6._moe_gate_up5[(I // BN, TOPK)](
    xps[l], idss[l], w13w, s13, actps[l], H=H, I=I, TOPK=TOPK, BN=BN, NW=NW, PERM_OUT=True, num_warps=W, warp_size=WS)) for l in range(L)],
    [dict(BN=b, NW=n, W=w, WS=s) for b, n, w, s in
     [(1, 64, 1, 16), (1, 128, 1, 16), (1, 256, 1, 16), (2, 32, 1, 16), (2, 64, 1, 16), (2, 64, 1, 32), (2, 128, 1, 16),
      (2, 128, 1, 32), (4, 32, 1, 16), (4, 64, 1, 16), (4, 64, 1, 32), (4, 64, 2, 32), (8, 32, 1, 16), (8, 64, 2, 32)]])
sweep('down5', lambda BN, W, WS: [(lambda l=l: d6._moe_down5[(H // BN, 1)](
    actps[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True,
    num_warps=W, warp_size=WS)) for l in range(L)],
    [dict(BN=b, W=w, WS=s) for b, w, s in [(1, 1, 16), (1, 1, 32), (2, 1, 16), (2, 1, 32), (4, 1, 16), (4, 1, 32), (4, 2, 32), (8, 2, 32)]])
(root / 'dec8test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
