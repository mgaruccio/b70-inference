"""Round-3 decode kernel tests + cold-weight graph microbench on the B70."""
import faulthandler, json, pathlib, sys, time
faulthandler.dump_traceback_later(1100, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))
import xpu_decode3 as d3
root = pathlib.Path(sys.argv[1]); dev = 'xpu'; torch.manual_seed(0)
R = {}
def log(k, v):
    R[k] = v; print(k, v, flush=True)

def graph_us(fns, iters=10):
    """Capture fns (list of callables) once in an XPU graph; us per callable."""
    for f in fns[:2]: f()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for f in fns: f()
    g.replay(); torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.xpu.synchronize()
    return round(s.elapsed_time(e) / (iters * len(fns)) * 1000, 2)

H, E, TOPK, I, L = 2048, 256, 6, 512, 47
# ---------------- bandwidth / launch floor
big = torch.empty(512 * 2**20 // 2, dtype=torch.bfloat16, device=dev).normal_()
dst = torch.empty_like(big)
us = graph_us([lambda: dst.copy_(big)], iters=10)
log('copy_512MB_GBps', round(2 * big.numel() * 2 / us / 1e3, 1))
del big, dst
tiny = torch.zeros(16, device=dev)
log('tiny_add_us', graph_us([lambda: tiny.add_(1.0)] * 200))

# ---------------- router correctness
Wr = (torch.randn(L, E, H, device=dev) * 0.02).to(torch.bfloat16)
xb = torch.randn(37, H, device=dev).to(torch.bfloat16)
ref = (xb.double().cpu() @ Wr[0].double().cpu().T)
cfgs = [dict(BN=b, KS=k, RB=1, W=w) for b, k, w in
        [(8, 4, 4), (4, 8, 4), (8, 8, 4), (16, 8, 4), (16, 4, 4), (4, 4, 2), (8, 2, 4),
         (2, 8, 2), (4, 16, 2), (8, 16, 2), (16, 16, 4), (2, 4, 1), (8, 4, 8)]]
router_res = {}
for c in cfgs:
    key = 'bn{BN}_ks{KS}_w{W}'.format(**c)
    try:
        lg, ids, wts = d3.router_topk(xb, Wr[0], TOPK, **c)
        torch.xpu.synchronize()
        err = (lg.double().cpu() - ref).abs().max().item()
        inv = all(torch.equal(d3.router_topk(xb[i:i + 1], Wr[0], TOPK, **c)[0], lg[i:i + 1]) for i in range(0, 37, 5))
        inv8 = torch.equal(d3.router_topk(xb[:8], Wr[0], TOPK, **c)[0], lg[:8])
        det = torch.equal(d3.router_topk(xb, Wr[0], TOPK, **c)[0], lg)
        x1 = xb[:1]
        cold = graph_us([(lambda l=l: d3.router_topk(x1, Wr[l], TOPK, **c)) for l in range(L)])
        part_only = graph_us([(lambda l=l: d3._router_partial[(E // c['BN'], c['KS'], 1)](
            x1, Wr[l], torch.empty(1, c['KS'], E, device=dev), 1, N=E, K=H, BN=c['BN'], KS=c['KS'], RB=1,
            num_warps=c['W'])) for l in range(L)])
        router_res[key] = dict(err=err, inv=inv, inv8=inv8, det=det, cold_us=cold, partial_cold_us=part_only)
    except Exception as ex:
        router_res[key] = repr(ex)[:300]
    print('router', key, router_res[key], flush=True)
R['router'] = router_res
# top-k semantics vs fp64 reference
lg, ids, wts = d3.router_topk(xb, Wr[0], TOPK)
rs, ri = torch.sort(ref, dim=1, descending=True, stable=True)
gap_ok = (rs[:, TOPK - 1] - rs[:, TOPK]) > 1e-4
ids_ok = sum(int(set(ids[i].tolist()) == set(ri[i, :TOPK].tolist())) for i in range(37) if gap_ok[i])
order_ok = sum(int(ids[i].tolist() == ri[i, :TOPK].tolist()) for i in range(37))
sg = torch.sigmoid(rs[:, :TOPK]); rw = sg / sg.sum(1, keepdim=True)
log('topk', dict(rows=37, gap_rows=int(gap_ok.sum()), set_match=ids_ok, order_match=order_ok,
                 wts_maxerr=(wts.double().cpu() - rw).abs().max().item()))
# non-finite row: all NaN, and partial inf
xn = xb[:3].clone(); xn[0] = float('nan'); xn[1, 5] = float('inf')
lgn, idn, wtn = d3.router_topk(xn, Wr[0], TOPK)
log('nonfinite', dict(ids=idn.tolist(), wts=[[round(v, 4) for v in r] for r in wtn.tolist()]))
# current pipeline (router tl.dot + fused_topk + sanitize) for comparison
try:
    import xpu_router as xr
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
    def cur(l):
        lgc = xr.flint_router_logits(xb[:1], Wr[l])
        tw, ti, _ = fused_topk(xb[:1], lgc, TOPK, True, scoring_func='sigmoid')
        v = (ti >= 0) & (ti < E)
        ti = torch.where(v, ti, torch.zeros_like(ti)); tw = torch.where(v, tw, torch.zeros_like(tw))
        return ti, tw
    ti, tw = cur(0)
    log('cur_vs_new_ids_equal', torch.equal(ti[0].to(torch.int32).cpu(), ids[0].cpu()))
    log('cur_pipeline_cold_us', graph_us([(lambda l=l: cur(l)) for l in range(L)]))
    log('cur_router_only_cold_us', graph_us([(lambda l=l: xr.flint_router_logits(xb[:1], Wr[l])) for l in range(L)]))
except Exception as ex:
    log('cur_pipeline_err', repr(ex)[:300])
del Wr

# ---------------- MoE
import xpu_moe_decode as md
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
xs = [torch.randn(1, H, device=dev).to(torch.bfloat16) for _ in range(L)]
idss = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
wtss = []
for _ in range(L):
    w = torch.rand(1, TOPK, device=dev); wtss.append((w / w.sum(-1, keepdim=True)).float())
shared = torch.randn(1, H, device=dev).to(torch.bfloat16)
# fp64 reference for layer 0
def deq(q, s):
    lo = ((q.to(torch.int16) << 12) >> 12).double(); hi = (q.to(torch.int16) >> 4).double()
    v = torch.stack([lo, hi], -1).flatten(-2)
    return v * s.double().repeat_interleave(64, -1)
e0 = idss[0][0].long().cpu()
W13 = deq(w13[e0].cpu(), s13[e0].cpu()); W2 = deq(w2[e0].cpu(), s2[e0].cpu())
gu = torch.einsum('eik,k->ei', W13, xs[0][0].double().cpu())
a = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
r = torch.einsum('ehi,ei->eh', W2, a)
refo = (r * wtss[0][0].double().cpu()[:, None]).sum(0) + shared[0].double().cpu()
o = d3.moe_decode_v3(xs[0], idss[0], wtss[0], w13, s13, w2, s2, shared=shared)
o2 = md.moe_decode_v2(xs[0], idss[0], wtss[0], w13, s13, w2, s2) + shared
scale = refo.abs().max().item()
log('moe_v3_relerr', (o[0].double().cpu() - refo).abs().max().item() / scale)
log('moe_v2_relerr', (o2[0].double().cpu() - refo).abs().max().item() / scale)
log('moe_v3_det', torch.equal(o, d3.moe_decode_v3(xs[0], idss[0], wtss[0], w13, s13, w2, s2, shared=shared)))
act = torch.empty(TOPK, I, dtype=torch.float32, device=dev)
out = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
acts = [torch.empty(TOPK, I, dtype=torch.float32, device=dev) for _ in range(L)]
for l in range(L):
    md.moe_decode_v2(xs[l], idss[l], wtss[l], w13, s13, w2, s2)
log('v2_full_cold_us', graph_us([(lambda l=l: md.moe_decode_v2(xs[l], idss[l], wtss[l], w13, s13, w2, s2)) for l in range(L)]))
log('v2_full_hot_us', graph_us([(lambda: md.moe_decode_v2(xs[0], idss[0], wtss[0], w13, s13, w2, s2))] * L))
gres = {}
for BN, GPB, W in [(16, 16, 4), (16, 32, 4), (8, 16, 2), (8, 32, 2), (8, 32, 4), (4, 32, 2), (4, 16, 1),
                   (16, 8, 2), (32, 16, 4), (32, 32, 8), (8, 8, 1), (4, 32, 1)]:
    k = f'bn{BN}_gpb{GPB}_w{W}'
    try:
        f = lambda l, BN=BN, GPB=GPB, W=W: d3._moe_gate_up_act_v3[(I // BN, TOPK)](
            xs[l], idss[l], w13, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=BN, GPB=GPB, num_warps=W)
        gres[k] = graph_us([(lambda l=l: f(l)) for l in range(L)])
    except Exception as ex:
        gres[k] = repr(ex)[:200]
    print('gate_up', k, gres[k], flush=True)
R['gate_up_cold'] = gres
dres = {}
for BN, GPB, W in [(16, 8, 2), (8, 8, 1), (8, 8, 2), (4, 8, 1), (4, 8, 2), (16, 8, 4), (32, 8, 4), (2, 8, 1),
                   (8, 4, 1), (16, 4, 2), (4, 4, 1)]:
    k = f'bn{BN}_gpb{GPB}_w{W}'
    try:
        f = lambda l, BN=BN, GPB=GPB, W=W: d3._moe_down_sum_v3[(H // BN, 1)](
            acts[l], idss[l], wtss[l], w2, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, GPB=GPB,
            ADD_SHARED=True, num_warps=W)
        dres[k] = graph_us([(lambda l=l: f(l)) for l in range(L)])
    except Exception as ex:
        dres[k] = repr(ex)[:200]
    print('down', k, dres[k], flush=True)
R['down_cold'] = dres
(root / 'dec3test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
