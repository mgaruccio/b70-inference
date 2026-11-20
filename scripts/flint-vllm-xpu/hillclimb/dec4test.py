"""Word-load MoE GEMV + top-k(1 warp) tests and cold-weight graph microbench."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1100, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode3 as d3
import xpu_decode4 as d4
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
    g.replay(); torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.xpu.synchronize()
    return round(s.elapsed_time(e) / (iters * len(fns)) * 1000, 2)

H, E, TOPK, I, L = 2048, 256, 6, 512, 47
# router top-k kernel variants
Wr = (torch.randn(L, E, H, device=dev) * 0.02).to(torch.bfloat16)
x1 = torch.randn(1, H, device=dev).to(torch.bfloat16)
KS, BN = 4, 8
part = [torch.empty(1, KS, E, device=dev) for _ in range(L)]
lg = torch.empty(1, E, device=dev); ids = torch.empty(1, TOPK, dtype=torch.int32, device=dev)
wts = torch.empty(1, TOPK, device=dev)
def rp(l): d3._router_partial[(E // BN, KS, 1)](x1, Wr[l], part[l], 1, N=E, K=H, BN=BN, KS=KS, RB=1, num_warps=4)
for l in range(L): rp(l)
log('router_partial_cold', graph_us([(lambda l=l: rp(l)) for l in range(L)]))
for W in (1, 2, 4):
    log(f'topk1_w{W}', graph_us([(lambda l=l: d4._router_topk1[(1,)](part[l], lg, ids, wts, N=E, KS=KS, TOPK=TOPK, TK2=8, num_warps=W)) for l in range(L)]))
    log(f'topk_d3_w{W}', graph_us([(lambda l=l: d3._router_topk[(1,)](part[l], lg, ids, wts, N=E, KS=KS, TOPK=TOPK, TK2=8, num_warps=W)) for l in range(L)]))
    log(f'router_pair_w{W}', graph_us([f for l in range(L) for f in ((lambda l=l: rp(l)), (lambda l=l: d4._router_topk1[(1,)](part[l], lg, ids, wts, N=E, KS=KS, TOPK=TOPK, TK2=8, num_warps=W)))]) * 2)
ref_ids = d3.router_topk(x1, Wr[0], TOPK)[1]
d4._router_topk1[(1,)](part[0], lg, ids, wts, N=E, KS=KS, TOPK=TOPK, TK2=8, num_warps=1)
log('topk1_eq', torch.equal(ids, ref_ids))
del Wr

w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
w13w = w13.view(torch.int32); w2w = w2.view(torch.int32)
xs = [torch.randn(1, H, device=dev).to(torch.bfloat16) for _ in range(L)]
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
refo = (torch.einsum('ehi,ei->eh', W2, a_ref) * wtss[0][0].double().cpu()[:, None]).sum(0) + shared[0].double().cpu()
acts = [torch.empty(TOPK, I, dtype=torch.float32, device=dev) for _ in range(L)]
out = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
def gate(l, BN, NW, W):
    d4._moe_gate_up_w[(I // BN, TOPK)](xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=BN, NW=NW, num_warps=W)
def down3(l, BN, W):
    d4._moe_down_w[(H // BN, 1)](acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True, num_warps=W)
def downl(l, BN, W):
    d4._moe_down_wl[(H // BN, 1)](acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True, num_warps=W)
# correctness
gate(0, 8, 64, 2)
log('gate_relerr', ((acts[0].double().cpu() - a_ref).abs().max() / a_ref.abs().max()).item())
for nm, fn in (('down3', down3), ('downl', downl)):
    fn(0, 4, 2)
    log(f'{nm}_relerr', ((out[0].double().cpu() - refo).abs().max() / refo.abs().max()).item())
# baseline in this run
f = lambda l: d3._moe_gate_up_act_v3[(I // 16, TOPK)](xs[l], idss[l], w13, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=16, GPB=16, num_warps=4)
log('base_gate_v3_bn16_gpb16_w4', [graph_us([(lambda l=l: f(l)) for l in range(L)]) for _ in range(2)])
f2 = lambda l: d3._moe_down_sum_v3[(H // 2, 1)](acts[l], idss[l], wtss[l], w2, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=2, GPB=8, ADD_SHARED=True, num_warps=1)
log('base_down_v3_bn2_w1', [graph_us([(lambda l=l: f2(l)) for l in range(L)]) for _ in range(2)])
gres = {}
for BN, NW, W in [(8, 64, 2), (8, 128, 2), (8, 256, 4), (4, 64, 1), (4, 128, 2), (4, 256, 2), (16, 64, 4),
                  (16, 128, 4), (2, 128, 1), (2, 256, 2), (8, 32, 1), (4, 32, 1), (16, 32, 2), (32, 64, 8)]:
    k = f'bn{BN}_nw{NW}_w{W}'
    try:
        gres[k] = graph_us([(lambda l=l: gate(l, BN, NW, W)) for l in range(L)])
    except Exception as ex:
        gres[k] = repr(ex)[:200]
    print('gate', k, gres[k], flush=True)
R['gate_w_cold'] = gres
dres = {}
for nm, fn in (('down3', down3), ('downl', downl)):
    for BN, W in [(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4), (16, 4), (16, 8)]:
        k = f'{nm}_bn{BN}_w{W}'
        try:
            dres[k] = graph_us([(lambda l=l: fn(l, BN, W)) for l in range(L)])
        except Exception as ex:
            dres[k] = repr(ex)[:200]
        print('down', k, dres[k], flush=True)
R['down_w_cold'] = dres
(root / 'dec4test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
