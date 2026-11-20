"""3D-x MoE GEMV sweep for MoE GEMVs."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1100, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode4 as d4
from vllm.triton_utils import tl, triton
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


L = 47
H, E, TOPK, I = 2048, 256, 6, 512
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
acts = [torch.empty(TOPK, I, dtype=torch.float32, device=dev) for _ in range(L)]
out = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
def gate(l, BN, NW, W, WS=32):
    d4._moe_gate_up_3[(I // BN, TOPK)](xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=BN, NW=NW, num_warps=W, warp_size=WS)
def down(l, BN, W, WS=32):
    d4._moe_down_3[(H // BN, 1)](acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True, num_warps=W, warp_size=WS)
for l in range(L): gate(l, 4, 64, 1)
log('gate3_relerr', ((acts[0].double().cpu() - a_ref).abs().max() / a_ref.abs().max()).item())
refo = (torch.einsum('ehi,ei->eh', W2, acts[0].double().cpu()) * wtss[0][0].double().cpu()[:, None]).sum(0) + shared[0].double().cpu()
down(0, 1, 1)
log('down3_relerr', ((out[0].double().cpu() - refo).abs().max() / refo.abs().max()).item())
gres = {}
for BN, NW, W, WS in [(1, 64, 1, 32), (1, 128, 1, 32), (1, 256, 1, 32), (2, 64, 1, 32), (2, 128, 1, 32), (4, 32, 1, 32),
                      (4, 64, 1, 32), (4, 64, 2, 32), (4, 128, 2, 32), (8, 32, 1, 32), (8, 64, 2, 32), (8, 64, 4, 32),
                      (16, 32, 2, 32), (2, 64, 1, 16), (4, 64, 1, 16), (1, 128, 1, 16)]:
    k = f'bn{BN}_nw{NW}_w{W}_ws{WS}'
    try:
        gres[k] = graph_us([(lambda l=l: gate(l, BN, NW, W, WS)) for l in range(L)])
    except Exception as ex:
        gres[k] = repr(ex)[:200]
    print('gate', k, gres[k], flush=True)
R['gate3'] = gres
dres = {}
for BN, W, WS in [(1, 1, 32), (1, 1, 16), (2, 1, 32), (2, 1, 16), (4, 1, 32), (4, 2, 32), (8, 2, 32), (8, 4, 32), (16, 4, 32)]:
    k = f'bn{BN}_w{W}_ws{WS}'
    try:
        dres[k] = graph_us([(lambda l=l: down(l, BN, W, WS)) for l in range(L)])
    except Exception as ex:
        dres[k] = repr(ex)[:200]
    print('down', k, dres[k], flush=True)
R['down3'] = dres
(root / 'dec6test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
