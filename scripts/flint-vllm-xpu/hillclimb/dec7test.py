"""GEMV core v4 vs v3: accuracy (fp64) and cold-weight graph timing, MoE + GPTQ shapes."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
import xpu_decode as xd
import xpu_decode4 as d4
import xpu_decode5 as d5
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

# ---------------- GPTQ shapes
def gptq_ref(x, qw, sc):
    q = qw.cpu().to(torch.int64); N, K = q.shape[0], q.shape[1] * 8
    nib = torch.stack([(q >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).double() - 8
    return x.cpu().double() @ (nib * sc.cpu().double().t().repeat_interleave(64, 1)).t()
for name, N, K, L, silu in [('qkv', 5120, 2048, 49, False), ('o', 2048, 4096, 49, False),
                            ('sh_gu', 2048, 2048, 47, True), ('sh_down', 2048, 1024, 47, False),
                            ('dense_gu', 12288, 2048, 6, True), ('dense_down', 2048, 6144, 6, False)]:
    qws = [torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev) for _ in range(L)]
    scs = [(torch.rand(K // 64, N, device=dev) * 0.01).to(torch.bfloat16) for _ in range(L)]
    x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
    rows = N // 2 if silu else N
    o = torch.empty(1, rows, device=dev, dtype=torch.bfloat16)
    ref = gptq_ref(x, qws[0], scs[0])[0]
    if silu:
        ref = torch.nn.functional.silu(ref[:rows]) * ref[rows:]
    d5._gptq_gemv4[(rows // 2, 1)](x, qws[0], scs[0], o, N=N, K=K, BN=2, NW=64, SILU_MUL=silu, num_warps=1)
    log(f'{name}_v4_relerr', ((o[0].double().cpu() - ref).abs().max() / ref.abs().max()).item())
    def mk(kern, BN, NW, W, WS):
        nw = min(NW, K // 8)
        return [(lambda l=l: kern[(rows // BN, 1)](x, qws[l], scs[l], o, N=N, K=K, BN=BN, NW=nw, SILU_MUL=silu,
                                                    num_warps=W, warp_size=WS)) for l in range(L)]
    cfgs = [dict(BN=b, NW=n, W=w, WS=s) for b, n, w, s in
            [(1, 64, 1, 16), (1, 128, 1, 16), (2, 32, 1, 16), (2, 64, 1, 16), (2, 64, 1, 32), (2, 128, 1, 32),
             (4, 32, 1, 16), (4, 64, 1, 16), (4, 64, 1, 32), (4, 64, 2, 32), (8, 32, 1, 16), (8, 64, 2, 32),
             (16, 32, 2, 32), (8, 32, 2, 16)]]
    sweep(f'{name}_v4', lambda **c: mk(d5._gptq_gemv4, **c), cfgs)
    sweep(f'{name}_v3', lambda **c: mk(d4._gptq_gemv3, **c), [dict(BN=2, NW=64, W=1, WS=16), dict(BN=2, NW=64, W=1, WS=32), dict(BN=4, NW=32, W=1, WS=32)])
    del qws, scs

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
d5._moe_gate_up4[(I // 2, TOPK)](xs[0], idss[0], w13w, s13, acts[0], H=H, I=I, TOPK=TOPK, BN=2, NW=64, num_warps=1)
log('gate4_relerr', ((acts[0].double().cpu() - a_ref).abs().max() / a_ref.abs().max()).item())
refo = (torch.einsum('ehi,ei->eh', W2, acts[0].double().cpu()) * wtss[0][0].double().cpu()[:, None]).sum(0) + shared[0].double().cpu()
d5._moe_down4[(H, 1)](acts[0], idss[0], wtss[0], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=1, ADD_SHARED=True, num_warps=1)
log('down4_relerr', ((out[0].double().cpu() - refo).abs().max() / refo.abs().max()).item())
for l in range(L):
    xd.moe_decode(xs[l], idss[l], wtss[l], w13, s13, w2, s2, shared=shared)
log('cur_moe_full', graph_us([(lambda l=l: xd.moe_decode(xs[l], idss[l], wtss[l], w13, s13, w2, s2, shared=shared)) for l in range(L)]))
log('cur_gate', graph_us([(lambda l=l: xd._moe_gate_up[(I // 2, TOPK)](xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=2, NW=64, num_warps=1, warp_size=16)) for l in range(L)]))
log('cur_down', graph_us([(lambda l=l: xd._moe_down[(H, 1)](acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=1, ADD_SHARED=True, num_warps=1)) for l in range(L)]))
sweep('gate4', lambda BN, NW, W, WS: [(lambda l=l: d5._moe_gate_up4[(I // BN, TOPK)](
    xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=BN, NW=NW, num_warps=W, warp_size=WS)) for l in range(L)],
    [dict(BN=b, NW=n, W=w, WS=s) for b, n, w, s in
     [(1, 64, 1, 16), (1, 128, 1, 16), (2, 32, 1, 16), (2, 64, 1, 16), (2, 64, 1, 32), (2, 128, 1, 32),
      (4, 32, 1, 16), (4, 64, 1, 16), (4, 64, 1, 32), (4, 64, 2, 32), (8, 32, 1, 16), (8, 64, 2, 32)]])
sweep('down4', lambda BN, W, WS: [(lambda l=l: d5._moe_down4[(H // BN, 1)](
    acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True,
    num_warps=W, warp_size=WS)) for l in range(L)],
    [dict(BN=b, W=w, WS=s) for b, w, s in [(1, 1, 16), (1, 1, 32), (2, 1, 16), (2, 1, 32), (4, 1, 16), (4, 1, 32), (4, 2, 32), (8, 2, 32)]])
(root / 'dec7test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
