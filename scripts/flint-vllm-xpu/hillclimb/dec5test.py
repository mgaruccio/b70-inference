"""Roofline for small cold reads + BN=1/threads_per_warp sweep for MoE GEMVs."""
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

@triton.jit
def _readsum(p, o, CH: tl.constexpr, NWI: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, NWI)
    acc = tl.zeros((NWI,), dtype=tl.int32)
    for c in range(0, CH, NWI):
        acc += tl.load(p + pid * CH + c + offs)
    tl.store(o + pid, tl.sum(acc, axis=0))

L = 47
buf = torch.randint(-2**31, 2**31 - 1, (L, 6 * 2**20 // 4 * 1 + 2**18), dtype=torch.int32, device=dev)  # ~6.3MB per slot
osum = torch.empty(65536, dtype=torch.int32, device=dev)
roof = {}
for MB, words in (('6.3', 6 * 1024 * 1024 // 4 + 2**18 - 2**18 + 1572864 - 1572864), ('3.1', 786432)):
    words = 1572864 if MB == '6.3' else 786432
    for CH, NWI, W, TPW in [(256, 256, 1, 32), (512, 256, 1, 32), (1024, 256, 2, 32), (2048, 512, 4, 32),
                            (4096, 1024, 4, 32), (256, 128, 1, 16), (512, 256, 1, 16), (1024, 256, 2, 16),
                            (2048, 512, 2, 16), (1024, 1024, 4, 32)]:
        k = f'{MB}MB_ch{CH}_nwi{NWI}_w{W}_tpw{TPW}'
        try:
            P = words // CH
            us = graph_us([(lambda l=l: _readsum[(P,)](buf[l], osum, CH=CH, NWI=NWI, num_warps=W, warp_size=TPW)) for l in range(L)])
            roof[k] = (us, round(words * 4 / us / 1e3, 1))
        except Exception as ex:
            roof[k] = repr(ex)[:200]
        print('roof', k, roof[k], flush=True)
R['roof'] = roof
del buf

H, E, TOPK, I = 2048, 256, 6, 512
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
w13w = w13.view(torch.int32); w2w = w2.view(torch.int32)
xs = [torch.randn(1, H, device=dev).to(torch.bfloat16) for _ in range(L)]
idss = [torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32) for _ in range(L)]
wtss = [torch.full((1, TOPK), 1.0 / TOPK, device=dev) for _ in range(L)]
shared = torch.randn(1, H, device=dev).to(torch.bfloat16)
acts = [torch.randn(TOPK, I, dtype=torch.float32, device=dev) for _ in range(L)]
out = torch.empty(1, H, dtype=torch.bfloat16, device=dev)
gres = {}
for BN, NW, W, TPW in [(1, 256, 1, 32), (1, 128, 1, 32), (1, 256, 1, 16), (1, 128, 1, 16), (2, 256, 1, 16),
                       (2, 128, 1, 16), (2, 256, 2, 32), (4, 128, 1, 16), (8, 64, 2, 32), (8, 128, 2, 16),
                       (4, 256, 2, 16), (1, 64, 1, 16)]:
    k = f'bn{BN}_nw{NW}_w{W}_tpw{TPW}'
    try:
        gres[k] = graph_us([(lambda l=l: d4._moe_gate_up_w[(I // BN, TOPK)](
            xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=BN, NW=NW, num_warps=W,
            warp_size=TPW)) for l in range(L)])
    except Exception as ex:
        gres[k] = repr(ex)[:200]
    print('gate', k, gres[k], flush=True)
R['gate'] = gres
dres = {}
for BN, W, TPW in [(1, 1, 32), (1, 1, 16), (2, 1, 16), (2, 1, 32), (4, 1, 16), (1, 2, 16), (4, 2, 16)]:
    k = f'downl_bn{BN}_w{W}_tpw{TPW}'
    try:
        dres[k] = graph_us([(lambda l=l: d4._moe_down_wl[(H // BN, 1)](
            acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=BN, ADD_SHARED=True,
            num_warps=W, warp_size=TPW)) for l in range(L)])
    except Exception as ex:
        dres[k] = repr(ex)[:200]
    print('down', k, dres[k], flush=True)
R['down'] = dres
# repeatability of the best-known configs
R['repeat'] = {
    'gate_bn8_nw64_w2': [graph_us([(lambda l=l: d4._moe_gate_up_w[(I // 8, TOPK)](
        xs[l], idss[l], w13w, s13, acts[l], H=H, I=I, TOPK=TOPK, BN=8, NW=64, num_warps=2)) for l in range(L)]) for _ in range(3)],
    'downl_bn1_w1': [graph_us([(lambda l=l: d4._moe_down_wl[(H, 1)](
        acts[l], idss[l], wtss[l], w2w, s2, shared, out, H=H, I=I, TOPK=TOPK, BN=1, ADD_SHARED=True,
        num_warps=1)) for l in range(L)]) for _ in range(3)]}
print('repeat', R['repeat'], flush=True)
(root / 'dec5test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
