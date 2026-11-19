import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(500, exit=True)
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent / 'ktest'))
import xpu_moe_decode as md
root = pathlib.Path(sys.argv[1]); dev = 'xpu'; torch.manual_seed(0)
E, H, I, TOPK = 256, 2048, 512, 6
w13 = torch.randint(-128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device=dev)
w2 = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device=dev)
s13 = (torch.rand(E, 2 * I, H // 64, device=dev) * 0.01).to(torch.bfloat16)
s2 = (torch.rand(E, H, I // 64, device=dev) * 0.01).to(torch.bfloat16)
def graph_us(fn, reps=20, iters=20):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(reps): fn()
    g.replay(); torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.xpu.synchronize()
    return s.elapsed_time(e) / (iters * reps) * 1000
M = 1
x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
ids = torch.randperm(E, device=dev)[:TOPK].view(1, TOPK).to(torch.int32)
wts = torch.rand(M, TOPK, device=dev); wts = (wts / wts.sum(-1, keepdim=True)).float()
act = torch.empty((M * TOPK, I), dtype=torch.float32, device=dev)
out = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
ref_act = act.clone(); ref_out = out.clone()
md._moe_gate_up_act_v2[(TOPK, I // 8)](x, ids, w13, s13, ref_act, H=H, I=I, TOPK=TOPK, BN=8, GPB=8, num_warps=2)
md._moe_down_sum_v2[(1, H // 32)](ref_act, ids, wts, w2, s2, ref_out, H=H, I=I, TOPK=TOPK, BN=32, GPB=8, num_warps=4)
res = {'gate_up': {}, 'down': {}}
for BN, GPB, W in [(8, 8, 2), (8, 16, 2), (8, 32, 4), (4, 8, 1), (4, 16, 2), (4, 32, 2), (16, 16, 4), (8, 8, 1), (8, 16, 4), (2, 32, 1), (16, 32, 8)]:
    k = f'bn={BN} gpb={GPB} w={W}'
    try:
        f = lambda: md._moe_gate_up_act_v2[(TOPK, I // BN)](x, ids, w13, s13, act, H=H, I=I, TOPK=TOPK, BN=BN, GPB=GPB, num_warps=W)
        f(); torch.xpu.synchronize()
        res['gate_up'][k] = (round(graph_us(f), 2), ((act - ref_act).abs().max() / ref_act.abs().max()).item())
    except Exception as ex:
        res['gate_up'][k] = repr(ex)[:160]
    print('gate_up', k, res['gate_up'][k], flush=True)
for BN, GPB, W in [(32, 8, 4), (16, 8, 2), (64, 8, 4), (32, 8, 2), (16, 8, 4), (8, 8, 1), (8, 8, 2), (16, 8, 1), (4, 8, 1)]:
    k = f'bn={BN} gpb={GPB} w={W}'
    try:
        f = lambda: md._moe_down_sum_v2[(1, H // BN)](ref_act, ids, wts, w2, s2, out, H=H, I=I, TOPK=TOPK, BN=BN, GPB=GPB, num_warps=W)
        f(); torch.xpu.synchronize()
        res['down'][k] = (round(graph_us(f), 2), ((out.float() - ref_out.float()).abs().max()).item())
    except Exception as ex:
        res['down'][k] = repr(ex)[:160]
    print('down', k, res['down'][k], flush=True)
print('ideal us: gate_up', TOPK * (2 * I * H // 2 + 2 * I * H // 32) / 550e3, 'down', TOPK * (H * I // 2 + H * I // 32) / 550e3, flush=True)
(root / 'moebench2.json').write_text(json.dumps(res))
(root / 'offline.done').write_text('ok\n')
