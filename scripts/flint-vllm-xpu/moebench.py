import faulthandler, itertools, json, pathlib, sys
faulthandler.dump_traceback_later(400, exit=True)
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
res = {}
for M in (1, 8):
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(M)]).to(torch.int32)
    w = torch.rand(M, TOPK, device=dev); w = w / w.sum(-1, keepdim=True)
    base = md.moe_decode(x, ids, w, w13, s13, w2, s2)
    r = {'v1': graph_us(lambda: md.moe_decode(x, ids, w, w13, s13, w2, s2))}
    for BN1, GPB1, W1, BN2, GPB2, W2 in [(16, 4, 2, 32, 4, 2), (16, 8, 4, 32, 8, 4), (8, 4, 2, 16, 4, 2), (32, 4, 4, 64, 4, 4), (16, 2, 2, 16, 8, 2), (8, 8, 2, 32, 8, 4)]:
        key = f'v2 bn1={BN1} gpb1={GPB1} w1={W1} bn2={BN2} gpb2={GPB2} w2={W2}'
        try:
            f = lambda: md.moe_decode_v2(x, ids, w, w13, s13, w2, s2, BN1, GPB1, W1, BN2, GPB2, W2)
            o = f(); torch.xpu.synchronize()
            rel = ((o.float() - base.float()).norm() / base.float().norm()).item()
            r[key] = (round(graph_us(f), 2), rel)
        except Exception as ex:
            r[key] = repr(ex)[:200]
        print(M, key, r[key], flush=True)
    res[M] = r
    print(M, 'v1', r['v1'], flush=True)
# bandwidth reference: bytes touched per token
bytes_tok = TOPK * (2 * I * H // 2 + 2 * I * H // 64 * 2 + H * I // 2 + H * I // 64 * 2)
res['bytes_per_token'] = bytes_tok
print('bytes/token', bytes_tok, '-> us at 550GB/s', bytes_tok / 550e3, flush=True)
(root / 'moebench.json').write_text(json.dumps(res, default=str))
(root / 'offline.done').write_text('ok\n')
