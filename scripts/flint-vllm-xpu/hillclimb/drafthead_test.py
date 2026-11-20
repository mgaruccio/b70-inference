"""int8 draft lm_head: accuracy vs bf16 (real Flint lm_head weights) and graph timing."""
import faulthandler, json, pathlib, sys
faulthandler.dump_traceback_later(1200, exit=True)
import torch
from safetensors import safe_open
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here))
import xpu_draft_head as dh
root = pathlib.Path(sys.argv[1]); dev = 'xpu'; torch.manual_seed(0)
R = {}
def log(k, v):
    R[k] = v; print(k, v, flush=True)
gp = pathlib.Path.home() / 'flint-vllm-spike/gptq'
idx = json.loads((gp / 'model.safetensors.index.json').read_text())['weight_map']
with safe_open(str(gp / idx['lm_head.weight']), 'pt') as f:
    w = f.get_tensor('lm_head.weight').to(dev)
log('w', [list(w.shape), str(w.dtype)])
q, s = dh.quantize_rows(w)
def graph_us(fn, iters=20):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    for _ in range(5): g.replay()
    torch.xpu.synchronize()
    res = []
    for _ in range(5):
        a, b = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        a.record()
        for _ in range(iters): g.replay()
        b.record(); torch.xpu.synchronize()
        res.append(a.elapsed_time(b) / iters * 1000)
    return round(sorted(res)[2], 1)
# hidden states with a realistic scale: rows of the embedding-like weight itself, perturbed
for M in (1, 3, 4, 7, 8):
    x = (w[torch.randint(0, w.shape[0], (M,), device=dev)].float() * 20
         + torch.randn(M, w.shape[1], device=dev)).to(torch.bfloat16).contiguous()
    ref = (x.float() @ w.float().t())
    bf = x @ w.t()
    i8 = dh.i8_head(x, q, s)
    rel = ((i8.float() - ref).abs().max() / ref.abs().max()).item()
    relbf = ((bf.float() - ref).abs().max() / ref.abs().max()).item()
    top_ref = ref.topk(16, dim=1).indices; top_i8 = i8.float().topk(16, dim=1).indices
    ov = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_ref, top_i8)) / (16 * M)
    top1 = (ref.argmax(1) == i8.float().argmax(1)).float().mean().item()
    t_bf = graph_us(lambda: x @ w.t())
    t_i8 = graph_us(lambda: dh.i8_head(x, q, s))
    log(f'M{M}', dict(relerr_i8=rel, relerr_bf16=relbf, top16_overlap=ov, top1=top1,
                      us_bf16=t_bf, us_i8=t_i8))
x = (torch.randn(4, w.shape[1], device=dev)).to(torch.bfloat16).contiguous()
sw = {}
for bn, bk, nw in [(64, 64, 4), (128, 64, 4), (64, 128, 4), (128, 128, 8), (256, 64, 8),
                   (64, 64, 2), (32, 128, 2), (128, 32, 4)]:
    dh.BN, dh.BK, dh.NWARPS = bn, bk, nw
    try:
        sw[f'{bn}_{bk}_{nw}'] = graph_us(lambda: dh.i8_head(x, q, s))
    except Exception as ex:
        sw[f'{bn}_{bk}_{nw}'] = repr(ex)[:120]
    print(bn, bk, nw, sw[f'{bn}_{bk}_{nw}'], flush=True)
log('sweep_M4', sw)
(root / 'drafthead_test.json').write_text(json.dumps(R, indent=1))
(root / 'offline.done').write_text('ok\n')
