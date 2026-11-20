"""Real-weight gate for the shared-expert pseudo-expert path (moe_decode_sh) vs the
current path (oneDNN shared + moe_decode), fp64 shared reference, invariance, timing."""
import faulthandler, json, os, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here))
from vllm import LLM
root = pathlib.Path(sys.argv[1]); dev = 'xpu'
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

llm = LLM(model=str(pathlib.Path.cwd() / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN',
          enforce_eager=True, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=512,
          enable_prefix_caching=False, gpu_memory_utilization=0.90)

def work(model):
    import xpu_decode as xd
    torch.manual_seed(0)
    blocks = [l.mlp for l in model.model.layers if hasattr(l.mlp, 'shared_expert')]
    def qw(m):
        q = m.qweight
        return (q.t() if not q.is_contiguous() and q.t().is_contiguous() else q).data, m.scales.data
    packs, cur, new = [], [], []
    for b in blocks:
        ex = b.experts.routed_experts
        gq, gs = qw(b.shared_expert.gate_up_proj); dq, ds = qw(b.shared_expert.down_proj)
        packs.append(xd.pack_shared_expert(gq, gs, dq, ds, ex.w13_qweight.shape[1] // 2))
    log('pack_shapes', [(tuple(t.shape), str(t.dtype)) for t in packs[0]])
    xs = [(torch.randn(1, 2048, device=dev) * 0.7).to(torch.bfloat16) for _ in blocks]
    def run_cur(i):
        b = blocks[i]; ex = b.experts.routed_experts
        _, ids, wts = xd.router(xs[i], b.gate.weight, b.top_k)
        sh = b.shared_expert(xs[i])
        return xd.moe_decode(xs[i], ids, wts, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales, shared=sh)
    def run_new(i, x=None):
        b = blocks[i]; ex = b.experts.routed_experts
        x = xs[i] if x is None else x
        _, ids, wts = xd.router(x, b.gate.weight, b.top_k)
        return xd.moe_decode_sh(x, ids, wts, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales, packs[i])
    rel = []
    for i in range(len(blocks)):
        a = run_cur(i).double(); c = run_new(i).double()
        rel.append(((a - c).abs().max() / a.abs().max()).item())
    log('new_vs_cur_relerr_max', max(rel)); log('new_vs_cur_relerr_med', sorted(rel)[len(rel) // 2])
    # shared-only fp64 reference (routed weights zeroed)
    b = blocks[3]; ex = b.experts.routed_experts
    _, ids, _ = xd.router(xs[3], b.gate.weight, b.top_k)
    z = torch.zeros(1, b.top_k, device=dev)
    so = xd.moe_decode_sh(xs[3], ids, z, ex.w13_qweight, ex.w13_scales, ex.w2_qweight, ex.w2_scales, packs[3])
    def deq(m):
        q, s = qw(m); q = q.cpu().to(torch.int64); N, K = q.shape[0], q.shape[1] * 8
        nib = torch.stack([(q >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).double() - 8
        return nib * s.cpu().double().t().repeat_interleave(64, 1)
    Wgu = deq(b.shared_expert.gate_up_proj); Wd = deq(b.shared_expert.down_proj)
    gu = xs[3].cpu().double() @ Wgu.t(); IS = gu.shape[1] // 2
    ref = (torch.nn.functional.silu(gu[:, :IS]) * gu[:, IS:]) @ Wd.t()
    dn = b.shared_expert(xs[3]).cpu().double()
    log('shared_only_relerr_new', ((so.cpu().double() - ref).abs().max() / ref.abs().max()).item())
    log('shared_only_relerr_onednn', ((dn - ref).abs().max() / ref.abs().max()).item())
    # batch invariance + determinism
    X = torch.cat([xs[0], xs[1], xs[2]], 0)
    o3 = run_new(0, X)
    log('inv_m3', all(torch.equal(run_new(0, X[t:t + 1]), o3[t:t + 1]) for t in range(3)))
    log('det', torch.equal(run_new(0, X), o3))
    n = len(blocks)
    log('cur_path_us', graph_us([(lambda i=i: run_cur(i)) for i in range(n)]))
    log('new_path_us', graph_us([(lambda i=i: run_new(i)) for i in range(n)]))
    for bn, nw, w, ws in [(1, 64, 1, 16), (2, 64, 1, 16), (1, 128, 1, 16), (1, 64, 1, 32)]:
        xd.SH_GATE_BN, xd.SH_GATE_NW, xd.SH_GATE_W, xd.SH_GATE_WS = bn, nw, w, ws
        log(f'new_path_us_gate_bn{bn}_nw{nw}_w{w}_ws{ws}', graph_us([(lambda i=i: run_new(i)) for i in range(n)]))
    return True

llm.apply_model(work)
(root / 'realsh.json').write_text(json.dumps(R, indent=1, default=str))
(root / 'offline.done').write_text('ok\n')
