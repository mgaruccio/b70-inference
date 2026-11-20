"""Real-weight check: GPTQ W4A16 linears (oneDNN) vs Triton GEMV; layouts, accuracy, cold timing."""
import faulthandler, json, os, pathlib, sys
faulthandler.dump_traceback_later(1500, exit=True)
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here)); sys.path.insert(0, str(here.parent))
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
    import xpu_decode4 as d4
    torch.manual_seed(0)
    layers = model.model.layers
    moe = [l for l in layers if hasattr(l.mlp, 'shared_expert')]
    dense = [l for l in layers if not hasattr(l.mlp, 'shared_expert')]
    mods = {'qkv': [l.self_attn.qkv_proj for l in layers], 'o': [l.self_attn.o_proj for l in layers],
            'dense_gate_up': [l.mlp.gate_up_proj for l in dense], 'dense_down': [l.mlp.down_proj for l in dense],
            'sh_gate_up': [l.mlp.shared_expert.gate_up_proj for l in moe],
            'sh_down': [l.mlp.shared_expert.down_proj for l in moe]}
    m0 = mods['qkv'][0]
    info = {n: (tuple(p.shape), str(p.dtype), tuple(p.stride()), p.is_contiguous()) for n, p in m0.named_parameters()}
    info['kernel'] = type(getattr(m0.quant_method, 'kernel', None)).__name__
    info['kernel_names'] = [getattr(getattr(m0.quant_method, 'kernel', None), a, None) for a in ('w_q_name', 'w_s_name', 'w_zp_name')]
    info['qm'] = type(m0.quant_method).__name__
    info['dense0'] = type(layers[0].mlp).__name__
    info['lm_head'] = (type(model.lm_head).__name__, tuple(model.lm_head.weight.shape), str(model.lm_head.weight.dtype))
    log('info', info)

    def wq(m):
        qw = m.qweight; sc = m.scales
        if not qw.is_contiguous() and qw.t().is_contiguous():
            qw = qw.t()
        return qw.data, sc.data
    res = {}
    for name, ms in mods.items():
        try:
            qw, sc = wq(ms[0])
            N, K = qw.shape[0], qw.shape[1] * 8
            ent = {'qw': (tuple(qw.shape), str(qw.dtype), qw.is_contiguous()), 'sc': (tuple(sc.shape), str(sc.dtype), sc.is_contiguous())}
            x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
            if name.endswith('down'):
                x = (torch.randn(1, K, device=dev) * 0.05).to(torch.bfloat16)
            y_dnn = ms[0](x)[0]
            # fp64 reference from the packed layout
            q = qw.cpu().to(torch.int64)
            nib = torch.stack([(q >> (4 * i)) & 15 for i in range(8)], -1).reshape(N, K).double() - 8
            W = nib * sc.cpu().double().t().repeat_interleave(64, 1)
            ref = x.cpu().double() @ W.t()
            y_tri = d4.gptq_gemv(x, qw, sc)
            y3 = torch.empty_like(y_tri); d4._gptq_gemv3[(N // 4, 1)](x, qw, sc, y3, N=N, K=K, BN=4, NW=64, SILU_MUL=False, num_warps=1)
            scale = ref.abs().max().item()
            ent['dnn_relerr'] = (y_dnn.cpu().double() - ref).abs().max().item() / scale
            ent['tri3_relerr'] = (y3.cpu().double() - ref).abs().max().item() / scale
            ent['tri_relerr'] = (y_tri.cpu().double() - ref).abs().max().item() / scale
            ent['dnn_us'] = graph_us([(lambda m=m: m(x)) for m in ms])
            ent['bytes_MB'] = round((qw.numel() * 4 + sc.numel() * sc.element_size()) / 1e6, 2)
            pairs = [wq(m) for m in ms]
            sweep = {}
            for BN, NW, W, WS in [(1, 64, 1, 32), (1, 128, 1, 32), (2, 64, 1, 32), (2, 128, 1, 32), (4, 64, 1, 32),
                                  (4, 128, 2, 32), (8, 64, 2, 32), (1, 128, 1, 16), (2, 64, 1, 16), (4, 64, 1, 16),
                                  (2, 256, 2, 32), (4, 32, 1, 32)]:
                k = f'bn{BN}_nw{NW}_w{W}_ws{WS}'
                try:
                    rows = N // 2 if name.endswith('gate_up') else N
                    o = torch.empty(1, rows, device=dev, dtype=torch.bfloat16)
                    nw = min(NW, K // 8)
                    for kn_, kern in (('v1', d4._gptq_gemv), ('v3', d4._gptq_gemv3)):
                        sweep[kn_ + '_' + k] = graph_us([(lambda p=p, kern=kern: kern[(rows // BN, 1)](
                            x, p[0], p[1], o, N=N, K=K, BN=BN, NW=nw, SILU_MUL=name.endswith('gate_up'),
                            num_warps=W, warp_size=WS)) for p in pairs])
                except Exception as ex:
                    sweep[k] = repr(ex)[:160]
            ent['tri'] = sweep
            if name.endswith('gate_up'):
                gu = ref[0]; I = N // 2
                refa = torch.nn.functional.silu(gu[:I]) * gu[I:]
                ya = d4.gptq_gemv(x, qw, sc, silu_mul=True)
                ent['silu_relerr'] = (ya[0].cpu().double() - refa).abs().max().item() / refa.abs().max().item()
            res[name] = ent
            print(name, ent, flush=True)
        except Exception as ex:
            import traceback; res[name] = traceback.format_exc()[-800:]; print(name, res[name], flush=True)
    log('linears', res)
    # lm_head bf16
    w = model.lm_head.weight
    x = torch.randn(1, w.shape[1], device=dev).to(torch.bfloat16)
    log('lm_head_us', graph_us([lambda: torch.nn.functional.linear(x, w)] * 4))
    return True

llm.apply_model(work)
(root / 'realtest.json').write_text(json.dumps(R, indent=1, default=str))
(root / 'offline.done').write_text('ok\n')
