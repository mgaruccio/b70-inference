import faulthandler, json, os, pathlib, sys
faulthandler.dump_traceback_later(600, exit=True)
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch, torch.nn.functional as F
from safetensors import safe_open
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import vllm_xpu_kernels.fused_moe_interface as _fmi
sys.path.insert(0, str(pathlib.Path(__file__).parent / 'ktest'))
import xpu_moe_decode as md
root = pathlib.Path(sys.argv[1]); here = pathlib.Path.cwd()
ref = json.loads((here / 'probe-ref.json').read_text())
llm = LLM(model=str(here / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN', enforce_eager=True,
          max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096, enable_prefix_caching=False,
          gpu_memory_utilization=0.90)
LAYERS = (3, 24)
CAP = {'cur': None, 'on': False, 'rec': {}}
_orig = _fmi.XpuFusedMoe.apply
def _apply(self, output, hidden_states, topk_weights, topk_ids, *a, **k):
    r = _orig(self, output, hidden_states, topk_weights, topk_ids, *a, **k)
    if CAP['on'] and CAP['cur'] is not None:
        CAP['rec'].setdefault(CAP['cur'], []).append({'x': hidden_states.clone(), 'ids': topk_ids.clone(),
                                                      'w': topk_weights.clone(), 'out': output.clone(), 'impl': self})
    return r
_fmi.XpuFusedMoe.apply = _apply
def install(model):
    info = {}
    for l in LAYERS:
        mlp = model.model.layers[l].mlp
        def ph(m, a, l=l): CAP['cur'] = l
        def fh(m, a, o): CAP['cur'] = None
        mlp.register_forward_pre_hook(ph); mlp.register_forward_hook(fh)
        info[l] = [(n, list(p.shape), str(p.dtype), p.is_contiguous()) for n, p in mlp.experts.named_parameters()]
    return info
print('params', json.dumps(llm.apply_model(install))[:3000], flush=True)
CAP['on'] = True
llm.generate([TokensPrompt(prompt_token_ids=ref[1]['prompt_ids'])], SamplingParams(temperature=0, max_tokens=9, ignore_eos=True))
CAP['on'] = False
mdir = here / 'gptq'
idx = json.loads((mdir / 'model.safetensors.index.json').read_text())['weight_map']
_h = {}
def T(n):
    fn = idx[n]
    if fn not in _h: _h[fn] = safe_open(str(mdir / fn), 'pt', device='cpu')
    return _h[fn].get_tensor(n)
SH = torch.arange(0, 32, 4, dtype=torch.int32)
def deq(p):
    qw, qz, sc = T(p + '.qweight'), T(p + '.qzeros'), T(p + '.scales').float()
    K8, N = qw.shape
    q = ((qw.unsqueeze(1) >> SH.view(1, 8, 1)) & 0xF).reshape(K8 * 8, N).float()
    z = ((qz.unsqueeze(2) >> SH.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1).float()
    g = (K8 * 8) // sc.shape[0]
    return ((q - (z + 1).repeat_interleave(g, 0)) * sc.repeat_interleave(g, 0)).t().contiguous()
def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm()).item()
def bench(f, n=100):
    f(); torch.xpu.synchronize()
    s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.xpu.synchronize(); return s.elapsed_time(e) / n * 1000
results = {}
for l in LAYERS:
    recs = CAP['rec'][l]
    impl = recs[0]['impl']
    w13, w2, s13, s2 = impl.w13, impl.w2, impl.gemm1_wei_scales, impl.gemm2_wei_scales
    meta = {k: (list(t.shape), str(t.dtype), t.is_contiguous()) for k, t in (('w13', w13), ('w2', w2), ('s13', s13), ('s2', s2))}
    print(l, 'impl tensors', meta, flush=True)
    s13c, s2c = s13.contiguous(), s2.contiguous()
    for tag, r in (('prefill', recs[0]), ('decode', recs[1])):
        x, ids, w = r['x'], r['ids'], r['w']
        mine = md.moe_decode(x.contiguous(), ids, w, w13, s13c, w2, s2c)
        torch.xpu.synchronize()
        # fp32 reference with the same routing
        X = x.float().cpu(); I_ = ids.cpu(); W = w.float().cpu()
        refo = torch.zeros_like(X); cache = {}
        pf = f'model.layers.{l}.mlp.experts'
        for t in range(X.shape[0]):
            for j in range(I_.shape[1]):
                e = int(I_[t, j])
                if e not in cache:
                    cache[e] = (torch.cat([deq(f'{pf}.{e}.gate_proj'), deq(f'{pf}.{e}.up_proj')]), deq(f'{pf}.{e}.down_proj'))
                gu, dn = cache[e]
                g_, u_ = (X[t] @ gu.t()).chunk(2)
                refo[t] += W[t, j] * ((F.silu(g_) * u_) @ dn.t())
        res = {'M': x.shape[0], 'mine_vs_fp32': rel(mine.cpu(), refo), 'cutlass_vs_fp32': rel(r['out'].cpu(), refo),
               'mine_vs_cutlass': rel(mine.cpu(), r['out'].cpu())}
        if tag == 'decode':
            x8 = recs[1]['x'].repeat(8, 1).contiguous(); ids8 = ids.repeat(8, 1); w8 = w.repeat(8, 1)
            m8 = md.moe_decode(x8, ids8, w8, w13, s13c, w2, s2c)
            res['batch_invariant_M8'] = bool(all(torch.equal(m8[i], mine[0]) for i in range(8)))
            res['deterministic'] = bool(torch.equal(md.moe_decode(x.contiguous(), ids, w, w13, s13c, w2, s2c), mine))
            res['us_mine_M1'] = bench(lambda: md.moe_decode(x.contiguous(), ids, w, w13, s13c, w2, s2c))
            res['us_mine_M8'] = bench(lambda: md.moe_decode(x8, ids8, w8, w13, s13c, w2, s2c))
            o = torch.empty_like(r['out'])
            res['us_cutlass_M1'] = bench(lambda: _orig(impl, o, x, w, ids))
            o8 = torch.empty((8, x.shape[1]), dtype=x.dtype, device=x.device)
            res['us_cutlass_M8'] = bench(lambda: _orig(impl, o8, x8, w8, ids8))
        results[f'{l}-{tag}'] = res
        print(l, tag, json.dumps(res), flush=True)
(root / 'moetest.json').write_text(json.dumps(results))
(root / 'offline.done').write_text('ok\n')
