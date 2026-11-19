"""Run identical greedy requests repeatedly in-process; locate first (step, layer, op) that differs."""
import json, os, pathlib, sys
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import vllm_xpu_kernels.fused_moe_interface as _fmi

root = pathlib.Path(sys.argv[1]); here = pathlib.Path.cwd()
ref = json.loads((here / 'probe-ref.json').read_text())
spec = os.environ.get('REPEAT_SPEC')
kw = dict(model=str(here / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN', enforce_eager=True,
          max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=int(os.environ.get('REPEAT_MNBT', '512')), enable_prefix_caching=False,
          gpu_memory_utilization=0.90)
if spec:
    kw['speculative_config'] = {'method': 'dflash', 'model': str(here / 'dflash2'), 'num_speculative_tokens': 7}
llm = LLM(**kw)
CAP = {'on': False, 'calls': [], 'cur': None}

_orig = _fmi.XpuFusedMoe.apply
def _apply(self, output, hidden_states, topk_weights, topk_ids, *a, **k):
    if CAP['on'] and CAP['cur'] is not None:
        c = CAP['calls'][-1]
        c.setdefault('tids', {})[CAP['cur']] = topk_ids.detach().cpu().clone()
        c.setdefault('w', {})[CAP['cur']] = topk_weights.detach().float().cpu().clone()
    return _orig(self, output, hidden_states, topk_weights, topk_ids, *a, **k)
_fmi.XpuFusedMoe.apply = _apply

def install(model):
    m = model.model if hasattr(model, 'model') else model
    n = 0
    for i, layer in enumerate(m.layers):
        def lh(mod, inp, out, i=i):
            if CAP['on'] and CAP['calls']:
                h, r = out
                CAP['calls'][-1].setdefault('res', {})[i] = (h.float() + r.float()).cpu()
        layer.register_forward_hook(lh); n += 1
        def ah(mod, inp, out, i=i):
            if CAP['on'] and CAP['calls']:
                CAP['calls'][-1].setdefault('attn', {})[i] = out.detach().float().cpu()
                CAP['calls'][-1].setdefault('attn_in', {})[i] = inp[1].detach().float().cpu()
        layer.self_attn.register_forward_hook(ah)
        if hasattr(layer.mlp, 'gate'):
            def gh(mod, inp, out, i=i):
                if CAP['on'] and CAP['calls']:
                    o = out[0] if isinstance(out, tuple) else out
                    CAP['calls'][-1].setdefault('gate', {})[i] = o.detach().float().cpu()
                    CAP['calls'][-1].setdefault('gate_in', {})[i] = inp[0].detach().float().cpu()
            layer.mlp.gate.register_forward_hook(gh)
            def ph(mod, args, i=i): CAP['cur'] = i
            def fh(mod, args, out): CAP['cur'] = None
            layer.mlp.register_forward_pre_hook(ph); layer.mlp.register_forward_hook(fh)
    def pre(mod, args, kwargs):
        if CAP['on']:
            pos = kwargs.get('positions', args[1] if len(args) > 1 else None)
            ids = kwargs.get('input_ids', args[0] if args else None)
            CAP['calls'].append({'pos': pos.detach().cpu().clone(), 'ids': None if ids is None else ids.detach().cpu().clone()})
    m.register_forward_pre_hook(pre, with_kwargs=True)
    return n
print('hooks', llm.apply_model(install), flush=True)

def _zero_kv(model):
    m = model.model if hasattr(model, 'model') else model
    n = 0
    for layer in m.layers:
        kv = layer.self_attn.attn.kv_cache
        if torch.is_tensor(kv) and kv.numel():
            kv.zero_(); n += 1
    torch.xpu.synchronize()
    return n
def run(ids, n=int(os.environ.get('REPEAT_N', '48'))):
    if os.environ.get('REPEAT_ZERO_KV'):
        print('zeroed kv layers', llm.apply_model(_zero_kv), flush=True)
    CAP['calls'] = []; CAP['on'] = True
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    CAP['on'] = False
    return list(out[0].outputs[0].token_ids), CAP['calls']

P1, P2 = ref[0]['prompt_ids'], ref[1]['prompt_ids']
seq = [('A', P2), ('B', P1), ('A', P2), ('A', P2), ('B', P1), ('A', P2)]
if os.environ.get('REPEAT_LONG'):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(here / 'gptq'))
    prompts = json.loads((here / 'benchmark-prompts.json').read_text())
    lp = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
    P2 = tk(lp)['input_ids']
    seq = [('A', P2), ('B', P1), ('A', P2), ('A', P2), ('B', P1), ('A', P2)]
runs = []
for name, ids in seq:
    t, calls = run(ids)
    runs.append((name, t, calls))
    print(name, t[:12], flush=True)

def cmp(c1, c2):
    """first differing thing between two call lists"""
    for s, (a, b) in enumerate(zip(c1, c2)):
        if not torch.equal(a['pos'], b['pos']) or (a['ids'] is not None and not torch.equal(a['ids'], b['ids'])):
            return {'step': s, 'what': 'inputs', 'pos': a['pos'].tolist()[:4], 'ids_a': a['ids'].tolist()[:8], 'ids_b': b['ids'].tolist()[:8]}
        for l in sorted(a.get('res', {})):
            if l in a.get('attn_in', {}) and not torch.equal(a['attn_in'][l], b['attn_in'][l]):
                d = (a['attn_in'][l] - b['attn_in'][l]).abs()
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'attn_input(hidden)', 'maxabs': d.max().item(), 'rows': d.amax(-1).nonzero().flatten().tolist()[:8]}
            if l in a.get('attn', {}) and not torch.equal(a['attn'][l], b['attn'][l]):
                d = (a['attn'][l] - b['attn'][l]).abs()
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'attn_output(same input)', 'maxabs': d.max().item(), 'rows': d.amax(-1).nonzero().flatten().tolist()[:8], 'pos_first': a['pos'][0].item()}
            if l in a.get('gate_in', {}) and not torch.equal(a['gate_in'][l], b['gate_in'][l]):
                d = (a['gate_in'][l] - b['gate_in'][l]).abs()
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'gate_input', 'maxabs': d.max().item(), 'rows': d.amax(-1).nonzero().flatten().tolist()[:8]}
            if l in a.get('gate', {}) and not torch.equal(a['gate'][l], b['gate'][l]):
                d = (a['gate'][l] - b['gate'][l]).abs()
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'gate_logits(same input)', 'maxabs': d.max().item(), 'rows': d.amax(-1).nonzero().flatten().tolist()[:8]}
            if l in a.get('tids', {}) and not torch.equal(a['tids'][l], b['tids'][l]):
                rows = (a['tids'][l] != b['tids'][l]).any(-1).nonzero().flatten().tolist()
                r = rows[0]
                g = a['gate'][l][r]
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'topk_ids(same logits)', 'rows': rows[:8],
                        'ids_a': a['tids'][l][r].tolist(), 'ids_b': b['tids'][l][r].tolist(),
                        'true_top8': torch.topk(g, 8).indices.tolist(), 'true_top8_vals': torch.topk(g, 8).values.tolist(),
                        'w_a': a['w'][l][r].tolist(), 'w_b': b['w'][l][r].tolist()}
            if l in a.get('w', {}) and not torch.equal(a['w'][l], b['w'][l]):
                return {'step': s, 'layer': l, 'what': 'topk_weights(same ids)', 'maxabs': (a['w'][l] - b['w'][l]).abs().max().item()}
            if not torch.equal(a['res'][l], b['res'][l]):
                d = (a['res'][l] - b['res'][l]).abs()
                return {'step': s, 'M': a['pos'].numel(), 'layer': l, 'what': 'layer_output', 'maxabs': d.max().item(), 'rows': d.amax(-1).nonzero().flatten().tolist()[:8]}
    return None

# also: check vllm topk ids against true fp32 topk on every call of every run
bad = []
for k, (name, t, calls) in enumerate(runs):
    for s, c in enumerate(calls):
        for l, ids in c.get('tids', {}).items():
            tr = torch.topk(c['gate'][l], ids.shape[1], -1).indices
            for r in range(ids.shape[0]):
                if set(ids[r].tolist()) != set(tr[r].tolist()):
                    g = c['gate'][l][r]
                    bad.append({'run': k, 'step': s, 'M': ids.shape[0], 'layer': l, 'row': r, 'vllm': ids[r].tolist(),
                                'true': tr[r].tolist(), 'vals_true_top8': torch.topk(g, 8).values.tolist(),
                                'vllm_vals': g[ids[r].long()].tolist() if ids[r].min() >= 0 else None})
def cascade(c1, c2, step=0):
    a, b = c1[step], c2[step]
    out = []
    for l in sorted(a.get('res', {})):
        x, y = a['res'][l], b['res'][l]
        row = (x - y).norm(dim=-1) / y.norm(dim=-1)
        ent = {'l': l, 'max_row_rel': row.max().item(), 'argrow': int(row.argmax())}
        if l in a.get('tids', {}):
            fl = (a['tids'][l].sort(-1).values != b['tids'][l].sort(-1).values).any(-1).nonzero().flatten().tolist()
            ent['route_flip_rows'] = fl[:6]
            if fl:
                r = fl[0]; g = a['gate'][l][r]; v = torch.topk(g, 8)
                ent['flip0'] = {'a': sorted(a['tids'][l][r].tolist()), 'b': sorted(b['tids'][l][r].tolist()),
                                'top8': v.indices.tolist(), 'top8v': [round(x, 7) for x in v.values.tolist()],
                                'gate_maxabs_row': (a['gate'][l][r] - b['gate'][l][r]).abs().max().item()}
        out.append(ent)
    return out
summary = {'tokens': [(n, t) for n, t, _ in runs], 'topk_vs_true_mismatches': len(bad), 'bad_first': bad[:10],
           'A0_vs_A2': cmp(runs[0][2], runs[2][2]), 'A2_vs_A3': cmp(runs[2][2], runs[3][2]),
           'A0_vs_A5': cmp(runs[0][2], runs[5][2]), 'B1_vs_B4': cmp(runs[1][2], runs[4][2]), 'cascade_A0_A2': cascade(runs[0][2], runs[2][2]), 'cascade_A0_A3': cascade(runs[0][2], runs[3][2])}
(root / 'repeat.json').write_text(json.dumps(summary, default=str))
print(json.dumps({k: v for k, v in summary.items() if k != 'tokens'}, default=str)[:4000], flush=True)
(root / 'offline.done').write_text('ok\n')
