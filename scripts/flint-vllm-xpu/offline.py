"""Layerwise shape-consistency harness: decode (M=1) vs batched prefill on identical tokens."""
import json, os, pathlib, sys, time
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

root = pathlib.Path(sys.argv[1])
here = pathlib.Path.cwd()
ref = json.loads((here / 'probe-ref.json').read_text())

llm = LLM(model=str(here / 'gptq'), dtype='bfloat16', attention_backend='TRITON_ATTN',
          enforce_eager=True, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=512,
          enable_prefix_caching=False, gpu_memory_utilization=0.90)

CAP = {'on': False, 'calls': []}

def install(model):
    layers = model.model.layers
    hooks = []
    for i, layer in enumerate(layers):
        def lh(mod, inp, out, i=i):
            if not CAP['on']:
                return
            h, r = out
            CAP['calls'][-1].setdefault('res', {})[i] = (h.float() + r.float()).cpu()
        hooks.append(layer.register_forward_hook(lh))
        gate = getattr(layer.mlp, 'gate', None)
        if gate is not None:
            def gh(mod, inp, out, i=i):
                if not CAP['on']:
                    return
                o = out[0] if isinstance(out, tuple) else out
                CAP['calls'][-1].setdefault('gate', {})[i] = o.float().cpu()
            hooks.append(gate.register_forward_hook(gh))
    def pre(mod, args, kwargs):
        if CAP['on']:
            ids = kwargs.get('input_ids', args[0] if args else None)
            pos = kwargs.get('positions', args[1] if len(args) > 1 else None)
            CAP['calls'].append({'ids': None if ids is None else ids.cpu(), 'pos': None if pos is None else pos.cpu()})
    hooks.append(model.model.register_forward_pre_hook(pre, with_kwargs=True))
    return len(hooks)

print('hooks', llm.apply_model(install), flush=True)

def run(ids, n):
    CAP['calls'] = []
    CAP['on'] = True
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    CAP['on'] = False
    return list(out[0].outputs[0].token_ids), CAP['calls']

def by_position(calls):
    """position -> {'res': [L, H] per layer dict, 'gate': ...} keyed by absolute position"""
    table = {}
    for c in calls:
        pos = c['pos'].tolist()
        for row, p in enumerate(pos):
            table[p] = {'res': {l: t[row] for l, t in c.get('res', {}).items()},
                        'gate': {l: t[row] for l, t in c.get('gate', {}).items()},
                        'M': len(pos)}
    return table

summary = []
for case in ref:
    P = case['prompt_ids']
    toks, calls = run(P, 48)
    dec = by_position(calls)
    full = P + toks
    _, calls2 = run(full, 1)
    pre = by_position(calls2)
    # also a prefill truncated right after the case-specific divergence point
    rec = {'case': case['case'], 'decode_tokens': toks, 'positions': []}
    for p in range(len(P) - 1, len(full) - 1):
        a, b = dec[p], pre[p]
        layers = []
        for l in sorted(a['res']):
            x, y = a['res'][l], b['res'][l]
            rel = ((x - y).norm() / y.norm()).item()
            ent = {'l': l, 'rel': rel}
            if l in a['gate']:
                ga, gb = a['gate'][l], b['gate'][l]
                ta = set(torch.topk(ga, 6).indices.tolist()); tb = set(torch.topk(gb, 6).indices.tolist())
                s = torch.sort(gb, descending=True).values
                ent.update(route_same=ta == tb, gate_rel=((ga - gb).norm() / gb.norm()).item(),
                           margin6_7=(s[5] - s[6]).item())
            layers.append(ent)
        rec['positions'].append({'p': p, 'out_idx': p - len(P) + 1, 'M_dec': a['M'], 'M_pre': b['M'], 'layers': layers})
    summary.append(rec)
    torch.save({'dec': dec, 'pre': pre, 'full': full}, root / f"case{case['case']}.pt")

(root / 'offline-summary.json').write_text(json.dumps(summary))
(root / 'offline.done').write_text('ok\n')
print('done', flush=True)
