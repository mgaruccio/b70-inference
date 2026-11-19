"""Attention correctness on a long single-chunk prefill vs fp32 reference (sinks, sliding window)."""
import faulthandler, json, os, pathlib, sys
faulthandler.dump_traceback_later(700, exit=True)
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
root = pathlib.Path(sys.argv[1]); here = pathlib.Path.cwd(); mdir = here / 'gptq'
cfg = json.loads((mdir / 'config.json').read_text())
mnbt = int(os.environ.get('ATT_MNBT', '4096'))
llm = LLM(model=str(mdir), dtype='bfloat16', attention_backend='TRITON_ATTN', enforce_eager=True,
          max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=mnbt, enable_prefix_caching=False,
          gpu_memory_utilization=0.90)
LAYERS = [0, 2, 3, 5]
CAP = {'on': False, 'rec': {}, 'pos': None}
def install(model):
    for l in LAYERS:
        a = model.model.layers[l].self_attn.attn
        def hp(m, args, l=l):
            if CAP['on']: CAP['pending'] = [t.detach().float().cpu() for t in args[:3]]
        def h(m, args, out, l=l):
            if CAP['on']:
                CAP['rec'].setdefault(l, []).append({'pos': CAP['pos'].clone(), 'qkv': CAP['pending'], 'out': out.detach().float().cpu()})
        a.register_forward_pre_hook(hp); a.register_forward_hook(h)
    def pre(m, args, kwargs):
        if CAP['on']:
            CAP['pos'] = kwargs.get('positions', args[1] if len(args) > 1 else None).detach().cpu()
    model.model.register_forward_pre_hook(pre, with_kwargs=True)
    impl = model.model.layers[2].self_attn.attn.impl
    return {'window_2': getattr(impl, 'sliding_window', None), 'window_0': getattr(model.model.layers[0].self_attn.attn.impl, 'sliding_window', None)}
print('install', llm.apply_model(install), flush=True)
tk = AutoTokenizer.from_pretrained(str(mdir))
prompts = json.loads((here / 'benchmark-prompts.json').read_text())
lp = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
ids = tk(lp)['input_ids']
CAP['on'] = True
llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
CAP['on'] = False
idx = json.loads((mdir / 'model.safetensors.index.json').read_text())['weight_map']
def T(n):
    return safe_open(str(mdir / idx[n]), 'pt', device='cpu').get_tensor(n)
NQ, NKV = cfg['num_attention_heads'], cfg['num_key_value_heads']
res = {}
for l in LAYERS:
    rows = {}
    for c in CAP['rec'][l]:
        q, k, v = c['qkv']; M = c['pos'].numel()
        for r_, p in enumerate(c['pos'].tolist()):
            rows[p] = (q[r_], k[r_], v[r_], c['out'][r_].reshape(-1), M)
    allp = sorted(rows); n = len(allp)
    q = torch.stack([rows[p][0] for p in allp]).view(n, NQ, -1)
    k = torch.stack([rows[p][1] for p in allp]).view(n, NKV, -1).repeat_interleave(NQ // NKV, 1)
    v = torch.stack([rows[p][2] for p in allp]).view(n, NKV, -1).repeat_interleave(NQ // NKV, 1)
    s = torch.einsum('qhd,khd->hqk', q, k) * (q.shape[-1] ** -0.5)
    i = torch.arange(n).view(-1, 1); j = torch.arange(n).view(1, -1)
    mask = j <= i
    if cfg['layer_types'][l] == 'sliding_attention':
        mask &= (i - j) < cfg['sliding_window']
    s = s.masked_fill(~mask, float('-inf'))
    sinks = T(f'model.layers.{l}.self_attn.softmax_offset').float()
    s = torch.cat([s, sinks.view(-1, 1, 1).expand(-1, n, 1)], -1)
    pr = torch.softmax(s, -1)[..., :-1]
    ref = torch.einsum('hqk,khd->qhd', pr, v).reshape(n, -1)
    out = torch.stack([rows[p][3] for p in allp])
    per = ((out - ref).norm(dim=-1) / ref.norm(dim=-1)).tolist()
    bins = {}
    for p, e in zip(allp, per):
        b = f'{(p // 128) * 128}-{(p // 128) * 128 + 127}'
        bins.setdefault(b, []).append(e)
    res[l] = {'type': cfg['layer_types'][l], 'M_calls': sorted({rows[p][4] for p in allp}),
              'bins': {b: [round(sum(v_) / len(v_), 5), round(max(v_), 4)] for b, v_ in bins.items()},
              'worst': sorted([(round(e, 4), p) for p, e in zip(allp, per)], reverse=True)[:6]}
    print(l, json.dumps(res[l]), flush=True)
(root / 'attncheck.json').write_text(json.dumps(res))
(root / 'offline.done').write_text('ok\n')
