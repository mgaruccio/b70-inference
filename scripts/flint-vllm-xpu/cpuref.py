"""FP32 CPU reference forward for Flint GPTQ checkpoint using HF modeling_flint layers.

Weights are dequantized (sym int4, group 64, gptq v1 zero+1) to fp32 one layer at a time.
Outputs reference next-token log-probs at every generated position of each test sequence.
"""
import json, os, pathlib, sys, time
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.flint.configuration_flint import FlintConfig
from transformers.models.flint.modeling_flint import FlintDecoderLayer, FlintRotaryEmbedding, FlintRMSNorm

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get('REF_THREADS', '10')))
here = pathlib.Path(__file__).resolve().parent
mdir = here / 'gptq'
out_path = pathlib.Path(sys.argv[1])
cfg = FlintConfig.from_pretrained(str(mdir))
cfg._attn_implementation = 'eager'
if hasattr(cfg, '_experts_implementation'):
    cfg._experts_implementation = 'eager'
idx = json.loads((mdir / 'model.safetensors.index.json').read_text())['weight_map']
_h = {}
def T(name):
    fn = idx[name]
    if fn not in _h:
        _h[fn] = safe_open(str(mdir / fn), 'pt', device='cpu')
    return _h[fn].get_tensor(name)
SH = torch.arange(0, 32, 4, dtype=torch.int32)
def deq(prefix):
    qw, qz, sc = T(prefix + '.qweight'), T(prefix + '.qzeros'), T(prefix + '.scales').float()
    K8, N = qw.shape
    q = ((qw.unsqueeze(1) >> SH.view(1, 8, 1)) & 0xF).reshape(K8 * 8, N).float()
    z = ((qz.unsqueeze(2) >> SH.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1).float()
    g = (K8 * 8) // sc.shape[0]
    return ((q - (z + 1).repeat_interleave(g, 0)) * sc.repeat_interleave(g, 0)).t().contiguous()

# ---- test sequences
prompts = json.loads((here / 'benchmark-prompts.json').read_text())
long_prompt = 'Review the following requests and summarize the shared engineering themes.\n\n' + '\n\n'.join(prompts * 2)
texts = prompts + [long_prompt]
tok = AutoTokenizer.from_pretrained(str(mdir))
div = json.loads((here / 'divergence-ref.json').read_text())
seqs = []
for d in div:
    pid = tok(texts[d['case']], add_special_tokens=True)['input_ids']
    seqs.append({'case': d['case'], 'k': d['k'], 'P': len(pid), 'ids': pid + d['prefix'] + [d['nospec_tok']]})
print('lens', [(s['case'], s['P'], len(s['ids'])) for s in seqs], flush=True)

emb = T('model.embed_tokens.weight').float()
H = [emb[torch.tensor(s['ids'])].unsqueeze(0) for s in seqs]
rot = FlintRotaryEmbedding(cfg)
PE, MASK = [], []
for s in seqs:
    n = len(s['ids']); pos = torch.arange(n).unsqueeze(0)
    PE.append(rot(H[0], pos)); 
    i = torch.arange(n).view(-1, 1); j = torch.arange(n).view(1, -1)
    causal = j <= i
    slide = causal & ((i - j) < cfg.sliding_window)
    MASK.append({'full': torch.where(causal, 0.0, float('-inf')).view(1, 1, n, n),
                 'slide': torch.where(slide, 0.0, float('-inf')).view(1, 1, n, n), 'pos': pos})

t0 = time.time()
for l in range(cfg.num_hidden_layers):
    layer = FlintDecoderLayer(cfg, l).float().eval()
    pf = f'model.layers.{l}'
    sd = {'input_layernorm.weight': T(f'{pf}.input_layernorm.weight').float(),
          'post_attention_layernorm.weight': T(f'{pf}.post_attention_layernorm.weight').float(),
          'self_attn.q_norm.weight': T(f'{pf}.self_attn.q_norm.weight').float(),
          'self_attn.k_norm.weight': T(f'{pf}.self_attn.k_norm.weight').float(),
          'self_attn.sinks': T(f'{pf}.self_attn.softmax_offset').float()}
    for p in 'qkvo':
        sd[f'self_attn.{p}_proj.weight'] = deq(f'{pf}.self_attn.{p}_proj')
    if l in cfg.mlp_only_layers:
        for p in ('gate', 'up', 'down'):
            sd[f'mlp.{p}_proj.weight'] = deq(f'{pf}.mlp.{p}_proj')
    else:
        sd['mlp.gate.weight'] = T(f'{pf}.mlp.gate.weight').float()
        for p in ('gate', 'up', 'down'):
            sd[f'mlp.shared_expert.{p}_proj.weight'] = deq(f'{pf}.mlp.shared_expert.{p}_proj')
        E = cfg.num_experts
        sd['mlp.experts.gate_up_proj'] = torch.stack([torch.cat([deq(f'{pf}.mlp.experts.{e}.gate_proj'), deq(f'{pf}.mlp.experts.{e}.up_proj')], 0) for e in range(E)])
        sd['mlp.experts.down_proj'] = torch.stack([deq(f'{pf}.mlp.experts.{e}.down_proj') for e in range(E)])
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not unexpected and not [m for m in missing if 'rotary' not in m], (missing, unexpected)
    kind = 'slide' if cfg.layer_types[l] == 'sliding_attention' else 'full'
    for si in range(len(seqs)):
        H[si] = layer(H[si], attention_mask=MASK[si][kind], position_ids=MASK[si]['pos'], position_embeddings=PE[si])
    del layer, sd
    print(f'layer {l} {kind} done {time.time() - t0:.0f}s', flush=True)

norm = FlintRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); norm.weight.data = T('model.norm.weight').float()
head = T('lm_head.weight').float()
res = []
for s, h in zip(seqs, H):
    lp = torch.log_softmax(norm(h)[0] @ head.t(), -1)
    P = s['P']; rows = []
    for i in range(P - 1, len(s['ids']) - 1):  # predicts token at output index i-P+1
        v, ix = torch.topk(lp[i], 10)
        rows.append({'out_idx': i - P + 1, 'top': [[int(a), float(b)] for a, b in zip(ix, v)],
                     'next_actual': int(s['ids'][i + 1]), 'lp_actual': float(lp[i, s['ids'][i + 1]]), 'full_lp_idx': i})
    res.append({'case': s['case'], 'k': s['k'], 'P': P, 'rows': rows,
                'lp_at_k': {str(t): float(lp[P - 1 + s['k'], t]) for t in torch.topk(lp[P - 1 + s['k']], 20).indices.tolist()}})
out_path.write_text(json.dumps(res))
print('done', time.time() - t0, flush=True)
