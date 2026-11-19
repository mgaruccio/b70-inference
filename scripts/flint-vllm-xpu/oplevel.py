"""Op-level accuracy vs FP32 dequantized reference, at decode (M=1) and prefill (M=N) shapes."""
import json, os, pathlib, sys
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_ALLOW_INSECURE_SERIALIZATION', '1')
import torch
import torch.nn.functional as F
from safetensors import safe_open
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

root = pathlib.Path(sys.argv[1])
here = pathlib.Path.cwd()
ref = json.loads((here / 'probe-ref.json').read_text())
mdir = here / 'gptq'
cfg = json.loads((mdir / 'config.json').read_text())
EPS = cfg['rms_norm_eps']
torch.set_num_threads(os.cpu_count())

llm = LLM(model=str(mdir), dtype='bfloat16', attention_backend='TRITON_ATTN',
          enforce_eager=True, max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=512,
          enable_prefix_caching=False, gpu_memory_utilization=0.90)

LAYERS = [int(x) for x in os.environ.get('OP_LAYERS', '3,24').split(',')]
CAP = {'on': False, 'pos': None, 'rec': {}}
STATIC = {}

def install(model):
    hooks = []
    def add(name, mod, kind):
        def hp(m, args, kwargs):
            if CAP['on']:
                CAP.setdefault('pending', {})[name] = [a.detach().float().cpu() if torch.is_tensor(a) else a for a in args]
        hooks.append(mod.register_forward_pre_hook(hp, with_kwargs=True))
        def h(m, args, kwargs, out):
            if not CAP['on']:
                return
            ins = CAP['pending'].pop(name)
            o = out[0] if isinstance(out, tuple) else out
            o2 = out[1] if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]) else None
            CAP['rec'].setdefault(name, []).append({'pos': CAP['pos'].clone(), 'in': ins,
                'out': o.detach().float().cpu(), 'out2': None if o2 is None else o2.detach().float().cpu()})
        hooks.append(mod.register_forward_hook(h, with_kwargs=True))
    for l in LAYERS:
        L = model.model.layers[l]
        a = L.self_attn
        add(f'{l}.qkv', a.qkv_proj, 'lin'); add(f'{l}.o', a.o_proj, 'lin'); add(f'{l}.attn', a.attn, 'attn')
        add(f'{l}.in_norm', L.input_layernorm, 'norm'); add(f'{l}.post_norm', L.post_attention_layernorm, 'norm')
        if hasattr(L.mlp, 'experts'):
            add(f'{l}.moe', L.mlp, 'moe'); add(f'{l}.gate', L.mlp.gate, 'lin')
            add(f'{l}.shared', L.mlp.shared_expert, 'mlp')
            add(f'{l}.sh_gu', L.mlp.shared_expert.gate_up_proj, 'lin'); add(f'{l}.sh_down', L.mlp.shared_expert.down_proj, 'lin')
        else:
            add(f'{l}.gu', L.mlp.gate_up_proj, 'lin'); add(f'{l}.down', L.mlp.down_proj, 'lin')
        STATIC[f'{l}.sinks'] = a.attn.impl.sinks.detach().float().cpu() if getattr(a.attn.impl, 'sinks', None) is not None else None
        STATIC[f'{l}.window'] = getattr(a.attn.impl, 'sliding_window', None)
        STATIC[f'{l}.scale'] = a.attn.impl.scale
        STATIC[f'{l}.gate_dtype'] = str(L.mlp.gate.weight.dtype) if hasattr(L.mlp, 'gate') else None
    def pre(m, args, kwargs):
        if CAP['on']:
            pos = kwargs.get('positions', args[1] if len(args) > 1 else None)
            CAP['pos'] = pos.detach().cpu()
    hooks.append(model.model.register_forward_pre_hook(pre, with_kwargs=True))
    return len(hooks)

import vllm_xpu_kernels.fused_moe_interface as _fmi
_orig_apply = _fmi.XpuFusedMoe.apply
def _apply(self, output, hidden_states, topk_weights, topk_ids, *a, **k):
    if CAP['on'] and CAP.get('cur_moe') is not None:
        CAP['rec'].setdefault(f"{CAP['cur_moe']}.topk", []).append({'pos': CAP['pos'].clone(),
            'in': [topk_ids.detach().cpu().to(torch.int64), topk_weights.detach().float().cpu()],
            'out': hidden_states.detach().float().cpu(), 'out2': None})
    return _orig_apply(self, output, hidden_states, topk_weights, topk_ids, *a, **k)
_fmi.XpuFusedMoe.apply = _apply
def install2(model):
    for l in LAYERS:
        mlp = model.model.layers[l].mlp
        def ph(m, args, l=l):
            CAP['cur_moe'] = l
        def fh(m, args, out):
            CAP['cur_moe'] = None
        mlp.register_forward_pre_hook(ph); mlp.register_forward_hook(fh)
    return True
print('hooks', llm.apply_model(install), llm.apply_model(install2), STATIC, flush=True)

def run(ids, n):
    CAP['rec'] = {}; CAP['on'] = True
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(temperature=0, max_tokens=n, ignore_eos=True))
    CAP['on'] = False
    return list(out[0].outputs[0].token_ids), CAP['rec']

# ---- reference weights
idx = json.loads((mdir / 'model.safetensors.index.json').read_text())['weight_map']
_h = {}
def T(name):
    fn = idx[name]
    if fn not in _h:
        _h[fn] = safe_open(str(mdir / fn), 'pt', device='cpu')
    return _h[fn].get_tensor(name)
ZINFO = {}
def deq(prefix):
    qw, qz, sc = T(prefix + '.qweight'), T(prefix + '.qzeros'), T(prefix + '.scales').float()
    K8, N = qw.shape
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K8 * 8, N).float()
    z = ((qz.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1).float()
    ZINFO.setdefault('z', sorted(set(z.unique().tolist())))
    g = (K8 * 8) // sc.shape[0]
    zz = (z + 1).repeat_interleave(g, 0); ss = sc.repeat_interleave(g, 0)
    return ((q - zz) * ss).t().contiguous()  # [N, K]
def deq_cat(prefixes):
    return torch.cat([deq(p) for p in prefixes], 0)

def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()

def rms(x, w):
    x = x.float(); return w.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))

def by_pos(calls):
    """flatten calls into rows: list of (pos, M, row_in(list), row_out, row_out2)"""
    rows = {}
    for c in calls:
        M = c['pos'].numel()
        for r, p in enumerate(c['pos'].tolist()):
            rows[p] = {'M': M, 'in': [t[r] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == M else t for t in c['in']],
                       'out': c['out'][r], 'out2': None if c['out2'] is None else c['out2'][r]}
    return rows

def stack(rows, ps, k):
    return torch.stack([rows[p][k] if k != 'in0' else rows[p]['in'][0] for p in ps])

results = []
for case in ref[1:]:  # case 2 (largest observed drift)
    P = case['prompt_ids']
    toks, dec_rec = run(P, 48)
    full = P + toks
    _, pre_rec = run(full, 1)
    for l in LAYERS:
        pf = f'model.layers.{l}'
        W = {
            'qkv': deq_cat([f'{pf}.self_attn.{x}_proj' for x in 'qkv']),
            'o': deq(f'{pf}.self_attn.o_proj'),
        }
        is_moe = f'{l}.moe' in dec_rec
        if is_moe:
            W['sh_gu'] = deq_cat([f'{pf}.mlp.shared_expert.{x}_proj' for x in ('gate', 'up')])
            W['sh_down'] = deq(f'{pf}.mlp.shared_expert.down_proj')
            W['gate'] = T(f'{pf}.mlp.gate.weight').float()
        else:
            W['gu'] = deq_cat([f'{pf}.mlp.{x}_proj' for x in ('gate', 'up')])
            W['down'] = deq(f'{pf}.mlp.down_proj')
        for run_name, rec in (('decode', dec_rec), ('prefill', pre_rec)):
            ps = list(range(len(P), len(full) - 1)) if run_name == 'decode' else list(range(len(P), len(full) - 1))
            ent = {'layer': l, 'run': run_name}
            for k in [x for x in W if x != 'gate'] + (['gate'] if is_moe else []):
                rows = by_pos(rec[f'{l}.{k}'])
                X = stack(rows, ps, 'in0'); Y = stack(rows, ps, 'out')
                R = X @ W[k].t()
                ent[k] = rel(Y, R)
                ent[k + '_M'] = sorted({rows[p]['M'] for p in ps})[:3]
            # norms: input (x, residual) -> (normed, residual')
            for nk, wn in (('in_norm', 'input_layernorm'), ('post_norm', 'post_attention_layernorm')):
                rows = by_pos(rec[f'{l}.{nk}'])
                w = T(f'{pf}.{wn}.weight')
                xs = torch.stack([rows[p]['in'][0] for p in ps])
                rs = [rows[p]['in'][1] if len(rows[p]['in']) > 1 and torch.is_tensor(rows[p]['in'][1]) else None for p in ps]
                if rs[0] is not None:
                    xs = xs + torch.stack(rs)
                ent[nk] = rel(torch.stack([rows[p]['out'] for p in ps]), rms(xs, w))
            # attention from this run's own q,k,v at all positions
            rows = by_pos(rec[f'{l}.attn'])
            allp = sorted(rows)
            q = torch.stack([rows[p]['in'][0] for p in allp]).view(len(allp), cfg['num_attention_heads'], -1)
            kk = torch.stack([rows[p]['in'][1] for p in allp]).view(len(allp), cfg['num_key_value_heads'], -1)
            vv = torch.stack([rows[p]['in'][2] for p in allp]).view(len(allp), cfg['num_key_value_heads'], -1)
            rep = q.shape[1] // kk.shape[1]
            kk = kk.repeat_interleave(rep, 1); vv = vv.repeat_interleave(rep, 1)
            s = torch.einsum('qhd,khd->hqk', q, kk) * STATIC[f'{l}.scale']
            n = len(allp)
            mask = torch.ones(n, n, dtype=torch.bool).tril()
            win = cfg['sliding_window'] if cfg['layer_types'][l] == 'sliding_attention' else None
            if win:
                mask &= ~torch.ones(n, n, dtype=torch.bool).tril(-win)
            s = s.masked_fill(~mask, float('-inf'))
            sinks = T(f'{pf}.self_attn.softmax_offset').float()
            s = torch.cat([s, sinks.view(-1, 1, 1).expand(-1, n, 1)], -1)
            pr = torch.softmax(s, -1)[..., :-1].to(torch.bfloat16).float()
            ao = torch.einsum('hqk,khd->qhd', pr, vv).reshape(n, -1)
            sel = [allp.index(p) for p in ps]
            ent['attn'] = rel(torch.stack([rows[p]['out'].reshape(-1) for p in ps]), ao[sel])
            # moe block: from its own input
            if is_moe:
                rows = by_pos(rec[f'{l}.moe'])
                X = torch.stack([rows[p]['in'][0] for p in ps])
                logits = X @ W['gate'].t()
                wts, sel_e = torch.topk(logits.sigmoid(), cfg['num_experts_per_tok'], -1)
                wts = wts / wts.sum(-1, keepdim=True)
                out = torch.zeros_like(X)
                cache = {}
                for t in range(X.shape[0]):
                    for j in range(sel_e.shape[1]):
                        e = int(sel_e[t, j])
                        if e not in cache:
                            ep = f'{pf}.mlp.experts.{e}'
                            cache[e] = (deq_cat([f'{ep}.gate_proj', f'{ep}.up_proj']), deq(f'{ep}.down_proj'))
                        gu, dn = cache[e]
                        g, u = (X[t] @ gu.t()).chunk(2)
                        out[t] += wts[t, j] * ((F.silu(g) * u) @ dn.t())
                gs, us = (X @ W['sh_gu'].t()).chunk(2, -1)
                out += (F.silu(gs) * us) @ W['sh_down'].t()
                Ymoe = torch.stack([rows[p]['out'] for p in ps])
                ent['moe'] = rel(Ymoe, out)
                srows = by_pos(rec[f'{l}.shared'])
                Ysh = torch.stack([srows[p]['out'] for p in ps])
                Xsh = torch.stack([srows[p]['in'][0] for p in ps])
                gs2, us2 = (Xsh @ W['sh_gu'].t()).chunk(2, -1)
                ref_sh = (F.silu(gs2) * us2) @ W['sh_down'].t()
                ent['shared'] = rel(Ysh, ref_sh)
                ent['shared_in_vs_moe_in'] = rel(Xsh, X)
                grows = by_pos(rec[f'{l}.gate'])
                glog = torch.stack([grows[p]['out'] for p in ps])
                w2, e2 = torch.topk(glog.sigmoid(), cfg['num_experts_per_tok'], -1)
                w2 = w2 / w2.sum(-1, keepdim=True)
                routed_ref = torch.zeros_like(X)
                for t in range(X.shape[0]):
                    for j in range(e2.shape[1]):
                        e = int(e2[t, j])
                        if e not in cache:
                            ep = f'{pf}.mlp.experts.{e}'
                            cache[e] = (deq_cat([f'{ep}.gate_proj', f'{ep}.up_proj']), deq(f'{ep}.down_proj'))
                        gu, dn = cache[e]
                        g, u = (X[t] @ gu.t()).chunk(2)
                        routed_ref[t] += w2[t, j] * ((F.silu(g) * u) @ dn.t())
                trows = by_pos(rec[f'{l}.topk'])
                vids = torch.stack([trows[p]['in'][0] for p in ps]); vw = torch.stack([trows[p]['in'][1] for p in ps])
                ent['vllm_vs_bf16topk_set_mismatch'] = [t for t in range(X.shape[0]) if set(vids[t].tolist()) != set(e2[t].tolist())]
                ent['vllm_vs_fp32topk_set_mismatch'] = [t for t in range(X.shape[0]) if set(vids[t].tolist()) != set(sel_e[t].tolist())]
                ent['vllm_w_vs_bf16w_rel'] = rel(torch.sort(vw, -1).values, torch.sort(w2, -1).values)
                kin = torch.stack([trows[p]['out'] for p in ps]); ent['kernel_in_vs_moe_in'] = rel(kin, X)
                routed_v_ref = torch.zeros_like(X)
                for t in range(X.shape[0]):
                    for j in range(vids.shape[1]):
                        e = int(vids[t, j])
                        if e not in cache:
                            ep = f'{pf}.mlp.experts.{e}'
                            cache[e] = (deq_cat([f'{ep}.gate_proj', f'{ep}.up_proj']), deq(f'{ep}.down_proj'))
                        gu, dn = cache[e]
                        g, u = (X[t] @ gu.t()).chunk(2)
                        routed_v_ref[t] += vw[t, j] * ((F.silu(g) * u) @ dn.t())
                routed_v = Ymoe - Ysh
                ent['routed_kernel_vs_ref_vllm_routing'] = rel(routed_v, routed_v_ref)
                ptv = ((routed_v - routed_v_ref).norm(dim=-1) / routed_v_ref.norm(dim=-1)).tolist()
                ent['routed_vllm_routing_per_token_top3'] = sorted([(round(v, 4), i) for i, v in enumerate(ptv)], reverse=True)[:3]
                bad = [t for t in range(X.shape[0]) if set(vids[t].tolist()) != set(e2[t].tolist())]
                ent['mismatch_detail'] = [{'t': t, 'vllm': vids[t].tolist(), 'bf16': e2[t].tolist(),
                                           'logits_vllm_ids': glog[t, vids[t]].tolist(), 'logits_bf16_ids': glog[t, e2[t]].tolist()} for t in bad[:4]]
                ent['routed_kernel_vs_ref_same_routing'] = rel(routed_v, routed_ref)
                ent['routed_ref_fp32route_vs_bf16route'] = rel(out - ref_sh, routed_ref)
                ent['route_set_mismatch_tokens'] = int(sum(set(e2[t].tolist()) != set(sel_e[t].tolist()) for t in range(X.shape[0])))
                pt = ((routed_v - routed_ref).norm(dim=-1) / routed_ref.norm(dim=-1)).tolist()
                ent['routed_per_token_rel_top5'] = sorted([(round(v, 4), i) for i, v in enumerate(pt)], reverse=True)[:5]
                ent['routed_per_token_rel_median'] = sorted(pt)[len(pt) // 2]
                ent['moe_routed_only_ref_norm'] = out.norm().item()
            results.append(ent)
            print(json.dumps(ent), flush=True)
            (root / 'oplevel-partial.jsonl').open('a').write(json.dumps(ent) + '\n')

(root / 'oplevel.json').write_text(json.dumps({'static': {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in STATIC.items()},
                                               'zeros': ZINFO, 'results': results}))
(root / 'offline.done').write_text('ok\n')
