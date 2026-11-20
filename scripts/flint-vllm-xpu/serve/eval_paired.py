"""Paired per-item comparison of two eval_flint.py result dirs (run in ~/flint-eval).
Usage: .venv/bin/python -I eval_paired.py results/A results/B"""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, str(pathlib.Path(__file__).parent / 'data'))
from instruction_following_eval import evaluation_lib  # noqa: E402
import eval_flint as ef  # noqa: E402

A, B = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])


def load(d, t):
    return {json.loads(l)['id']: json.loads(l) for l in open(d / f'{t}.jsonl')}


def gsm_ok(r):
    gold = ef.num(r['gold'].split('####')[-1])
    m = re.findall(r'Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)', r['response'])
    p = ef.num(m[-1]) if m else None
    if p is None:
        n = re.findall(r'-?[\d,]*\.?\d+', r['response']); p = ef.num(n[-1]) if n else None
    return p is not None and gold is not None and abs(p - gold) < 1e-6


inputs = {x.prompt: x for x in evaluation_lib.read_prompt_list(str(ef.DATA / 'ifeval_input_data.jsonl'))}


def ife_ok(r):
    return evaluation_lib.test_instruction_following_strict(
        inputs[r['prompt']], {r['prompt']: r['response']}).follow_all_instructions


def he(d):
    ev = json.load(open(d / 'humaneval_samples_eval_results.json'))['eval']
    out = {}
    for tid, v in ev.items():
        v = v[0] if isinstance(v, list) else v
        out[tid] = v.get('base_status') == 'pass' and v.get('plus_status') == 'pass'
    return out


res = {}
for task, fn in (('gsm8k', gsm_ok), ('ifeval_strict', ife_ok)):
    a, b = load(A, task.split('_')[0]), load(B, task.split('_')[0])
    pa = {k: fn(v) for k, v in a.items()}; pb = {k: fn(b[k]) for k in a}
    trunc = {k for k in a if a[k]['finish'] == 'length' or b[k]['finish'] == 'length'}
    nt = [k for k in a if k not in trunc]
    res[task] = dict(n=len(a), a=sum(pa.values()), b=sum(pb.values()),
                     only_a=sum(pa[k] and not pb[k] for k in a), only_b=sum(pb[k] and not pa[k] for k in a),
                     either_truncated=len(trunc),
                     excl_truncated=dict(n=len(nt), a=sum(pa[k] for k in nt), b=sum(pb[k] for k in nt)))
ha, hb = he(A), he(B)
res['humaneval_plus'] = dict(n=len(ha), a=sum(ha.values()), b=sum(hb.values()),
                             only_a=sum(ha[k] and not hb[k] for k in ha),
                             only_b=sum(hb[k] and not ha[k] for k in ha))
res['labels'] = dict(a=A.name, b=B.name)
print(json.dumps(res, indent=1))
