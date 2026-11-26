"""Join device GEMM -> XPU runtime correlation -> enclosing CPU operator and label.
Require matched callbacks on the same thread; do not attribute by device order.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze(path):
    with gzip.open(path, 'rt') as f:
        es = json.load(f)['traceEvents']
    launches = {e['args']['correlation']: e for e in es if e.get('cat') == 'xpu_runtime'
                and 'correlation' in e.get('args', {})}
    kernels = [e for e in es if e.get('cat') == 'kernel' and e['name'] == 'gemm_kernel']
    markers = [e for e in es if e.get('cat') == 'user_annotation' and e['name'].startswith('ink2/')]
    mm = [e for e in es if e.get('cat') == 'cpu_op' and e['name'] in
          {'aten::linear', 'aten::mm', 'aten::bmm', 'aten::addmm', 'aten::_scaled_mm', 'aten::matmul'}]
    def inside(outer, inner):
        return outer['tid'] == inner['tid'] and outer['pid'] == inner['pid'] and outer['ts'] <= inner['ts'] <= outer['ts'] + outer['dur']
    rows = defaultdict(list)
    for k in kernels:
        r = launches[k['args']['correlation']]
        label = min((e for e in markers if inside(e, r)), key=lambda e: e['dur'], default={})
        op = min((e for e in mm if inside(e, r)), key=lambda e: e['dur'], default={})
        key = label.get('name', 'unlabeled'), op.get('name', 'unknown'), str(op.get('args', {}).get('Input Dims', []))
        rows[key].append(k['dur'])
    out = [dict(scope=k[0], op=k[1], shapes=k[2], calls=len(v), total_us=round(sum(v), 2)) for k,v in rows.items()]
    return sorted(out, key=lambda row: -row['total_us'])


for arg in sys.argv[1:]:
    p = Path(arg)
    rows = analyze(p)
    p.with_suffix('').with_suffix('.dense.json').write_text(json.dumps(rows, indent=2)+'\n')
    scopes = defaultdict(float)
    for row in rows:
        scopes[row['scope']] += row['total_us']
    print(p.name, sum(row['calls'] for row in rows), sorted(scopes.items(), key=lambda x: -x[1]))
