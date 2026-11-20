"""Per-cycle kernel aggregation of a DFlash 2 torch-profiler XPU trace (eager).
Kernels launched inside an `execute_context_0(0)_generation_1(N)` annotation are the
target verify forward ("V"); kernels launched between consecutive verify annotations are
target lm_head + sampling + drafter ("O"). Usage: spec_agg.py trace.json.gz [top]"""
import collections, gzip, json, sys
tr = json.load(gzip.open(sys.argv[1])); top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ev = tr['traceEvents']
rt = {e['args'].get('correlation'): e['ts'] for e in ev if e.get('cat') == 'xpu_runtime'}
ks = sorted((e for e in ev if e.get('cat') == 'kernel'), key=lambda e: e['ts'])
gen = sorted((e for e in ev if e.get('cat') == 'user_annotation' and 'generation_1' in e['name']),
             key=lambda e: e['ts'])
n = len(gen) - 1
agg = {'V': collections.defaultdict(lambda: [0.0, 0]), 'O': collections.defaultdict(lambda: [0.0, 0])}
for k in ks:
    t = rt.get(k['args'].get('correlation'))
    if t is None or t < gen[0]['ts'] or t >= gen[-1]['ts']:
        continue  # whole cycles only: [first verify, last verify)
    lab = 'V' if any(a['ts'] <= t <= a['ts'] + a['dur'] for a in gen) else 'O'
    a = agg[lab][k['name'][:90]]; a[0] += k['dur']; a[1] += 1
print(f'cycles {n} ({gen[0]["name"]})')
for lab in 'VO':
    tot = sum(v[0] for v in agg[lab].values())
    print(f'{lab}: {tot / n / 1000:.3f} ms/cycle, {sum(v[1] for v in agg[lab].values()) / n:.0f} kernels')
    for nm, (d, c) in sorted(agg[lab].items(), key=lambda x: -x[1][0])[:top]:
        print(f'  {d / n / 1000:7.3f} ms {c / n:6.1f}x {d / c:8.2f} us  {nm}')
