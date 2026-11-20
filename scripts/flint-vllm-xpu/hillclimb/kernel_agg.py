"""Per-kernel decode-step aggregation of a torch-profiler XPU trace (steps split on >2 ms gaps;
step 0 = prefill, step 1 skipped as warm-up). Usage: kernel_agg.py trace.json.gz"""
import collections, gzip, json, sys
tr = json.load(gzip.open(sys.argv[1]))
k = sorted((e for e in tr['traceEvents'] if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_op', 'Kernel')),
           key=lambda e: e['ts'])
steps, cur = [], [k[0]]
for a, b in zip(k, k[1:]):
    if b['ts'] - (a['ts'] + a.get('dur', 0)) > 2000:
        steps.append(cur); cur = []
    cur.append(b)
steps.append(cur)
dec = steps[2:]; n = len(dec)
agg = collections.defaultdict(lambda: [0.0, 0])
for s in dec:
    for e in s:
        a = agg[e['name'][:80]]; a[0] += e['dur']; a[1] += 1
print(f"decode steps {n}; kernel busy {sum(v[0] for v in agg.values()) / n / 1000:.3f} ms/step; "
      f"{sum(v[1] for v in agg.values()) / n:.0f} kernels/step")
for nm, (d, c) in sorted(agg.items(), key=lambda x: -x[1][0])[:25]:
    print(f'{d / n / 1000:7.3f} ms {c / n:6.0f}x {d / c:7.2f} us  {nm}')
