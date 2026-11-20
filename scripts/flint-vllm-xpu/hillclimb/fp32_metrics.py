"""FP32-reference metrics: |vLLM lp - fp32 lp| and fp32-argmax agreement over a build's own
greedy tokens; plus matched-context comparison of two builds on their shared prefixes.
Usage: fp32_metrics.py parityA cpurefA [parityB cpurefB]"""
import json, statistics, sys
def load(par, ref):
    return [json.loads(l) for l in open(par)], json.load(open(ref))
def metrics(P, C):
    d, agree = [], 0
    for p, c in zip(P, C):
        assert p['case'] == c['case']
        for i, row in enumerate(c['rows'][:len(p['lp'])]):
            assert row['next_actual'] == p['token_ids'][i]
            d.append(abs(p['lp'][i] - row['lp_actual'])); agree += row['top'][0][0] == row['next_actual']
    d.sort()
    return dict(n=len(d), median=round(statistics.median(d), 4), p95=round(d[int(.95 * len(d))], 4),
                max=round(d[-1], 3), mean=round(sum(d) / len(d), 4), fp32_argmax_agree=agree)
A = load(sys.argv[1], sys.argv[2]); print('A', metrics(*A))
if len(sys.argv) > 4:
    B = load(sys.argv[3], sys.argv[4]); print('B', metrics(*B))
    da, db = [], []
    for a, b, ca, cb in zip(A[0], B[0], A[1], B[1]):
        k = 0
        while k < len(a['token_ids']) and a['token_ids'][k] == b['token_ids'][k]:
            k += 1
        for i in range(k):
            da.append(abs(a['lp'][i] - ca['rows'][i]['lp_actual'])); db.append(abs(b['lp'][i] - cb['rows'][i]['lp_actual']))
    for nm, x in (('A', da), ('B', db)):
        x = sorted(x)
        print(f'matched-context {nm}: n={len(x)} median={statistics.median(x):.4f} mean={sum(x) / len(x):.4f} '
              f'p90={x[int(.9 * len(x))]:.4f} max={x[-1]:.4f}')
    print('A better at', sum(a < b for a, b in zip(da, db)), 'worse at', sum(a > b for a, b in zip(da, db)))
