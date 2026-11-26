"""Aggregate device kernel events in an iteration-bounded Kineto trace. No CPU time.

Do not infer a decode/verify boundary from host gaps (especially in graph mode).
Grid distinguishes small-M dense GEMMs when the tool records launch geometry.
"""
import collections
import gzip
import json
import sys
from pathlib import Path


def summarize(path):
    with gzip.open(path, 'rt') as f:
        events = json.load(f)['traceEvents']
    stats = collections.defaultdict(list)
    for e in events:
        if e.get('ph') == 'X' and e.get('cat', '').lower() in ('gpu_op', 'kernel'):
            args = e.get('args', {})
            key = (e['name'], str(args.get('grid', '')))
            stats[key].append(e['dur'])
    rows = sorted((dict(name=k[0], grid=k[1], calls=len(v), total_us=round(sum(v), 2),
                        mean_us=round(sum(v)/len(v), 2)) for k, v in stats.items()),
                  key=lambda r: -r['total_us'])
    total = sum(r['total_us'] for r in rows)
    return dict(trace=path.name, sum_kernel_us=total, rows=rows)


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        path = Path(arg)
        summary = summarize(path)
        path.with_suffix('').with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(path.name, 'sum_kernel_ms=', round(summary['sum_kernel_us']/1000, 2))
        for row in summary['rows'][:14]:
            print(f"{100*row['total_us']/summary['sum_kernel_us']:5.1f}% {row['calls']:5}x "
                  f"{row['mean_us']:7.2f} us  {row['grid']}  {row['name'][:110]}")
