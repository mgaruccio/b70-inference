#!/usr/bin/env python3
"""Account for GPU work in the isolated, externally paused unitrace prefill trace."""
import collections
import json
import math
from pathlib import Path
import sys


def summarize(document):
    events = document['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'gpu_op' and e.get('ph') == 'X']
    if not kernels:
        raise ValueError('No gpu_op/X events; do not infer a bottleneck from an empty trace')
    durations = collections.defaultdict(float)
    counts = collections.Counter()
    intervals = []
    for event in kernels:
        start, duration = float(event['ts']), float(event['dur'])
        if not math.isfinite(start) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid trace timestamp/duration')
        name = event['name']
        durations[name] += duration
        counts[name] += 1
        intervals.append((start, start + duration))
    intervals.sort()
    left, right = intervals[0]
    union = 0.0
    for start, end in intervals[1:]:
        if start > right:
            union += right - left
            left, right = start, end
        else:
            right = max(right, end)
    union += right - left
    total = sum(durations.values())
    if total <= 0:
        raise ValueError('No positive GPU duration')
    ranked = sorted(durations, key=durations.get, reverse=True)
    return {
        'note': 'Diagnostic GPU-work shares, not wall-time speedups. Chrome trace timestamps/durations are microseconds.',
        'gpu_event_count': len(kernels),
        'summed_gpu_work_ms': total / 1000,
        'gpu_interval_union_ms': union / 1000,
        'gpu_span_ms': (max(end for _, end in intervals) - intervals[0][0]) / 1000,
        'kernels': [{'name': name, 'count': counts[name], 'duration_ms': durations[name] / 1000,
                     'gpu_work_percent': durations[name] / total * 100} for name in ranked],
    }


if __name__ == '__main__':
    result = summarize(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(result, indent=2))
