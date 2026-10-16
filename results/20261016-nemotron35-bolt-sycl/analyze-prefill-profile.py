#!/usr/bin/env python3
"""Summarize this experiment's retained unitrace data; no benchmark execution."""
import csv
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'prefill-router-r1-unitrace'
lines = gzip.decompress((OUT / 'unitrace.1.txt.gz').read_bytes()).decode().splitlines()
start = next(i for i, line in enumerate(lines) if line.strip().startswith('Kernel,'))
rows = []
for line in lines[start + 1:]:
    if not line.strip():
        break
    name, calls, ns, percentage, average, minimum, maximum = next(csv.reader([line.strip()], skipinitialspace=True))
    rows.append({'kernel': name, 'calls': int(calls), 'time_ns': int(ns), 'device_time_percent': float(percentage)})
trace = json.loads(gzip.decompress((OUT / 'llama-server.1.json.gz').read_bytes()))['traceEvents']
gpu = [e for e in trace if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
assert len(gpu) == sum(r['calls'] for r in rows)
assert abs(sum(e['dur'] for e in gpu) * 1000 - sum(r['time_ns'] for r in rows)) < 1000
intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in gpu)
first = previous_end = intervals[0][0]
busy_us = 0.0
gaps = []
for begin, end in intervals:
    if begin > previous_end:
        gaps.append(begin - previous_end)
        busy_us += previous_end - first
        first = begin
    previous_end = max(previous_end, end)
busy_us += previous_end - first
summary = {
    'tier': 'development-diagnostic',
    'source': 'unitrace.1.txt.gz + llama-server.1.json.gz',
    'request': {'prompt_tokens': 8270, 'output_tokens': 1, 'cache_n': 0},
    'device_events': len(gpu),
    'summed_device_time_ns': sum(r['time_ns'] for r in rows),
    'device_span_s': (previous_end - intervals[0][0]) / 1e6,
    'device_busy_union_s': busy_us / 1e6,
    'device_gap_sum_s': sum(gaps) / 1e6,
    'largest_device_gap_ms': max(gaps) / 1000,
    'top_kernels': rows[:25],
    'caveats': [
        'One instrumented prefill only; attribution, not a performance comparison.',
        'Summed device durations include overlapping events; union/span are reported separately.',
        'Host memory-copy API duration includes waiting for earlier device work. Do not add it to device time or interpret it as PCIe transfer time.',
        'Kernel names identify implementations, not necessarily individual model layers.',
    ],
}
print(json.dumps(summary, indent=2))
