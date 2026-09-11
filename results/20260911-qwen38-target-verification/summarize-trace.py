#!/usr/bin/env python3
"""Attribute each XPU kernel once using CPU External id and target module ranges."""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as source:
    events = json.load(source)['traceEvents']
threads = collections.defaultdict(list)
for event in events:
    if event.get('cat') == 'cpu_op' or (event.get('cat') == 'user_annotation' and event['name'].startswith('b70_target/')):
        threads[event['pid'], event['tid']].append(event)
lookup = {}
for rows in threads.values():
    stack = []
    for event in sorted(rows, key=lambda e: (e['ts'], 0 if e.get('cat') == 'user_annotation' else 1)):
        while stack and stack[-1]['ts'] + stack[-1]['dur'] < event['ts']:
            stack.pop()
        if event.get('cat') == 'user_annotation':
            stack.append(event)
        else:
            lookup[event.get('args', {}).get('External id')] = (event, stack[-1]['name'] if stack else None)
roots = [e for e in events if e.get('name', '').startswith('b70_target/root:')]
steps = len(roots)
assert steps > 0, 'No target root annotations'
aggregate = {name: collections.defaultdict(lambda: {'count': 0, 'total_us': 0.0}) for name in ('module_types', 'subsystems', 'operators', 'kernels')}
total_us = target_us = missing_us = outside_us = 0.0
kernel_count = missing_count = 0
queues = set()
for event in events:
    if event.get('cat') != 'kernel':
        continue
    kernel_count += 1
    total_us += event['dur']
    queues.add((event['pid'], event['tid']))
    cpu, label = lookup.get(event.get('args', {}).get('External id'), (None, None))
    if cpu is None:
        missing_us += event['dur']
        missing_count += 1
    if label is None:
        outside_us += event['dur']
        continue
    target_us += event['dur']
    subsystem = 'gdn' if '.linear_attn' in label else ('full_attention' if '.self_attn' in label else ('mlp' if '.mlp' in label else 'other'))
    keys = {'module_types': label.split(':')[-1], 'subsystems': subsystem,
            'operators': cpu['name'] + ' ' + json.dumps(cpu.get('args', {}).get('Input Dims')),
            'kernels': event['name']}
    for name, key in keys.items():
        aggregate[name][key]['count'] += 1
        aggregate[name][key]['total_us'] += event['dur']
result = {'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'target_forward_steps': steps,
          'decode_annotations': [e['name'] for e in events if e.get('name', '').startswith('execute_')],
          'kernel_count': kernel_count, 'kernel_queues': sorted(queues),
          'all_kernel_ms': total_us / 1000, 'target_kernel_ms': target_us / 1000,
          'outside_target_or_unmapped_ms': outside_us / 1000,
          'unmapped_cpu_external_id': {'count': missing_count, 'kernel_ms': missing_us / 1000},
          'caveat': 'Eager diagnostic only; each kernel attributed once through its CPU External id and innermost target module. No overlapping parent totals. Not graph-mode throughput.'}
for name, rows in aggregate.items():
    result[name] = [{'name': key, **value, 'ms_per_target_step': value['total_us'] / (1000 * steps)}
                    for key, value in sorted(rows.items(), key=lambda row: -row[1]['total_us'])]
print(json.dumps(result, indent=2))
