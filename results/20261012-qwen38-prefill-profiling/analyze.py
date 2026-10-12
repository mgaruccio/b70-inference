#!/usr/bin/env python3
"""Correlate native device kernels to launch operators; never time CPU scopes as GPU work."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import re
import statistics


def analyze(cell):
    summary = json.loads((cell / 'summary.json').read_text())
    ledger = json.loads((cell / 'iterations.json').read_text())
    files = sorted((cell / 'torch').glob('*.pt.trace.json*'))
    assert len(files) == 1, 'One worker/trace required'
    opener = gzip.open if files[0].suffix == '.gz' else open
    with opener(files[0], 'rt') as stream:
        events = json.load(stream)['traceEvents']
    cpu = {e.get('args', {}).get('External id'): e for e in events if e.get('cat') == 'cpu_op'}
    runtime = {e.get('args', {}).get('correlation'): e for e in events if e.get('cat') == 'xpu_runtime'}
    phases = []
    for event in events:
        if event.get('cat') != 'user_annotation':
            continue
        match = re.search(r'execute_context_\d+\((\d+)\)', event['name'])
        if match and int(match[1]):
            phases.append({'context_tokens': int(match[1]), 'cpu_start': event['ts'],
                           'cpu_end': event['ts'] + event['dur'], 'groups': {}})
    phases.sort(key=lambda p: p['cpu_start'])
    expected_chunks = [r['context_tokens'] for r in ledger if r['context_tokens']]
    assert [p['context_tokens'] for p in phases] == expected_chunks
    totals = collections.defaultdict(lambda: {'calls': 0, 'device_ms': 0.0})
    unmatched = 0
    for event in events:
        if event.get('cat') != 'kernel' or event.get('ph') != 'X' or event.get('dur', 0) <= 0:
            continue
        args = event.get('args', {})
        launch = runtime.get(args.get('correlation'), {})
        external = args.get('External id', launch.get('args', {}).get('External id'))
        operator = cpu.get(external, {})
        name = event['name']
        if name == 'gemm_kernel':
            group = 'INT4 GEMM' if operator.get('name') == '_xpu_C::int4_gemm_w4a16' else 'dense GEMM' if operator.get('name') == 'aten::mm' else 'unmapped GEMM'
        elif 'FMHAFwdKernel' in name:
            group = 'K64 attention'
        elif name.startswith('gdn::'):
            group = 'GDN'
        else:
            group = 'other kernels'
        duration = event['dur'] / 1000
        totals[group]['calls'] += 1
        totals[group]['device_ms'] += duration
        # Associate using the CPU launch/op identity, NOT asynchronous GPU start time.
        timestamp = operator.get('ts', launch.get('ts'))
        phase = next((p for p in phases if timestamp is not None and p['cpu_start'] <= timestamp <= p['cpu_end']), None)
        if phase is None:
            unmatched += 1
        else:
            values = phase['groups'].setdefault(group, {'calls': 0, 'device_ms': 0.0})
            values['calls'] += 1
            values['device_ms'] += duration
    latencies = [r['elapsed_s'] for r in summary['unprofiled']]
    median = statistics.median(latencies)
    return {'cell': cell.name, 'budget': summary['budget'], 'prefix_cache': summary['prefix_cache'],
            'length': summary['length'], 'chunks': expected_chunks, 'unprofiled_s': latencies,
            'median_s': median, 'range_s': [min(latencies), max(latencies)],
            'profiled_s': summary['profiled']['elapsed_s'],
            'profile_overhead_pct': 100 * (summary['profiled']['elapsed_s'] / median - 1),
            'device_groups': dict(totals), 'phases': phases, 'unmatched_phase_kernel_count': unmatched,
            'caveat': 'Kernel durations are device activity, not CPU scope time or elapsed latency. No tensor shapes were recorded; generic GEMMs are identified by external/correlation IDs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    cells = [args.root] if (args.root / 'summary.json').exists() else sorted(p for p in args.root.iterdir() if (p / 'summary.json').exists())
    print(json.dumps([analyze(p) for p in cells], indent=2, allow_nan=False))
