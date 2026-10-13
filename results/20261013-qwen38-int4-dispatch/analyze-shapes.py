#!/usr/bin/env python3
"""Exact native INT4 dimensions + correlated GPU activity; no speedup claim."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re


def parse_u4_creation(lines):
    records = []
    for raw in lines:
        if 'onednn_verbose' not in raw or not ('wei:u4' in raw or 'src_b:u4' in raw):
            continue
        line = raw[raw.index('onednn_verbose'):]
        parts = line.split(',')
        index = next((i for i, p in enumerate(parts) if p.startswith('create:')), None)
        if index is None:
            continue
        dispatch = parts[index] == 'create:dispatch'
        implementation = parts[index + (4 if dispatch else 3)]
        match = re.search(r',(\d+x\d+:\d+x\d+),', line)
        records.append({'action': parts[index], 'implementation': implementation,
                        'problem': match[1] if match else None,
                        'reason': parts[-2] if dispatch else None, 'raw': line})
    return records


def analyze(cell):
    files = sorted((cell / 'torch').glob('*.pt.trace.json*'))
    assert len(files) == 1, files
    opener = gzip.open if files[0].suffix == '.gz' else open
    with opener(files[0], 'rt') as stream:
        events = json.load(stream)['traceEvents']
    cpu = {e['args']['External id']: e for e in events
           if e.get('cat') == 'cpu_op' and 'External id' in e.get('args', {})}
    runtime = {e['args']['correlation']: e for e in events
               if e.get('cat') == 'xpu_runtime' and 'correlation' in e.get('args', {})}
    native = [e for e in events if e.get('cat') == 'cpu_op'
              and e.get('name') == '_xpu_C::int4_gemm_w4a16']
    phases = []
    for e in events:
        match = re.search(r'execute_context_\d+\((\d+)\)', e.get('name', ''))
        if e.get('cat') == 'user_annotation' and match and int(match[1]):
            phases.append({'context_tokens': int(match[1]), 'start': e['ts'],
                           'end': e['ts'] + e['dur']})
    phases.sort(key=lambda p: p['start'])
    ledger = json.loads((cell / 'iterations.json').read_text())
    chunks = [r['context_tokens'] for r in ledger if r['context_tokens']]
    assert [p['context_tokens'] for p in phases] == chunks
    groups = defaultdict(lambda: {'operator_calls': 0, 'kernel_calls': 0, 'device_ms': 0.0,
                                  'phase_operator_calls': Counter(), 'kernel_names': Counter(),
                                  'input_metadata': []})
    missing_shapes = 0
    dimensions = {}
    for e in native:
        args = e.get('args', {})
        dims = args.get('Input Dims', [])
        if len(dims) < 4 or len(dims[0]) != 2 or len(dims[1]) != 2:
            missing_shapes += 1
            continue
        m, k = dims[0]
        packed_k, n = dims[1]
        assert packed_k * 8 == k, dims
        assert len(dims[3]) == 2 and dims[3][1] == n and k % dims[3][0] == 0, dims
        key = (m, k, n)
        external = args.get('External id')
        if external is not None:
            dimensions[external] = key
        g = groups[key]
        g['operator_calls'] += 1
        phase = next((p['context_tokens'] for p in phases if p['start'] <= e['ts'] <= p['end']), None)
        g['phase_operator_calls'][str(phase)] += 1
        metadata = {name: args[name] for name in ('Input Dims', 'Input type', 'Input Strides', 'Concrete Inputs') if name in args}
        if metadata not in g['input_metadata']:
            g['input_metadata'].append(metadata)
    assert native and groups, 'No exact native INT4 shapes'
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X' and e.get('dur', 0) > 0]
    assert kernels, 'No device activity'
    generic_unmapped = 0
    native_without_dims = 0
    for e in kernels:
        args = e.get('args', {})
        launch = runtime.get(args.get('correlation'), {})
        external = args.get('External id', launch.get('args', {}).get('External id'))
        operator = cpu.get(external, {})
        if operator.get('name') != '_xpu_C::int4_gemm_w4a16':
            if e['name'] == 'gemm_kernel' and operator.get('name') != 'aten::mm':
                generic_unmapped += 1
            continue
        if external not in dimensions:
            native_without_dims += 1
            continue
        g = groups[dimensions[external]]
        g['kernel_calls'] += 1
        g['device_ms'] += e['dur'] / 1000
        g['kernel_names'][e['name']] += 1
    rows = [{'M': key[0], 'K': key[1], 'N': key[2], **value}
            for key, value in sorted(groups.items(), key=lambda kv: -kv[1]['device_ms'])]
    console = (cell / 'server.log').read_text(errors='replace')
    verbose = [line for line in console.splitlines() if 'onednn_verbose' in line]
    matmul_create = [line for line in verbose if 'matmul' in line and 'create' in line]
    dispatch = [line for line in verbose if 'dispatch' in line]
    u4 = parse_u4_creation(console.splitlines())
    selected = [r for r in u4 if r['action'] != 'create:dispatch']
    rejected = [r for r in u4 if r['action'] == 'create:dispatch']
    profile_window = cell / 'profile-window.log'
    window_u4 = parse_u4_creation(profile_window.read_text().splitlines()) if profile_window.exists() else []
    geometry = defaultdict(lambda: {'operator_calls': 0, 'kernel_calls': 0, 'device_ms': 0.0})
    total_int4_ms = sum(r['device_ms'] for r in rows)
    for r in rows:
        g = geometry[(r['K'], r['N'])]
        for field in ('operator_calls', 'kernel_calls', 'device_ms'):
            g[field] += r[field]
    geometry_rows = [{'K': k[0], 'N': k[1], **v,
                      'share_of_int4_device_pct': 100 * v['device_ms'] / total_int4_ms if total_int4_ms else None}
                     for k, v in sorted(geometry.items(), key=lambda kv: -kv[1]['device_ms'])]
    return {'tier': 'development diagnostic, not performance comparison', 'chunks': chunks,
            'native_operator_calls': len(native), 'native_ops_missing_shapes': missing_shapes,
            'total_device_kernel_count': len(kernels), 'unmapped_generic_gemm_kernels': generic_unmapped,
            'native_kernels_missing_shapes': native_without_dims,
            'shapes': rows, 'onednn_verbose_lines': len(verbose),
            'onednn_matmul_create_lines': len(matmul_create), 'onednn_dispatch_lines': len(dispatch),
            'onednn_create_samples': list(dict.fromkeys(matmul_create))[:24],
            'int4_device_ms': total_int4_ms, 'geometry_totals': geometry_rows,
            'u4_create_records': len(u4), 'u4_selected_implementations': dict(Counter(r['implementation'] for r in selected)),
            'u4_dispatch_reasons': dict(Counter(r['reason'] for r in rejected)),
            'profile_window_u4_create_records': len(window_u4),
            'u4_create_samples': list(dict.fromkeys(r['raw'] for r in selected))[:12],
            'u4_dispatch_samples': list(dict.fromkeys(r['raw'] for r in rejected))[:12],
            'caveats': ['GPU sums may overlap and are not response latency.',
                        'Shapes are dispatcher input geometry, not internal kernel tile sizes.',
                        'oneDNN create/dispatch records do not measure execution.',
                        'Creation/selection samples span startup and warmup, not necessarily the profiled request.',
                        'No performance candidate was changed or benchmarked.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cell', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cell), indent=2, allow_nan=False))
