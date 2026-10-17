#!/usr/bin/env python3
"""Attribute a single warmed real-API trace using validated logits-copy boundaries."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
COPY = 'zeCommandListAppendMemoryCopy(D2H)[524288]'
OUTPUT = 'q8_0_esimd_launch<8>'
PREFILL_ONLY = ('ggml_sycl_ssm_scan_prefill(', 'grouped_dequant_gemm_launch<', 'grouped_gemm_pack_b<')


def read(path):
    if path.exists():
        return path.read_text()
    return gzip.decompress(path.with_name(path.name + '.gz').read_bytes()).decode()


def implementation(name):
    return name.split('[SIMD', 1)[0]


def union_us(events):
    # Center before adding short durations: epoch-sized doubles lose sub-us precision.
    origin = min((e['ts'] for e in events), default=0)
    intervals = sorted((e['ts'] - origin, (e['ts'] - origin) + e['dur']) for e in events)
    total = 0.0
    end = float('-inf')
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def boundaries(gpu):
    indices = [i for i, e in enumerate(gpu) if e['name'] == COPY]
    assert len(indices) == 128, f'Expected128 logits copies, got{len(indices)}'
    output_indices = [i for i, e in enumerate(gpu) if e['name'].startswith(OUTPUT)
                      and '{65536; 1; 1} {8; 1; 1}' in e['name']]
    assert len(output_indices) == 128, f'Expected128 output projections, got{len(output_indices)}'
    for i in indices:
        assert i > 0 and gpu[i - 1]['name'].startswith(OUTPUT), 'Logits copy lacks immediate output projection'
        assert '{65536; 1; 1} {8; 1; 1}' in gpu[i - 1]['name'], 'Unexpected output projection geometry'
        assert gpu[i - 1]['ts'] + gpu[i - 1]['dur'] <= gpu[i]['ts'] + 1
    assert indices[-1] == len(gpu) - 1, 'Unexpected device work after final logits copy'
    assert any(e['name'].startswith('ggml_sycl_ssm_scan_prefill(') for e in gpu[:indices[0]])
    assert not any(e['name'].startswith(PREFILL_ONLY) for e in gpu[indices[0] + 1:]), 'Prefill kernels leaked into decode'
    for prev, end in zip(indices, indices[1:]):
        start_us = gpu[prev]['ts'] + gpu[prev]['dur']
        end_us = gpu[end]['ts'] + gpu[end]['dur']
        assert all(e['ts'] >= start_us - 1 and e['ts'] + e['dur'] <= end_us + 1 for e in gpu[prev + 1:end + 1])
    return indices


def top_events(events, denominator, group=True):
    totals = defaultdict(lambda: [0, 0.0])
    for e in events:
        name = implementation(e['name']) if group else e['name']
        totals[name][0] += 1
        totals[name][1] += e['dur']
    return [{'name': k, 'calls': v[0], 'sum_ms': v[1] / 1000,
             'percent_summed_device_time': v[1] / denominator * 100}
            for k, v in sorted(totals.items(), key=lambda kv: kv[1][1], reverse=True)]


def window(gpu, cpu, indices, first_copy):
    origin = gpu[0]['ts']
    gpu = [dict(e, ts=e['ts'] - origin) for e in gpu]
    cpu = [dict(e, ts=e['ts'] - origin) for e in cpu]
    left, right = indices[first_copy - 1], indices[-1]
    begin = gpu[left]['ts'] + gpu[left]['dur']
    end = gpu[right]['ts'] + gpu[right]['dur']
    events = gpu[left + 1:right + 1]
    summed = sum(e['dur'] for e in events)
    busy = union_us(events)
    cycles = []
    per_step = []
    for prev, last in zip(indices[first_copy - 1:-1], indices[first_copy:]):
        cycles.append((gpu[last]['ts'] + gpu[last]['dur'] - gpu[prev]['ts'] - gpu[prev]['dur']) / 1000)
        per_step.append(Counter(implementation(e['name']) for e in gpu[prev + 1:last + 1]))
    host = []
    for e in cpu:
        duration = min(e['ts'] + e['dur'], end) - max(e['ts'], begin)
        if duration > 0:
            host.append(dict(e, dur=duration))
    host_totals = top_events(host, summed, group=False)
    for e in host_totals:
        del e['percent_summed_device_time']
    top = top_events(events, summed)
    for e in top:
        counts = [step[e['name']] for step in per_step]
        e['calls_per_step_min'] = min(counts)
        e['calls_per_step_max'] = max(counts)
    return {'steps': len(cycles), 'after_copy': first_copy, 'through_copy': 128,
            'trace_origin_us': origin, 'start_relative_us': begin, 'end_relative_us': end, 'device_events': len(events),
            'summed_device_ms': summed / 1000, 'device_busy_union_ms': busy / 1000,
            'window_ms': (end - begin) / 1000, 'uncovered_ms': (end - begin - busy) / 1000,
            'copy_to_copy_ms': cycles, 'copy_to_copy_median_ms': statistics.median(cycles),
            'top_implementations': top, 'top_shaped_kernels': top_events(events, summed, group=False)[:30],
            'host_api_sums_ms_not_cpu_work': host_totals[:20]}


def main():
    run = ROOT / 'run-r1'
    summary = json.loads(read(run / 'summary.json'))
    if summary['status'] == 'failed':
        assert summary.get('restoration_error') == 'AssertionError()' and 'error' not in summary
        restored = json.loads(read(run / 'restoration-check.json'))
        assert restored['status'] == 'passed' and restored['same_mounts_order_independent']
        assert restored['hardware_unchanged'] and restored['same_config'] and restored['same_container_id']
    else:
        assert summary['status'] == 'passed' and summary['restored']
    assert set(summary['phases']) == {'baseline', 'profile'}
    assert all(x['status'] == 'passed' for x in summary['phases'].values())
    cell = run / 'profile'
    profile = summary['phases']['profile']
    for key in ('warmup', 'profile'):
        result = profile[key]
        assert result['usage']['prompt_tokens'] == 8270 and result['usage']['completion_tokens'] == 128
        assert result['timings']['cache_n'] == 0
    trace = json.loads(read(cell / 'llama-server.1.json'))['traceEvents']
    gpu = sorted((e for e in trace if e.get('ph') == 'X' and e.get('cat') == 'gpu_op'), key=lambda e: e['ts'])
    cpu = [e for e in trace if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
    assert gpu and all(e['dur'] >= 0 for e in gpu)
    # This unitrace build mixes BOOTTIME epoch offset with MONOTONIC_RAW values.
    # Validate relative phase durations instead of treating trace ts as wall-clock epoch.
    lines = read(cell / 'unitrace.1.txt').splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith('Kernel,'))
    rows = []
    for line in lines[start + 1:]:
        if not line.strip():
            break
        row = next(csv.reader([line.strip()], skipinitialspace=True))
        rows.append((int(row[1]), int(row[2])))
    assert len(gpu) == sum(n for n, _ in rows), 'Trace/table event count mismatch'
    assert abs(sum(e['dur'] for e in gpu) * 1000 - sum(ns for _, ns in rows)) < 1000, 'Trace/table duration mismatch'
    indices = boundaries(gpu)
    prefill_ms = (gpu[indices[0]]['ts'] - gpu[0]['ts'] + gpu[indices[0]]['dur']) / 1000
    decode_ms = (gpu[indices[-1]]['ts'] - gpu[indices[0]]['ts'] + gpu[indices[-1]]['dur'] - gpu[indices[0]]['dur']) / 1000
    timings = profile['profile']['timings']
    residuals = {'prefill_trace_ms': prefill_ms, 'decode_trace_ms': decode_ms,
                 'prefill_api_minus_trace_ms': timings['prompt_ms'] - prefill_ms,
                 'decode_api_minus_trace_ms': timings['predicted_ms'] - decode_ms}
    assert abs(residuals['prefill_api_minus_trace_ms']) < 50
    assert abs(residuals['decode_api_minus_trace_ms']) < 50
    assert abs(prefill_ms / timings['prompt_ms'] - 1) < .01
    assert abs(decode_ms / timings['predicted_ms'] - 1) < .01
    base = summary['phases']['baseline']
    metrics = {}
    for mode, count in (('bench', 55), ('long', 8270)):
        samples = base[mode]['measured']
        assert len(samples) == 3
        assert all(x['usage']['prompt_tokens'] == count and x['usage']['completion_tokens'] == 128 and x['timings']['cache_n'] == 0 for x in samples)
        metrics[mode] = {key: {'samples': [x['timings'][key] for x in samples],
                              'median': statistics.median(x['timings'][key] for x in samples)}
                         for key in ('prompt_ms', 'prompt_per_second', 'predicted_ms', 'predicted_per_second')}
    result = {'status': 'passed', 'tier': 'development-diagnostic', 'baseline': metrics,
              'profile_api_timings': profile['profile']['timings'], 'warmup_api_timings': profile['warmup']['timings'],
              'full_trace_device_events': len(gpu), 'logits_copies': len(indices), 'relative_phase_validation': residuals,
              'trace_minus_api_start_ms': (gpu[0]['ts'] - profile['profile_start_epoch_ns'] / 1000) / 1000,
              'controller_status': summary['status'], 'restoration_independently_verified': True,
              'decode_all': window(gpu, cpu, indices, 1), 'decode_steady': window(gpu, cpu, indices, 16),
              'caveats': ['One instrumented8270/128 request after full8270/128 warmup, not a speed comparison.',
                          'First logits copy ends prefill; copy2..128 end127 autoregressive graph executions.',
                          'Server predicted_n128 counts generated tokens, not127 post-prefill graph executions.',
                          'Device sums can include overlaps; union and gaps are separate.',
                          'Trace clock is not directly aligned to wall time; relative phases agree within50ms and1%.',
                          'Gaps include inter-step host sampling and substantial profiler overhead, not proof of production launch bottlenecks.',
                          'Host synchronize/copy API durations include waiting, not necessarily CPU work or PCIe transfer time.',
                          'No bandwidth, occupancy, maximum-context, memory-capacity or broad-quality claim.']}
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('decode_all', 'decode_steady')}, indent=2))
    for key in ('decode_all', 'decode_steady'):
        w = result[key]
        print(key, json.dumps({k: w[k] for k in ('steps', 'summed_device_ms', 'device_busy_union_ms', 'window_ms', 'uncovered_ms', 'copy_to_copy_median_ms')}))
        for e in w['top_implementations'][:15]:
            print(round(e['percent_summed_device_time'], 3), round(e['sum_ms'], 3), e['calls'], e['name'][:180])
    print('BOLT_WARMED_DECODE_ANALYSIS_PASSED')


if __name__ == '__main__':
    main()
