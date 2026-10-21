#!/usr/bin/env python3
"""Bounded analysis of unitrace JSON and explicit real-request time windows.

Intervals are measured inside an instrumented run, not inferred clean costs.
No kernel family is assigned to a model without a supplied source-backed rule.
"""
import argparse
import bisect
import collections
import json
import math
from pathlib import Path


def trace_us(stamp):
    """Pinned UniTimer = (REALTIME - BOOTTIME at init) + MONOTONIC_RAW.

    Caller must retain the three contemporaneous clock readings and their
    acquisition span. Raw Chrome values are not Linux REALTIME timestamps.
    See pti-gpu commit 887bba6e tools/unitrace/src/unitimer.h.
    """
    assert stamp['clock_read_span_ns'] >= 0
    return (stamp['realtime_ns'] - stamp['boottime_ns'] + stamp['monotonic_raw_ns']) / 1000


def events(path):
    """Stream the pinned unitrace's one-event-per-line format; reject truncation."""
    with Path(path).open() as stream:
        assert stream.readline().strip() == '{ "traceEvents":[', 'unexpected trace envelope'
        count = 0
        closed = False
        previous_comma = True
        for line in stream:
            line = line.strip()
            if line == ']':
                assert not previous_comma, 'trailing event comma'
                assert stream.readline().strip() == '}', 'unclosed trace envelope'
                closed = True
                assert not stream.read().strip(), 'trailing trace content'
                break
            assert previous_comma, 'missing event separator'
            assert line.startswith('{'), ('unexpected trace row', count)
            previous_comma = line.endswith(',')
            event = json.loads(line.removesuffix(','))
            assert isinstance(event, dict) and 'ph' in event
            count += 1
            yield event
        assert closed and count, 'empty or truncated trace'


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        assert math.isfinite(start) and math.isfinite(end) and end >= start
        if end == start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def partition(intervals, begin, end):
    """Disjoint wall-time buckets, preserving concurrent GPU/host activity."""
    changes = collections.defaultdict(collections.Counter)
    changes[begin]
    changes[end]
    for label, rows in intervals.items():
        for start, stop in union(rows):
            start, stop = max(begin, start), min(end, stop)
            if stop > start:
                changes[start][label] += 1
                changes[stop][label] -= 1
    active = collections.Counter()
    result = collections.Counter()
    previous = begin
    for timestamp in sorted(changes):
        result['+'.join(sorted(k for k, v in active.items() if v)) or 'untraced'] += timestamp - previous
        active.update(changes[timestamp])
        assert all(v >= 0 for v in active.values()), 'unbalanced intervals'
        previous = timestamp
    assert all(v == 0 for v in active.values())
    assert abs(sum(result.values()) - (end - begin)) < 0.01
    return {key: value / 1000 for key, value in sorted(result.items()) if value}


def cpu_kind(name):
    if 'Synchronize' in name or 'QueryStatus' in name:
        return 'host_sync_api'
    if 'Append' in name or 'ExecuteCommandLists' in name or 'SetGroupSize' in name or 'SetArgument' in name:
        return 'host_submission_api'
    return 'host_other_api'


def kernel_kind(name, rules):
    matches = [r['category'] for r in rules if r['substring'] in name]
    assert len(matches) <= 1, ('ambiguous kernel mapping', name, matches)
    return matches[0] if matches else 'shared_or_unattributed'


def analyze(trace, windows, rules=()):
    assert windows
    for rule in rules:
        assert rule['substring'] and rule['category'] and rule['evidence'], 'source evidence required'
    rows = []
    for window in sorted(windows, key=lambda w: w['start_us']):
        assert window['end_us'] > window['start_us'] > 0
        assert window['cell'] and window['point'] and window['stem']
        assert window.get('fidelity_passed') is True, 'unqualified request'
        assert window.get('traced') is True, 'cannot attribute untraced request'
        if rows:
            assert window['start_us'] >= rows[-1]['end_us'], 'overlapping request windows'
        rows.append(dict(**window, intervals=collections.defaultdict(list),
                         kernels=collections.defaultdict(lambda: [0, 0.0]),
                         api=collections.defaultdict(lambda: [0, 0.0]),
                         categories=collections.defaultdict(list), gpu_events=0, cpu_events=0))
    starts = [row['start_us'] for row in rows]
    total_events = 0
    selected_events = 0
    metadata = []
    for event in events(trace):
        total_events += 1
        if event['ph'] == 'M':
            metadata.append(event)
            continue
        if event['ph'] != 'X' or event.get('cat') not in ('gpu_op', 'cpu_op'):
            continue
        timestamp, elapsed = float(event['ts']), float(event['dur'])
        assert math.isfinite(timestamp) and math.isfinite(elapsed) and elapsed >= 0
        index = max(0, bisect.bisect_right(starts, timestamp) - 1)
        while index < len(rows) and rows[index]['start_us'] <= timestamp + elapsed:
            row = rows[index]
            begin, end = max(row['start_us'], timestamp), min(row['end_us'], timestamp + elapsed)
            if end > begin:
                selected_events += 1
                name = event['name']
                if event['cat'] == 'gpu_op':
                    row['gpu_events'] += 1
                    row['intervals']['gpu'].append((begin, end))
                    row['kernels'][name][0] += 1
                    row['kernels'][name][1] += end - begin
                    category = kernel_kind(name, rules)
                    row['categories'][category].append((begin, end))
                else:
                    row['cpu_events'] += 1
                    row['intervals'][cpu_kind(name)].append((begin, end))
                    row['api'][name][0] += 1
                    row['api'][name][1] += end - begin
            index += 1
    reports = []
    for row in rows:
        assert row['gpu_events'] and row['cpu_events'], ('empty real request window', row['stem'])
        intervals = row.pop('intervals')
        categories = row.pop('categories')
        kernels = row.pop('kernels')
        api = row.pop('api')
        wall = row['end_us'] - row['start_us']
        gpu = duration(intervals['gpu'])
        row.update(wall_ms=wall / 1000, gpu_busy_union_ms=gpu / 1000,
                   gpu_busy_wall_fraction=gpu / wall,
                   wall_partition_ms=partition(intervals, row['start_us'], row['end_us']),
                   gpu_category_union_ms={key: duration(value) / 1000 for key, value in sorted(categories.items())},
                   kernel_event_sum_ms=sum(v[1] for v in kernels.values()) / 1000,
                   top_kernels=[dict(name=name, count=value[0], event_sum_ms=value[1] / 1000)
                                for name, value in sorted(kernels.items(), key=lambda p: -p[1][1])[:40]],
                   host_api=[dict(name=name, count=value[0], event_sum_ms=value[1] / 1000)
                             for name, value in sorted(api.items(), key=lambda p: -p[1][1])])
        reports.append(row)
    return dict(diagnostic_only=True, clean_speed_claim=False, trace=str(trace),
                total_trace_events=total_events, selected_interval_events=selected_events,
                metadata=metadata, source_rules=list(rules), requests=reports,
                limitations=[
                    'Tracer changes event submission; durations describe the traced run only.',
                    'Kernel event sums and per-category unions overlap; neither is an additive latency partition.',
                    'Wall partition is additive and retains CPU/GPU overlap explicitly.',
                    'untraced means no observed GPU or Level Zero API activity, not proven CPU compute or avoidable overhead.',
                    'Request windows include HTTP/SSE delivery; no exact prefill/decode boundary is inferred.',
                    'USM traffic and host code outside Level Zero may be invisible.',
                    'No critical-path or model assignment is inferred from names alone.'
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--windows', type=Path, required=True)
    parser.add_argument('--rules', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve prior analysis'
    rules = json.loads(args.rules.read_text()) if args.rules else []
    result = analyze(args.trace, json.loads(args.windows.read_text()), rules)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(requests=len(result['requests']), events=result['total_trace_events'], output=str(args.output))))


if __name__ == '__main__':
    main()
