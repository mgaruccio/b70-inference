"""Attribute XPU kernels once via CPU external IDs and nested draft scopes."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys


def summarize(path):
    with gzip.open(path, 'rt') if path.suffix == '.gz' else path.open() as source:
        events = json.load(source)['traceEvents']
    threads = collections.defaultdict(list)
    for event in events:
        if event.get('ph') == 'X' and (event.get('cat') == 'cpu_op' or
                (event.get('cat') == 'user_annotation' and event['name'].startswith('b70_draft/'))):
            threads[event['pid'], event['tid']].append(event)
    lookup = {}
    for rows in threads.values():
        stack = []
        for event in sorted(rows, key=lambda e: (e['ts'], -e['dur'])):
            while stack and stack[-1]['ts'] + stack[-1]['dur'] <= event['ts']:
                stack.pop()
            if event['cat'] == 'user_annotation':
                stack.append(event)
                continue
            enclosing = [e['name'] for e in stack if e['ts'] <= event['ts'] and
                         event['ts'] + event['dur'] <= e['ts'] + e['dur'] + 0.001]
            phases = [name.split(':', 1)[1] for name in enclosing if name.startswith('b70_draft/phase:')]
            modules = [name for name in enclosing if not name.startswith('b70_draft/phase:')]
            external = event.get('args', {}).get('External id')
            if external is not None:
                key = (event['pid'], external)
                if key in lookup:
                    raise ValueError(f'Duplicate CPU External id: {key}')
                lookup[key] = (event, phases[-1] if phases else None, modules[-1] if modules else None)
    phase_counts = collections.Counter(e['name'].split(':', 1)[1] for e in events
                                       if e.get('name', '').startswith('b70_draft/phase:'))
    steps = phase_counts['generate']
    if not steps:
        raise ValueError('No draft generation scopes')
    aggregate = {key: collections.defaultdict(lambda: {'count': 0, 'total_us': 0.0})
                 for key in ('phases', 'modules', 'operators', 'kernels')}
    total_us = draft_us = missing_us = outside_us = 0.0
    missing_count = 0
    queues = set()
    phase_events = [e for e in events if e.get('name', '').startswith('b70_draft/phase:')]
    launches = {}
    for event in events:
        correlation = event.get('args', {}).get('correlation')
        if event.get('cat') == 'xpu_runtime' and correlation is not None:
            if correlation in launches:
                raise ValueError(f'Duplicate runtime correlation: {correlation}')
            launches[correlation] = event
    missing_audit = collections.defaultdict(lambda: {'count': 0, 'kernel_ms': 0.0})
    # GPU events use a different pid; external IDs must resolve uniquely across CPU processes.
    by_id = {}
    for (_, external), value in lookup.items():
        if external in by_id:
            raise ValueError(f'Cross-process External id collision: {external}')
        by_id[external] = value
    for event in events:
        if event.get('cat') != 'kernel':
            continue
        total_us += event['dur']
        queues.add((event['pid'], event['tid']))
        cpu, phase, module = by_id.get(event.get('args', {}).get('External id'), (None, None, None))
        if cpu is None:
            missing_us += event['dur']
            missing_count += 1
            launch = launches.get(event.get('args', {}).get('correlation'))
            audit_key = 'missing_runtime_launch'
            if launch is not None:
                scopes = [p for p in phase_events if p['pid'] == launch['pid'] and p['tid'] == launch['tid']
                          and p['ts'] <= launch['ts'] and launch['ts'] + launch['dur'] <= p['ts'] + p['dur']]
                audit_key = min(scopes, key=lambda p: p['dur'])['name'] if scopes else 'outside_draft'
            missing_audit[audit_key]['count'] += 1
            missing_audit[audit_key]['kernel_ms'] += event['dur'] / 1000
        if phase is None:
            outside_us += event['dur']
            continue
        draft_us += event['dur']
        keys = {'phases': phase, 'modules': phase + ' / ' + (module or '(phase only)'),
                'operators': phase + ' / ' + cpu['name'] + ' ' + json.dumps(cpu.get('args', {}).get('Input Dims')),
                'kernels': phase + ' / ' + event['name']}
        for category, key in keys.items():
            aggregate[category][key]['count'] += 1
            aggregate[category][key]['total_us'] += event['dur']
    result = {'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'draft_generation_steps': steps, 'phase_counts': dict(phase_counts),
              'decode_annotations': [e['name'] for e in events if e.get('name', '').startswith('execute_')],
              'kernel_queues': sorted(queues), 'all_kernel_ms': total_us / 1000,
              'draft_kernel_ms': draft_us / 1000, 'outside_draft_or_unmapped_ms': outside_us / 1000,
              'unmapped_cpu_external_id': {'count': missing_count, 'kernel_ms': missing_us / 1000},
              'unmapped_kernel_runtime_correlation_audit': dict(missing_audit),
              'caveat': 'Eager diagnostic, not graph timing or throughput. Each kernel counted once in its innermost phase; context KV is outside draft generation. Summed device work is not critical-path latency.'}
    for category, rows in aggregate.items():
        result[category] = [{'name': key, **value, 'ms_per_generation_step': value['total_us'] / (1000 * steps)}
                           for key, value in sorted(rows.items(), key=lambda row: -row[1]['total_us'])]
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
