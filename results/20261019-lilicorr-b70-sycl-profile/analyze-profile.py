#!/usr/bin/env python3
"""Fail-closed host-wall attribution for the second LiliCorr SYCL diagnostic."""
from collections import defaultdict
import re
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIRST = HERE.parent / '20261019-lilicorr-b70-profile'
BASE = runpy.run_path(str(FIRST / 'analyze-profile.py'))
MARKER = 'LLAMA_LILICORR_SYCL_PROFILE '
LILICORR_LATTICE = 'lilicorr_lattice'
LILICORR_WEIGHT_PREFIX = 'lilicorr.'

GATE_FIELDS = {'v', 'stage', 'graph', 'requested', 'enabled', 'marker', 'graph_nodes', 'marker_name'}
OP_FIELDS = {
    'v', 'stage', 'graph', 'first_index', 'last_index', 'count', 'begin_us', 'end_us',
    'elapsed_host_us', 'group', 'first_op', 'last_op', 'first_name', 'last_name',
    'src0_name', 'src0_type', 'src1_name', 'src1_type', 'dst_type',
    'ne0', 'ne1', 'ne2', 'ne3', 'src0_ne0', 'src0_ne1', 'src0_ne2', 'src0_ne3',
    'dst_ne0', 'dst_ne1', 'dst_ne2', 'dst_ne3',
}
GRAPH_FIELDS = {
    'v', 'stage', 'graph', 'requested', 'enabled', 'marker', 'graph_nodes', 'covered_nodes',
    'uncovered_nodes', 'skipped_nodes', 'spans', 'first_index', 'last_index', 'begin_us',
    'end_us', 'core_wall_us', 'event_wall_us', 'uncounted_wall_us', 'log_wall_us',
    'covered_ranges', 'uncovered_ranges', 'marker_name',
}
INTEGER_FIELDS = {
    'v', 'graph', 'requested', 'enabled', 'marker', 'graph_nodes', 'first_index', 'last_index',
    'count', 'begin_us', 'end_us', 'elapsed_host_us', 'ne0', 'ne1', 'ne2', 'ne3',
    'src0_ne0', 'src0_ne1', 'src0_ne2', 'src0_ne3', 'dst_ne0', 'dst_ne1', 'dst_ne2', 'dst_ne3',
    'covered_nodes', 'uncovered_nodes', 'skipped_nodes', 'spans', 'core_wall_us',
    'event_wall_us', 'uncounted_wall_us', 'log_wall_us',
}


def _parse_fields(line):
    assert line.count(MARKER) == 1
    fields = line.split(MARKER, 1)[1].strip().split()
    row = {}
    for field in fields:
        key, sep, value = field.partition('=')
        assert sep and key and value, field
        assert key not in row, 'duplicate SYCL profile field: ' + key
        row[key] = value
    return row


def _valid_marker_name(name):
    return name == LILICORR_LATTICE or name.startswith(LILICORR_WEIGHT_PREFIX)


def event(line):
    """Parse one backend record and validate its shape without inferring device time."""
    row = _parse_fields(line)
    stage = row.get('stage')
    expected = {'gate': GATE_FIELDS, 'op': OP_FIELDS, 'graph': GRAPH_FIELDS}.get(stage)
    assert expected is not None, stage
    assert set(row) == expected, (stage, sorted(set(row) ^ expected))
    for key in set(row) & INTEGER_FIELDS:
        assert re.fullmatch(r'-?\d+', row[key]), (key, row[key])
        row[key] = int(row[key])
    assert row['v'] == 1 and row['graph'] > 0
    if stage != 'op':
        assert row['requested'] == 1
    if stage == 'gate':
        assert row['graph_nodes'] > 0
        assert row['enabled'] in (0, 1) and row['marker'] == row['enabled']
        if row['enabled']:
            assert _valid_marker_name(row['marker_name'])
        else:
            assert row['marker_name'] == '-'
        return row
    if stage == 'op':
        assert row['graph'] > 0 and 0 <= row['first_index'] <= row['last_index']
        assert row['count'] == row['last_index'] - row['first_index'] + 1
        assert 0 < row['begin_us'] <= row['end_us']
        assert row['elapsed_host_us'] == row['end_us'] - row['begin_us']
        assert row['group'] not in ('', '-')
        assert all(row[key] >= 0 for key in row
                   if key.endswith('_ne') or key in ('ne0', 'ne1', 'ne2', 'ne3'))
        return row
    assert row['graph_nodes'] > 0
    assert row['covered_nodes'] >= 0 and row['uncovered_nodes'] >= 0
    assert row['covered_nodes'] + row['uncovered_nodes'] == row['graph_nodes']
    assert row['skipped_nodes'] >= 0 and row['skipped_nodes'] <= row['uncovered_nodes']
    assert row['spans'] >= 0
    assert row['core_wall_us'] >= 0 and row['event_wall_us'] >= 0
    assert row['uncounted_wall_us'] >= 0 and row['log_wall_us'] >= 0
    if row['enabled']:
        assert row['marker'] == 1 and row['begin_us'] > 0
        assert row['end_us'] >= row['begin_us']
        assert row['end_us'] - row['begin_us'] == row['core_wall_us']
        assert row['event_wall_us'] <= row['core_wall_us']
        assert row['uncounted_wall_us'] == row['core_wall_us'] - row['event_wall_us']
        assert _valid_marker_name(row['marker_name'])
        assert 0 <= row['first_index'] <= row['last_index'] < row['graph_nodes']
        assert row['covered_ranges'] != '-' and row['uncovered_ranges'] != 'all'
    else:
        assert row['marker'] == 0 and row['enabled'] == 0
        assert row['covered_nodes'] == 0 and row['spans'] == 0
        assert row['first_index'] == row['last_index'] == -1
        assert row['begin_us'] == row['end_us'] == row['core_wall_us'] == 0
        assert row['event_wall_us'] == row['uncounted_wall_us'] == 0
        assert row['marker_name'] == '-'
        assert row['covered_ranges'] == '-' and row['uncovered_ranges'] == 'all'
    return row


def _ranges(value, size):
    if value == '-':
        return set()
    if value == 'all':
        return set(range(size))
    result = set()
    for part in value.split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        assert match, part
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        assert 0 <= first <= last < size
        before = len(result)
        result.update(range(first, last + 1))
        assert len(result) == before + last - first + 1, 'overlapping coverage range'
    return result


def parse_sycl(lines, first_line=1):
    result = []
    for offset, line in enumerate(lines):
        if MARKER in line:
            row = event(line)
            row['line'] = first_line + offset
            result.append(row)
    return result


def _graph_bundles(events):
    by_graph = defaultdict(list)
    for row in events:
        by_graph[row['graph']].append(row)
    bundles = []
    for graph_id, rows in sorted(by_graph.items(), key=lambda item: min(r['line'] for r in item[1])):
        gates = [r for r in rows if r['stage'] == 'gate']
        graphs = [r for r in rows if r['stage'] == 'graph']
        ops = [r for r in rows if r['stage'] == 'op']
        assert len(gates) == 1 and len(graphs) == 1, 'missing/duplicate graph gate or summary'
        gate, summary = gates[0], graphs[0]
        ordered = sorted(rows, key=lambda row: row['line'])
        assert ordered[0]['stage'] == 'gate' and ordered[-1]['stage'] == 'graph'
        assert all(row['stage'] == 'op' for row in ordered[1:-1])
        assert gate['enabled'] == summary['enabled']
        assert gate['marker'] == summary['marker']
        assert gate['graph_nodes'] == summary['graph_nodes']
        assert gate['marker_name'] == summary['marker_name']
        if not summary['enabled']:
            assert not ops
            bundles.append(dict(graph=graph_id, gate=gate, summary=summary, ops=[]))
            continue
        assert gate['enabled'] == 1 and len(ops) == summary['spans']
        coverage = _ranges(summary['covered_ranges'], summary['graph_nodes'])
        uncovered = _ranges(summary['uncovered_ranges'], summary['graph_nodes'])
        assert coverage.isdisjoint(uncovered)
        assert coverage | uncovered == set(range(summary['graph_nodes']))
        assert len(coverage) == summary['covered_nodes']
        assert len(uncovered) == summary['uncovered_nodes']
        ops = sorted(ops, key=lambda row: row['begin_us'])
        op_coverage = set()
        event_sum = 0
        for previous, current in zip(ops, ops[1:]):
            assert previous['end_us'] <= current['begin_us'], 'overlapping operation host intervals'
        for op in ops:
            assert summary['begin_us'] <= op['begin_us'] <= op['end_us'] <= summary['end_us']
            span = set(range(op['first_index'], op['last_index'] + 1))
            assert span.isdisjoint(op_coverage), 'overlapping consumed node spans'
            op_coverage |= span
            event_sum += op['elapsed_host_us']
        assert op_coverage == coverage, 'missing or extra operation coverage'
        assert event_sum == summary['event_wall_us']
        bundles.append(dict(graph=graph_id, gate=gate, summary=summary, ops=ops))
    assert bundles
    return bundles


def _submit_interval(ubatch):
    begin = ubatch['begin_us'] + ubatch['build_alloc_us'] + ubatch['inputs_us']
    return begin, begin + ubatch['submit_us']


def _contains_interval(parent, child):
    return parent['begin_us'] <= child['begin_us'] <= child['end_us'] <= parent['end_us']


def _phase_owner(ubatch, phase_events):
    candidates = [row for row in phase_events
                  if row['stage'] in ('draft_batch', 'process_chunk')
                  and _contains_interval(row, ubatch)]
    assert len(candidates) == 1, 'ubatch missing/ambiguous draft/process phase owner'
    parent = candidates[0]
    if parent['stage'] == 'draft_batch':
        assert ubatch['rows'] == parent['rows'] == 8, 'draft graph is not an eight-row ubatch'
    return parent


def _scheduler_owners(phase_events):
    ubatches = [row for row in phase_events if row['stage'] == 'ubatch']
    assert ubatches, 'missing scheduler ubatches'
    owners = {}
    for ubatch in ubatches:
        assert ubatch['line'] not in owners, 'duplicate scheduler ubatch line'
        owners[ubatch['line']] = (ubatch, _phase_owner(ubatch, phase_events))
    return owners


def _phase_parent(bundle, phase_events):
    summary = bundle['summary']
    owners = []
    for ubatch, parent in _scheduler_owners(phase_events).values():
        begin, end = _submit_interval(ubatch)
        if begin <= summary['begin_us'] <= summary['end_us'] <= end:
            owners.append((ubatch, parent))
    assert len(owners) == 1, 'SYCL graph missing/ambiguous phase ubatch owner'
    ubatch, parent = owners[0]
    assert _contains_interval(parent, summary), 'SYCL graph is outside its phase parent interval'
    return dict(stage='draft' if parent['stage'] == 'draft_batch' else 'process',
                parent_stage=parent['stage'], parent_line=parent['line'],
                parent_rows=parent['rows'], ubatch_line=ubatch['line'],
                ubatch_rows=ubatch['rows'], ubatch_gtype=ubatch['gtype'],
                ubatch_nodes=ubatch['nodes'])


def _operator_totals(bundles):
    totals = {}
    for bundle in bundles:
        for op in bundle['ops']:
            shape = tuple(op[f'ne{i}'] for i in range(4))
            key = (op['group'], op['first_op'], op['last_op'], op['src0_name'],
                   op['src0_type'], op['src1_name'], op['src1_type'], op['dst_type'], shape)
            item = totals.setdefault(key, dict(group=op['group'], first_op=op['first_op'],
                last_op=op['last_op'], src0_name=op['src0_name'], src0_type=op['src0_type'],
                src1_name=op['src1_name'], src1_type=op['src1_type'], dst_type=op['dst_type'],
                shape=list(shape), calls=0, consumed_nodes=0, host_wall_us=0, raw_refs=[]))
            item['calls'] += 1
            item['consumed_nodes'] += op['count']
            item['host_wall_us'] += op['elapsed_host_us']
            item['raw_refs'].append(dict(line=op['line'], graph=op['graph'],
                                         first_index=op['first_index'], last_index=op['last_index'],
                                         begin_us=op['begin_us'], end_us=op['end_us']))
    return sorted(totals.values(), key=lambda item: (item['group'], item['first_op'], item['src0_name'], item['shape']))


def _empty_sycl_report():
    return dict(graph_count=0, enabled_graph_count=0, disabled_graph_count=0, event_count=0,
                graphs=[], disabled_graphs=[], graph_core_wall_us=0, operation_wall_us=0,
                uncounted_wall_us=0, logging_wall_us=0, operator_totals=[], raw_refs=[],
                scheduler_coverage=dict(expected_ubatches=0, covered_ubatches=0,
                                        uncovered_ubatches=0, expected_draft_ubatches=0,
                                        covered_draft_ubatches=0, uncovered_draft_ubatches=0,
                                        expected_nodes=0, covered_nodes=0, uncovered_nodes=0))


def _unattributed_sycl(events):
    if not events:
        return _empty_sycl_report()
    bundles = _graph_bundles(events)
    enabled_bundles = [bundle for bundle in bundles if bundle['summary']['enabled']]
    graph_reports = []
    for bundle in enabled_bundles:
        summary = bundle['summary']
        graph_reports.append(dict(graph=bundle['graph'], line=summary['line'],
            graph_nodes=summary['graph_nodes'], covered_nodes=summary['covered_nodes'],
            uncovered_nodes=summary['uncovered_nodes'], covered_ranges=summary['covered_ranges'],
            uncovered_ranges=summary['uncovered_ranges'], core_wall_us=summary['core_wall_us'],
            event_wall_us=summary['event_wall_us'], uncounted_wall_us=summary['uncounted_wall_us'],
            log_wall_us=summary['log_wall_us'], operation_refs=[dict(line=op['line'],
                first_index=op['first_index'], last_index=op['last_index'], group=op['group'],
                elapsed_host_us=op['elapsed_host_us']) for op in bundle['ops']]))
    disabled_graphs = [dict(graph=bundle['graph'], line=bundle['summary']['line'],
                             graph_nodes=bundle['summary']['graph_nodes'],
                             uncovered_nodes=bundle['summary']['uncovered_nodes'],
                             reason='lilicorr marker absent')
                       for bundle in bundles if not bundle['summary']['enabled']]
    expected_nodes = sum(item['graph_nodes'] for item in graph_reports)
    covered_nodes = sum(item['covered_nodes'] for item in graph_reports)
    uncovered_nodes = sum(item['uncovered_nodes'] for item in graph_reports)
    assert expected_nodes == covered_nodes + uncovered_nodes
    return dict(graph_count=len(bundles), enabled_graph_count=len(graph_reports),
                disabled_graph_count=len(disabled_graphs), event_count=len(events),
                graphs=graph_reports, disabled_graphs=disabled_graphs,
                graph_core_wall_us=sum(item['core_wall_us'] for item in graph_reports),
                operation_wall_us=sum(op['elapsed_host_us'] for bundle in enabled_bundles
                                      for op in bundle['ops']),
                uncounted_wall_us=sum(item['uncounted_wall_us'] for item in graph_reports),
                logging_wall_us=sum(item['log_wall_us'] for item in graph_reports),
                operator_totals=_operator_totals(enabled_bundles),
                raw_refs=[dict(line=row['line'], stage=row['stage'], graph=row['graph'])
                          for row in events],
                scheduler_coverage=dict(expected_ubatches=len(enabled_bundles),
                    covered_ubatches=len(enabled_bundles), uncovered_ubatches=0,
                    expected_draft_ubatches=0, covered_draft_ubatches=0,
                    uncovered_draft_ubatches=0, expected_nodes=expected_nodes,
                    covered_nodes=covered_nodes, uncovered_nodes=uncovered_nodes))


def request_sycl(request, phase_request):
    events = parse_sycl(request['lines'], request['first_line'])
    assert events, 'missing SYCL diagnostic events'
    bundles = _graph_bundles(events)
    enabled_bundles = [bundle for bundle in bundles if bundle['summary']['enabled']]
    assert enabled_bundles, 'profile gate did not cover any LiliCorr graph'
    phase_events = phase_request['events']
    expected = _scheduler_owners(phase_events)
    owner_by_ubatch = {}
    graph_reports = []
    for bundle in enabled_bundles:
        parent = _phase_parent(bundle, phase_events)
        owner_line = parent['ubatch_line']
        assert owner_line not in owner_by_ubatch, 'duplicate SYCL graph owner'
        summary = bundle['summary']
        assert summary['graph_nodes'] == parent['ubatch_nodes'], 'graph/scheduler node count mismatch'
        owner_by_ubatch[owner_line] = (bundle, parent)
        graph_reports.append(dict(graph=bundle['graph'], line=summary['line'],
            graph_nodes=summary['graph_nodes'], covered_nodes=summary['covered_nodes'],
            uncovered_nodes=summary['uncovered_nodes'], covered_ranges=summary['covered_ranges'],
            uncovered_ranges=summary['uncovered_ranges'], core_wall_us=summary['core_wall_us'],
            event_wall_us=summary['event_wall_us'], uncounted_wall_us=summary['uncounted_wall_us'],
            log_wall_us=summary['log_wall_us'], phase=parent,
            operation_refs=[dict(line=op['line'], first_index=op['first_index'],
                last_index=op['last_index'], group=op['group'],
                elapsed_host_us=op['elapsed_host_us']) for op in bundle['ops']]))
    expected_lines = set(expected)
    assert set(owner_by_ubatch) == expected_lines, 'missing/extra scheduler graph owner'
    assert not [bundle for bundle in bundles if not bundle['summary']['enabled']], \
        'disabled graph owner in measured request'
    expected_nodes = sum(ubatch['nodes'] for ubatch, _ in expected.values())
    covered_nodes = sum(item['covered_nodes'] for item in graph_reports)
    uncovered_nodes = sum(item['uncovered_nodes'] for item in graph_reports)
    assert expected_nodes == covered_nodes + uncovered_nodes
    expected_drafts = {line for line, (_, parent) in expected.items()
                       if parent['stage'] == 'draft_batch'}
    covered_drafts = expected_drafts & set(owner_by_ubatch)
    graph_core = sum(item['core_wall_us'] for item in graph_reports)
    return dict(graph_count=len(bundles), enabled_graph_count=len(graph_reports),
        disabled_graph_count=0, event_count=len(events), graphs=sorted(graph_reports,
            key=lambda item: item['line']), disabled_graphs=[], graph_core_wall_us=graph_core,
        operation_wall_us=sum(op['elapsed_host_us'] for bundle in enabled_bundles
                              for op in bundle['ops']),
        uncounted_wall_us=sum(item['uncounted_wall_us'] for item in graph_reports),
        logging_wall_us=sum(item['log_wall_us'] for item in graph_reports),
        operator_totals=_operator_totals(enabled_bundles),
        raw_refs=[dict(line=row['line'], stage=row['stage'], graph=row['graph']) for row in events],
        scheduler_coverage=dict(expected_ubatches=len(expected), covered_ubatches=len(owner_by_ubatch),
            uncovered_ubatches=len(expected) - len(owner_by_ubatch),
            expected_draft_ubatches=len(expected_drafts), covered_draft_ubatches=len(covered_drafts),
            uncovered_draft_ubatches=len(expected_drafts - covered_drafts), expected_nodes=expected_nodes,
            covered_nodes=covered_nodes, uncovered_nodes=uncovered_nodes))


def _startup_sycl(lines):
    return _unattributed_sycl(parse_sycl(lines, 1))


def _nonmeasured_sycl(request):
    parsed = _unattributed_sycl(parse_sycl(request['lines'], request['first_line']))
    assert parsed['enabled_graph_count'] == 0, 'timed SYCL graph outside measured schedule'
    return parsed


def analyze(out, c):
    out = Path(out)
    report = dict(status='failed', tier='development', gpu_time_claim=False,
                  speed_comparison_permitted=False, nested_operation_not_additive=True,
                  requests=[], errors=[])
    try:
        inherited = BASE['analyze'](out, c)
        assert inherited['status'] == 'analyzed' and not inherited['errors']
        report = inherited
        report['gpu_time_claim'] = False
        report['speed_comparison_permitted'] = False
        report['nested_operation_not_additive'] = True
        report['sycl_diagnostic'] = dict(marker=MARKER, host_wall_only=True, graphs=[],
                                         warmup=[], startup={}, public=[])
        for cell in c['CELLS']:
            directory = out / cell
            stderr = (directory / 'server.stderr.log').read_text()
            stdout = (directory / 'server.stdout.log').read_text()
            assert MARKER not in stdout, 'unexpected SYCL diagnostic stream'
            requests, _startup = BASE['requests'](stderr)
            lines = stderr.splitlines()
            first_request = requests[0]['first_line']
            startup = _startup_sycl(lines[:first_request - 1])
            report['startup_events'][cell] = startup
            report['sycl_diagnostic']['startup'][cell] = startup
            public = [_nonmeasured_sycl(request) for request in requests[:25]]
            report['sycl_diagnostic']['public'].append(dict(cell=cell, requests=public))
            if c['role_for'](cell) == 'A':
                assert not any(item['enabled_graph_count'] for item in public)
                assert startup['enabled_graph_count'] == 0, 'no-draft baseline was profiled'
                for request in requests[25:]:
                    assert _nonmeasured_sycl(request)['enabled_graph_count'] == 0, \
                        'target received timed SYCL spans'
                continue
            schedule = [job for job in c['schedule']() if job['cell'] == cell]
            assert len(schedule) == 8
            phase_requests = [row for row in report['requests'] if row['cell'] == cell]
            assert len(phase_requests) == 8
            for index, job in enumerate(schedule, 25):
                request = requests[index]
                phase_request = phase_requests[index - 25]
                events = parse_sycl(request['lines'], request['first_line'])
                if not events:
                    parsed = _empty_sycl_report()
                elif job['phase'] == 'warmup' and not any(r['stage'] == 'op' for r in events):
                    parsed = _unattributed_sycl(events)
                else:
                    parsed = request_sycl(request, phase_request)
                assert request['prompts'] == [phase_request['counters']['prompt']]
                phase_request['sycl'] = parsed
                record = dict(cell=cell, point=job['point'], phase=job['phase'],
                              ordinal=job['ordinal'], **parsed)
                if job['phase'] == 'warmup':
                    report['sycl_diagnostic']['warmup'].append(record)
                else:
                    report['sycl_diagnostic']['graphs'].append(record)
        assert len(report['requests']) == 16
        assert report['sycl_diagnostic']['graphs']
        report['status'] = 'analyzed'
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        report['status'] = 'failed'
        report.setdefault('errors', []).append(repr(exc))
    c['save'](out / 'analysis-r1/profile.json', report)
    return report
