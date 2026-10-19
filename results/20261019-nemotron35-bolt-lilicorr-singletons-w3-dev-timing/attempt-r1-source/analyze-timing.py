#!/usr/bin/env python3
"""Independent post-restoration raw timing/fidelity analysis.

This process is intentionally separate from capture.  It first performs fresh
live restoration checks, then rereads raw public HTTP/SSE, event timings,
responses, telemetry, and prevalidation evidence before deriving descriptive
ABBA medians and type-7 IQRs.
"""
import argparse
import hashlib
import json
import math
import runpy
import socket
from pathlib import Path


HERE = Path(__file__).resolve().parent
C = runpy.run_path(str(HERE / 'contracts.py'))
R = runpy.run_path(str(HERE / 'controller.py'))


def _artifact(stem, suffix):
    return Path(str(stem) + suffix)


def _cell_reference(p, h, out, cell, job):
    if cell == 'VSPF-A1':
        directory = C['qualified_dir']('A')
    else:
        directory = out / 'VSPF-A1'
    return p['read_result'](h, directory, job)


def _frozen_reference(p, h, job):
    return p['read_result'](h, h['off']['ROOT'] / 'run-r1/A0', job)


def _parsed_sse_bodies(raw):
    bodies, done = [], 0
    for line in raw.splitlines():
        if not line.startswith(b'data: '):
            continue
        data = line[6:].strip()
        if data == b'[DONE]':
            done += 1
            continue
        bodies.append(json.loads(data))
    assert done == 1
    return bodies


def _event_timing(stem, raw, transport):
    events = C['load'](_artifact(stem, '.events.json'))
    bodies = _parsed_sse_bodies(raw)
    assert [event['body'] for event in events] == bodies
    elapsed_total = transport['elapsed_s']
    assert isinstance(elapsed_total, (int, float))
    assert math.isfinite(elapsed_total) and elapsed_total > 0
    previous = 0.0
    first_content = None
    for event in events:
        elapsed = event['elapsed_s']
        assert isinstance(elapsed, (int, float)) and math.isfinite(elapsed)
        assert 0 <= elapsed <= elapsed_total
        assert elapsed >= previous
        previous = elapsed
        if first_content is None and any(
                choice.get('delta', {}).get('content')
                for choice in event['body'].get('choices', [])):
            first_content = elapsed
    assert transport['complete'] and transport['status'] == 200
    assert transport['bytes'] == len(raw)
    assert first_content is not None
    assert transport['ttfc_s'] == first_content
    assert 0 <= transport['ttfc_s'] < elapsed_total
    return events


def _augment_timing(stem, raw, protocol_result, transport):
    """Attach only independently reread HTTP/SSE clocks to protocol fields."""
    assert 'elapsed_s' not in protocol_result and 'ttfc_s' not in protocol_result
    _event_timing(stem, raw, transport)
    result = dict(protocol_result)
    result.update(elapsed_s=transport['elapsed_s'], ttfc_s=transport['ttfc_s'])
    return result


def _compare_stored_result(stem, result):
    stored = C['load'](_artifact(stem, '.result.json'))
    assert result == stored, 'normalized raw response plus reread clocks differs from saved result'
    return stored


def check_runtime_evidence(p, h, out, cell):
    """Recheck actual startup/PID/container/native evidence without rewriting it."""
    directory = out / cell
    role = C['role_for'](cell)
    assert C['load'](directory / 'create-argv.json') == C['make_create'](cell, h)
    base = p['make_create'](C['base_cell'](cell), h)
    assert C['load'](directory / 'argv-diff.json') == C['argv_diff'](base, C['make_create'](cell, h))
    for phase in ('before', 'after'):
        p['C']['check_pid'](C['load'](directory / ('pid-' + phase + '.json')), role, h)
        C['check_container'](C['load'](directory / ('container-' + phase + '.json')), cell, h)
    assert C['same_container'](C['load'](directory / 'container-before.json'),
                               C['load'](directory / 'container-after.json'))
    raw_startup = ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                          for stream in ('stdout', 'stderr'))
    p['check_startup'](raw_startup, C['base_cell'](cell), h)
    phases = {}
    for phase in ('native-and-startup', 'server', 'restored-final'):
        phases[phase] = [
            (directory / (phase + '.' + stream + '.log')).read_bytes()
            for stream in ('stdout', 'stderr')]
        h['off']['check_off_log'](b''.join(phases[phase]).decode(errors='replace'))
    for index in range(2):
        assert phases['server'][index].startswith(phases['native-and-startup'][index])
        assert phases['restored-final'][index].startswith(phases['server'][index])
    for tag, filename in p['NATIVE_LOGS'].items():
        ready = (directory / ('ready-' + filename)).read_bytes()
        restored = (directory / ('restored-' + filename)).read_bytes()
        assert ready == restored
        p['check_native_log'](tag, ready.decode())
    return dict(passed=True, raw_startup_bytes=len(raw_startup.encode()))


def reread_prevalidation(p, h, out, cell):
    """Rebuild the P 25-row report independently from the saved raw artifacts."""
    base = C['base_cell'](cell)
    report = p['new_report'](base)
    if cell == 'VSPF-A1':
        fresh_dir, label = C['qualified_dir']('A'), 'qualified VSB-A'
    else:
        fresh_dir, label = out / 'VSPF-A1', 'fresh timing VSPF-A1'
    report.update(timing_cell=cell, timing_reference=label,
                  frozen_reference='original A0', actual_cell=cell)
    jobs = C['qualification_jobs'](h)
    for job in jobs:
        p['read_public'](h, out / cell, job)
        actual = p['read_result'](h, out / cell, job)
        frozen = _frozen_reference(p, h, job)
        fresh = p['read_result'](h, fresh_dir, job)
        p['add_result'](report, job, actual, frozen, fresh)
    p['finish_report'](report)
    saved = C['load'](out / cell / 'prevalidation.json')
    assert report == saved, 'independent P prevalidation report differs from capture'
    assert report['passed'] and len(report['jobs']) == 25
    return report


def reread_measurement(p, h, out, job):
    cell, point, ordinal = job['cell'], job['point'], job['ordinal']
    stem = out / cell / point / f"measured-{ordinal:02}"
    payload = C['timing_jobs'](h)[point]['payload']
    raw = h['analysis']['read_http'](stem, payload, '/v1/chat/completions')
    protocol_result = h['analysis']['normalize'](raw, payload)
    transport = C['load'](_artifact(stem, '.http.json'))
    result = _augment_timing(stem, raw, protocol_result, transport)
    stored = _compare_stored_result(stem, result)
    assert stored['elapsed_s'] == transport['elapsed_s']
    assert stored['ttfc_s'] == transport['ttfc_s']
    metrics = C['timing_metrics'](result)
    telemetry = C['load'](_artifact(stem, '.telemetry.json'))
    assert C['telemetry'](telemetry['samples'], telemetry['window_duration_s']) == telemetry
    row = dict(job, artifact=str(stem), passed=True,
               actual=C['result_view'](result), timings=result['timings'],
               metrics=metrics, equals_no_draft=True,
               telemetry=telemetry)
    saved_check = C['load'](_artifact(stem, '.check.json'))
    assert saved_check['passed']
    saved_rows = C['load'](out / cell / 'timing-rows.json')
    matching = [saved for saved in saved_rows
                if saved['point'] == point and saved['ordinal'] == ordinal]
    assert len(matching) == 1, (cell, point, ordinal, matching)
    for key in ('cell', 'point', 'phase', 'ordinal', 'artifact', 'passed',
                'actual', 'timings', 'metrics', 'equals_no_draft'):
        assert matching[0][key] == row[key], key
    reference = _cell_reference(p, h, out, cell, C['timing_jobs'](h)[point])
    C['check_output'](result, C['timing_jobs'](h)[point],
                      reference['result'], h['off']['reference_prompt']('A0', C['timing_jobs'](h)[point]))
    row['equals_fresh_a'] = C['result_view'](result) == reference['actual']
    return row


def _delta(left, right):
    result = {}
    for metric in C['METRICS']:
        a = left[metric]['summary']
        b = right[metric]['summary']
        result[metric] = None if a is None or b is None else dict(
            baseline_median=a['median'], candidate_median=b['median'],
            absolute=b['median'] - a['median'],
            percent=None if a['median'] == 0 else 100 * (b['median'] / a['median'] - 1),
        )
    return result


def summarize(rows):
    """Return descriptive per-point/per-arm summaries for exactly 24 rows."""
    assert len(rows) == 24 and all(row['passed'] for row in rows)
    report = dict(per_cell={}, per_arm={}, deltas={}, output_matching={},
                  quartiles='Hyndman-Fan type 7: linear interpolation at (n-1)*p',
                  measured_rows=24, samples_per_arm_point=6,
                  ci=None, significance_claim=False, outlier_removal=False,
                  first_content_decode_proxy_is_not_decode_speed=True,
                  headline_server_metric='predicted_per_second')
    for point in C['POINTS']:
        selected = [row for row in rows if row['point'] == point]
        assert len(selected) == 12
        per_cell = {}
        for cell in C['CELLS']:
            cell_rows = [row for row in selected if row['cell'] == cell]
            assert len(cell_rows) == 3
            per_cell[cell] = C['aggregate'](cell_rows)
        report['per_cell'][point] = per_cell
        arms = {
            'A': [row for row in selected if C['ROLES'][row['cell']] == 'A'],
            'W3': [row for row in selected if C['ROLES'][row['cell']] == 'W3'],
        }
        assert all(len(value) == 6 for value in arms.values())
        report['per_arm'][point] = {arm: C['aggregate'](value)
                                    for arm, value in arms.items()}
        report['deltas'][point] = {'A-to-W3': _delta(report['per_arm'][point]['A'],
                                                     report['per_arm'][point]['W3'])}
        report['output_matching'][point] = {
            arm: dict(rows=len(value), matching=sum(row['equals_fresh_a'] for row in value),
                      text_sha256=[hashlib.sha256(row['actual']['text'].encode()).hexdigest()
                                   for row in value], raw_f32_equality_claimed=False)
            for arm, value in arms.items()
        }
    return report

def _validate_timing_rows(rows, cell):
    expected = [job for job in C['expected_measurements']() if job['cell'] == cell]
    keys = [(row['point'], row['ordinal']) for row in rows]
    expected_keys = [(job['point'], job['ordinal']) for job in expected]
    assert len(rows) == 6, (cell, len(rows))
    assert len(set(keys)) == 6, (cell, keys)
    assert set(keys) == set(expected_keys), (cell, keys, expected_keys)
    assert all(row['cell'] == cell and row['phase'] == 'measured' and row['passed']
               for row in rows)


def _timing_rows_schema(out, cell):
    rows = C['load'](Path(out) / cell / 'timing-rows.json')
    _validate_timing_rows(rows, cell)
    for point in C['POINTS']:
        for phase, ordinals in (('warmup', (0,)), ('measured', range(3))):
            for ordinal in ordinals:
                stem = Path(out) / cell / point / f'{phase}-{ordinal:02}'
                for suffix in ('.request.json', '.request-body', '.http-request.json',
                               '.http-body', '.http.json', '.events.json', '.result.json',
                               '.check.json'):
                    assert _artifact(stem, suffix).is_file(), str(_artifact(stem, suffix))
                if phase == 'measured':
                    assert _artifact(stem, '.telemetry.json').is_file(), str(stem)
    return rows


def _timing_schema(out):
    rows = [_timing_rows_schema(out, cell) for cell in C['CELLS']]
    assert sum(len(value) for value in rows) == len(C['expected_measurements']()) == 24


_summarize = summarize


def analyze(out, p, h):
    out = Path(out)
    summary = C['load'](out / 'summary.json')
    report = dict(status='failed', independently_restored=True, prevalidation={},
                  rows=[], errors=[], statistics={}, timing_permitted=True,
                  promotion_permitted=False, significance_claim=False,
                  raw_f32_equality_claimed=False, output_matching_only=True)
    try:
        assert summary['status'] == 'captured' and summary['restored']
        assert summary['order'] == summary['completed'] == list(C['CELLS'])
        expected = C['expected_measurements']()
        assert len(expected) == 24
        attempts = summary['measured_attempts']
        assert len(attempts) == 24
        assert [dict((key, row[key]) for key in ('cell', 'point', 'phase', 'ordinal'))
                for row in attempts] == expected
        assert all(row['phase'] == 'measured' and row['passed'] for row in attempts)
        assert C['load'](out / 'restoration.json')['passed']
        smoke = C['load'](out / 'restored-smoke.result.json')
        assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
        _timing_schema(out)
    except BaseException as exc:
        report['errors'].append(dict(scope='completion/restoration', error=repr(exc)))

    for cell in C['CELLS']:
        try:
            report['prevalidation'][cell] = reread_prevalidation(p, h, out, cell)
            check_runtime_evidence(p, h, out, cell)
        except BaseException as exc:
            report['errors'].append(dict(cell=cell, scope='prevalidation/evidence', error=repr(exc)))

    for job in C['expected_measurements']():
        row = dict(job, artifact=str(out / job['cell'] / job['point'] /
                                    f"measured-{job['ordinal']:02}"), passed=False)
        report['rows'].append(row)  # Retain every expected row, including failures.
        try:
            row.update(reread_measurement(p, h, out, job))
        except BaseException as exc:
            row['error'] = repr(exc)
            report['errors'].append(dict(job=job, scope='raw-timing', error=repr(exc)))

    valid = (not report['errors'] and len(report['rows']) == 24 and
             all(row['passed'] for row in report['rows']) and
             all(report['prevalidation'].get(cell, {}).get('passed') for cell in C['CELLS']))
    if valid:
        report['statistics'] = summarize(report['rows'])
        report['status'] = 'analyzed'
    else:
        report['statistics'] = dict(available=False,
                                    unavailable_reason='incomplete or invalid raw/fidelity evidence',
                                    significance_claim=False)
    C['save'](out / 'analysis-r1' / 'analysis.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-timing-restore', action='store_true', required=True)
    parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert HERE == C['ROOT'], (HERE, C['ROOT'])
    p, k, h = R['load_dependencies']()
    C['configure'](p)
    out = C['ROOT'] / 'run-r1'
    # This independent live check is deliberately before analysis-r1 exists and
    # before any raw timing file is read.
    R['Runtime'](h, out).verify_restored(out / 'analysis-live-restoration.json')
    analysis_dir = out / 'analysis-r1'
    analysis_dir.mkdir(exist_ok=False)
    report = analyze(out, p, h)
    assert report['status'] == 'analyzed', report['errors']
    print('W3_ABBA_TIMING_ANALYZED_RESTORED_DEVELOPMENT_DESCRIPTIVE_ONLY', flush=True)


if __name__ == '__main__':
    main()
