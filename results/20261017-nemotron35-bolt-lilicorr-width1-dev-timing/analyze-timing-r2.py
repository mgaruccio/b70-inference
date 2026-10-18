#!/usr/bin/env python3
"""LEAD ONLY, separate post-restoration process: descriptive six-sample summaries."""
import argparse
import hashlib
import json
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
ROOT, ORDER, POINTS = C['ROOT'], C['ORDER'], C['POINTS']
load, save = C['load'], C['save']


def artifact(stem, suffix):
    return Path(str(stem) + suffix)


def public_ids(h, cell, job):
    stem = cell / 'public' / job['stem']
    read = h['analysis']['read_http']
    template = h['off']['L']['apply_request'](job['payload'])
    rendered = json.loads(read(artifact(stem, '.template'), template, '/apply-template'))
    tokens = h['off']['L']['tokenize_request'](rendered)
    response = json.loads(read(artifact(stem, '.tokenize'), tokens, '/tokenize'))
    ids = h['off']['L']['token_ids'](response)
    assert ids == h['off']['reference_prompt'](C['label'](cell.name) or 'A0', job)
    return ids


def check_cell(h, cell, original):
    pre, points = C['jobs'](h['off'])
    read, normalize = (h['analysis'][k] for k in ('read_http', 'normalize'))
    off = h['off']
    log = ''.join((cell / ('native-and-startup.' + stream + '.log')).read_text() for stream in ('stdout', 'stderr'))
    arm = C['label'](cell.name)
    if arm:
        assert load(cell / 'create-argv.json') == C['make_create'](cell.name, off)
        off['check_startup'](log, arm)
        for phase in ('before', 'after'):
            off['check_pid'](load(cell / ('pid-' + phase + '.json')), arm)
    else:
        C['check_original_log'](log, C['original_contract'](original, h))
        for phase in ('before', 'after'):
            current = load(cell / ('container-' + phase + '.json'))
            for key in ('Id', 'Config', 'HostConfig'):
                assert current[key] == original[key], ('original mutated', key)
            # Docker may reorder Mounts on inspect/restart. Preserve every field
            # and duplicate count; only list ordering is irrelevant.
            canonical_mounts = lambda mounts: sorted(json.dumps(m, sort_keys=True) for m in mounts)
            assert canonical_mounts(current['Mounts']) == canonical_mounts(original['Mounts']), ('original mutated', 'Mounts')
        assert load(cell / 'stopped-log-prefix.json')
    quality = []
    for job in pre:
        ids = public_ids(h, cell, job)
        result = normalize(read(cell / job['stem'], job['payload'], '/v1/chat/completions'), job['payload'])
        assert result['usage']['prompt_tokens'] == len(ids) and result['timings']['cache_n'] == 0
        assert (off['L']['functional_anchor'](result) if job['stem'] == 'anchor' else
                off['L']['functional_quality'](job['stem'].split('/')[1], result))
        quality.append(dict(stem=job['stem'], actual=C['view'](result)))
    for point in POINTS:
        ordinary = points[point]['ordinary']
        public_ids(h, cell, ordinary)
        if arm:
            diagnostic = points[point]['diagnostic']
            ids = public_ids(h, cell, diagnostic)
            actual = off['diagnostic_view'](json.loads(read(cell / diagnostic['stem'], diagnostic['payload'], '/v1/chat/completions')), ids)
            expected = off['diagnostic_view'](json.loads(C['reference'](h, arm, diagnostic)[0]), ids)
            assert all(actual[k] == expected[k] for k in ('ids', 'pieces', 'text', 'finish_reason', 'usage'))
    return quality


def summarize(rows, valid):
    report = dict(per_cell={}, per_arm={}, deltas={}, identity={}, descriptive_only=True,
                  quartiles='Hyndman-Fan type 7: linear interpolation at (n-1)*p',
                  independent_loads_per_arm=2, correlated_measurements_per_arm_point=6,
                  ci=None, significance_claim=False, outlier_removal=False)
    if not valid:
        report['unavailable_reason'] = 'incomplete or invalid run; no selection of good measurements'
        return report
    assert len(rows) == 36 and all(r['passed'] for r in rows)
    for point in POINTS:
        selected = [r for r in rows if r['point'] == point]
        report['per_cell'][point] = {cell: C['aggregate']([r for r in selected if r['cell'] == cell]) for cell in ORDER}
        arms = {arm: [r for r in selected if r['cell'][0] == arm] for arm in ('O', 'A', 'C')}
        assert all(len(rs) == 6 for rs in arms.values())
        report['per_arm'][point] = {arm: C['aggregate'](rs) for arm, rs in arms.items()}
        report['identity'][point] = {arm: dict(text_variants=len({r['actual']['text'] for r in rs}),
            text_sha256=[hashlib.sha256(r['actual']['text'].encode()).hexdigest() for r in rs],
            equal_candidate_reference=sum(r['equals_candidate_reference'] for r in rs),
            actual_emission_ids_claimed=False) for arm, rs in arms.items()}
        report['deltas'][point] = {}
        for left, right in (('O', 'A'), ('A', 'C'), ('O', 'C')):
            changes = {}
            for metric in C['METRICS']:
                a, b = (report['per_arm'][point][arm][metric]['summary'] for arm in (left, right))
                changes[metric] = None if a is None or b is None else dict(
                    baseline_median=a['median'], candidate_median=b['median'],
                    absolute=b['median'] - a['median'],
                    percent=None if a['median'] == 0 else 100 * (b['median'] / a['median'] - 1))
            report['deltas'][point][left + '-to-' + right] = changes
    return report


def analyze(out, h):
    run = load(out / 'summary.json')
    report = dict(status='failed', rows=[], errors=[], quality={}, raw_vector_equality_claimed=False,
                  comparability='O-to-A runtime/source/compiled-hook bundle; A-to-C drafting cost; O-to-C overall bundle. O identity not assumed.')
    try:
        assert run['status'] == 'captured' and run['restored']
        assert run['order'] == run['completed'] == list(ORDER)
        assert run['measured_attempts'] == [j for j in C['schedule']() if j['phase'] == 'measured']
        assert load(out / 'restoration.json')['passed']
        smoke = load(out / 'restored-smoke.result.json')
        assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    except (AssertionError, KeyError, OSError) as exc:
        report['errors'].append(dict(scope='completion/restoration', error=repr(exc)))
    original = load(out / 'original-container.json')
    for cell in ORDER:
        try:
            report['quality'][cell] = check_cell(h, out / cell, original)
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            report['errors'].append(dict(cell=cell, scope='prevalidation', error=repr(exc)))
    points = C['jobs'](h['off'])[1]
    for job in C['schedule']():
        stem = out / job['cell'] / job['point'] / f"{job['phase']}-{job['ordinal']:02}"
        row = dict(job, artifact=str(stem), passed=False)
        if job['phase'] == 'measured':
            report['rows'].append(row)  # Includes every missing/failed measurement, never filtered.
        try:
            payload = points[job['point']]['ordinary']['payload']
            raw = h['analysis']['read_http'](stem, payload, '/v1/chat/completions')
            result = h['analysis']['normalize'](raw, payload)
            stored = load(artifact(stem, '.result.json'))
            assert all(result[k] == stored[k] for k in result)
            transport = load(artifact(stem, '.http.json'))
            assert stored['elapsed_s'] == transport['elapsed_s'] and stored['ttfc_s'] == transport['ttfc_s']
            arm = C['label'](job['cell'])
            ref = C['reference'](h, arm or 'A0', points[job['point']]['ordinary'])[1]
            C['check_result'](stored, job['point'], ref if arm else None)
            row.update(actual=C['view'](stored), timings=stored['timings'], metrics=C['timing_metrics'](stored),
                       equals_candidate_reference=C['view'](stored) == C['view'](ref))
            if job['phase'] == 'measured':
                sampled = load(artifact(stem, '.telemetry.json'))
                row['telemetry'] = C['telemetry'](sampled['samples'], sampled['window_duration_s'])
                assert row['telemetry'] == sampled
            row['passed'] = True
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            row['error'] = repr(exc)
            report['errors'].append(dict(job, error=repr(exc)))
    valid = not report['errors'] and all(r['passed'] for r in report['rows'])
    report['statistics'] = summarize(report['rows'], valid)
    report['status'] = 'passed' if valid else 'failed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-restore', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    C['check_frozen'](h)
    out = ROOT / 'run-r1'
    analysis = out / 'analysis-r2'
    analysis.mkdir(exist_ok=False)
    runner = runpy.run_path(str(ROOT / 'run-timing.py'))
    # Independent fresh live health, exact original, ownership, hardware, sources/five hashes
    # before statistical analysis. This is NOT satisfied by a saved controller assertion.
    runner['Runtime'](h, out).verify_restored(analysis / 'live-restoration.json')
    report = analyze(out, h)
    save(analysis / 'analysis.json', report)
    assert report['status'] == 'passed', report['errors']
    print('DEVELOPMENT_DESCRIPTIVE_THREE_ARM_PASSED_NO_PROMOTION_NO_PUBLICATION', flush=True)


if __name__ == '__main__':
    main()
