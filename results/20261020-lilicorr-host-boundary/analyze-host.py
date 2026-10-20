#!/usr/bin/env python3
"""Host-boundary diagnostic using existing strict schemas; GPU durations unavailable."""
import json
import runpy
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE.parent / '20261020-lilicorr-device-timing/analyze-device.py'))


def check_request(stderr):
    rows, components = D['_parse_records'](stderr)
    coverage = [r for r in rows if r['stage'] == 'coverage']
    events = [r for r in rows if r['stage'] == 'event']
    libraries = [r for r in rows if r['stage'] == 'library_submit']
    drains = [r for r in rows if r['stage'] == 'drain']
    assert coverage and events and drains, 'missing host/launch metadata'
    assert not any(r['stage'] == 'discard' for r in rows), 'metadata discarded'
    assert all(r['status'] == 'partial' and not r['dropped'] and not r['submit_errors'] for r in coverage)
    owners = {(r['graph'], r['backend_ctx'], r['device']): r for r in coverage}
    assert len(owners) == len(coverage), 'duplicate coverage owner'
    seen = set()
    for r in events + libraries:
        key = (r['graph'], r['ordinal'])
        assert key not in seen, 'duplicate submission'
        seen.add(key)
        owner = owners[(r['graph'], r['backend_ctx'], r['device'])]
        assert 0 <= r['node'] < owner['graph_nodes'] and r['ordinal'] <= owner['submitted']
        assert all(r[k] == 'unavailable' for k in ('device_start_ns', 'device_end_ns', 'device_elapsed_ns')), 'GPU timing is not host mode'
        assert r['status'] == ('unsupported_profiling' if r['stage'] == 'event' else 'device_unavailable_library')
    for key, owner in owners.items():
        assert sum((r['graph'], r['backend_ctx'], r['device']) == key for r in events + libraries) == owner['retained']
    assert sum(r['records'] for r in drains) == len(events) + len(libraries)
    assert any(r['ncols'] == 8 for r in events), 'no scoped n8 metadata'
    chunks = [r for r in components if r['stage'] == 'chunk']
    logging = [r for r in components if r['stage'] == 'logging']
    assert chunks and all(r['status'] == 'ok' and r['process_rc'] == 0 for r in chunks)
    chunk_keys = [D['_component_key'](r) for r in chunks]
    log_keys = [D['_component_key'](r) for r in logging]
    assert len(set(chunk_keys)) == len(chunks) and len(set(log_keys)) == len(logging)
    assert set(chunk_keys) == set(log_keys), 'component/log ownership mismatch'
    routes = Counter((r['route'], r['weight'], r['ncols']) for r in events + libraries)
    return dict(status='ok', gpu_time_claim=False, gpu_timestamps='unavailable',
                host=D['_component_report'](chunks, logging),
                device=dict(coverage=coverage),
                launch_metadata=[dict(route=k[0], weight=k[1], width=k[2], count=v)
                                 for k, v in sorted(routes.items())])


def analyze(out, c):
    out = Path(out)
    output = out / 'analysis-r1/host-analysis.json'
    assert not output.exists(), 'preserve previous analysis, including failed results'
    report = dict(status='failed', diagnostic_only=True, gpu_time_claim=False,
                  gpu_timestamps='unavailable', speed_comparison_permitted=False,
                  requests=[], errors=[])
    try:
        # The unchanged analyzer validates 24 measured outputs but returns only
        # the 16 W3 phase records; do not apply the device wrapper's row-count check.
        shadow = dict(c, save=lambda *_a: None)
        context = D['CONTEXT_ANALYZER']['analyze'](out, shadow)
        assert context.get('status') == 'analyzed' and not context.get('errors'), 'unchanged context validation failed'
        measured = c['load'](out / 'analysis-r1/analysis.json')['rows']
        assert len(measured) == 24 and all(r['passed'] for r in measured)
        selected, excluded = D['_selected_requests'](out, c)
        report['context_measured_outputs'] = len(measured)
        report['excluded_records'] = excluded
        for meta in selected:
            parsed = check_request(meta['text'])
            old = D['_context_graph_record'](context, meta)
            D['_validate_graph_links'](parsed, old)
            report['requests'].append(dict(
                **{k: meta[k] for k in ('cell', 'point', 'phase', 'ordinal', 'request_index', 'first_line', 'last_line') if k in meta},
                host=parsed['host'], launch_metadata=parsed['launch_metadata'],
                graph_coverage=parsed['device']['coverage'], gpu_timestamps='unavailable'))
        report['status'] = 'analyzed'
    except (AssertionError, KeyError, OSError, TypeError, ValueError, UnicodeError) as error:
        report['errors'].append(repr(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report
