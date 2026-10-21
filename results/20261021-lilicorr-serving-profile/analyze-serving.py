#!/usr/bin/env python3
"""Analyze qualified real-serving traces, calibrated request windows and tracer overhead."""
import argparse
import json
import runpy
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = runpy.run_path(str(HERE / 'analyze-trace.py'))
CELLS = ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2')
TRACED = ('VSPF-W3b', 'VSPF-A2')


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    assert not path.exists(), ('preserve previous analysis', str(path))
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def clock_window(sidecar, result, cell, point, ordinal):
    assert sidecar['cell'] == cell
    assert len(sidecar['requests']) == 1
    span = sidecar['requests'][0]
    assert span['kind'] == 'timed-measured'
    assert span['resumed'] and span['request_succeeded'] and span['pause_succeeded']
    samples = [span[k] for k in ('before_resume', 'after_resume', 'before_request',
                                'after_request', 'before_pause', 'after_pause')]
    epoch = [s['realtime_ns'] - s['boottime_ns'] for s in samples]
    read_span = max(s['clock_read_span_ns'] for s in samples)
    assert 0 <= read_span <= 1_000_000, ('excessive clock-read uncertainty', read_span)
    assert max(epoch) - min(epoch) < 1_000_000, ('unstable REALTIME-BOOTTIME', epoch)
    first, last = span['before_request'], span['after_request']
    begin, end = T['trace_us'](first), T['trace_us'](last)
    assert end > begin
    wall_ms = (last['monotonic_ns'] - first['monotonic_ns']) / 1e6
    http_ms = result['elapsed_s'] * 1000
    assert abs(wall_ms - http_ms) < max(100, http_ms * .01), ('window/HTTP mismatch', wall_ms, http_ms)
    assert abs((end - begin) / 1000 - wall_ms) < max(5, wall_ms * .001), 'RAW/monotonic duration mismatch'
    return dict(cell=cell, point=point, stem=f'measured-{ordinal:02}', start_us=begin, end_us=end,
                fidelity_passed=True, traced=True,
                clock=dict(domain='pinned_Unitimer_epoch_plus_monotonic_raw',
                           boottime_minus_raw_begin_ns=first['boottime_ns'] - first['monotonic_raw_ns'],
                           boottime_minus_raw_end_ns=last['boottime_ns'] - last['monotonic_raw_ns'],
                           epoch_minus_boot_spread_ns=max(epoch) - min(epoch),
                           max_clock_read_span_ns=read_span,
                           nominal_uncertainty_us=read_span / 1000,
                           monotonic_window_ms=wall_ms, http_elapsed_ms=http_ms))


def callback_deltas(capture, cell):
    """Use existing cumulative host timer; never rename it GPU or all draft cost."""
    records = load(capture / cell / 'server-duration-stats.json')['records']
    # 25 public completions, then short warmup/3 measured, long warmup/3 measured.
    # This is the existing controller's exact sequential journey, not guessed timing.
    assert len(records) == 33, (cell, 'unexpected callback-stat count', len(records))
    report = []
    previous = dict(calls=dict(b=0, g=0, a=0), generated=dict(drafts=0, tokens=0),
                    accepted=dict(drafts=0, tokens=0), duration_ms=dict(b=0, g=0, a=0))
    for index, current in enumerate(records):
        assert current['cumulative']
        assert current['calls']['b'] == previous['calls']['b'] + 1, 'request order no longer uniquely identified'
        if index in (26, 27, 28, 30, 31, 32):
            point = 'short' if index < 29 else 'long'
            ordinal = index - (26 if point == 'short' else 30)
            result = load(capture / cell / point / f'measured-{ordinal:02}.result.json')
            proposed = current['generated']['tokens'] - previous['generated']['tokens']
            accepted = current['accepted']['tokens'] - previous['accepted']['tokens']
            expected = (129, 84) if point == 'short' else (133, 82)
            assert (proposed, accepted) == expected
            assert proposed == result['timings']['draft_n'] and accepted == result['timings']['draft_n_accepted']
            timing = {key: current['duration_ms'][key] - previous['duration_ms'][key] for key in ('b', 'g', 'a')}
            assert all(value >= 0 for value in timing.values())
            report.append(dict(cell=cell, point=point, ordinal=ordinal, cumulative_record_index=index,
                               calls={key: current['calls'][key] - previous['calls'][key] for key in ('b', 'g', 'a')},
                               proposed=proposed, accepted=accepted,
                               callback_host_ms=timing,
                               draft_callback_fraction_of_server_decode=timing['g'] / result['timings']['predicted_ms'],
                               server_predicted_ms=result['timings']['predicted_ms'],
                               source_log_record=current['text']))
        previous = current
    return dict(rows=report,
                scope='Existing common_time_meas around impl->draft, plus begin/accept timers; host time, includes waits.',
                exclusions='Not pure GPU duration or all drafter overhead; process callbacks and target verification are outside this timer.',
                source='20261020-lilicorr-context-profile/fixtures/common/speculative.cpp.gz lines2943-2948 and3080-3083; unchanged timer in qualified source.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=HERE / 'capture-r1')
    parser.add_argument('--output', type=Path, default=HERE / 'analysis-r1')
    args = parser.parse_args()
    capture = args.capture.resolve()
    summary = load(capture / 'summary.json')
    assert summary['status'] == 'captured' and summary['restored'] and summary['mode'] == 'profiling'
    assert not summary['timing_permitted'] and not summary['promotion_permitted']
    assert tuple(summary['completed']) == CELLS
    restored = load(capture / 'independent-restoration-live.json')
    assert restored['passed'] and not restored['errors']
    fidelity = load(capture / 'qualification-analysis.json')
    assert not fidelity['timing_permitted'] and len(fidelity['rows']) == 24
    assert all(row['passed'] and row['equals_fresh_a'] for row in fidelity['rows'])
    for cell in CELLS:
        assert fidelity['native'][cell]['passed']
        assert fidelity['public'][cell]['passed'] and len(fidelity['public'][cell]['jobs']) == 25
    args.output.mkdir(exist_ok=False)
    rules = load(HERE / 'kernel-rules.json')
    traced_reports = {}
    for cell in TRACED:
        assert load(capture / cell / 'native-trace-validation.json')['passed']
        assert load(capture / cell / 'server-trace-validation.json')['passed']
        windows = []
        for point in ('short', 'long'):
            for ordinal in range(3):
                stem = capture / cell / point / f'measured-{ordinal:02}'
                result = load(str(stem) + '.result.json')
                assert load(str(stem) + '.check.json')['passed']
                windows.append(clock_window(load(str(stem) + '.trace-sidecar.json'), result, cell, point, ordinal))
        epoch_samples = []
        for path in (capture / cell).rglob('*.trace-sidecar.json'):
            for span in load(path)['requests']:
                for key in ('before_resume', 'after_resume', 'before_request', 'after_request', 'before_pause', 'after_pause'):
                    s = span[key]
                    epoch_samples.append(s['realtime_ns'] - s['boottime_ns'])
        assert max(epoch_samples) - min(epoch_samples) < 1_000_000, 'clock epoch instability across capture'
        traces = list((capture / cell / 'profile/server').rglob('chrome_trace.json'))
        assert len(traces) == 1, ('ambiguous server trace', cell, traces)
        save(args.output / (cell + '-windows.json'), windows)
        report = T['analyze'](traces[0], windows, rules)
        report['capture_epoch_minus_boot_spread_ns'] = max(epoch_samples) - min(epoch_samples)
        save(args.output / (cell + '-trace-analysis.json'), report)
        traced_reports[cell] = report
        print('ANALYZED_REAL_REQUESTS', cell, len(windows), report['total_trace_events'], flush=True)
    overhead = []
    for clean, traced in (('VSPF-A1', 'VSPF-A2'), ('VSPF-W3a', 'VSPF-W3b')):
        for point in ('short', 'long'):
            samples = {}
            for cell in (clean, traced):
                samples[cell] = []
                for ordinal in range(3):
                    r = load(capture / cell / point / f'measured-{ordinal:02}.result.json')
                    samples[cell].append(dict(ordinal=ordinal, elapsed_ms=r['elapsed_s'] * 1000,
                                              prompt_ms=r['timings']['prompt_ms'],
                                              predicted_ms=r['timings']['predicted_ms'],
                                              predicted_per_second=r['timings']['predicted_per_second']))
            medians = {cell: {key: statistics.median(row[key] for row in rows)
                              for key in ('elapsed_ms', 'prompt_ms', 'predicted_ms', 'predicted_per_second')}
                       for cell, rows in samples.items()}
            overhead.append(dict(clean_cell=clean, traced_cell=traced, point=point, samples=samples,
                                 medians=medians,
                                 elapsed_overhead_percent=100 * (medians[traced]['elapsed_ms'] / medians[clean]['elapsed_ms'] - 1),
                                 predicted_latency_overhead_percent=100 * (medians[traced]['predicted_ms'] / medians[clean]['predicted_ms'] - 1)))
    callbacks = {cell: callback_deltas(capture, cell) for cell in ('VSPF-W3a', 'VSPF-W3b')}
    aggregate = []
    for cell, report in traced_reports.items():
        for point in ('short', 'long'):
            rows = [r for r in report['requests'] if r['point'] == point]
            assert len(rows) == 3
            partitions = sorted({key for r in rows for key in r['wall_partition_ms']})
            categories = sorted({key for r in rows for key in r['gpu_category_union_ms']})
            aggregate.append(dict(cell=cell, point=point, samples=3,
                                  mean_wall_ms=statistics.mean(r['wall_ms'] for r in rows),
                                  mean_gpu_busy_ms=statistics.mean(r['gpu_busy_union_ms'] for r in rows),
                                  mean_wall_partition_ms={key: statistics.mean(r['wall_partition_ms'].get(key, 0) for r in rows) for key in partitions},
                                  nonadditive_mean_gpu_category_union_ms={key: statistics.mean(r['gpu_category_union_ms'].get(key, 0) for r in rows) for key in categories},
                                  gpu_events=[r['gpu_events'] for r in rows], cpu_events=[r['cpu_events'] for r in rows]))
    result = dict(status='analyzed', diagnostic_only=True, clean_speed_claim=False,
                  public_qualification_jobs=100, measured_requests=24, attributed_measured_requests=12,
                  independent_restoration_passed=True, tracing_overhead=overhead,
                  traced_wall_breakdown=aggregate, callback_host_timers=callbacks,
                  limitations=[
                      'Clean/traced cells are one startup each, not replicated interleaved profiler blocks; overhead estimates include temporal drift.',
                      'No pooling traced and clean arms into a performance comparison.',
                      'Kernel family origin is not a complete drafter/target phase or critical-path attribution.',
                      'Shared/unattributed GPU activity and untraced wall gaps remain explicit.',
                      'No exact GPU prefill/decode boundary is inferred; server prompt/decode durations are separately reported.',
                      'Host callback timer is not all drafter overhead and is not additive with nested GPU durations.'
                  ])
    save(args.output / 'summary.json', result)
    print('SERVING_PROFILE_ANALYZED_WITH_CALIBRATED_WINDOWS_AND_OVERHEAD', args.output, flush=True)


if __name__ == '__main__':
    main()
