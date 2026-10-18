#!/usr/bin/env python3
"""LEAD ONLY: independent post-restoration descriptive timing analysis."""
import argparse
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
K = runpy.run_path(str(Path(__file__).with_name('controller.py')))


def _arm(cell):
    if cell.startswith('A'):
        return 'A'
    return 'W' + str(C['width_for'](cell))


def _summarize(rows):
    return C['aggregate'](rows)


def _delta(left, right):
    result = {}
    for metric in C['METRICS']:
        a = left[metric]['summary']
        b = right[metric]['summary']
        result[metric] = None if a is None or b is None else dict(
            baseline_median=a['median'], candidate_median=b['median'],
            absolute=b['median'] - a['median'],
            percent=None if a['median'] == 0 else 100 * (b['median'] / a['median'] - 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-timing-restore', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == C['ROOT']
    h = C['helpers']()
    C['check_tested'](h)
    out = C['TIMING_DIR']
    selection = C['load'](out / 'selection.json')
    summary = C['load'](out / 'summary.json')
    widths = tuple(selection['qualified_widths'])
    if summary['status'] == 'blocked':
        report = dict(status='blocked', timing_not_launched=True,
                      reason=summary['reason'], qualified_widths=list(widths),
                      ci=None, significance_claim=False, promotion_permitted=False)
        C['save'](out / 'analysis.json', report)
        print('TIMING_BLOCKED_NO_ANALYSIS', flush=True)
        return
    runtime = K['Runtime'](h, out, 'timing', widths)
    runtime.verify_restored(out / 'analysis-live-restoration.json')
    assert summary['status'] == 'captured' and summary['restored']
    selected = tuple(selection['selected_order'])
    assert selected == C['eligible_order'](widths)
    assert summary['completed'] == list(selected)
    rows = summary['measured_attempts']
    assert len(rows) == len(C['expected_measurements'](widths))
    assert all(row['phase'] == 'measured' and row['passed'] for row in rows)
    assert all(row['cell'] in selected for row in rows)
    assert all(row['metrics']['elapsed_s'] > 0 for row in rows)
    assert not any(row['phase'] == 'warmup' for row in rows), 'warmups entered measured summary'

    per_cell, per_arm, deltas = {}, {}, {}
    for point in C['POINTS']:
        point_rows = [row for row in rows if row['point'] == point]
        per_cell[point] = {}
        for cell in selected:
            cell_rows = [row for row in point_rows if row['cell'] == cell]
            assert len(cell_rows) == 3
            per_cell[point][cell] = _summarize(cell_rows)
        arms = sorted({_arm(cell) for cell in selected})
        per_arm[point] = {}
        for arm in arms:
            arm_rows = [row for row in point_rows if _arm(row['cell']) == arm]
            assert len(arm_rows) == 6
            per_arm[point][arm] = _summarize(arm_rows)
        deltas[point] = {}
        baseline = per_arm[point]['A']
        for arm in arms:
            if arm != 'A':
                deltas[point]['A-to-' + arm] = _delta(baseline, per_arm[point][arm])

    duration = {}
    for cell in selected:
        path = out / cell / 'server-duration-stats.json'
        duration[cell] = C['load'](path) if path.exists() else {
            'available': False, 'records': [],
            'source': 'server logs only; no diagnostic/profile hook added',
        }
    report = dict(status='passed', restored=True, independently_restored=True,
                  qualified_widths=list(widths), selected_order=list(selected),
                  per_cell=per_cell, per_arm=per_arm, deltas=deltas,
                  speculative_duration_stats_from_server_logs=duration,
                  warmup_excluded=True, samples_per_cell_point=3,
                  samples_per_arm_point=6, quartiles='Hyndman-Fan type 7',
                  ci=None, significance_claim=False, promotion_permitted=False,
                  raw_capture=False, raw_f32_equality_claimed=False,
                  no_profile_hooks_added=True)
    C['save'](out / 'analysis.json', report)
    print('TIMING_ANALYZED_RESTORED_DEVELOPMENT_DESCRIPTIVE_ONLY', flush=True)


if __name__ == '__main__':
    main()
