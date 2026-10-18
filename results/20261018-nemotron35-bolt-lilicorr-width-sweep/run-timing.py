#!/usr/bin/env python3
"""LEAD ONLY: balanced timing cells after independently confirmed qualification."""
import argparse
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
K = runpy.run_path(str(Path(__file__).with_name('controller.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-time-eligible', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == C['ROOT']
    h = C['helpers']()
    C['check_tested'](h)
    qualification = C['load'](C['QUALIFY_DIR'] / 'qualification.json')
    assert qualification['status'] == 'passed'
    assert qualification['restored'] and qualification['independently_restored']
    # This is a fresh live check, not trust in the saved screening assertion.
    qualification_runtime = K['Runtime'](h, C['QUALIFY_DIR'], 'qualification')
    qualification_runtime.verify_restored(C['QUALIFY_DIR'] / 'timing-preflight-restoration.json')
    widths = tuple(sorted(set(int(width) for width in qualification['qualified_widths'])))
    assert set(widths) <= {3, 7}
    selected = C['eligible_order'](widths)
    out = C['TIMING_DIR']
    out.mkdir(exist_ok=False)
    (out / 'declaration.md').write_bytes((C['ROOT'] / 'README.md').read_bytes())
    C['save'](out / 'selection.json', dict(qualified_widths=list(widths),
                                           selected_order=list(selected),
                                           predeclared_order=list(C['TIMING_ORDER']),
                                           warmup_per_point=1, measured_per_point=3,
                                           points={'short': 55, 'long': 8270},
                                           selection_only_by_qualification=True,
                                           no_raw_capture=True))
    if not widths:
        C['save'](out / 'summary.json', dict(status='blocked', restored=True,
                                             reason='no width passed full qualification; timing not launched',
                                             qualified_widths=[], selected_order=[]))
        print('TIME_BLOCKED_NO_QUALIFIED_WIDTH', flush=True)
        return

    runtime = K['Runtime'](h, out, 'timing', widths)
    summary = K['capture'](runtime, selected, lambda rt, cell: rt.run_timing_cell(cell))
    assert summary['completed'] == list(selected)
    C['save'](out / 'timing-summary.json', dict(status='captured', restored=summary['restored'],
                                                selected_order=list(selected),
                                                qualified_widths=list(widths),
                                                measured_rows=len(summary['measured_attempts']),
                                                measured_requests_per_arm_point=6,
                                                warmups_excluded=True,
                                                timing_interpretation='development descriptive only'))
    print('TIME_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)


if __name__ == '__main__':
    main()
