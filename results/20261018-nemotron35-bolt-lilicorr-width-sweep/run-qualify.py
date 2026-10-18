#!/usr/bin/env python3
"""LEAD ONLY: fresh A, W3, W7 qualification; restore before timing."""
import argparse
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
K = runpy.run_path(str(Path(__file__).with_name('controller.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-qualification', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == C['ROOT']
    h = C['helpers']()
    C['check_tested'](h)
    out = C['QUALIFY_DIR']
    out.mkdir(exist_ok=False)
    (out / 'declaration.md').write_bytes((C['ROOT'] / 'README.md').read_bytes())
    runtime = K['Runtime'](h, out, 'qualification')
    runtime.qualification_reports = []

    def callback(rt, cell):
        report = rt.run_qualification_cell(cell)
        rt.qualification_reports.append(report)
        return []

    summary = K['capture'](runtime, C['QUALIFY_ORDER'], callback)
    reports = {report['cell']: report for report in runtime.qualification_reports}
    qualified = [C['width_for'](cell) for cell in ('W3', 'W7')
                 if reports.get(cell, {}).get('passed')]
    C['save'](out / 'screening.json', dict(status='captured', restored=summary['restored'],
                                           order=list(C['QUALIFY_ORDER']), reports=reports,
                                           qualified_widths=qualified,
                                           timing_not_started=True,
                                           timing_interpretation_permitted=False,
                                           raw_f32_equality_claimed=False,
                                           no_synthetic_acceptance=True))
    print('QUALIFICATION_CAPTURED_RESTORED_TIMING_NOT_STARTED', flush=True)


if __name__ == '__main__':
    main()
