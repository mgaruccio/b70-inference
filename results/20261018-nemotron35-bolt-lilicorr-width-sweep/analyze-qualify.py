#!/usr/bin/env python3
"""LEAD ONLY: independently verify restored qualification and select widths."""
import argparse
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
K = runpy.run_path(str(Path(__file__).with_name('controller.py')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-after-qualification-restore', action='store_true', required=True)
    parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == C['ROOT']
    h = C['helpers']()
    C['check_tested'](h)
    out = C['QUALIFY_DIR']
    summary = C['load'](out / 'summary.json')
    screening = C['load'](out / 'screening.json')
    assert summary['status'] == 'captured' and summary['restored']
    assert summary['completed'] == list(C['QUALIFY_ORDER'])
    assert screening['status'] == 'captured' and screening['restored']
    runtime = K['Runtime'](h, out, 'qualification')
    runtime.verify_restored(out / 'independent-restoration.json')

    reports = {}
    for cell in C['QUALIFY_ORDER']:
        report = C['load'](out / cell / 'qualification.json')
        assert report['cell'] == cell
        assert len(report['jobs']) == 25
        assert report['behavior']['natural_stop_cases'] == 21
        assert report['behavior']['fixed_length_cases'] == 2
        assert report['errors'] == []
        reports[cell] = report
    assert reports['A']['passed'], 'fresh no-draft A did not reproduce frozen A'
    qualified = [C['width_for'](cell) for cell in ('W3', 'W7') if reports[cell]['passed']]
    C['save'](out / 'qualification.json', dict(status='passed', restored=True,
                                                independently_restored=True,
                                                baseline='A', qualified_widths=qualified,
                                                reports=reports,
                                                output_contract='exact text/finish/usage; diagnostics exact actual IDs/pieces/text/finish/usage',
                                                counts_compared=False,
                                                probabilities_compared=False,
                                                raw_f32_equality_claimed=False,
                                                timing_interpretation_permitted=False,
                                                no_synthetic_acceptance=True))
    print('QUALIFICATION_ANALYZED_RESTORED_WIDTHS=' + ','.join(map(str, qualified)), flush=True)


if __name__ == '__main__':
    main()
