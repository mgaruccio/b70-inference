#!/usr/bin/env python3
"""LEAD ONLY: independently verify restored qualification and select widths."""
import argparse
import runpy
import socket
from pathlib import Path

C = runpy.run_path(str(Path(__file__).with_name('contracts.py')))
K = runpy.run_path(str(Path(__file__).with_name('controller.py')))


def select_qualified(reports):
    """Reject failed widths without blocking a fully qualified sibling."""
    assert set(reports) == set(C['QUALIFY_ORDER'])
    for cell in C['QUALIFY_ORDER']:
        report = reports[cell]
        assert report['cell'] == cell and report['width'] == C['width_for'](cell)
        assert len(report['jobs']) == 25
        assert len({row['stem'] for row in report['jobs']}) == 25
        assert report['errors'] == []
        assert type(report['passed']) is bool
        if report['passed']:
            assert all(row['passed'] for row in report['jobs'])
            assert report['fidelity_failures'] == []
            assert report['behavior']['natural_stop_cases'] == 21
            assert report['behavior']['fixed_length_cases'] == 2
            assert report['behavior']['diagnostics_exercised'] == 2
            if C['width_for'](cell) is not None:
                assert report['behavior']['rejected_draft_observed'] > 0
        else:
            assert cell != 'A', 'fresh no-draft A did not reproduce frozen A'
            assert report['fidelity_failures'], 'unexplained rejection'
    return [C['width_for'](cell) for cell in ('W3', 'W7') if reports[cell]['passed']]


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

    reports = {cell: C['load'](out / cell / 'qualification.json')
               for cell in C['QUALIFY_ORDER']}
    qualified = select_qualified(reports)
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
