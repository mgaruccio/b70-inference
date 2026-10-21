#!/usr/bin/env python3
"""Independently reread the completed clean controls from a failed trace capture.

This does not convert the failed attempt into a completed GPU profile.
"""
import json
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-profile.py'))
assert __debug__ and socket.gethostname() == 'inference-host' and HERE == R['ROOT']
prior, runtime, analysis, c = R['bind']()
p, _k, h = runtime['load_dependencies']()
out = R['OUT']
summary = c['load'](out / 'summary.json')
assert summary['status'] == 'failed' and summary['completed'] == ['VSPF-A1', 'VSPF-W3a']
assert summary['restored'] and c['load'](out / 'independent-restoration-live.json')['passed']
result = dict(diagnostic_only=True, full_profile_passed=False, traced_serving_qualified=False,
              native={}, public={}, rows=[], capture_failure=summary['error'])
for cell in summary['completed']:
    result['native'][cell] = analysis['check_runtime_evidence'](p, h, out, cell)
    report = analysis['reread_prevalidation'](p, h, out, cell)
    assert report['passed'] and len(report['jobs']) == 25
    result['public'][cell] = dict(passed=True, jobs=25)
    for point in ('short', 'long'):
        for ordinal in range(3):
            row = analysis['reread_measurement'](p, h, out, dict(cell=cell, point=point,
                phase='measured', ordinal=ordinal))
            assert row['passed'] and row['equals_fresh_a']
            result['rows'].append(row)
assert len(result['rows']) == 12
path = out / 'clean-controls-analysis.json'
assert not path.exists(), 'preserve previous rechecks'
c['save'](path, result)
print('CLEAN_CONTROLS_50_PUBLIC_12_MEASURED_INDEPENDENTLY_REREAD_NOT_A_GPU_PROFILE')
