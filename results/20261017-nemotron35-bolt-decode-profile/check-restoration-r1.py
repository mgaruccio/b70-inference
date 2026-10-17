#!/usr/bin/env python3
"""One-off independent verification after the retained mount-list-order assertion failure."""
import json
from pathlib import Path
import runpy
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
C = runpy.run_path(str(ROOT / 'run-decode-profile.py'))
assert subprocess.check_output(['hostname'], text=True).strip() == 'inference-host'
run = ROOT / 'run-r1'
summary = json.loads((run / 'summary.json').read_text())
assert summary['status'] == 'failed' and summary.get('restoration_error') == 'AssertionError()'
assert 'error' not in summary
assert all(x['status'] == 'passed' for x in summary['phases'].values())
old = json.loads((run / 'original-container.json').read_text())
now = json.loads(subprocess.check_output(['docker', 'inspect', C['ORIGINAL']]))[0]
assert now['State']['Running'] and old['Id'] == now['Id'] and old['Config'] == now['Config']
assert sorted(now['Mounts'], key=lambda m: m['Destination']) == sorted(old['Mounts'], key=lambda m: m['Destination'])
helper = runpy.run_path(str(C['GROUPED'] / 'run-grouped-diagnostic.py'))
runner = runpy.run_path(str(C['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
hashes = helper['binaries'](runner['PREFILL_SSM'])
assert hashes == C['EXPECTED']
hardware = runner['hardware']()
assert hardware == json.loads((run / 'hardware-before.json').read_text())
with urllib.request.urlopen(C['URL'] + '/health', timeout=10) as response:
    health = json.load(response)
assert health == {'status': 'ok'}
assert set(subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()) == {C['ORIGINAL']}
smoke = json.loads((run / 'restored-smoke.result.json').read_text())
assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
C['save'](run / 'restored-container-independent.json', now)
C['save'](run / 'hardware-after-independent.json', hardware)
C['save'](run / 'restoration-check.json', {
    'status': 'passed', 'original_run_status': 'failed', 'reason': 'Mount list ordering only; all mount fields match after sorting by Destination.',
    'same_container_id': True, 'same_config': True, 'same_mounts_order_independent': True,
    'health': health, 'accepted_hashes': hashes, 'hardware_unchanged': True, 'ready_smoke': True,
    'baseline_phase': summary['phases']['baseline']['status'], 'profile_phase': summary['phases']['profile']['status'],
    'capture_repeated': False,
})
print('BOLT_RESTORATION_INDEPENDENT_CHECK_PASSED')
