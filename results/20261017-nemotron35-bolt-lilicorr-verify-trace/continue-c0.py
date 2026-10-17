#!/usr/bin/env python3
"""Single-use missing-C0 continuation; never rerun or overwrite the saved S0."""
import runpy
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-verify-trace')
NEXT = ROOT / 'continuation-c0'


def controller_source():
    source = (ROOT / 'run-trace.py').read_text()
    changes = {
        f"ROOT = Path('{ROOT}')": f"ROOT = Path('{NEXT}')",
        "WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'":
            "WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')",
        "PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-deterministic'":
            "PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-deterministic')",
        "ORDER = ('S0', 'C0')": "ORDER = ('C0',)",
        "PREFIX = 'bolt-lilicorr-verify-trace-r1-'": "PREFIX = 'bolt-lilicorr-verify-trace-c0-r1-'",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    analysis = runpy.run_path(str(ROOT / 'analyze-trace.py'))
    old = runpy.run_path(str(control['WIDTH1'] / 'run-controls.py'))
    saved = control['load'](ROOT / 'run-r1/summary.json')
    assert saved['status'] == 'failed' and saved['restored']
    assert saved['error'] == "AssertionError('unexpected history mutation')"
    assert set(saved['phases']) == {'S0'} and not (ROOT / 'run-r1/C0').exists()
    left = analysis['analyze_arm'](ROOT / 'run-r1/S0', 'S0', old)
    NEXT.mkdir(exist_ok=False)
    for name in ('README.md', 'analyze-trace.py', 'accepted-binaries.json', 'parent-binaries.json',
                 'candidate-binaries.json', 'source-diff.json', 'post-build-verification.log',
                 'runtime-precheck.log'):
        shutil.copyfile(ROOT / name, NEXT / name)
    (NEXT / 'run-trace.py').write_text(controller_source())
    subprocess.run(['python3', '-B', str(NEXT / 'run-trace.py')], check=True)
    continued = control['load'](NEXT / 'run-r1/summary.json')
    assert continued['status'] == 'completed' and continued['restored']
    assert continued['order'] == ['C0'] and set(continued['phases']) == {'C0'}
    right = analysis['analyze_arm'](NEXT / 'run-r1/C0', 'C0', old)

    # Independent restoration checks against the original failed pair as well.
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    import json
    command = helper['command']
    live = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
    for out in (ROOT / 'run-r1', NEXT / 'run-r1'):
        control['check_restored'](control['load'](out / 'original-container.json'), live)
        assert runner['hardware']() == control['load'](out / 'hardware-before.json') == control['load'](out / 'hardware-after.json')
        for label in ('accepted', 'parent', 'candidate'):
            assert control['binaries'](control[label.upper()]) == control['load'](out / (label + '-binaries.json'))
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
    assert api['request'](old['URL'], '/health')['status'] == 'ok'
    assert control['load'](NEXT / 'run-r1/restored-smoke.result.json')['text'].strip() == 'READY'
    result = analysis['compare'](left, right)
    result.update(restoration_verified=True, original_failed_run_preserved=True,
                  S0_source=str(ROOT / 'run-r1/S0'), C0_source=str(NEXT / 'run-r1/C0'))
    control['save'](ROOT / 'continuation-comparison.json', result)
    print('BOLT_VERIFY_MISSING_C0_COMPLETED_AND_COMPARED_NO_TIMING')


if __name__ == '__main__':
    main()
