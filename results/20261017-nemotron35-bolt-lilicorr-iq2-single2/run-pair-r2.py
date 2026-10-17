#!/usr/bin/env python3
"""One-use fresh pair after an IQ2 native-log parser failure, before model requests."""
import json
import runpy
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2')
NEXT = ROOT / 'pair-r2'


def rewritten(name):
    source = (ROOT / name).read_text()
    changes = {
        f"ROOT = Path('{ROOT}')": f"ROOT = Path('{NEXT}')",
        "WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'":
            "WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')",
        "PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-attention-dmmv2-view'":
            "PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-dmmv2-view')",
    }
    if name == 'run-trace.py':
        changes["PREFIX = 'bolt-lilicorr-iq2-single2-r1-'"] = "PREFIX = 'bolt-lilicorr-iq2-single2-r2-'"
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    old = runpy.run_path(str(control['WIDTH1'] / 'run-controls.py'))
    native = runpy.run_path(str(control['WIDTH1'] / 'analyze-controls.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    failed = ROOT / 'run-r1'
    saved = control['load'](failed / 'summary.json')
    assert saved['status'] == 'failed' and saved['restored'] and saved['error'] == 'AssertionError()'
    assert set(saved['phases']) == {'S0'} and saved['phases']['S0']['status'] == 'started'
    assert not (failed / 'C0').exists() and not (failed / 'S0/quality').exists()
    assert not any((failed / 'S0/capture').iterdir())
    log = (failed / 'S0/server.log').read_text()
    assert control['MARKER'] not in log and 'AssertionError: []' in log
    native['parse_native_gate'](log, 'C0')
    assert 'call ggml_sycl_ssm_scan_prefill:' not in log
    focused = log.split('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[1].split('Traceback (most recent call last):', 1)[0]
    control['check_iq2_native_output'](focused)
    live = json.loads(subprocess.check_output(['docker', 'inspect', old['ORIGINAL']]))[0]
    control['check_restored'](control['load'](failed / 'original-container.json'), live)
    assert set(subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).splitlines()) == {old['ORIGINAL']}
    assert runner['hardware']() == control['load'](failed / 'hardware-before.json') == control['load'](failed / 'hardware-after.json')
    for name in ('accepted', 'parent', 'candidate'):
        assert control['binaries'](control[name.upper()]) == control['load'](failed / (name + '-binaries.json')) == control['load'](ROOT / (name + '-binaries.json'))
    assert api['request'](old['URL'], '/health')['status'] == 'ok'
    smoke = control['load'](failed / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    NEXT.mkdir(exist_ok=False)
    for name in ('README.md', 'accepted-binaries.json', 'parent-binaries.json', 'candidate-binaries.json',
                 'source-diff.json', 'post-build-verification.log', 'runtime-precheck.log'):
        shutil.copyfile(ROOT / name, NEXT / name)
    for name in ('run-trace.py', 'analyze-trace.py'):
        (NEXT / name).write_text(rewritten(name))
    subprocess.run(['python3', '-B', str(NEXT / 'run-trace.py')], check=True)
    subprocess.run(['python3', '-B', str(NEXT / 'analyze-trace.py')], check=True)
    print('BOLT_IQ2_PAIR_R2_ANALYZED_ORIGINAL_FAILURE_PRESERVED_NO_TIMING')


if __name__ == '__main__':
    main()
