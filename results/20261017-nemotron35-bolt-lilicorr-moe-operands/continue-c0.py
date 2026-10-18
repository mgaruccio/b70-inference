#!/usr/bin/env python3
"""One-use missing-C0 continuation after saved S0 passes corrected log analysis."""
import json
import runpy
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-moe-operands')
NEXT = ROOT / 'continuation-c0'


def controller_source():
    source = (ROOT / 'run-trace.py').read_text()
    changes = {
        f"ROOT = Path('{ROOT}')": f"ROOT = Path('{NEXT}')",
        "WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'":
            "WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')",
        "PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-iq2-single2' / 'pair-r2'":
            "PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2/pair-r2')",
        "ORDER = ('S0', 'C0')": "ORDER = ('C0',)",
        "PREFIX = 'bolt-lilicorr-moe-operands-r1-'": "PREFIX = 'bolt-lilicorr-moe-operands-c0-r1-'",
    }
    for before, after in changes.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source


def validate_arm(control, analysis, old, native, cell, label):
    log = (cell / 'server.log').read_text()
    control['check_startup'](log, label, old, native)
    assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
    assert (cell / 'diagnostic-window.log').read_text() == control['diagnostic_window'](
        (cell / 'pre-diagnostic.log').read_text(), (cell / 'through-diagnostic.log').read_text())
    quality = control['load'](cell / 'quality/summary.json')
    assert quality['pass'] and len(quality['checks']) == 20 and all(row['pass'] for row in quality['checks'])
    result = analysis['analyze_arm'](cell, label, old)
    assert result['observer_parent_invariant_passed'] and result['operand_analysis']['complete']
    return result


def validate_saved_s0(control, analysis, old, native, runner):
    out = ROOT / 'run-r1'
    saved = control['load'](out / 'summary.json')
    assert saved['status'] == 'failed' and saved['restored']
    assert saved['error'] == 'AssertionError()'
    failure = (ROOT / 'controller-r1.log').read_text()
    assert "assert row['type'] == expected_type and row['reorder'] == expected_reorder" in failure
    assert 'in operand_record' in failure
    assert saved['order'] == ['S0', 'C0'] and set(saved['phases']) == {'S0'}
    assert saved['phases']['S0']['status'] == 'started' and not (out / 'C0').exists()
    for name in ('accepted', 'parent', 'candidate'):
        assert control['binaries'](control[name.upper()]) == control['load'](out / (name + '-binaries.json')) == control['load'](ROOT / (name + '-binaries.json'))
    assert control['load'](out / 'frozen-payload.json') == control['load'](control['WIDTH1'] / 'run-r1/frozen-payload.json')
    base = control['load'](old['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    expected = control['make_create'](base, 'S0', runner['IMAGE'], out / 'S0/capture', old)
    assert control['load'](out / 'S0/create-argv.json') == expected
    result = validate_arm(control, analysis, old, native, out / 'S0', 'S0')
    assert len(result['samples']) == result['parent_comparison']['compared_through_ordinal'] == 128
    return result


def check_restoration(control, old, runner, outputs):
    helper = runpy.run_path(str(old['GROUPED'] / 'run-grouped-diagnostic.py'))
    api = runpy.run_path(str(old['OLD'] / 'probe-api.py'))
    command = helper['command']
    live = json.loads(command('docker', 'inspect', old['ORIGINAL']))[0]
    hardware = runner['hardware']()
    control['check_protected_sources']()
    for out in outputs:
        control['check_restored'](control['load'](out / 'original-container.json'), live)
        assert hardware == control['load'](out / 'hardware-before.json') == control['load'](out / 'hardware-after.json')
        for name in ('accepted', 'parent', 'candidate'):
            assert control['binaries'](control[name.upper()]) == control['load'](out / (name + '-binaries.json'))
        smoke = control['load'](out / 'restored-smoke.result.json')
        assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {old['ORIGINAL']}
    assert api['request'](old['URL'], '/health')['status'] == 'ok'


def main():
    assert socket.gethostname() == 'inference-host'
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    analysis = runpy.run_path(str(ROOT / 'analyze-trace.py'))
    old = runpy.run_path(str(control['WIDTH1'] / 'run-controls.py'))
    native = runpy.run_path(str(control['WIDTH1'] / 'analyze-controls.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    left = validate_saved_s0(control, analysis, old, native, runner)
    check_restoration(control, old, runner, [ROOT / 'run-r1'])
    NEXT.mkdir(exist_ok=False)
    control['save'](ROOT / 'reanalysis-S0.json', left)
    for name in ('README.md', 'analyze-trace.py', 'accepted-binaries.json', 'parent-binaries.json',
                 'candidate-binaries.json', 'source-diff.json', 'protected-source-hashes.json',
                 'post-build-verification.log', 'runtime-precheck.log'):
        shutil.copyfile(ROOT / name, NEXT / name)
    (NEXT / 'run-trace.py').write_text(controller_source())
    subprocess.run(['python3', '-B', str(NEXT / 'run-trace.py')], check=True)
    continued = control['load'](NEXT / 'run-r1/summary.json')
    assert continued['status'] == 'completed' and continued['restored']
    assert continued['order'] == ['C0'] and set(continued['phases']) == {'C0'}
    right = validate_arm(control, analysis, old, native, NEXT / 'run-r1/C0', 'C0')
    check_restoration(control, old, runner, [ROOT / 'run-r1', NEXT / 'run-r1'])
    result = analysis['compare'](left, right)
    result.update(restoration_verified=True, original_failed_run_preserved=True, attribution_gates_passed=True,
                  S0_source=str(ROOT / 'run-r1/S0'), C0_source=str(NEXT / 'run-r1/C0'),
                  observer_parent_invariant_passed=True,
                  operand_comparison=analysis['compare_operands']({'S0': left, 'C0': right}),
                  route_analysis={label: arm['route_analysis'] for label, arm in (('S0', left), ('C0', right))},
                  parent_comparison={label: arm['parent_comparison'] for label, arm in (('S0', left), ('C0', right))})
    control['save'](ROOT / 'continuation-comparison.json', result)
    print(json.dumps({k: result[k] for k in ('first_output_difference_ordinal', 'compared_through_ordinal',
                                           'restoration_verified', 'attribution_gates_passed')}))
    print(json.dumps(result['operand_comparison']))
    print('BOLT_MOE_MISSING_C0_COMPLETED_SAVED_S0_PRESERVED_NO_TIMING')


if __name__ == '__main__':
    main()
