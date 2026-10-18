#!/usr/bin/env python3
"""One-use missing-C0 continuation after corrected saved-S0 route attribution."""
import json
import runpy
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2')
NEXT = ROOT / 'continuation-c0'


def controller_source():
    source = (ROOT / 'run-trace.py').read_text()
    replacements = [
        ((f"ROOT = Path('{ROOT}')",), f"ROOT = Path('{NEXT}')"),
        (("WIDTH1 = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-width1'",
          "WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')"),
         "WIDTH1 = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1')"),
        (("PARENT_RESULTS = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-iq2-single2' / 'pair-r2'",
          "PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2/pair-r2')"),
         "PARENT_RESULTS = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2/pair-r2')"),
        (("ORDER = ('S0', 'C0')",), "ORDER = ('C0',)"),
        (("PREFIX = 'bolt-lilicorr-router-single2-r1-'",),
         "PREFIX = 'bolt-lilicorr-router-single2-c0-r1-'")]
    for options, after in replacements:
        matches = [before for before in options if source.count(before) == 1]
        assert len(matches) == 1, options
        source = source.replace(matches[0], after)
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
    assert result['route_analysis']['actual_route_presence']
    assert len(result['samples']) == 128
    if label == 'S0':
        analysis['assert_parent_observer_invariant'](result['parent_comparison'])
        assert result['s0_parent_invariant_passed']
    else:
        assert result['c0_parent_report_only']
    return result


def validate_saved_s0(control, analysis, old, native, runner):
    out = ROOT / 'run-r1'
    saved = control['load'](out / 'summary.json')
    assert saved['status'] == 'failed' and saved['restored']
    assert saved['error'] == "AssertionError('missing or ambiguous router target backend')"
    assert saved['order'] == ['S0', 'C0'] and set(saved['phases']) == {'S0'}
    assert saved['phases']['S0']['status'] == 'started' and not (out / 'C0').exists()
    failure = (ROOT / 'controller-r1.log').read_text()
    traceback = failure.split('Traceback (most recent call last):', 1)[-1]
    for line in (
            'File "/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2/run-trace.py", line 650, in main',
            "checked = analyze['analyze_arm'](cell, label, old)",
            'File "/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2/analyze-trace.py", line 632, in analyze_arm',
            "route_analysis['router_batches'] = assert_router_model_routes((cell / 'diagnostic-window.log').read_text(), label)",
            'File "/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2/analyze-trace.py", line 340, in assert_router_model_routes',
            "assert len(contexts) == 1, 'missing or ambiguous router target backend'",
            'AssertionError: missing or ambiguous router target backend'):
        assert line in traceback, line
    assert saved['timing_permitted'] is False and saved['promotion_permitted'] is False
    for name in ('accepted', 'parent', 'candidate'):
        assert control['binaries'](control[name.upper()]) == control['load'](out / (name + '-binaries.json')) == control['load'](ROOT / (name + '-binaries.json'))
    assert control['load'](out / 'frozen-payload.json') == control['load'](control['WIDTH1'] / 'run-r1/frozen-payload.json')
    base = control['load'](old['OLD'] / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json')
    make_create = control['make_create']
    original_root = make_create.__globals__['ROOT']
    make_create.__globals__['ROOT'] = out.resolve().parent
    try:
        expected = make_create(base, 'S0', runner['IMAGE'], out.resolve() / 'S0/capture', old)
    finally:
        make_create.__globals__['ROOT'] = original_root
    assert control['load'](out / 'S0/create-argv.json') == expected
    result = validate_arm(control, analysis, old, native, out / 'S0', 'S0')
    assert result['parent_comparison']['compared_through_ordinal'] == 128
    assert result['parent_comparison']['first_output_difference_ordinal'] is None
    assert len(result['route_analysis']['router_batches']) == 127
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
    assert not (ROOT / 'reanalysis-S0.json').exists(), 'S0 reanalysis already exists; continuation is one-use'
    assert not NEXT.exists(), 'continuation directory already exists; continuation is one-use'
    control['save'](ROOT / 'reanalysis-S0.json', left)
    NEXT.mkdir(exist_ok=False)
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
    result.update(
        restoration_verified=True,
        original_failed_run_preserved=True,
        attribution_gates_passed=True,
        timing_permitted=False,
        promotion_permitted=False,
        S0_source=str(ROOT / 'run-r1/S0'),
        C0_source=str(NEXT / 'run-r1/C0'),
        reanalysis_S0=str(ROOT / 'reanalysis-S0.json'),
        original_failed_summary=str(ROOT / 'run-r1/summary.json'),
        s0_parent_invariant_passed=left['s0_parent_invariant_passed'],
        c0_parent_report_only=right['c0_parent_report_only'],
        route_analysis={label: arm['route_analysis'] for label, arm in (('S0', left), ('C0', right))},
        parent_comparison={label: arm['parent_comparison'] for label, arm in (('S0', left), ('C0', right))})
    assert result['s0_parent_invariant_passed'] and result['c0_parent_report_only']
    control['save'](ROOT / 'continuation-comparison.json', result)
    print(json.dumps({k: result[k] for k in ('first_output_difference_ordinal', 'compared_through_ordinal',
                                             'restoration_verified', 'attribution_gates_passed',
                                             'timing_permitted', 'promotion_permitted')}))
    print('BOLT_ROUTER_MISSING_C0_COMPLETED_SAVED_S0_PRESERVED_NO_TIMING')


if __name__ == '__main__':
    main()
