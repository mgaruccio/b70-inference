#!/usr/bin/env python3
"""Matched native/shared-KV HTTP benchmarks at 230 W; restore 275 W afterward.

Same retained inputs and serving settings as the earlier comparison, except the
power cap and graph statistics (enabled identically in both arms). Does not port
the candidate's shape-specific seam to the published 131072/.88 configuration.
"""
import hashlib
import json
from pathlib import Path
import re
import runpy
import shlex
import signal
import subprocess

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
CAMPAIGN = ROOT / '20261009-qwen38-community-protocol'
PRIOR = ROOT / '20260913-qwen38-native-grouped-verify'
SUITE = CAMPAIGN / 'power-230w-01'
POWER = Path('/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap')
PRODUCTION = Path('/home/mike/inference/launchers/start-qwen38.sh')
PRODUCTION_SHA = '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def set_power(value):
    subprocess.run(['sudo', '-n', 'tee', str(POWER)], input=value + '\n',
                   text=True, stdout=subprocess.PIPE, check=True, timeout=15)
    assert POWER.read_text().strip() == value, 'Power-cap readback mismatch'


def abort(signum, frame):
    raise InterruptedError(f'received signal {signum}')


def main():
    assert digest(PRODUCTION) == PRODUCTION_SHA
    previous = ROOT / '20261009-qwen38-native-shared-kv/run-cell.py'
    assert digest(previous) == '58fc22ce410d84c6559c07109973568078e6ac80004a2c8c0769a25dc277fd09'
    sources = runpy.run_path(str(previous))['verify_sources']()
    adapter = runpy.run_path(str(PRIOR / 'run-serving.py'))
    library, patch = adapter['DEFAULT_LIBRARY'], adapter['CANONICAL_PATCH']
    assert digest(library) == adapter['EXPECTED_LIBRARY_SHA256']
    assert digest(patch) == adapter['EXPECTED_CANONICAL_PATCH_SHA256']
    common = runpy.run_path(str(CAMPAIGN / 'run.py'))
    assert digest(common['GUARD']) == common['GUARD_SHA256']
    original_guard = runpy.run_path(str(common['GUARD']))
    original_cap = POWER.read_text().strip()
    assert original_cap == '275000000', 'Unexpected initial power; do not overwrite another setting'
    for name in ('native-230w-01', 'shared-kv-230w-01'):
        assert not (CAMPAIGN / name).exists(), f'Output already exists: {name}'
    SUITE.mkdir(exist_ok=False)
    (SUITE / 'run-230w.py').write_bytes(Path(__file__).read_bytes())
    before = SUITE / 'before'
    before.mkdir()
    original_guard['preconditions'](before, 'qwen38')
    guard_text = common['GUARD'].read_text()
    assert guard_text.count('275000000') == 2 and guard_text.count('275 W') == 1
    guard_text = guard_text.replace('275000000', '230000000').replace('275 W', '230 W')
    guard_path = SUITE / 'guard-230w.py'
    guard_path.write_text(guard_text)
    compile(guard_text, str(guard_path), 'exec')
    original = PRODUCTION.read_text()
    marker = '--language-model-only'
    assert original.count(marker) == 1
    native = original.replace(marker, marker + ' --cudagraph-metrics')
    mounts = adapter['_candidate_mounts'](PRIOR / 'serving-overlay.py', patch, PRIOR / 'grouped_verify.py', library)
    environment = ['PYTHONPATH=/experiment', 'B70_STEP_TIMING=1', 'B70_GROUPED_SERVING=1',
                   'B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so']
    additions = ''.join('  -v ' + shlex.quote(f"{m['host']}:{m['container']}:ro") + ' \\\n' for m in mounts)
    additions += ''.join('  -e ' + shlex.quote(value) + ' \\\n' for value in environment)
    hook = '; /opt/venv/bin/python -P /experiment/qwen38_step_timing_patch.py; exec vllm serve '
    assert native.count('  --entrypoint bash') == native.count('; exec vllm serve ') == 1
    candidate = native.replace('  --entrypoint bash', additions + '  --entrypoint bash').replace('; exec vllm serve ', hook)
    assert candidate.replace(additions, '', 1).replace(hook, '; exec vllm serve ', 1) == native
    for name, text in (('native', native), ('shared-kv', candidate)):
        launcher = SUITE / (name + '.sh')
        launcher.write_text(text)
        launcher.chmod(0o700)
        subprocess.run(['bash', '-n', str(launcher)], check=True)
    status = {'original_cap_microwatts': original_cap, 'requested_cap_microwatts': '230000000',
              'success': False, 'completed_arms': [], 'guard_sha256': digest(guard_path),
              'settings': 'unchanged 212992/.95/C1/cache-on/thinking-on; graph metrics enabled in both arms'}
    put(SUITE / 'power.json', status)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, abort)
    try:
        set_power('230000000')
        print('Verified GPU cap: 230 W', flush=True)
        for arm in ('native', 'shared-kv'):
            out = CAMPAIGN / (arm + '-230w-01')

            def check_build(directory, final):
                assert POWER.read_text().strip() == '230000000'
                text = (directory / 'server.log').read_text(errors='replace')
                assert 'cudagraph_metrics=True' in text, 'Graph metrics not enabled'
                if arm == 'shared-kv':
                    eligible = [line for line in text.splitlines()
                                if '[B70_GROUPED_SERVING]' in line and '"event":"eligible-dispatch"' in line]
                    assert any(adapter['EXPECTED_LIBRARY_SHA256'] in line for line in eligible), 'Candidate dispatch missing'
                    assert '"event":"unsupported-q5"' not in text, 'Unsupported startup candidate shape'
                    put(directory / 'build-contract.json', {
                        'candidate_library_sha256': adapter['EXPECTED_LIBRARY_SHA256'],
                        'source_hashes': sources, 'mounts': mounts, 'environment': environment,
                        'base_launcher_sha256': PRODUCTION_SHA, 'startup_eligible_dispatch': eligible,
                        'final_check': final})
                    if final:
                        adapter['_candidate_execution_evidence'](directory)
                else:
                    assert '[B70_GROUPED_SERVING]' not in text, 'Native arm unexpectedly contains candidate'
                if final:
                    full_lines = [line for line in text.splitlines() if re.search(r'\|\s*FULL\s*\|', line)]
                    assert full_lines, 'Missing measured runtime FULL graph statistics'
                    put(directory / 'graph-runtime-evidence.json', {'full_runtime_statistics': full_lines})
                print(f'Verified {arm} at 230 W, final={final}', flush=True)

            run = runpy.run_path(str(CAMPAIGN / 'run.py'))
            run['main'].__globals__.update(
                OUT=out, LAUNCHER=SUITE / (arm + '.sh'), GUARD=guard_path,
                GUARD_SHA256=digest(guard_path), CHECK_BUILD=check_build,
                PREFILL_PROMPTS=CAMPAIGN / 'current-serving-02/prefill-prompts.json')
            print('Starting ' + arm + ' arm', flush=True)
            run['main']()
            assert digest(PRODUCTION) == PRODUCTION_SHA
            runpy.run_path(str(previous))['verify_sources']()
            status['completed_arms'].append(arm)
            put(SUITE / 'power.json', status)
        status['success'] = True
    except BaseException as exc:
        status['error'] = repr(exc)
        raise
    finally:
        try:
            set_power(original_cap)
            after = SUITE / 'restored'
            after.mkdir()
            original_guard['preconditions'](after, 'qwen38')
            status['restoration_verified'] = True
            print('Restored and verified original 275 W cap and idle host', flush=True)
        finally:
            status['final_cap_microwatts'] = POWER.read_text().strip()
            put(SUITE / 'power.json', status)


if __name__ == '__main__':
    main()
