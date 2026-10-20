#!/usr/bin/env python3
"""Lead-only bounded device/host diagnostic through the existing qualified journey."""
import argparse
import copy
import re
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-device-timing')
OUT = ROOT / 'profile-r1'
PREFIX = 'bolt-lilicorr-device-timing-r1-'
CONTEXT = HERE.parent / '20261020-lilicorr-context-profile'
CTX = runpy.run_path(str(CONTEXT / 'run-profile.py'))
# Keep an immutable helper namespace: the configured adapter's HERE changes below.
ORIGINAL_SCOPE_CHECK = runpy.run_path(str(CONTEXT / 'run-profile.py'))['check_first_job_scope']
BATCH = runpy.run_path(str(HERE.parent / '20261020-lilicorr-iq2-batch4/run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
Q, PROFILE_ENV = CTX['Q'], CTX['PROFILE_ENV']
QUALIFIER_BIND = str(ROOT.parent / '20261020-lilicorr-iq2-batch4') + ':/iq2-batch4:ro'
OLD_QUALIFIER, NEW_QUALIFIER = BATCH['OLD_QUALIFIER'], BATCH['NEW_QUALIFIER']
OWN_FILES = (
    'README.md', '.gitignore', 'device-timing.patch', 'source-pins.json',
    'source-proof.py', 'test-source-proof.py', 'build-candidate.sh',
    'build-three-modules.py', 'test-three-module-build.py', 'build-three-module-inputs.json.gz',
    'cpu-checks.txt', 'build-cpu-checks.txt', 'test-device-timing.py',
    'fixtures/event-driver.cpp', 'fixtures/process-driver.cpp',
    'run-profile.py', 'test-runtime.py', 'analyze-device.py', 'test-analyze-device.py',
    'device-analysis-cpu-checks.txt',
    '../20261020-lilicorr-iq2-batch4/run-performance.py',
    '../20261020-lilicorr-iq2-batch4/native-batch4.py',
    '../20261020-lilicorr-iq2-batch4/source-proof.py',
    str(CONTEXT / 'run-profile.py'), str(CONTEXT / 'source-proof.py'),
    str(CONTEXT / 'analyze-profile.py'),
) + tuple(str(CONTEXT / name) for name in CTX['OWN_FILES'])
_BUILD = runpy.run_path(str(HERE / 'build-three-modules.py'))
_LABELS = ('compiler-version', 'ninja-version', 'cmake-version', 'targets') + tuple(
    stem + '-' + _BUILD['_label'](target)
    for target in _BUILD['MODULE_ORDER']
    for stem in ('generated-commands', 'queries', 'module-commands', 'module-plan', 'module-build', 'module-link'))
BUILD_FILES = tuple('build-attempt-r1/' + name for name in (
    'candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
    'post-build-verification.log', 'runtime-precheck.log', 'exit-code.txt', 'image.json',
    'seed-built-before.json', 'seed-built-after.json', 'candidate-source-before.json',
    'candidate-source-after.json', 'parent-source.json', 'build.log',
)) + tuple('build-attempt-r1/' + label + suffix for label in _LABELS
          for suffix in ('.stdout.log', '.stderr.log', '.argv.json')) + tuple(
    'build-attempt-r1/module-objects-' + _BUILD['_label'](target) + '.json'
    for target in _BUILD['MODULE_ORDER'])


def qualifier_argv(argv):
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    assert result[-1].count(OLD_QUALIFIER) == 1 and NEW_QUALIFIER not in result[-1]
    assert QUALIFIER_BIND not in result
    result[-1] = result[-1].replace(OLD_QUALIFIER, NEW_QUALIFIER)
    index = len(result) - 4  # Docker options must precede IMAGE.
    result[index:index] = ['-v', QUALIFIER_BIND]
    reverse = list(result)
    del reverse[index:index + 2]
    reverse[-1] = reverse[-1].replace(NEW_QUALIFIER, OLD_QUALIFIER)
    assert reverse == list(argv), 'unintended qualifier argv delta'
    return result


def check_container(c, record, cell, h):
    if c['role_for'](cell) == 'A':
        return Q['check_container'](c, record, cell, h)
    reference = c['load'](c['qualified_dir']('W3') / 'container-before.json')
    assert record['Name'] == '/' + c['name'](cell)
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert record['Image'] == reference['Image']
    assert record['Config']['Cmd'] == c['make_create'](cell, h)[-3:]
    env = c['_env_map'](reference['Config']['Env'])
    assert PROFILE_ENV not in env and env['ONEDNN_VERBOSE'] == '0'
    env[PROFILE_ENV] = '1'
    assert c['_env_map'](record['Config']['Env']) == env
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {k: v for k, v in record['Config'].items() if k not in excluded} == {
        k: v for k, v in reference['Config'].items() if k not in excluded}
    expected, actual = copy.deepcopy(reference['HostConfig']), copy.deepcopy(record['HostConfig'])
    for old, new in ((Q['old_draft_bind'](), Q['candidate_draft_bind']()),
                     (str(S['PARENT']) + ':/src:ro', str(S['CANDIDATE']) + ':/src:ro')):
        assert expected['Binds'].count(old) == 1
        expected['Binds'] = [new if item == old else item for item in expected['Binds']]
    expected['Binds'].append(QUALIFIER_BIND)
    for host in (expected, actual):
        host['Binds'] = sorted(host.get('Binds') or [])
        assert host.pop('OomKillDisable', None) in (None, False)
    assert actual == expected
    mounts = Q['_expected_mounts'](reference, cell)
    sources = [m for m in mounts if m['Destination'] == '/src']
    assert len(sources) == 1 and sources[0]['Source'] == str(S['PARENT'])
    assert sources[0]['Type'] == 'bind' and sources[0]['RW'] is False
    sources[0]['Source'] = str(S['CANDIDATE'])
    mounts.append(dict(Type='bind', Source=str(ROOT.parent / '20261020-lilicorr-iq2-batch4'),
                       Destination='/iq2-batch4', Mode='ro', RW=False, Propagation='rprivate'))
    assert c['canonical_mounts'](record['Mounts']) == c['canonical_mounts'](mounts)
    return True


def check_first_job_scope(stderr):
    proof = ORIGINAL_SCOPE_CHECK(stderr)
    lines = stderr.splitlines()
    current = '\n'.join(lines[proof['request_start_line'] - 1:])
    device = runpy.run_path(str(HERE / 'analyze-device.py'))['check_request'](current)
    assert device.get('status') == 'ok', 'first public job lacks positively validated device/host timing'
    return dict(**proof, bounded_device=device)


def bind():
    globals_ = CTX['bind'].__globals__
    globals_.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, S=S,
                    OWN_FILES=OWN_FILES, BUILD_FILES=BUILD_FILES,
                    check_first_job_scope=check_first_job_scope)
    runtime, analysis, c = CTX['bind']()
    assert c['ROOT'] == ROOT and c['OUT'] == OUT and c['PREFIX'] == PREFIX
    base_make = c['make_create']
    base_runtime_check, base_measurement = analysis['check_runtime_evidence'], analysis['reread_measurement']

    def make_create(cell, h):
        base = base_make(cell, h)
        result = qualifier_argv(base) if c['role_for'](cell) == 'W3' else base
        assert c['server_argv'](result) == c['server_argv'](base)
        return result

    class DeviceTimingRuntime(runtime['Runtime']):
        def preserve_native(self, cell, phase, required=False):
            if c['role_for'](cell) == 'A':
                return super().preserve_native(cell, phase, required=required)
            super().preserve_native(cell, phase, required=False)
            if required:
                directory = self.out / cell
                assert not c['load'](directory / (phase + '-native-copy.json'))['errors']
                public = c['_public']()
                for tag, filename in public['NATIVE_LOGS'].items():
                    BATCH['check_native'](public, tag, (directory / (phase + '-' + filename)).read_text())

    def check_runtime_evidence(p, h, out, cell):
        if c['role_for'](cell) == 'A':
            return base_runtime_check(p, h, out, cell)
        private = dict(p)
        private['check_native_log'] = lambda tag, log: BATCH['check_native'](p, tag, log)
        return base_runtime_check(private, h, out, cell)

    def reread_measurement(p, h, out, job):
        return BATCH['check_acceptance'](base_measurement(p, h, out, job))

    c.update(make_create=make_create,
             check_container=lambda record, cell, h: check_container(c, record, cell, h))
    runtime['Runtime'] = DeviceTimingRuntime
    for key, value in (('check_runtime_evidence', check_runtime_evidence),
                       ('reread_measurement', reread_measurement)):
        analysis['analyze'].__globals__[key] = value
        analysis[key] = value
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--lead-run-profile', action='store_true')
    mode.add_argument('--lead-after-profile-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified ancestry must not be rebound'
    if args.lead_run_profile:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored'], result
        print('LILICORR_DEVICE_TIMING_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        legacy = runpy.run_path(str(CONTEXT / 'analyze-profile.py'))['analyze'](OUT, c)
        assert legacy['status'] == 'analyzed', legacy['errors']
        diagnostic = runpy.run_path(str(HERE / 'analyze-device.py'))['analyze'](OUT, c)
        assert diagnostic['status'] == 'analyzed', diagnostic.get('errors')
        print('LILICORR_DEVICE_TIMING_ANALYZED_RESTORED_DIAGNOSTIC_ONLY', flush=True)


if __name__ == '__main__':
    main()
