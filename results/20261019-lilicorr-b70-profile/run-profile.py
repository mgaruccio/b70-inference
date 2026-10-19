#!/usr/bin/env python3
"""Lead-only timing-profile extension of the existing Q8 W3 ABBA adapter."""
import argparse
import copy
import json
import re
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-b70-profile')
OUT = ROOT / 'profile-r1'
PREFIX = 'bolt-lilicorr-b70-profile-r1-'
Q = runpy.run_path(str(HERE.parent / '20261019-lilicorr-q8-drafter/run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
PROFILE_ENV = 'LLAMA_LILICORR_PROFILE'
OWN_FILES = ('README.md', '.gitignore', 'run-profile.py', 'source-proof.py', 'build-candidate.sh',
             'analyze-profile.py', 'test-profile.py', 'source-hashes.json', 'lilicorr-profile.patch',
             'fixtures/common/speculative.cpp.gz', 'fixtures/src/llama-context.cpp.gz')
BUILD_FILES = ('candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
               'post-build-verification.log', 'runtime-precheck.log', 'build-attempt-r1/exit-code.txt',
               'build-attempt-r1/image.json')
PROFILE_PID_PROBE = (
    "import json; from pathlib import Path; "
    "print(json.dumps([v.decode().split('=',1) for v in Path('/proc/1/environ').read_bytes().split(b'\\0') "
    "if v.startswith(b'LLAMA_LILICORR_PROFILE=')]))"
)


def source_bind(source):
    return str(source) + ':/src:ro'


def replace_source_bind(argv):
    result = list(argv)
    old, new = source_bind(S['PARENT']), source_bind(S['CANDIDATE'])
    assert result.count(old) == 1 and new not in result
    index = result.index(old)
    assert result[index - 1] == '-v'
    assert not any(PROFILE_ENV in arg for arg in result)
    result[index] = new
    result[-4:-4] = ['-e', PROFILE_ENV + '=1']
    reverse = list(result)
    del reverse[-6:-4]
    reverse[index] = old
    assert reverse == list(argv)
    return result


def check_profile_pid(record, cell):
    assert record['pid'] == 1 and record['origin'] == '/proc/1/environ'
    assert record['profile_environment_origin'] == '/proc/1/environ'
    expected = [[PROFILE_ENV, '1']] if cell.startswith('VSPF-W3') else []
    assert record['profile_environment'] == expected, 'actual PID1 profiling opt-in differs'
    return True


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
    assert PROFILE_ENV not in env
    env[PROFILE_ENV] = '1'
    assert c['_env_map'](record['Config']['Env']) == env
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {k: v for k, v in record['Config'].items() if k not in excluded} == {
        k: v for k, v in reference['Config'].items() if k not in excluded}
    expected_host, actual_host = copy.deepcopy(reference['HostConfig']), copy.deepcopy(record['HostConfig'])
    for old, new in ((Q['old_draft_bind'](), Q['candidate_draft_bind']()),
                     (source_bind(S['PARENT']), source_bind(S['CANDIDATE']))):
        assert expected_host['Binds'].count(old) == 1
        expected_host['Binds'] = [new if item == old else item for item in expected_host['Binds']]
    for host in (expected_host, actual_host):
        host['Binds'] = sorted(host.get('Binds') or [])
        assert host.pop('OomKillDisable', None) in (None, False)
    assert actual_host == expected_host
    mounts = Q['_expected_mounts'](reference, cell)
    sources = [m for m in mounts if m['Destination'] == '/src']
    assert len(sources) == 1 and sources[0]['Source'] == str(S['PARENT'])
    assert sources[0]['Type'] == 'bind' and sources[0]['RW'] is False
    sources[0]['Source'] = str(S['CANDIDATE'])
    assert c['canonical_mounts'](record['Mounts']) == c['canonical_mounts'](mounts)
    return True


def bind():
    runtime, analysis, c = Q['bind']()
    # Only the Q8 adapter's private contract map changes; P and its C are untouched.
    c.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX)
    base_make = c['make_create']
    base_code = c['adapter_code_hashes']
    base_hashes = c['source_binary_hashes']
    base_evidence = c['source_evidence']
    base_runtime_check = analysis['check_runtime_evidence']

    def make_create(cell, h):
        base = base_make(cell, h)
        derived = replace_source_bind(base) if c['role_for'](cell) == 'W3' else base
        assert c['server_argv'](derived) == c['server_argv'](base)
        return derived

    def code_hashes():
        return dict(q8_adapter=base_code(), profile_adapter={
            str(HERE / name): c['sha'](HERE / name) for name in (*OWN_FILES, *BUILD_FILES)})

    def source_binary_hashes(h):
        return dict(q8_parent=base_hashes(h), profile=S['proof'](c['_public']()))

    def source_evidence(h):
        return dict(q8_parent=base_evidence(h), profile=S['proof'](c['_public']()))

    class ProfileRuntime(runtime['Runtime']):
        def pid(self, cell, phase):
            # Preserve the existing PID argv/diagnostic validation and observation.
            record = super().pid(cell, phase)
            record['profile_environment'] = json.loads(self.command(
                'docker', 'exec', self._cell_name(cell), 'python3', '-B', '-c', PROFILE_PID_PROBE))
            record['profile_environment_origin'] = '/proc/1/environ'
            c['save'](self.out / cell / ('pid-' + phase + '.json'), record)
            check_profile_pid(record, cell)
            return record

    def check_runtime_evidence(p, h, out, cell):
        result = base_runtime_check(p, h, out, cell)
        for phase in ('before', 'after'):
            check_profile_pid(c['load'](out / cell / ('pid-' + phase + '.json')), cell)
        return result

    c.update(make_create=make_create,
             check_container=lambda record, cell, h: check_container(c, record, cell, h),
             adapter_code_hashes=code_hashes, source_binary_hashes=source_binary_hashes,
             source_evidence=source_evidence)
    runtime['Runtime'] = ProfileRuntime
    analysis['analyze'].__globals__['check_runtime_evidence'] = check_runtime_evidence
    analysis['check_runtime_evidence'] = check_runtime_evidence
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-profile', action='store_true')
    action.add_argument('--lead-after-profile-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified ancestry must not be rebound'
    if args.lead_run_profile:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored']
        print('LILICORR_PROFILE_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        profile = runpy.run_path(str(HERE / 'analyze-profile.py'))['analyze'](OUT, c)
        assert profile['status'] == 'analyzed', profile['errors']
        print('LILICORR_PROFILE_ANALYZED_RESTORED_DEVELOPMENT_ONLY_NO_SPEED_CLAIM', flush=True)


if __name__ == '__main__':
    main()
