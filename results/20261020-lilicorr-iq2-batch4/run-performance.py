#!/usr/bin/env python3
"""Lead-only clean ABBA for the private four-token IQ2 candidate."""
import argparse
import copy
import re
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-iq2-batch4')
OUT = ROOT / 'candidate-r1'
PREFIX = 'bolt-lilicorr-iq2-batch4-candidate-r1-'
Q = runpy.run_path(str(HERE.parent / '20261019-lilicorr-q8-drafter/run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
N = runpy.run_path(str(HERE / 'native-batch4.py'))
QUALIFIER_BIND = str(ROOT) + ':/iq2-batch4:ro'
OLD_QUALIFIER = 'python3 -B /verify-singletons/native.py --kind "$kind"'
NEW_QUALIFIER = 'python3 -B /iq2-batch4/native-batch4.py --kind "$kind"'
OWN_FILES = (
    'README.md', 'run-performance.py', 'test-performance.py',
    'native-batch4.py', 'test-native-batch4.py', 'iq2-batch4.patch',
    'source-pins.sha256', 'source-proof.py', 'test-candidate.py',
    'build-candidate.sh', 'build-module.py', 'build-inputs.json.gz', 'test-build.py',
)
BUILD_FILES = tuple('build-attempt-r1/' + name for name in (
    'candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
    'post-build-verification.log', 'runtime-precheck.log', 'exit-code.txt', 'image.json',
    'seed-built-before.json', 'seed-built-after.json',
    'candidate-source-before.json', 'candidate-source-after.json',
    'build.log', 'module-objects.json', 'module-link.argv.json',
    'module-plan.stdout.log', 'module-build.stdout.log', 'module-build.stderr.log',
    'module-commands.stdout.log', 'queries.stdout.log', 'targets.stdout.log',
    'generated-commands.stdout.log', 'compiler-version.stdout.log',
    'ninja-version.stdout.log', 'cmake-version.stdout.log',
))


def source_bind(source):
    return str(source) + ':/src:ro'


def derive_candidate_argv(argv):
    result = list(argv)
    old, new = source_bind(S['PARENT']), source_bind(S['CANDIDATE'])
    assert result[-3:-1] == ['bash', '-lc']
    assert result.count(old) == 1 and new not in result and QUALIFIER_BIND not in result
    index = result.index(old)
    assert result[index - 1] == '-v'
    assert result[-1].count(OLD_QUALIFIER) == 1 and NEW_QUALIFIER not in result[-1]
    result[index] = new
    result[-1] = result[-1].replace(OLD_QUALIFIER, NEW_QUALIFIER)
    # Docker options must precede IMAGE, not bash.
    mount_index = len(result) - 4
    result[mount_index:mount_index] = ['-v', QUALIFIER_BIND]
    reverse = list(result)
    del reverse[mount_index:mount_index + 2]
    reverse[index] = old
    reverse[-1] = reverse[-1].replace(NEW_QUALIFIER, OLD_QUALIFIER)
    assert reverse == list(argv), 'unintended source/qualifier argv delta'
    return result


def candidate_proof(load):
    value = S['proof']()
    attempt = HERE / 'build-attempt-r1'
    before = load(attempt / 'candidate-source-before.json')
    after = load(attempt / 'candidate-source-after.json')
    assert before == after
    assert value['source'] == after['candidate'], 'source differs from completed build'
    assert S['source_map'](S['PARENT']) == after['parent']
    assert value['binaries'] == load(attempt / 'candidate-binaries.json')
    assert value['parent_binaries'] == load(attempt / 'parent-binaries.json')
    return value


def check_acceptance(row):
    if row['cell'].startswith('VSPF-W3'):
        expected = {'short': (129, 84), 'long': (133, 82)}[row['point']]
        actual = tuple(row['timings'][key] for key in ('draft_n', 'draft_n_accepted'))
        assert actual == expected, (row['cell'], row['point'], actual, expected)
    return row


def check_native(public, tag, log):
    assert tag in public['NATIVE_LOGS']
    return N['FROZEN']['check'](log, tag) if tag == 'iq2' else public['check_native_log'](tag, log)


def check_container(c, record, cell, h):
    if c['role_for'](cell) == 'A':
        return Q['check_container'](c, record, cell, h)
    reference = c['load'](c['qualified_dir']('W3') / 'container-before.json')
    assert record['Name'] == '/' + c['name'](cell)
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert record['Image'] == reference['Image']
    assert record['Config']['Cmd'] == c['make_create'](cell, h)[-3:]
    assert c['_env_map'](record['Config']['Env']) == c['_env_map'](reference['Config']['Env'])
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {k: v for k, v in record['Config'].items() if k not in excluded} == {
        k: v for k, v in reference['Config'].items() if k not in excluded}
    expected, actual = copy.deepcopy(reference['HostConfig']), copy.deepcopy(record['HostConfig'])
    for old, new in ((Q['old_draft_bind'](), Q['candidate_draft_bind']()),
                     (source_bind(S['PARENT']), source_bind(S['CANDIDATE']))):
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
    mounts.append(dict(Type='bind', Source=str(ROOT), Destination='/iq2-batch4',
                       Mode='ro', RW=False, Propagation='rprivate'))
    assert c['canonical_mounts'](record['Mounts']) == c['canonical_mounts'](mounts)
    return True


def bind():
    runtime, analysis, c = Q['bind']()  # Fresh private namespaces; qualified P is never changed.
    c.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX)
    base_make, base_code = c['make_create'], c['adapter_code_hashes']
    base_hashes, base_evidence = c['source_binary_hashes'], c['source_evidence']
    base_runtime_check, base_measurement = analysis['check_runtime_evidence'], analysis['reread_measurement']

    def make_create(cell, h):
        base = base_make(cell, h)
        result = derive_candidate_argv(base) if c['role_for'](cell) == 'W3' else base
        assert c['server_argv'](result) == c['server_argv'](base)
        return result

    def code_hashes():
        return dict(q8_adapter=base_code(), iq2_batch4={
            str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES + BUILD_FILES})

    def source_binary_hashes(h):
        return dict(q8_parent=base_hashes(h), iq2_batch4=candidate_proof(c['load']))

    def source_evidence(h):
        return dict(q8_parent=base_evidence(h), iq2_batch4=candidate_proof(c['load']))

    class BatchRuntime(runtime['Runtime']):
        def preserve_native(self, cell, phase, required=False):
            if c['role_for'](cell) == 'A':
                return super().preserve_native(cell, phase, required=required)
            # Inherit the exact raw-copy/error preservation path; change only the IQ2 checker.
            super().preserve_native(cell, phase, required=False)
            if required:
                directory = self.out / cell
                assert not c['load'](directory / (phase + '-native-copy.json'))['errors']
                public = c['_public']()
                for tag, filename in public['NATIVE_LOGS'].items():
                    check_native(public, tag, (directory / (phase + '-' + filename)).read_bytes().decode())

    def check_runtime_evidence(p, h, out, cell):
        if c['role_for'](cell) == 'A':
            return base_runtime_check(p, h, out, cell)
        private = dict(p)
        private['check_native_log'] = lambda tag, log: check_native(p, tag, log)
        return base_runtime_check(private, h, out, cell)

    def reread_measurement(p, h, out, job):
        return check_acceptance(base_measurement(p, h, out, job))

    c.update(make_create=make_create,
             check_container=lambda record, cell, h: check_container(c, record, cell, h),
             adapter_code_hashes=code_hashes, source_binary_hashes=source_binary_hashes,
             source_evidence=source_evidence)
    runtime['Runtime'] = BatchRuntime
    for key, value in (('check_runtime_evidence', check_runtime_evidence),
                       ('reread_measurement', reread_measurement)):
        analysis['analyze'].__globals__[key] = value
        analysis[key] = value
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-timing', action='store_true')
    action.add_argument('--lead-after-timing-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified ancestry must remain original'
    if args.lead_run_timing:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored'], result
        print('IQ2_BATCH4_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        print('IQ2_BATCH4_ANALYZED_RESTORED_DEVELOPMENT_ONLY', flush=True)


if __name__ == '__main__':
    main()
