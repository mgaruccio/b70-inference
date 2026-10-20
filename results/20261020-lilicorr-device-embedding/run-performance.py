#!/usr/bin/env python3
"""Lead-only CLEAN original-Q8 / placement-candidate captures; one mode per process."""
import argparse
import copy
import re
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-device-embedding')
Q = runpy.run_path(str(HERE.parent / '20261019-lilicorr-q8-drafter/run-performance.py'))
S = runpy.run_path(str(HERE / 'source-proof.py'))
OWN_FILES = (
    'README.md', '.gitignore', 'run-performance.py', 'source-proof.py',
    'build-candidate.sh', 'build-candidate-r2.sh', 'build-module-r2.py', 'build-r2-inputs.json.gz',
    'test-device-embedding.py', 'test-build-r2.py', 'source-hashes.json',
    'device-embedding.patch', 'fixtures/embedding-driver.cpp', 'fixtures/support.tar.gz',
    *(str(Path('fixtures') / (n + '.gz')) for n in S['SOURCE_SCOPE']),
    '../20261019-lilicorr-q8-drafter/run-performance.py',
    '../20261019-lilicorr-b70-profile/source-proof.py',
)
BUILD_FILES = tuple('build-attempt-r2/' + name for name in (
    'candidate-binaries.json', 'parent-binaries.json', 'source-diff.json',
    'post-build-verification.log', 'runtime-precheck.log', 'exit-code.txt', 'image.json',
    'seed-built-before.json', 'seed-built-after.json',
    'candidate-source-before.json', 'candidate-source-after.json',
    'build.log', 'module-objects.json', 'module-link.argv.json',
    'module-plan.stdout.log', 'module-build.stdout.log', 'module-commands.stdout.log',
))
PLACEMENT = re.compile(r'lilicorr_device_embedding: active backend=(SYCL\d+) buffer=(SYCL\d+) '
                       r'type=q8_0 tensor_bytes=(\d+) buffer_bytes=(\d+)')


def source_bind(source):
    return str(source) + ':/src:ro'


def replace_source_bind(argv):
    result = list(argv)
    old, new = source_bind(S['PARENT']), source_bind(S['CANDIDATE'])
    assert result.count(old) == 1 and new not in result
    index = result.index(old)
    assert result[index - 1] == '-v'
    result[index] = new
    reverse = list(result)
    reverse[index] = old
    assert reverse == list(argv)
    return result


def check_placement(text, active):
    lines = [line for line in text.splitlines() if 'lilicorr_device_embedding:' in line]
    if not active:
        assert not lines, 'original source/A must not contain candidate activation'
        return dict(active=False)
    assert len(lines) == 1, 'missing, repeated, or inactive placement: do not time as optimized'
    match = PLACEMENT.search(lines[0])
    assert match, lines[0]
    backend, buffer, tensor_bytes, buffer_bytes = match.groups()
    assert backend == buffer == 'SYCL0'
    assert int(tensor_bytes) == 570425344  # Qualified shared Q8 table: 544 MiB.
    assert int(buffer_bytes) >= int(tensor_bytes)
    return dict(active=True, backend=backend, buffer=buffer,
                tensor_bytes=int(tensor_bytes), buffer_bytes=int(buffer_bytes))


def check_acceptance(row):
    if row['cell'].startswith('VSPF-W3'):
        expected = {'short': (129, 84), 'long': (133, 82)}[row['point']]
        actual = tuple(row['timings'][key] for key in ('draft_n', 'draft_n_accepted'))
        assert actual == expected, (row['cell'], row['point'], actual, expected)
    return row


def check_container(c, record, cell, h, mode):
    if mode == 'baseline' or c['role_for'](cell) == 'A':
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


def bind(mode):
    assert mode in ('baseline', 'candidate')
    runtime, analysis, c = Q['bind']()  # Fresh private namespaces; never change qualified P.
    out = ROOT / (mode + '-r1')
    prefix = 'bolt-lilicorr-device-embedding-' + mode + '-r1-'
    c.update(ROOT=ROOT, OUT=out, PREFIX=prefix)
    base_make = c['make_create']
    base_code = c['adapter_code_hashes']
    base_hashes = c['source_binary_hashes']
    base_evidence = c['source_evidence']
    base_runtime_check = analysis['check_runtime_evidence']
    base_measurement = analysis['reread_measurement']

    def make_create(cell, h):
        base = base_make(cell, h)
        derived = replace_source_bind(base) if mode == 'candidate' and c['role_for'](cell) == 'W3' else base
        assert c['server_argv'](derived) == c['server_argv'](base)
        return derived

    def code_hashes():
        files = OWN_FILES + (BUILD_FILES if mode == 'candidate' else ())
        return dict(q8_adapter=base_code(), placement_adapter={
            str(HERE / name): c['sha'](HERE / name) for name in files})

    def source_binary_hashes(h):
        result = dict(q8_parent=base_hashes(h))
        if mode == 'candidate':
            result['placement'] = S['proof'](c['_public']())
        return result

    def source_evidence(h):
        result = dict(q8_parent=base_evidence(h))
        if mode == 'candidate':
            result['placement'] = S['proof'](c['_public']())
        return result

    def placement(out, cell):
        directory = out / cell
        text = ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                       for stream in ('stdout', 'stderr'))
        return check_placement(text, mode == 'candidate' and c['role_for'](cell) == 'W3')

    class PlacementRuntime(runtime['Runtime']):
        def begin(self, cell):
            super().begin(cell)
            # Before any public/timed request; reuse the actual inherited startup logs.
            c['save'](self.out / cell / 'startup-placement.json', placement(self.out, cell))

    def check_runtime_evidence(p, h, out, cell):
        result = base_runtime_check(p, h, out, cell)
        assert c['load'](out / cell / 'startup-placement.json') == placement(out, cell)
        return result

    def reread_measurement(p, h, out, job):
        return check_acceptance(base_measurement(p, h, out, job))

    c.update(make_create=make_create,
             check_container=lambda record, cell, h: check_container(c, record, cell, h, mode),
             adapter_code_hashes=code_hashes, source_binary_hashes=source_binary_hashes,
             source_evidence=source_evidence)
    runtime['Runtime'] = PlacementRuntime
    for key, value in [('check_runtime_evidence', check_runtime_evidence),
                       ('reread_measurement', reread_measurement)]:
        analysis['analyze'].__globals__[key] = value
        analysis[key] = value
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('baseline', 'candidate'))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-timing', action='store_true')
    action.add_argument('--lead-after-timing-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, analysis, c = bind(args.mode)
    out = c['OUT']
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == S['PARENT'], 'qualified source ancestry must remain original'
    if args.lead_run_timing:
        out.mkdir(exist_ok=False)
        (out / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, out))
        assert result['status'] == 'captured' and result['restored']
        print('DEVICE_EMBEDDING_' + args.mode.upper() + '_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        # MUST be a new invocation after EACH capture: original identity/source/health/sole owner.
        runtime['Runtime'](h, out).verify_restored(out / 'analysis-live-restoration.json')
        (out / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](out, p, h)
        assert result['status'] == 'analyzed', result['errors']
        print('DEVICE_EMBEDDING_' + args.mode.upper() + '_ANALYZED_RESTORED_DEVELOPMENT_ONLY', flush=True)

if __name__ == '__main__':
    main()
