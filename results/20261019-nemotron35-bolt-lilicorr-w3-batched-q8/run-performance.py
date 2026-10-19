#!/usr/bin/env python3
"""Lead-only candidate binding for the existing, exercised W3 ABBA runner."""
import argparse
import copy
import json
import re
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-batched-q8')
BASE = HERE.parent / '20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-batched-q8-r1')
OUT = ROOT / 'timing-r1'
PREFIX = 'bolt-lilicorr-q8-batched-timing-r1-'
OWN_FILES = ('batched-q8.patch', 'source-hashes.json', 'check-source.py',
             'build-candidate.sh', 'test-contract.py', 'README.md',
             'run-performance.py', 'test-performance.py')


def candidate_proof():
    """Read actual source/build evidence; no native/public claim from hashes."""
    assert CANDIDATE.is_dir() and not CANDIDATE.is_symlink()
    assert (HERE / 'build-attempt-r1/exit-code.txt').read_text().strip() == '0'
    gate = runpy.run_path(str(HERE / 'check-source.py'))
    source = gate['check'](PARENT, CANDIDATE)
    phase = runpy.run_path(str(HERE.parent /
        '20261018-nemotron35-bolt-lilicorr-router-verify-phase/run-trace.py'))
    before, after = phase['source_hashes'](PARENT), phase['source_hashes'](CANDIDATE)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert changed == gate['SOURCE_SCOPE'], changed
    binaries = {}
    for label, path in (('parent', PARENT), ('candidate', CANDIDATE)):
        actual = phase['binaries'](path)
        expected = json.loads((HERE / (label + '-binaries.json')).read_text())
        assert set(actual) == set(phase['ARTIFACTS'])
        assert actual == expected, label + ' binary drift'
        binaries[label] = actual
    delta = sorted(k for k in binaries['parent']
                   if binaries['parent'][k] != binaries['candidate'][k])
    assert delta == ['libggml-sycl.so.0'], delta
    assert json.loads((HERE / 'changed-binaries.json').read_text()) == delta
    return dict(parent=str(PARENT), candidate=str(CANDIDATE), source=source,
                changed_source_files=changed, binaries=binaries, changed_binaries=delta)


def replace_source_bind(argv):
    """The only candidate launch change beyond the inherited fresh name."""
    result = list(argv)
    old, new = str(PARENT) + ':/src:ro', str(CANDIDATE) + ':/src:ro'
    assert result.count(old) == 1 and new not in result
    index = result.index(old)
    assert result[index - 1] == '-v'
    result[index] = new
    return result


def check_container(c, record, cell, h):
    """Compare actual Docker data to an explicitly derived expected binding."""
    reference = c['load'](c['qualified_dir'](c['role_for'](cell)) / 'container-before.json')
    assert record['Name'] == '/' + c['name'](cell)
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert record['Image'] == reference['Image']
    assert record['Config']['Cmd'] == c['make_create'](cell, h)[-3:]
    assert c['_env_map'](record['Config']['Env']) == c['_env_map'](reference['Config']['Env'])
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {k: v for k, v in record['Config'].items() if k not in excluded} == {
        k: v for k, v in reference['Config'].items() if k not in excluded}
    expected_host = copy.deepcopy(reference['HostConfig'])
    actual_host = copy.deepcopy(record['HostConfig'])
    old, new = str(PARENT) + ':/src:ro', str(CANDIDATE) + ':/src:ro'
    assert expected_host['Binds'].count(old) == 1
    expected_host['Binds'] = [new if item == old else item for item in expected_host['Binds']]
    for host in (expected_host, actual_host):
        host['Binds'] = sorted(host.get('Binds') or [])
        oom = host.pop('OomKillDisable', None)
        assert oom is None or oom is False
    assert actual_host == expected_host
    expected_mounts = copy.deepcopy(reference['Mounts'])
    source_mounts = [m for m in expected_mounts if m['Destination'] == '/src']
    assert len(source_mounts) == 1
    assert source_mounts[0]['Source'] == str(PARENT) and source_mounts[0]['RW'] is False
    source_mounts[0]['Source'] = str(CANDIDATE)
    assert c['canonical_mounts'](record['Mounts']) == c['canonical_mounts'](expected_mounts)
    return True


def bind():
    """Configure only freshly loaded private namespaces, never frozen P globals."""
    runtime = runpy.run_path(str(BASE / 'controller.py'))
    # runpy returns a copy. All contract functions must share their execution
    # globals, and both the derived and inherited runtime must receive that map.
    c = runtime['C']['configure'].__globals__
    runtime['Runtime'].__init__.__globals__['C'] = c
    runtime['C'] = c
    base_make = c['make_create']
    base_code = c['adapter_code_hashes']
    base_hashes = c['source_binary_hashes']
    base_evidence = c['source_evidence']
    c.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX)

    def code_hashes():
        inherited = base_code()
        pinned = c['load'](BASE / 'run-r3/adapter-code-hashes-before.json')
        assert inherited == pinned, 'completed baseline benchmark code drift'
        return dict(inherited, **{str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES})

    c.update(
        make_create=lambda cell, h: replace_source_bind(base_make(cell, h)),
        check_container=lambda record, cell, h: check_container(c, record, cell, h),
        adapter_code_hashes=code_hashes,
        source_binary_hashes=lambda h: dict(qualified_parent=base_hashes(h), candidate=candidate_proof()),
        source_evidence=lambda h: dict(qualified_parent=base_evidence(h), candidate=candidate_proof()),
    )
    runtime['_bind_inherited_contract_namespace'](runtime['_BOOT_K'])
    analysis = runpy.run_path(str(BASE / 'analyze-timing.py'))
    analysis['analyze'].__globals__['C'] = c
    analysis['C'] = c
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-timing', action='store_true')
    action.add_argument('--lead-after-timing-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert HERE == ROOT
    runtime, analysis, c = bind()
    p, k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == PARENT, 'do not rebind qualification ancestry'
    if args.lead_run_timing:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored']
        print('BATCHED_Q8_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        print('BATCHED_Q8_ANALYZED_RESTORED_DEVELOPMENT_ONLY', flush=True)


if __name__ == '__main__':
    main()
