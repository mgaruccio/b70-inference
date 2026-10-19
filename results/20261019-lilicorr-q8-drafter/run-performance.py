#!/usr/bin/env python3
"""Lead-only Q8-drafter binding for the qualified W3 ABBA runner.

The inherited singleton controller remains the lifecycle owner.  This adapter changes
only fresh container names and the W3 read-only drafter-file bind; all target argv,
public requests, native gates, and restoration checks remain inherited.
"""
import argparse
import copy
import hashlib
import re
import runpy
import socket
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter')
BASE = HERE.parent / '20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing'
QUALIFIED_SOURCE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')
CANDIDATE = Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter/lilicorr-q8-25.gguf')
CANDIDATE_SIZE = 1437595904
CANDIDATE_SHA256 = 'd420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7'
ORIGINAL_SHA256 = '080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194'
DRAFT_DIR = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr/draft')
DRAFT_DEST = '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf'
OUT = ROOT / 'timing-r1'
PREFIX = 'bolt-lilicorr-q8-drafter-timing-r1-'
OWN_FILES = ('README.md', 'run-performance.py', 'test-performance.py')


def file_pin(path, expected_size, expected_sha256):
    """Return a strong regular-file pin, rejecting symlinks and drift while hashing."""
    path = Path(path)
    assert path.is_file() and not path.is_symlink(), ('candidate is not a regular file', path)
    before = path.stat()
    assert stat.S_ISREG(before.st_mode), ('candidate is not a regular file', path)
    assert before.st_size == expected_size, (before.st_size, expected_size)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    assert stat.S_ISREG(after.st_mode) and after.st_size == expected_size
    actual = digest.hexdigest()
    assert actual == expected_sha256, (actual, expected_sha256)
    return dict(path=str(path), size_bytes=after.st_size, sha256=actual)


def candidate_file_proof(path=None, expected_size=None, expected_sha256=None):
    if path is None:
        path = CANDIDATE
    if expected_size is None:
        expected_size = CANDIDATE_SIZE
    if expected_sha256 is None:
        expected_sha256 = CANDIDATE_SHA256
    return file_pin(path, expected_size, expected_sha256)


def old_draft_bind():
    return str(DRAFT_DIR) + ':/draft:ro'


def candidate_draft_bind():
    return str(CANDIDATE) + ':' + DRAFT_DEST + ':ro'


def replace_draft_bind(argv):
    """Replace exactly the old W3 directory bind with the pinned file bind."""
    result = list(argv)
    old, new = old_draft_bind(), candidate_draft_bind()
    assert result.count(old) == 1 and new not in result
    index = result.index(old)
    assert index > 0 and result[index - 1] == '-v'
    result[index] = new
    return result


def _expected_mounts(reference, cell):
    mounts = copy.deepcopy(reference.get('Mounts') or [])
    if cell.startswith('VSPF-W3'):
        draft_mounts = [mount for mount in mounts if mount['Destination'] == '/draft']
        assert len(draft_mounts) == 1
        mount = draft_mounts[0]
        assert mount['Type'] == 'bind'
        assert mount['Source'] == str(DRAFT_DIR) and mount['RW'] is False
        mount['Source'] = str(CANDIDATE)
        mount['Destination'] = DRAFT_DEST
    return mounts


def check_container(c, record, cell, h):
    """Check actual Docker identity and exact candidate bind/mount without rewriting it."""
    reference = c['load'](c['qualified_dir'](c['role_for'](cell)) / 'container-before.json')
    assert record['Name'] == '/' + c['name'](cell)
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert record['Image'] == reference['Image']
    assert record['Config']['Cmd'] == c['make_create'](cell, h)[-3:]
    assert c['_env_map'](record['Config']['Env']) == c['_env_map'](reference['Config']['Env'])
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {key: value for key, value in record['Config'].items() if key not in excluded} == {
        key: value for key, value in reference['Config'].items() if key not in excluded}

    expected_host = copy.deepcopy(reference['HostConfig'])
    actual_host = copy.deepcopy(record['HostConfig'])
    if cell.startswith('VSPF-W3'):
        old, new = old_draft_bind(), candidate_draft_bind()
        assert expected_host['Binds'].count(old) == 1
        expected_host['Binds'] = [new if item == old else item
                                  for item in expected_host['Binds']]
    for host in (expected_host, actual_host):
        host['Binds'] = sorted(host.get('Binds') or [])
        oom = host.pop('OomKillDisable', None)
        assert oom is None or oom is False
    assert actual_host == expected_host
    assert c['canonical_mounts'](record['Mounts']) == c['canonical_mounts'](
        _expected_mounts(reference, cell))
    if cell.startswith('VSPF-W3'):
        actual_draft = [mount for mount in record['Mounts'] if mount['Destination'] == DRAFT_DEST]
        assert len(actual_draft) == 1
        assert actual_draft[0]['Source'] == str(CANDIDATE)
        assert actual_draft[0]['RW'] is False
    return True


def bind():
    """Configure only freshly loaded private namespaces, never qualified P globals."""
    runtime = runpy.run_path(str(BASE / 'controller.py'))
    # runpy returns a namespace copy. All contract functions share this C globals map,
    # and inherited runtime methods must receive the same private map.
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
        candidate_pin = candidate_file_proof()
        own = {str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES}
        return dict(inherited, **own, candidate_file=candidate_pin)

    def source_binary_hashes(h):
        return dict(qualified_parent=base_hashes(h), candidate_file=candidate_file_proof())

    def source_evidence(h):
        return dict(qualified_parent=base_evidence(h), candidate_file=candidate_file_proof())

    def make_create(cell, h):
        # The inherited singleton make_create already derives the fresh name from
        # this adapter's private PREFIX.  Only W3 receives the candidate bind.
        base = list(base_make(cell, h))
        derived = list(base)
        if c['role_for'](cell) == 'W3':
            derived = replace_draft_bind(derived)
        else:
            assert old_draft_bind() not in derived
            assert candidate_draft_bind() not in derived
        assert c['server_argv'](base) == c['server_argv'](derived)
        expected = [] if c['role_for'](cell) == 'A' else [
            dict(index=derived.index(candidate_draft_bind()),
                 base=old_draft_bind(), derived=candidate_draft_bind())]
        assert c['argv_diff'](base, derived) == expected
        return derived

    c.update(
        make_create=make_create,
        check_container=lambda record, cell, h: check_container(c, record, cell, h),
        adapter_code_hashes=code_hashes,
        source_binary_hashes=source_binary_hashes,
        source_evidence=source_evidence,
        candidate_file_proof=candidate_file_proof,
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
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == QUALIFIED_SOURCE, 'do not rebind qualified source ancestry'
    if args.lead_run_timing:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored']
        print('Q8_DRAFTER_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        print('Q8_DRAFTER_ANALYZED_RESTORED_DEVELOPMENT_ONLY', flush=True)


if __name__ == '__main__':
    main()
