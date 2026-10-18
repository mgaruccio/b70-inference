#!/usr/bin/env python3
"""Lead-only paired, seeded native N=9 control for the fresh callsite candidate.

This is a bounded reuse of the existing N=9 control.  It is not a
qualification waiver, timing run, or CPU-tolerance change.
"""
import argparse
import json
import re
import runpy
import shlex
import socket
import subprocess
from pathlib import Path

if not __debug__:
    raise RuntimeError('do not disable assertions in this experimental gate')

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite/n9')
CASE_ROOT = ROOT.parent
BASE_ROOT = CASE_ROOT.parent
SOURCE = BASE_ROOT / '20261018-nemotron35-bolt-lilicorr-router-verify-phase'
SEEDED_ROOT = BASE_ROOT / '20261018-nemotron35-bolt-lilicorr-router-n9-control'
WIDTH_ROOT = BASE_ROOT / '20261018-nemotron35-bolt-lilicorr-width-sweep'

# The new runtime pair is deliberately explicit.  The seeded test runner and
# its five-artifact/source proof remain the existing read-only control.
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1')
TEST_TREE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-n9-control-r2')
TEST_BASE_TREE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1')
BUILD_PROOF = SEEDED_ROOT / 'r2'
BUILD_ATTEMPT = CASE_ROOT / 'build-attempt-r1'

# Load the frozen width-sweep controller directly so its exception classes and
# lifecycle identity are unchanged.  The phase analyzers are read-only parsers.
K = runpy.run_path(str(WIDTH_ROOT / 'controller.py'))
C = K['C']
N = runpy.run_path(str(SOURCE / 'run-trace.py'))
A = runpy.run_path(str(SOURCE / 'analyze-trace.py'))
N9 = runpy.run_path(str(ROOT / 'n9-equivalence.py'))
CALLSITE = runpy.run_path(str(CASE_ROOT / 'check-source.py'))

OUT = ROOT / 'run-r1'
BACKENDS = {'RCSN9-PARENT': PARENT, 'RCSN9-CANDIDATE': CANDIDATE}
TEST_BINARY = TEST_TREE / 'build-sycl/bin/test-backend-ops'
TEST_ARTIFACTS = ('llama-server', 'libllama-server-impl.so', 'libggml-sycl.so.0',
                  'test-backend-ops', 'libllama-common.so')
FILTER = N9['FILTER']
COMMAND = r'''set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
export GGML_SYCL_ENABLE_DNN=1 GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_ENABLE_VMM=1
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_ENABLE_GRAPH=0
export BOLT_TRACE_TENSORS=0 ONEDNN_VERBOSE=0 GGML_SYCL_DEBUG=1
mkdir /tmp/router-n9-capture
export BOLT_VERIFY_CAPTURE_DIR=/tmp/router-n9-capture
ldd /src/build-sycl/bin/test-backend-ops
exec /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p ''' + shlex.quote(FILTER)


def name(cell):
    assert cell in BACKENDS
    return C['PREFIX'] + cell.lower()


def _manifest(path):
    value = C['load'](path)
    assert set(value) == set(N['ARTIFACTS'])
    return value


def candidate_proof():
    """Read and verify the fresh six-artifact callsite build contract."""
    assert (BUILD_ATTEMPT / 'exit-code.txt').read_text().strip() == '0'
    assert 'BOLT_VERIFY_CALLSITE_BUILD_PASSED' in (CASE_ROOT / 'post-build-verification.log').read_text()
    assert 'BOLT_VERIFY_CALLSITE_DEPENDENCIES_PASSED' in (CASE_ROOT / 'runtime-precheck.log').read_text()

    # The new parent is the already verified phase candidate.  Keep that
    # protected source/binary identity read-only and require the fresh root's
    # manifests to bind both runtime trees to this pair.
    N['check_protected_sources']()
    N['check_protected_binaries']()
    N['check_consumed_sources'](PARENT, 'candidate')
    parent_sources = N['source_hashes'](PARENT)
    assert parent_sources == N['source_hashes'](N['CANDIDATE'])
    CALLSITE['check'](PARENT, CANDIDATE, (CASE_ROOT / 'router-callsite.patch').read_text())
    candidate_sources = N['source_hashes'](CANDIDATE)
    assert sorted(k for k in parent_sources.keys() | candidate_sources.keys()
                  if parent_sources.get(k) != candidate_sources.get(k)) == [
        'src/models/nemotron-h.cpp']

    parent_binaries = N['binaries'](PARENT)
    candidate_binaries = N['binaries'](CANDIDATE)
    assert parent_binaries == _manifest(CASE_ROOT / 'parent-binaries.json')
    assert parent_binaries == C['load'](SOURCE / 'candidate-binaries.json')
    assert candidate_binaries == _manifest(CASE_ROOT / 'candidate-binaries.json')
    assert {name for name in N['ARTIFACTS']
            if parent_binaries[name] != candidate_binaries[name]} == {'libllama.so.0.7.0'}
    assert C['load'](CASE_ROOT / 'source-diff.json')['changed_relative_to_parent'] == [
        'src/models/nemotron-h.cpp']
    return dict(source=candidate_sources, binaries=candidate_binaries)


def inputs(h):
    """Return the unchanged input snapshot schema used by the old N=9 control."""
    candidate = candidate_proof()
    original = (TEST_BASE_TREE / 'tests/test-backend-ops.cpp').read_bytes()
    expected = N9['expected_test_source'](original)
    assert (TEST_TREE / 'tests/test-backend-ops.cpp').read_bytes() == expected
    test_sources = N['source_hashes'](TEST_TREE)
    before = N['source_hashes'](TEST_BASE_TREE)
    assert sorted(k for k in before.keys() | test_sources.keys()
                  if before.get(k) != test_sources.get(k)) == ['tests/test-backend-ops.cpp']
    build_binaries = C['load'](BUILD_PROOF / 'candidate-binaries.json')
    assert set(build_binaries) == set(TEST_ARTIFACTS)
    test_binaries = {
        key: N9['sha256_bytes']((TEST_TREE / 'build-sycl/bin' / key).read_bytes())
        for key in TEST_ARTIFACTS
    }
    assert test_binaries == build_binaries
    backend_hashes = {
        'parent': N['binaries'](PARENT),
        'candidate': N['binaries'](CANDIDATE),
    }
    return dict(candidate=candidate, test_sources=test_sources, test_binaries=test_binaries,
                backend_hashes=backend_hashes,
                controller_sha256=N9['sha256_bytes']((ROOT / 'run-control.py').read_bytes()),
                seed=N9['SEED'])


def argv_for(cell, h):
    argv = C['make_create']('A', h)
    old_mount = str(N['PARENT']) + ':/src:ro'
    assert argv.count(old_mount) == 1
    argv[argv.index('--name') + 1] = name(cell)
    argv[argv.index(old_mount)] = str(BACKENDS[cell]) + ':/src:ro'
    argv[-4:-4] = ['-v', str(TEST_BINARY) + ':/src/build-sycl/bin/test-backend-ops:ro']
    argv[-1] = COMMAND
    return argv


def check_container(record, cell, h):
    assert record['Name'] == '/' + name(cell)
    assert record['Config']['Cmd'] == argv_for(cell, h)[-3:]
    for destination, source in (('/src', BACKENDS[cell]),
                                ('/src/build-sycl/bin/test-backend-ops', TEST_BINARY)):
        mounts = [m for m in record['Mounts'] if m['Destination'] == destination]
        assert len(mounts) == 1 and mounts[0]['Source'] == str(source)
        assert mounts[0]['Type'] == 'bind' and mounts[0]['RW'] is False
        assert 'ro' in mounts[0]['Mode'].split(',')
        assert record['HostConfig']['Binds'].count(str(source) + ':' + destination + ':ro') == 1


def analyze_native(raw, exit_code):
    routes = A['route_records'](raw)
    key = ('blk.48.ffn_gate_inp.weight', 496, 0, 9)
    assert len(routes) == 2
    assert all(r['dst_name'] == N9['TARGET_ROUTE'] for r in routes)
    A['assert_router_operations'](routes, key)
    assert all(r['accum_prec'] == 0 for r in routes)
    output = N9['marker_bytes'](raw)
    totals = re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', raw)
    assert totals in ([('0', '1')], [('1', '1')]), totals
    passed = totals == [('1', '1')]
    assert exit_code == (0 if passed else 1)
    errors = re.findall(r'\[MUL_MAT\] ERR = ([0-9.eE+-]+) > ([0-9.eE+-]+)', raw)
    assert len(errors) == (0 if passed else 1)
    if errors:
        # Preserve the inherited CPU-reference error as observed data.  The
        # shared 1e-7 tolerance remains fixed; it is not a qualification gate.
        assert float(errors[0][1]) == 1e-7
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', raw) == [
        ('2' if passed else '1', '2')]
    return dict(numerical_pass=passed, reported_error=errors[0][0] if errors else None,
                tolerance=1e-7, n9_output_byte_length=len(output), routes=routes,
                original_random_operands_replayed=False,
                raw_output_byte_equality_claimed=False, qualification_waiver=False)


class Runtime(K['Runtime']):
    def preflight(self):
        self.capacity('initial')
        self.input_snapshot = inputs(self.h)
        C['save'](self.out / 'inputs-before.json', self.input_snapshot)
        self.hashes = C['source_binary_hashes'](self.h)
        self.before_hw = self.h['runner']['hardware']()
        assert self.before_hw == C['load'](self.off['REFERENCE'] / 'run-r1/hardware-before.json')
        live = self.inspect(self.original_name)
        assert C['same_container'](C['load'](self.off['ROOT'] / 'run-r1/original-container.json'), live)
        assert live['State']['Running']
        self.original, self.original_was_running = live, True
        C['save'](self.out / 'original-container.json', live)
        C['save'](self.out / 'hash-checks-before.json', self.hashes)
        C['save'](self.out / 'hardware-before.json', self.before_hw)
        assert self.h['api']['request'](self.off['OLD']['URL'], '/health')['status'] == 'ok'
        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        assert not (set(existing) & {name(cell) for cell in BACKENDS})
        pair = [argv_for(cell, self.h) for cell in BACKENDS]
        differences = C['argv_diff'](*pair)
        assert len(differences) == 2
        C['save'](self.out / 'paired-argv-difference.json', differences)
        self.observe('run-before', self.original_name)

    def begin(self, cell):
        self.capacity(cell)
        if not self.prefixes:
            self._stop_original_for_case()
        else:
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
        directory = self.out / cell
        directory.mkdir()
        argv = argv_for(cell, self.h)
        C['save'](directory / 'create-argv.json', argv)
        self.created.append(cell)
        self.command(*argv, timeout=60)
        check_container(self.inspect(name(cell)), cell, self.h)
        self.observe(cell + '-before', None)
        self.command('docker', 'start', name(cell))
        exit_code = int(self.command('docker', 'wait', name(cell), timeout=300).strip())
        C['save'](directory / 'exit-code.json', exit_code)
        raw = subprocess.run(['docker', 'logs', name(cell)], capture_output=True, check=True, timeout=90)
        for stream in ('stdout', 'stderr'):
            (directory / (stream + '.log')).write_bytes(getattr(raw, stream))
        result = analyze_native((raw.stdout + raw.stderr).decode(), exit_code)
        C['save'](directory / 'native-result.json', result)
        live = self.inspect(name(cell))
        check_container(live, cell, self.h)
        assert not live['State']['Running']
        C['save'](directory / 'container-before.json', live)
        self.active = cell

    def end(self, cell):
        self.save_logs(cell, 'server')
        self.observe(cell + '-after', None)
        assert inputs(self.h) == self.input_snapshot
        self.active = None

    def stopped_cells(self):
        result = {}
        for cell in BACKENDS:
            if not (self.out / cell).exists():
                continue
            live = self.inspect(name(cell))
            check_container(live, cell, self.h)
            assert not live['State']['Running']
            before = self.out / cell / 'container-before.json'
            if before.exists():
                assert C['same_container'](C['load'](before), live)
            result[cell] = live
        return result

    def verify_restored(self, destination):
        record = super().verify_restored(destination)
        try:
            record['preserved_cells'] = self.stopped_cells()
            assert inputs(self.h) == C['load'](self.out / 'inputs-before.json')
            record['seeded_test_inputs_unchanged'] = True
        except Exception as exc:
            record['errors'].append(dict(check='seeded-test-inputs', error=repr(exc)))
            record['passed'] = False
            C['save'](destination, record)
            raise
        C['save'](destination, record)
        return record


def raw_artifacts(cell):
    directory = OUT / cell
    return {stream: (directory / (stream + '.log')).read_bytes()
            for stream in ('stdout', 'stderr')}


def _pair_result(snapshot):
    raw = {cell: raw_artifacts(cell) for cell in BACKENDS}
    reports = {}
    for cell in BACKENDS:
        directory = OUT / cell
        combined = raw[cell]['stdout'] + raw[cell]['stderr']
        reports[cell] = analyze_native(combined.decode(), C['load'](directory / 'exit-code.json'))
        assert reports[cell] == C['load'](directory / 'native-result.json')
    images = {C['load'](OUT / cell / 'container-before.json')['Image'] for cell in BACKENDS}
    assert images == {C['load'](BUILD_ATTEMPT / 'image.json')[0]['Id']}
    raw_hashes = {cell: {stream: N9['sha256_bytes'](value)
                         for stream, value in artifacts.items()}
                  for cell, artifacts in raw.items()}
    proof = N9['make_proof'](
        raw['RCSN9-PARENT']['stdout'] + raw['RCSN9-PARENT']['stderr'],
        raw['RCSN9-CANDIDATE']['stdout'] + raw['RCSN9-CANDIDATE']['stderr'],
        snapshot['backend_hashes'], snapshot['test_binaries'], snapshot['test_sources'],
        snapshot['controller_sha256'], raw_hashes)
    return raw, reports, proof


def check_current_equivalence():
    """Read-only public gate for the saved fresh N=9 pair.

    It performs no Docker/API calls and writes no evidence.  The returned dict
    is the existing N9 proof schema, after rechecking current source/binary and
    seeded-runner bindings, route/accumulation controls, raw marker bytes, and
    independent restoration evidence.
    """
    summary = C['load'](OUT / 'summary.json')
    assert summary['status'] == 'captured'
    assert summary['order'] == list(BACKENDS)
    assert summary['completed'] == list(BACKENDS)
    assert summary['restored'] is True
    assert summary['timing_permitted'] is False
    assert summary['promotion_permitted'] is False
    assert summary['raw_f32_equality_claimed'] is False
    assert summary['restoration_errors'] == []

    restoration = C['load'](OUT / 'independent-restoration.json')
    assert restoration['passed'] is True
    assert restoration['errors'] == []
    assert restoration['independently_live'] is True
    assert restoration['seeded_test_inputs_unchanged'] is True

    snapshot = C['load'](OUT / 'inputs-before.json')
    assert inputs(C['helpers']()) == snapshot
    h = C['helpers']()
    current_argv = [argv_for(cell, h) for cell in BACKENDS]
    assert C['argv_diff'](*current_argv) == C['load'](OUT / 'paired-argv-difference.json')
    raw, reports, proof = _pair_result(snapshot)
    saved_proof = C['load'](OUT / 'equivalence.json')
    assert saved_proof == proof
    raw_hashes = {cell: {stream: N9['sha256_bytes'](value)
                         for stream, value in artifacts.items()}
                  for cell, artifacts in raw.items()}
    N9['check_proof'](
        saved_proof,
        raw['RCSN9-PARENT']['stdout'] + raw['RCSN9-PARENT']['stderr'],
        raw['RCSN9-CANDIDATE']['stdout'] + raw['RCSN9-CANDIDATE']['stderr'],
        snapshot['backend_hashes'], snapshot['test_binaries'], snapshot['test_sources'],
        snapshot['controller_sha256'], raw_hashes)
    analysis = C['load'](OUT / 'analysis.json')
    assert analysis['reports'] == reports
    assert analysis['independently_restored'] is True
    assert analysis['baseline_equivalent'] is True
    assert analysis['equivalence'] == 'equivalence.json'
    assert analysis['original_random_operands_replayed'] is False
    assert analysis['qualification_waiver'] is False
    assert analysis['timing_permitted'] is False
    return saved_proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-control', action='store_true')
    action.add_argument('--lead-after-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    if args.lead_run_control:
        inputs(h)
        OUT.mkdir(exist_ok=False)
        K['capture'](Runtime(h, OUT, 'native-n9-control-callsite-r1'), tuple(BACKENDS),
                     lambda rt, cell: [])
    else:
        summary = C['load'](OUT / 'summary.json')
        assert summary['status'] == 'captured' and summary['restored']
        assert summary['completed'] == list(BACKENDS) and not summary['timing_permitted']
        Runtime(h, OUT, 'native-n9-control-callsite-r1').verify_restored(
            OUT / 'independent-restoration.json')
        snapshot = C['load'](OUT / 'inputs-before.json')
        _, reports, equivalence = _pair_result(snapshot)
        C['save'](OUT / 'equivalence.json', equivalence)
        C['save'](OUT / 'analysis.json', dict(
            reports=reports, independently_restored=True,
            baseline_equivalent=equivalence['baseline_equivalent'],
            equivalence='equivalence.json', original_random_operands_replayed=False,
            qualification_waiver=False, timing_permitted=False))
        assert check_current_equivalence() == equivalence
        print(json.dumps({cell: {key: value for key, value in result.items() if key != 'routes'}
                          for cell, result in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
