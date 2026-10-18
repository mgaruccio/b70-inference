#!/usr/bin/env python3
"""Lead-only paired, seeded native diagnosis. Never a qualification waiver."""
import argparse
import json
import re
import runpy
import shlex
import socket
import subprocess
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control')
SOURCE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-verify-phase'
TEST_TREE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-n9-control-r2')
# Reuse the already-built seeded executable; its libraries are never mounted.
TEST_BASE_TREE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1')
BUILD_PROOF = ROOT / 'r2'
BUILD_ATTEMPT = ROOT / 'build-attempt-r2'
P = runpy.run_path(str(SOURCE / 'public-gate.py'))
C, K, N, A = (P[key] for key in ('C', 'K', 'N', 'A'))
N9 = runpy.run_path(str(ROOT / 'n9-equivalence.py'))
OUT = ROOT / 'run-r3'
BACKENDS = {'R8N9R3-PARENT': N['PARENT'], 'R8N9R3-CANDIDATE': N['CANDIDATE']}
TEST_BINARY = TEST_TREE / 'build-sycl/bin/test-backend-ops'
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


def inputs(h):
    # The public candidate and its protected inputs are pinned independently of
    # the new paired proof; the public CPU freeze is intentionally a later gate.
    candidate = P['candidate_proof']()
    N['check_consumed_sources'](N['CANDIDATE'], 'candidate')
    original = (TEST_BASE_TREE / 'tests/test-backend-ops.cpp').read_bytes()
    expected = N9['expected_test_source'](original)
    assert (TEST_TREE / 'tests/test-backend-ops.cpp').read_bytes() == expected
    test_sources = N['source_hashes'](TEST_TREE)
    before, after = N['source_hashes'](TEST_BASE_TREE), test_sources
    assert sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)) == [
        'tests/test-backend-ops.cpp']
    build_binaries = C['load'](BUILD_PROOF / 'candidate-binaries.json')
    test_binaries = {key: N9['sha256_bytes']((TEST_TREE / 'build-sycl/bin' / key).read_bytes()) for key in build_binaries}
    assert test_binaries == build_binaries
    backend_hashes = {
        'parent': N['binaries'](N['PARENT']),
        'candidate': N['binaries'](N['CANDIDATE']),
    }
    # The phase candidate intentionally also changes libllama's graph builder.
    # Its six runtime artifacts are pinned by candidate_proof(), not assumed equal.
    assert (BUILD_ATTEMPT / 'exit-code.txt').read_text().strip() == '0'
    assert 'BOLT_VERIFY_BUILD_PASSED' in (BUILD_ATTEMPT / 'post-build-verification.log').read_text()
    assert 'BOLT_VERIFY_DEPENDENCIES_PASSED' in (BUILD_ATTEMPT / 'runtime-precheck.log').read_text()
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
    for destination, source in (('/src', BACKENDS[cell]), ('/src/build-sycl/bin/test-backend-ops', TEST_BINARY)):
        mounts = [m for m in record['Mounts'] if m['Destination'] == destination]
        assert len(mounts) == 1 and mounts[0]['Source'] == str(source)
        assert mounts[0]['Type'] == 'bind' and mounts[0]['RW'] is False


def analyze_native(raw, exit_code):
    routes = A['route_records'](raw)
    key = ('blk.48.ffn_gate_inp.weight', 496, 0, 9)
    assert len(routes) == 2
    assert all(r['dst_name'] == N9['TARGET_ROUTE'] for r in routes)
    A['assert_router_operations'](routes, key)
    output = N9['marker_bytes'](raw)
    totals = re.findall(r'(?m)^\s*(\d+)/(\d+) tests passed\s*$', raw)
    assert totals in ([('0', '1')], [('1', '1')]), totals
    passed = totals == [('1', '1')]
    assert exit_code == (0 if passed else 1)
    errors = re.findall(r'\[MUL_MAT\] ERR = ([0-9.eE+-]+) > ([0-9.eE+-]+)', raw)
    assert len(errors) == (0 if passed else 1)
    if errors:
        assert float(errors[0][1]) == 1e-7
    assert re.findall(r'(?m)^(\d+)/(\d+) backends passed\s*$', raw) == [('2' if passed else '1', '2')]
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
        assert len(differences) == 2  # Container name and backend /src binding only.
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
    return {stream: (directory / (stream + '.log')).read_bytes() for stream in ('stdout', 'stderr')}


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
        K['capture'](Runtime(h, OUT, 'native-n9-control-r3'), tuple(BACKENDS), lambda rt, cell: [])
    else:
        summary = C['load'](OUT / 'summary.json')
        assert summary['status'] == 'captured' and summary['restored']
        assert summary['completed'] == list(BACKENDS) and not summary['timing_permitted']
        Runtime(h, OUT, 'native-n9-control-r3').verify_restored(OUT / 'independent-restoration.json')
        reports = {}
        raw = {}
        for cell in BACKENDS:
            directory = OUT / cell
            raw[cell] = raw_artifacts(cell)
            combined = raw[cell]['stdout'] + raw[cell]['stderr']
            reports[cell] = analyze_native(combined.decode(), C['load'](directory / 'exit-code.json'))
            assert reports[cell] == C['load'](directory / 'native-result.json')
        images = {C['load'](OUT / cell / 'container-before.json')['Image'] for cell in BACKENDS}
        assert images == {C['load'](BUILD_ATTEMPT / 'image.json')[0]['Id']}
        snapshot = C['load'](OUT / 'inputs-before.json')
        raw_hashes = {cell: {stream: N9['sha256_bytes'](value) for stream, value in artifacts.items()}
                      for cell, artifacts in raw.items()}
        equivalence = N9['make_proof'](
            raw['R8N9R3-PARENT']['stdout'] + raw['R8N9R3-PARENT']['stderr'],
            raw['R8N9R3-CANDIDATE']['stdout'] + raw['R8N9R3-CANDIDATE']['stderr'],
            snapshot['backend_hashes'], snapshot['test_binaries'], snapshot['test_sources'],
            snapshot['controller_sha256'], raw_hashes)
        C['save'](OUT / 'equivalence.json', equivalence)
        C['save'](OUT / 'analysis.json', dict(reports=reports, independently_restored=True,
                  baseline_equivalent=equivalence['baseline_equivalent'], equivalence='equivalence.json',
                  original_random_operands_replayed=False, qualification_waiver=False,
                  timing_permitted=False))
        print(json.dumps({cell: {key: value for key, value in result.items() if key != 'routes'}
                          for cell, result in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
