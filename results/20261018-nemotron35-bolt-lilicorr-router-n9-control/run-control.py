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
SOURCE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-columns8'
TEST_TREE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-n9-control-r1')
P = runpy.run_path(str(SOURCE / 'public-gate.py'))
C, K, N, A = (P[key] for key in ('C', 'K', 'N', 'A'))
OUT = ROOT / 'run-r1'
BACKENDS = {'R8N9-PARENT': N['PARENT'], 'R8N9-CANDIDATE': N['CANDIDATE']}
TEST_BINARY = TEST_TREE / 'build-sycl/bin/test-backend-ops'
FILTER = (r'^router_columns8=1,router_single2=1,weight=blk\.48\.ffn_gate_inp\.weight,'
          r'layout=0,type_a=f32,type_b=f32,m=496,n=9,')
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
    P['check_tested'](h)
    assert (ROOT / 'build-attempt-r1/exit-code.txt').read_text().strip() == '0'
    original = (N['CANDIDATE'] / 'tests/test-backend-ops.cpp').read_text()
    old = 'thread_local std::default_random_engine gen(std::random_device{}());'
    new = 'std::default_random_engine gen(static_cast<unsigned int>(0x5A17u + start));'
    assert original.count(old) == 1
    assert (TEST_TREE / 'tests/test-backend-ops.cpp').read_text() == original.replace(old, new)
    before, after = N['source_hashes'](N['CANDIDATE']), N['source_hashes'](TEST_TREE)
    assert sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)) == ['tests/test-backend-ops.cpp']
    binaries, test_binaries = N['binaries'](N['CANDIDATE']), N['binaries'](TEST_TREE)
    assert [k for k in binaries if binaries[k] != test_binaries[k]] == ['test-backend-ops']
    return dict(candidate=P['candidate_proof'](), test_sources=after, test_binaries=test_binaries,
                controller_sha256=C['sha'](ROOT / 'run-control.py'), seed='0x5A17 + chunk start',
                original_random_operands_replayed=False)


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
    assert all(r['dst_name'] == 'router_columns8-m496-l0-n9' for r in routes)
    A['assert_router_operations'](routes, key)
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
                tolerance=1e-7, routes=routes, original_random_operands_replayed=False,
                raw_output_byte_equality_claimed=False, qualification_waiver=False)


class Runtime(P['Runtime']):
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

    def preserve_native(self, cell, phase, required=False):
        # The inherited restoration already preserves complete docker stdout/stderr.
        # There are no model-era /tmp native fixture files in these native-only arms.
        pass

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
            assert inputs(self.h) == C['load'](self.out / 'inputs-before.json')
            record['seeded_test_inputs_unchanged'] = True
        except Exception as exc:
            record['errors'].append(dict(check='seeded-test-inputs', error=repr(exc)))
            record['passed'] = False
            C['save'](destination, record)
            raise
        C['save'](destination, record)
        return record


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
        K['capture'](Runtime(h, OUT, 'native-n9-control'), tuple(BACKENDS), lambda rt, cell: [])
    else:
        summary = C['load'](OUT / 'summary.json')
        assert summary['status'] == 'captured' and summary['restored']
        assert summary['completed'] == list(BACKENDS) and not summary['timing_permitted']
        Runtime(h, OUT, 'native-n9-control').verify_restored(OUT / 'independent-restoration.json')
        reports = {}
        for cell in BACKENDS:
            directory = OUT / cell
            raw = b''.join((directory / (stream + '.log')).read_bytes() for stream in ('stdout', 'stderr')).decode()
            reports[cell] = analyze_native(raw, C['load'](directory / 'exit-code.json'))
            assert reports[cell] == C['load'](directory / 'native-result.json')
        images = {C['load'](OUT / cell / 'container-before.json')['Image'] for cell in BACKENDS}
        assert images == {C['load'](ROOT / 'build-attempt-r1/image.json')[0]['Id']}
        C['save'](OUT / 'analysis.json', dict(reports=reports, independently_restored=True,
                  original_random_operands_replayed=False, qualification_waiver=False, timing_permitted=False))
        print(json.dumps({cell: {key: value for key, value in result.items() if key != 'routes'}
                          for cell, result in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
