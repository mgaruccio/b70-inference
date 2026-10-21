#!/usr/bin/env python3
"""Bounded clean/traced reproduction of the first native gate, not serving attribution."""
import argparse
import copy
import hashlib
import json
import runpy
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-profile.py'))
ROOT = Path('/home/mike/b70-evals/20261021-lilicorr-serving-profile')
OUT = ROOT / 'native-pilot-240-r1'
PREFIX = 'bolt-lilicorr-native-pilot-240-r1-'
TOOLS = Path('/home/mike/needle-jax-probe/tools/unitrace-2.4.0')
HASHES = {'bin/unitrace': '8945080853cce50d87be4b73982c261fbe204475139cbd7ed0278666e51e5fcc',
          'lib/libunitrace_tool.so': 'c55e5df84cbcd4215d0756e775cc71fdaeb762343a6c196d5e4dac02d05af3bb'}
CELL = 'VSPF-W3a'
BINDS = ((TOOLS, '/profiler240', 'ro'), (OUT / 'raw', '/profile', 'rw'))
COMPARATOR_PATH = next(HERE.parent / name / 'verify-native-perf-restored.py'
    for name in ('20261020-lilicorr-iq2-packed', '20261020-lilicorr-iq2-packed-r2')
    if (HERE.parent / name / 'verify-native-perf-restored.py').is_file())
OOM_COMPARATOR = runpy.run_path(str(COMPARATOR_PATH))['native_oom_comparator']
BODY = r'''
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
export BOLT_VERIFY_CAPTURE_DIR=/tmp
export GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_OPT=1
/profiler240/bin/unitrace --version > /profile/version.txt
ldd /profiler240/lib/libunitrace_tool.so > /profile/dependencies.txt
! grep -F 'not found' /profile/dependencies.txt
printf 'FIRST_NATIVE_GATE_CLEAN_BEGIN\n'
/src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p nvfp4_reorder=1 > /profile/clean.log 2>&1
grep -Fx '  56/56 tests passed' /profile/clean.log
! grep -Ei 'not supported|skipping large tensors' /profile/clean.log
grep -F 'Calling reorder_mul_mat_vec_nvfp4_q8_1_sycl' /profile/clean.log > /dev/null
printf 'FIRST_NATIVE_GATE_TRACED_BEGIN\n'
export ZE_ENABLE_TRACING_LAYER=1
unset UNITRACE_KernelMetrics UNITRACE_ChromeKmdLogging
/profiler240/bin/unitrace -d -h -s -v --chrome-kernel-logging --result-dir /profile/trace /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT -p nvfp4_reorder=1 > /profile/traced.log 2>&1
grep -Fx '  56/56 tests passed' /profile/traced.log
! grep -Ei 'not supported|skipping large tensors' /profile/traced.log
grep -F 'Calling reorder_mul_mat_vec_nvfp4_q8_1_sycl' /profile/traced.log > /dev/null
printf 'FIRST_NATIVE_GATE_PAIRED_PASSED_NOT_SERVING_QUALIFICATION\n'
'''


def tool_proof():
    actual = {name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() for name in HASHES}
    assert actual == HASHES
    return actual


def derive(argv):
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    result[-4:-4] = [value for source, destination, mode in BINDS
                    for value in ('-v', f'{source}:{destination}:{mode}')]
    result[-1] = BODY
    return result


def expected(reference):
    value = copy.deepcopy(reference)
    for source, destination, mode in BINDS:
        assert not any(m['Destination'] == destination for m in value['Mounts'])
        value['HostConfig']['Binds'].append(f'{source}:{destination}:{mode}')
        value['Mounts'].append(dict(Type='bind', Source=str(source), Destination=destination,
                                   Mode=mode, RW=mode == 'rw', Propagation='rprivate'))
    return value


def bind():
    prior, runtime, analysis, c = R['bind']()
    c.update(OUT=OUT, PREFIX=PREFIX)
    old_make, old_check, old_codes, old_hashes = (c[key] for key in
        ('make_create', 'check_container', 'adapter_code_hashes', 'source_binary_hashes'))
    original_load = c['load']
    c['make_create'] = lambda cell, h: derive(old_make(cell, h))
    def check(record, cell, h):
        reference_path = c['qualified_dir']('W3') / 'container-before.json'
        def expected_load(path):
            return expected(original_load(path)) if Path(path) == reference_path else original_load(path)
        saved = copy.deepcopy(record)
        c['load'] = expected_load
        try:
            result = old_check(record, cell, h)
            assert record == saved
            return result
        finally:
            c['load'] = original_load
    c['check_container'] = check
    c['adapter_code_hashes'] = lambda: dict(inherited=old_codes(), pilot={str(path): c['sha'](path) for path in (HERE / 'native-pilot-240.py', COMPARATOR_PATH)})
    c['source_binary_hashes'] = lambda h: dict(inherited=old_hashes(h), profiler240=tool_proof())
    base = runtime['Runtime']
    class Pilot(base):
        def stopped_cells(self):
            path = self.out / CELL / 'container-before.json'
            if not path.exists():
                return super().stopped_cells()
            before = c['load'](path)
            assert before['Name'] == '/' + c['name'](CELL)
            original = c['same_container']
            c['same_container'] = OOM_COMPARATOR(original, before['Id'])
            try:
                return super().stopped_cells()
            finally:
                c['same_container'] = original
        def begin(self, cell):
            assert cell == CELL
            self._stop_original_for_case()
            directory = self.out / cell
            directory.mkdir()
            argv = c['make_create'](cell, self.h)
            c['save'](directory / 'create-argv.json', argv)
            self.created.append(cell)
            ident = self.command(*argv).strip()
            self.active = cell
            before = self.inspect(c['name'](cell))
            assert before['Id'] == ident
            c['check_container'](before, cell, self.h)
            c['save'](directory / 'container-before.json', before)
            try:
                with (directory / 'native.stdout.log').open('w') as out, (directory / 'native.stderr.log').open('w') as err:
                    process = subprocess.run(['docker', 'start', '-a', c['name'](cell)], stdout=out, stderr=err,
                                             timeout=180, check=False)
                c['save'](directory / 'native-exit.json', dict(returncode=process.returncode, deadline_seconds=180))
                assert process.returncode == 0
                for filename in ('clean.log', 'traced.log'):
                    text = (self.out / 'raw' / filename).read_text()
                    assert '  56/56 tests passed' in text
                    assert any('BOLT_ROUTE route=LIBRARY_MKL_F32 ' in line and 'src1_ne1=9 ' in line
                               for line in text.splitlines()), 'failing n9 MKL route not exercised'
                traces = list((self.out / 'raw/trace').rglob('chrome_trace.json'))
                assert len(traces) == 1
                trace = json.loads(traces[0].read_text())
                gpu_events = sum(event.get('ph') == 'X' and event.get('cat') == 'gpu_op'
                                 for event in trace['traceEvents'])
                assert gpu_events > 0, 'no actual GPU trace; numerical pass alone is insufficient'
                c['save'](directory / 'first-gate-validation.json', dict(passed=True, clean_cases=56,
                    traced_cases=56, traced_gpu_events=gpu_events, n9_mkl_observed=True,
                    serving_qualification=False))
            except subprocess.TimeoutExpired:
                c['save'](directory / 'native-exit.json', dict(timed_out=True, deadline_seconds=180))
                raise
            finally:
                c['save'](directory / 'container-after-command.json', self.inspect(c['name'](cell)))
        def end(self, cell):
            self._stop_if_running(c['name'](cell))
            self.save_logs(cell, 'native-final')
            self.active = None
            self.ownership(None)
    return runtime, c, Pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--lead-run', action='store_true')
    mode.add_argument('--lead-verify-restored', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == ROOT
    runtime, c, cls = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == R['Q8']['S']['PARENT']
    if args.lead_run:
        OUT.mkdir(exist_ok=False)
        (OUT / 'raw').mkdir()
        c['save'](OUT / 'declaration.json', dict(diagnostic_only=True, journey='clean then traced exact first native 56-case NVFP4 gate;180s command deadline;no serving claim', tools=tool_proof(), flags=R['TRACE_FLAGS']))
        result = runtime['_BOOT_K']['capture'](cls(h, OUT), (CELL,), lambda *_: [])
        assert result['status'] == 'captured' and result['restored'] and not result['timing_permitted']
        print('PROFILER240_FIRST_GATE_PAIRED_CAPTURED_RESTORED', flush=True)
    else:
        record = cls(h, OUT).verify_restored(OUT / 'independent-restoration-live.json')
        assert record['passed'] and not record['errors']
        print('PROFILER240_PILOT_INDEPENDENT_RESTORATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
