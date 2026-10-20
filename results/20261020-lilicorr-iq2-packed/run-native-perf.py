#!/usr/bin/env python3
"""Development-only existing backend perf cases; unchanged binaries and exclusive lifecycle."""
import argparse
import copy
import runpy
import socket
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-performance.py'))
OUT = R['ROOT'] / 'native-perf-r1'
PREFIX = 'bolt-lilicorr-iq2-native-perf-r1-'
CELLS = ('VSPF-W3a',)
BASE_BIND = str(R['S']['SEED']) + ':/baseline:ro'
BASE_MOUNT = dict(Type='bind', Source=str(R['S']['SEED']), Destination='/baseline',
                  Mode='ro', RW=False, Propagation='rprivate')
SEQUENCE = ('base', 'packed', 'packed', 'base') * 3
OWN_FILES = ('run-native-perf.py', 'test-native-perf.py', 'native-perf-cases.txt', 'NATIVE_PERF_PLAN.md')
NATIVE_BODY = r'''
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
unset LLAMA_LILICORR_PROFILE SYCL_UR_TRACE SYCL_PI_TRACE LD_DEBUG
export GGML_SYCL_ENABLE_DNN=1 GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_ENABLE_VMM=1
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_ENABLE_GRAPH=0
export BOLT_TRACE_TENSORS=0 ONEDNN_VERBOSE=0
root=/tmp/lilicorr-native-perf
test ! -e "$root"
mkdir "$root"
for variant in base packed; do
 if test "$variant" = base; then bin=/baseline/build-sycl/bin; else bin=/src/build-sycl/bin; fi
 directory="$root/$variant"
 mkdir -p "$directory/capture"
 for kind in q8 iq2; do
  log="$directory/verify-singletons-$kind.log"
  if env LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" GGML_SYCL_DEBUG=1 BOLT_VERIFY_CAPTURE_DIR="$directory/capture" "$bin/test-backend-ops" test -b SYCL0 -p "^verify_singletons=1,case=$kind-" -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
  printf '%s\n' "$rc" >"$log.exit-code"
  test "$rc" = 0
  python3 -B /iq2-batch4/native-batch4.py --kind "$kind" "$log" >"$log.check.json"
 done
 # Host loader/UR call tracing only. No event profiling, queue flags or adapter changes.
 log="$directory/representative-diagnostic.log"
 if env LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" GGML_SYCL_DEBUG=1 LD_DEBUG=libs SYCL_UR_TRACE=2 "$bin/test-backend-ops" test -b SYCL0 -o MUL_MAT_ID --test-file /iq2-batch4/native-perf-cases.txt -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
 printf '%s\n' "$rc" >"$log.exit-code"
 test "$rc" = 0
 python3 -B - "$log" "$bin" <<'PY'
import sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
loaded=[line for line in text.splitlines() if 'calling init:' in line and 'libggml-sycl.so' in line]
assert loaded and all(sys.argv[2]+'/' in line for line in loaded), loaded
assert 'urEnqueueKernelLaunch' in text, 'UR launch trace missing'
assert 'UR_QUEUE_FLAG_PROFILING_ENABLE' not in text, 'unexpected profiling queue'
assert '2/2 tests passed' in text, 'representative CPU comparisons missing'
assert not any(word in text.lower() for word in ('not supported', 'skipping large tensors'))
print('NATIVE_PERF_DLL_KERNEL_ROUTE_AND_TWO_CPU_COMPARISONS_PASSED',sys.argv[2])
PY
done
index=0
for variant in base packed packed base base packed packed base base packed packed base; do
 if test "$variant" = base; then bin=/baseline/build-sycl/bin; else bin=/src/build-sycl/bin; fi
 log="$root/perf-$index-$variant.log"
 if env -u BOLT_VERIFY_CAPTURE_DIR LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" GGML_SYCL_DEBUG=0 "$bin/test-backend-ops" perf -b SYCL0 -o MUL_MAT_ID --test-file /iq2-batch4/native-perf-cases.txt >"$log" 2>&1; then rc=0; else rc=$?; fi
 printf '%s\n' "$rc" >"$log.exit-code"
 test "$rc" = 0
 python3 -B - "$log" >"$log.checked.json" <<'PERF'
import json,math,re,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
lines=[line for line in text.splitlines() if line.lstrip().startswith('MUL_MAT_ID(')]
assert len(lines)==2, ('expected two measured operations',lines)
rows=[]
for name in ('lilicorr-iq2-up-n4','lilicorr-iq2-down-n4'):
 matched=[line for line in lines if 'name='+name+',' in line]
 assert len(matched)==1, (name,matched)
 match=re.search(r'(\d+) runs\s*-\s*(\S+) us/run\s*-',matched[0])
 assert match, ('missing performance result',matched[0])
 runs=int(match[1]);time_us=float(match[2])
 assert runs>0 and math.isfinite(time_us) and time_us>0,(runs,time_us)
 rows.append(dict(name=name,runs=runs,operator_graph_wall_us=time_us))
assert not any(word in text.lower() for word in ('not supported','skipping large tensors'))
print(json.dumps(rows,sort_keys=True))
PERF
 printf 'NATIVE_PERF_FINISHED index=%s variant=%s log=%s\n' "$index" "$variant" "$log"
 index=$((index + 1))
done
echo NATIVE_IQ2_OPERATOR_PERF_CAPTURED_NOT_SERVING_SPEEDUP
'''


def derive(argv):
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    assert result.count(str(R['S']['CANDIDATE']) + ':/src:ro') == 1
    assert BASE_BIND not in result
    index = len(result) - 4
    result[index:index] = ['-v', BASE_BIND]
    result[-1] = NATIVE_BODY
    reverse = list(result)
    del reverse[index:index + 2]
    reverse[-1] = argv[-1]
    assert reverse == list(argv), 'unexpected native argv delta'
    return result


def without_baseline_mount(record):
    result = copy.deepcopy(record)
    assert result['HostConfig']['Binds'].count(BASE_BIND) == 1
    result['HostConfig']['Binds'].remove(BASE_BIND)
    mounts = [m for m in result['Mounts'] if m['Destination'] == '/baseline']
    assert mounts == [BASE_MOUNT], ('baseline mount drift', mounts)
    result['Mounts'].remove(mounts[0])
    return result


def bind():
    runtime, _analysis, c = R['bind']()
    c.update(OUT=OUT, PREFIX=PREFIX)
    capture = runtime['_BOOT_K']['capture']
    assert capture.__globals__['C'] is c
    original_make, original_check = c['make_create'], c['check_container']
    old_code = c['adapter_code_hashes']
    c['make_create'] = lambda cell, host: derive(original_make(cell, host))
    c['check_container'] = lambda record, cell, host: original_check(without_baseline_mount(record), cell, host)
    c['adapter_code_hashes'] = lambda: dict(inherited=old_code(), native_perf={str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES})

    class NativePerf(runtime['Runtime']):
        def preflight(self):
            saved = self.mode
            self.mode = 'timing'
            try:
                return super().preflight()
            finally:
                self.mode = saved

        def begin(self, cell):
            self._stop_original_for_case()
            directory = self.out / cell
            directory.mkdir()
            argv = c['make_create'](cell, self.h)
            c['save'](directory / 'create-argv.json', argv)
            ident = self.command(*argv).strip()
            self.created.append(cell)
            self.active = cell
            before = self.inspect(c['name'](cell))
            assert before['Id'] == ident and before['Config']['Cmd'] == argv[-3:]
            c['check_container'](before, cell, self.h)
            c['save'](directory / 'container-before.json', before)
            with (directory / 'native-perf.stdout.log').open('w') as stdout, (directory / 'native-perf.stderr.log').open('w') as stderr:
                process = subprocess.run(['docker', 'start', '-a', c['name'](cell)], stdout=stdout, stderr=stderr, check=False)
            after = self.inspect(c['name'](cell))
            c['save'](directory / 'container-after.json', after)
            # Preserve raw native/diagnostic/perf outputs before asserting success.
            self.command('docker', 'cp', c['name'](cell) + ':/tmp/lilicorr-native-perf', str(directory / 'raw'))
            assert process.returncode == 0 and not after['State']['Running'] and after['State']['ExitCode'] == 0

        def end(self, cell):
            self._stop_if_running(c['name'](cell))
            self.save_logs(cell, 'native-perf')
            self.active = None
            self.ownership(None)

    return runtime, c, capture, NativePerf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-native-perf', action='store_true')
    action.add_argument('--lead-verify-restored', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == R['ROOT']
    runtime, c, capture, cls = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == R['S']['PARENT'], 'immutable public-gate ancestry changed'
    proof = R['S']['proof']()
    assert proof['source']['tests/test-backend-ops.cpp'] == '3a2167a0b4d9e26feea23ef299e11fe6468baf19be6e0304ebde57ea240e1c6d'
    assert proof['source']['ggml/include/ggml.h'] == '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3'
    control = cls(h, OUT)
    control.mode = 'native-iq2-operator-perf'
    if args.lead_run_native_perf:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'NATIVE_PERF_PLAN.md').read_text())
        result = capture(control, CELLS, lambda _rt, _cell: None)
        assert result['restored'], result
        print('NATIVE_IQ2_PERF_CAPTURED_RESTORED_ANALYSIS_PENDING')
    else:
        control.verify_restored(OUT / 'independent-original-restoration.json')
        print('NATIVE_IQ2_PERF_SEPARATE_ORIGINAL_RESTORATION_PASSED')


if __name__ == '__main__':
    main()
