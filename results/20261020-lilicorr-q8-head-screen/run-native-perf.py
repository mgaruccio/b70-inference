#!/usr/bin/env python3
"""Lead-only Q8 head cost screen; one qualified W3 native container, no serving claim."""
import argparse
import copy
import socket
import subprocess
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUALIFIED_CASE = HERE.parent / '20261020-lilicorr-q8-target-batch4'
R = runpy.run_path(str(QUALIFIED_CASE / 'run-performance.py'))

# Keep the qualified target runner's source/build/protected-config ancestry and
# reference root.  Only the output root, container prefix, and native package
# are new for this bounded screen.
QUALIFIED_HERE = R['HERE']
QUALIFIED_ROOT = R['ROOT']
QUALIFIED_OWN_FILES = tuple(R['OWN_FILES'])
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-q8-head-screen')
OUT = ROOT / 'native-perf-r1'
PREFIX = 'bolt-lilicorr-q8-head-screen-native-perf-r1-'
CELLS = ('VSPF-W3a',)
PACKAGE_DEST = '/q8-head-screen'
PACKAGE_BIND = str(HERE) + ':' + PACKAGE_DEST + ':ro'
PACKAGE_MOUNT = dict(Type='bind', Source=str(HERE), Destination=PACKAGE_DEST,
                     Mode='ro', RW=False, Propagation='rprivate')
BASE_QUALIFIER_BIND = R['QUALIFIER_BIND']
OWN_FILES = ('run-native-perf.py', 'test-native-perf.py',
             'native-perf-cases.txt', 'NATIVE_PERF_PLAN.md')
# The qualified target package remains the read-only source of native-batch4.py;
# this package contributes only the case file/checks and is mounted separately.
R['OUT'] = OUT
R['PREFIX'] = PREFIX
R['OWN_FILES'] = QUALIFIED_OWN_FILES + tuple(str(HERE / name) for name in OWN_FILES)
assert R['HERE'] == QUALIFIED_HERE
assert R['ROOT'] == QUALIFIED_ROOT
assert R['QUALIFIER_BIND'] == BASE_QUALIFIER_BIND

CASE_NAMES = ('q8-head-n1', 'q8-head-n4', 'q8-head-n8')

# This is intentionally a source-supported diagnostic, not a claim that a UR
# launch line names a particular C++ kernel.  Raw logs retain any handles/names.
DIAGNOSTIC_CHECKER = r'''import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
expected = ('q8-head-n1', 'q8-head-n4', 'q8-head-n8')
loaded = [line for line in text.splitlines()
          if 'calling init:' in line and 'libggml-sycl.so' in line]
assert loaded and all(sys.argv[2] + '/libggml-sycl.so' in line for line in loaded), loaded
assert 'urEnqueueKernelLaunch' in text, 'UR launch trace missing'
assert 'UR_QUEUE_FLAG_PROFILING_ENABLE' not in text, 'unexpected profiling queue'
assert re.search(r'(?m)^\s*3/3 tests passed\s*$', text), 'CPU comparisons missing'
assert not re.search(r'(?i)not supported|skipping large tensors', text)
names = re.findall(r'(?m)^\s*MUL_MAT\(name=([^,\n)]+),', text)
assert names == list(expected), names
# Handle spellings vary by UR trace version; preserve them when present but do
# not make a C++ kernel-name claim from this host-level trace.
handles = sorted(set(re.findall(r'(?:hKernel|\.hKernel)\s*[=:]\s*(0x[0-9a-fA-F]+)', text)))
print(json.dumps(dict(loaded_dll=loaded, operation_names=names,
                      ur_launches=text.count('urEnqueueKernelLaunch'),
                      kernel_handles=handles,
                      source_mapping='shape/ABI diagnostic only; no C++ kernel-name claim'),
                 sort_keys=True))
'''

PERF_CHECKER = r'''import json
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
expected = ('q8-head-n1', 'q8-head-n4', 'q8-head-n8')
lines = [line for line in text.splitlines()
         if line.lstrip().startswith('MUL_MAT(')]
assert len(lines) == 3, ('expected three measured operations', lines)
rows = []
for name in expected:
    matched = [line for line in lines if 'name=' + name + ',' in line]
    assert len(matched) == 1, (name, matched)
    match = re.search(r'(\d+) runs\s*-\s*(\S+) us/run\s*-', matched[0])
    assert match, ('missing performance result', matched[0])
    runs = int(match[1])
    time_us = float(match[2])
    assert runs > 0 and math.isfinite(time_us) and time_us > 0, (runs, time_us)
    rows.append(dict(name=name, runs=runs, operator_graph_wall_us=time_us))
assert [row['name'] for row in rows] == list(expected)
assert not re.search(r'(?i)not supported|skipping large tensors', text)
print(json.dumps(rows, sort_keys=True))
'''

NATIVE_BODY = r'''set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
unset LLAMA_LILICORR_PROFILE SYCL_UR_TRACE SYCL_PI_TRACE LD_DEBUG GGML_SYCL_DEBUG
export GGML_SYCL_ENABLE_DNN=1 GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_ENABLE_VMM=1
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_ENABLE_GRAPH=0
export BOLT_TRACE_TENSORS=0 ONEDNN_VERBOSE=0
root=/tmp/lilicorr-q8-head-screen
test ! -e "$root"
mkdir "$root"
mkdir "$root/capture"
bin=/src/build-sycl/bin
for kind in q8 iq2; do
 log="$root/verify-singletons-$kind.log"
 if env LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" GGML_SYCL_DEBUG=1 BOLT_VERIFY_CAPTURE_DIR="$root/capture" "$bin/test-backend-ops" test -b SYCL0 -p "^verify_singletons=1,case=$kind-" -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
 printf '%s\n' "$rc" >"$log.exit-code"
 test "$rc" = 0
 python3 -B /iq2-batch4/native-batch4.py --kind "$kind" "$log" >"$log.check.json"
done
log="$root/representative-diagnostic.log"
if env LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" GGML_SYCL_DEBUG=1 LD_DEBUG=libs SYCL_UR_TRACE=2 "$bin/test-backend-ops" test -b SYCL0 -o MUL_MAT --test-file /q8-head-screen/native-perf-cases.txt -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >"$log.exit-code"
test "$rc" = 0
python3 -B - "$log" "$bin" >"$log.checked.json" <<'DIAG'
''' + DIAGNOSTIC_CHECKER + r'''DIAG
for index in 0 1 2; do
 log="$root/perf-$index.log"
 # The same immutable test-backend-ops executable runs the complete case file
 # three times; default perf mode supplies its built-in warmup/steady timing.
 if env -u LLAMA_LILICORR_PROFILE -u SYCL_UR_TRACE -u SYCL_PI_TRACE -u LD_DEBUG -u GGML_SYCL_DEBUG -u BOLT_VERIFY_CAPTURE_DIR LD_LIBRARY_PATH="$bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}" "$bin/test-backend-ops" perf -b SYCL0 -o MUL_MAT --test-file /q8-head-screen/native-perf-cases.txt >"$log" 2>&1; then rc=0; else rc=$?; fi
 printf '%s\n' "$rc" >"$log.exit-code"
 test "$rc" = 0
 python3 -B - "$log" >"$log.checked.json" <<'PERF'
''' + PERF_CHECKER + r'''PERF
done
python3 -B - "$root" >"$root/perf-summary.json" <<'SUMMARY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for index in range(3):
    rows.extend(json.loads((root / ('perf-%d.log.checked.json' % index)).read_text()))
assert len(rows) == 9, len(rows)
assert [row['name'] for row in rows] == ['q8-head-n1', 'q8-head-n4', 'q8-head-n8'] * 3
print(json.dumps(dict(processes=3, rows=rows, timing_scope='synchronous operator-graph wall time'), sort_keys=True))
SUMMARY
echo NATIVE_Q8_HEAD_SCREEN_PERF_FINISHED rows=9
'''


def derive(argv):
    """Add only the new read-only package and replace the server body."""
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    assert result.count(PACKAGE_BIND) == 0
    # Docker options must precede IMAGE, not bash.
    mount_index = len(result) - 4
    assert mount_index >= 0
    result[mount_index:mount_index] = ['-v', PACKAGE_BIND]
    result[-1] = NATIVE_BODY
    return result


def without_package_mount(record):
    """Project the diagnostic-only mount away before the inherited guard runs."""
    result = copy.deepcopy(record)
    binds = result['HostConfig']['Binds']
    assert binds.count(PACKAGE_BIND) == 1
    binds.remove(PACKAGE_BIND)
    mounts = [mount for mount in result['Mounts'] if mount['Destination'] == PACKAGE_DEST]
    assert mounts == [PACKAGE_MOUNT], ('package mount drift', mounts)
    result['Mounts'].remove(mounts[0])
    return result


def bind():
    """Reuse the qualified target lifecycle; wrap only one W3 native cell."""
    runtime, _analysis, c = R['bind']()
    base_make, base_check = c['make_create'], c['check_container']
    capture = runtime['_BOOT_K']['capture']
    assert capture.__globals__['C'] is c

    def make_create(cell, h):
        assert c['role_for'](cell) == 'W3'
        base = base_make(cell, h)
        result = derive(base)
        assert result.count(PACKAGE_BIND) == 1
        assert BASE_QUALIFIER_BIND in result
        assert '/baseline' not in '\n'.join(result)
        return result

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
            # Copy the raw tree before asserting success so failures retain all
            # available native, diagnostic, and perf logs.
            self.command('docker', 'cp', c['name'](cell) + ':/tmp/lilicorr-q8-head-screen', str(directory / 'raw'))
            assert process.returncode == 0 and not after['State']['Running'] and after['State']['ExitCode'] == 0

        def end(self, cell):
            self._stop_if_running(c['name'](cell))
            self.save_logs(cell, 'native-perf')
            self.active = None
            self.ownership(None)

    c.update(make_create=make_create,
             check_container=lambda record, cell, h: base_check(without_package_mount(record), cell, h))
    return runtime, c, capture, NativePerf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-native-perf', action='store_true')
    action.add_argument('--lead-verify-restored', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    runtime, c, capture, cls = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == R['S']['PARENT'], 'qualified ancestry must remain original'
    proof = R['S']['proof']()
    assert proof['source']['tests/test-backend-ops.cpp'] == '3a2167a0b4d9e26feea23ef299e11fe6468baf19be6e0304ebde57ea240e1c6d'
    assert proof['source']['ggml/include/ggml.h'] == '475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3'
    control = cls(h, OUT)
    control.mode = 'native-q8-head-screen'
    if args.lead_run_native_perf:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'NATIVE_PERF_PLAN.md').read_text())
        result = capture(control, CELLS, lambda _rt, _cell: None)
        assert result['restored'], result
        print('NATIVE_Q8_HEAD_SCREEN_CAPTURED_RESTORED_COST_SCREEN_PENDING')
    else:
        control.verify_restored(OUT / 'independent-original-restoration.json')
        print('NATIVE_Q8_HEAD_SCREEN_SEPARATE_ORIGINAL_RESTORATION_PASSED')


if __name__ == '__main__':
    main()
