#!/usr/bin/env python3
"""Fixed native-only failure discriminator, reusing the existing exclusive lifecycle."""
import argparse
import json
import runpy
import socket
import subprocess
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = runpy.run_path(str(HERE / 'run-profile.py'))
OUT = D['ROOT'] / 'native-router-controls-r1'
PREFIX = 'bolt-lilicorr-native-router-controls-r1-'
CELLS = ('VSPF-A1', 'VSPF-W3a')
MATRIX = tuple((profile, fusion) for profile in ('off', 'on') for fusion in (0, 1))
NATIVE_BODY = r'''
set -eo pipefail
source /opt/intel/oneapi/setvars.sh --force
set -u
export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}
export BOLT_VERIFY_CAPTURE_DIR=/tmp
for profile in off on; do
 for fusion in 0 1; do
  log="/tmp/native-router-$profile-fusion$fusion.log"
  test ! -e "$log"
  if test "$profile" = off; then
   if env -u LLAMA_LILICORR_PROFILE GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION="$fusion" /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o '^BOLT_ROUTER$' -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
  else
   if env LLAMA_LILICORR_PROFILE=1 GGML_SYCL_DEBUG=1 GGML_SYCL_ENABLE_FUSION="$fusion" /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o '^BOLT_ROUTER$' -j 1 >"$log" 2>&1; then rc=0; else rc=$?; fi
  fi
  printf '%s\n' "$rc" > "$log.exit-code"
  printf 'NATIVE_CONTROL_FINISHED profile=%s fusion=%s exit=%s log=%s\n' "$profile" "$fusion" "$rc" "$log"
 done
done
echo NATIVE_MATRIX_CAPTURE_ONLY_NOT_QUALIFICATION
'''


def derive(argv):
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    limit = len(result) - 4
    indexes = [i for i in range(limit - 1) if result[i:i + 2] == ['-e', D['PROFILE_ENV'] + '=1']]
    assert len(indexes) <= 1
    for index in reversed(indexes):
        del result[index:index + 2]
    result[-1] = NATIVE_BODY
    assert result[-4] == argv[-4], 'image changed'
    assert D['PROFILE_ENV'] + '=1' not in result[:-4]
    return result


def check_log(text, exit_code, fusion):
    checks = dict(exit_zero=exit_code == 0, tests_82=text.count('  82/82 tests passed') == 1,
                  floats_82=text.count('BOLT_ROUTER floats: mismatches=0 ') == 82,
                  indices_82=text.count('BOLT_ROUTER indices: mismatches=0 ') == 82,
                  allocator_5=text.count('BOLT_ROUTER allocator: logits=probs exact (2 graphs)') == 5,
                  consumer_2=text.count('BOLT_ROUTER consumer bytes: mismatches=0 ') == 2,
                  fused_calls=text.count('call ggml_sycl_op_router_fused:') == (67 if fusion else 0),
                  no_skip=not any(word in text.lower() for word in ('not supported', 'skipping large tensors')))
    return dict(passed=all(checks.values()), checks=checks)


def bind():
    runtime, _analysis, c = D['bind']()
    c.update(OUT=OUT, PREFIX=PREFIX)
    old_code = c['adapter_code_hashes']
    c['adapter_code_hashes'] = lambda: dict(inherited=old_code(), native_control={str(HERE / name): c['sha'](HERE / name) for name in ('native-router-controls.py', 'test-native-router-controls.py')})
    native_capture = runtime['_BOOT_K']['capture']
    assert native_capture.__globals__['C'] is c
    base_make = c['make_create']
    checker = D['BATCH']['check_container']
    checker_globals = dict(checker.__globals__, S=D['S'])
    native_checker = types.FunctionType(checker.__code__, checker_globals)
    def check_container(record, cell, h):
        expected = dict(c, make_create=lambda label, host: derive(base_make(label, host)))
        return native_checker(expected, record, cell, h)
    c['check_container'] = check_container

    class NativeControl(runtime['Runtime']):
        def preflight(self):
            saved = self.mode
            self.mode = 'timing'  # Reuse the existing planned-name/capacity checks.
            try:
                return super().preflight()
            finally:
                self.mode = saved

        def begin(self, cell):
            self._stop_original_for_case()
            directory = self.out / cell
            directory.mkdir()
            argv = derive(c['make_create'](cell, self.h))
            c['save'](directory / 'create-argv.json', argv)
            ident = self.command(*argv).strip()
            self.created.append(cell)
            self.active = cell
            before = self.inspect(c['name'](cell))
            assert before['Id'] == ident and before['Config']['Cmd'] == argv[-3:]
            c['check_container'](before, cell, self.h)
            c['save'](directory / 'container-before.json', before)
            with (directory / 'native-matrix.stdout.log').open('w') as stdout, (directory / 'native-matrix.stderr.log').open('w') as stderr:
                subprocess.run(['docker', 'start', '-a', c['name'](cell)], stdout=stdout, stderr=stderr, check=True)
            after = self.inspect(c['name'](cell))
            assert not after['State']['Running'] and after['State']['ExitCode'] == 0
            c['save'](directory / 'container-after.json', after)

        def matrix(self, cell):
            directory = self.out / cell
            rows = []
            for profile, fusion in MATRIX:
                filename = f'native-router-{profile}-fusion{fusion}.log'
                for suffix in ('', '.exit-code'):
                    self.command('docker', 'cp', c['name'](cell) + ':/tmp/' + filename + suffix, str(directory / (filename + suffix)))
                text = (directory / filename).read_text()
                exit_code = int((directory / (filename + '.exit-code')).read_text())
                rows.append(dict(cell=cell, source=str(D['S']['PARENT'] if cell == 'VSPF-A1' else D['S']['CANDIDATE']),
                                 profile=profile, fusion=fusion, exit_code=exit_code,
                                 native=check_log(text, exit_code, fusion), log=filename))
            c['save'](directory / 'native-matrix.json', dict(native_only=True, speed_claim=False, rows=rows))

        def end(self, cell):
            self._stop_if_running(c['name'](cell))
            self.save_logs(cell, 'native-matrix')
            self.active = None
            self.ownership(None)

    return runtime, c, native_capture, NativeControl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-native-controls', action='store_true')
    action.add_argument('--lead-verify-restored', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == D['ROOT']
    runtime, c, capture, cls = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == D['S']['PARENT']
    control = cls(h, OUT)
    control.mode = 'native-router-correctness'
    if args.lead_run_native_controls:
        OUT.mkdir(exist_ok=False)
        result = capture(control, CELLS, lambda rt, cell: rt.matrix(cell))
        assert result['restored']
        print('NATIVE_MATRIX_CAPTURED_ORIGINAL_RESTORED_NO_PUBLIC_REQUESTS')
    else:
        control.verify_restored(OUT / 'independent-original-restoration.json')
        print('NATIVE_MATRIX_SEPARATE_ORIGINAL_RESTORATION_PASSED')


if __name__ == '__main__':
    main()
