#!/usr/bin/env python3
"""Lead-only, profiling-only serving capture over the qualified Q8 lifecycle.

This adapter deliberately keeps the inherited public/native/source contracts and
changes only the declared unitrace mounts, executable wrappers, and collection
windows.  Traced timings are attribution evidence, never a speed result.
"""
import argparse
import copy
import hashlib
import json
import runpy
import shutil
import socket
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261021-lilicorr-serving-profile')
Q8_PATH = HERE.parent / '20261020-lilicorr-q8-target-batch4' / 'run-performance.py'
TOOL_ROOT = Path('/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-kernel-profile')
TOOLS = TOOL_ROOT / 'tools'
OUT = ROOT / 'capture-r1'
PREFIX = 'bolt-lilicorr-serving-profile-r1-'

CELLS = ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2')
CLEAN_CELLS = frozenset(('VSPF-A1', 'VSPF-W3a'))
TRACED_CELLS = frozenset(('VSPF-W3b', 'VSPF-A2'))
PAIRS = (('VSPF-A1', 'VSPF-A2'), ('VSPF-W3a', 'VSPF-W3b'))

SERVER_EXE = '/src/build-sycl/bin/llama-server'
SERVER_TOKEN = 'exec ' + SERVER_EXE + ' '
NATIVE_EXE = '/src/build-sycl/bin/test-backend-ops'
TRACE_FLAGS = ('-d', '-h', '-s', '-v', '--chrome-kernel-logging')
TRACE_FLAG_TEXT = ' '.join(TRACE_FLAGS)
TRACE_ENV = 'export ZE_ENABLE_TRACING_LAYER=1; unset UNITRACE_KernelMetrics UNITRACE_ChromeKmdLogging; '
TRACE_FILENAMES = ('device_timing.txt', 'host_timing.txt', 'device_submission.txt', 'chrome_trace.json')
MIN_FREE_BYTES = 20 * 1024 ** 3
TOOL_HASHES = {
    'unitrace': '99ffdad3f3ae86aeee8e52f4b687a9d41e5e67491993952de37c112544a0ff30',
    'libunitrace_tool.so': '4bb8d93a64d9091867babd6e14d3b8ed7647b1f4c649802cc1e8570ad0aa2d04',
}
OWN_FILES = ('README.md', 'run-profile.py', 'test-profile.py', 'lead-run.sh', '.gitignore')

# This is intentionally the existing qualified adapter, not a copy of its
# controller.  Its bind() is configured below through the function namespace.
Q8 = runpy.run_path(str(Q8_PATH))


def now():
    """Return both clocks needed to correlate client spans with Chrome unix-us."""
    return dict(realtime_ns=time.time_ns(), monotonic_ns=time.monotonic_ns())


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def session_name(cell):
    assert cell in CELLS
    value = 'boltserveprofile1' + cell.replace('-', '').lower()
    assert value.isalnum()
    return value


def extra_mounts(cell):
    """Return only the two declared profiling mounts for a traced cell."""
    assert cell in TRACED_CELLS
    return (
        (TOOLS, '/tools', 'ro'),
        (OUT / cell / 'profile', '/profile', 'rw'),
    )


def _native_command(cell, ordinal):
    native_session = session_name(cell) + 'n' + f'{ordinal:02d}'
    result_dir = f'/profile/native/native-{ordinal:02d}'
    return '/tools/unitrace ' + TRACE_FLAG_TEXT + f' --session {native_session} --result-dir {result_dir} ' + NATIVE_EXE


def _server_command(cell):
    return (TRACE_ENV + 'exec /tools/unitrace ' + TRACE_FLAG_TEXT +
            f' --session {session_name(cell)} --start-paused --result-dir /profile/server ' +
            SERVER_EXE + ' ')


def wrap_native_script(body, cell):
    """Wrap every inherited native test executable without touching its shell checks."""
    assert cell in TRACED_CELLS
    assert body.count(SERVER_TOKEN) == 1
    assert '/tools/unitrace' not in body
    count = body.count(NATIVE_EXE)
    assert count > 0, 'the inherited native gate lost test-backend-ops'
    result = body.replace(NATIVE_EXE, '__SERVING_PROFILE_NATIVE_SLOT__')
    for ordinal in range(1, count + 1):
        replacement = _native_command(cell, ordinal)
        assert '__SERVING_PROFILE_NATIVE_SLOT__' in result
        result = result.replace('__SERVING_PROFILE_NATIVE_SLOT__', replacement, 1)
    assert result.count('/tools/unitrace') == count
    assert result.count(NATIVE_EXE) == count
    # The tracer needs its own enablement, but all redirections, grep checks,
    # exit checks, and native log filenames remain byte-for-byte in place.
    return TRACE_ENV + result


def wrap_argv(argv, cell):
    """Apply the exact traced docker argv transform; clean cells are unchanged."""
    assert cell in CELLS
    result = list(argv)
    assert result[-3:-1] == ['bash', '-lc']
    body = result[-1]
    assert body.count(SERVER_TOKEN) == 1
    assert '/tools/unitrace' not in body
    if cell in CLEAN_CELLS:
        assert '/profile' not in body
        return result
    count = body.count(NATIVE_EXE)
    assert count > 0, 'the inherited native gate lost test-backend-ops'

    # Runtime.begin has already made OUT/cell before it calls make_create.  The
    # mount source must exist before docker create, while no output is removed.
    (OUT / cell / 'profile').mkdir(parents=True, exist_ok=True)
    bindings = [item for source, destination, mode in extra_mounts(cell)
                for item in ('-v', f'{source}:{destination}:{mode}')]
    result[-4:-4] = bindings
    body = wrap_native_script(body, cell)
    assert body.count(SERVER_TOKEN) == 1
    body = body.replace(SERVER_TOKEN, _server_command(cell))
    assert body.count('/tools/unitrace') == count + 1
    assert body.count(NATIVE_EXE) == count
    result[-1] = body
    return result


def tool_proof():
    """Verify the already-installed remote profiler, never build or replace it."""
    marker = TOOL_ROOT / 'tool-build-exit-code.txt'
    assert marker.read_text().strip() == '0'
    actual = {name: hashlib.sha256((TOOLS / name).read_bytes()).hexdigest()
              for name in TOOL_HASHES}
    assert actual == TOOL_HASHES, ('profiler binary drift', actual)
    return actual


def expected_reference(reference, cell):
    """Derive expected profiling mounts without modifying an observed record."""
    result = copy.deepcopy(reference)
    for source, destination, mode in extra_mounts(cell):
        assert not any(m['Destination'] == destination for m in result['Mounts'])
        result['HostConfig']['Binds'].append(f'{source}:{destination}:{mode}')
        result['Mounts'].append(dict(Type='bind', Source=str(source), Destination=destination,
                                     Mode=mode, RW=mode == 'rw', Propagation='rprivate'))
    return result


def check_declared_extra_mounts(record, reference, cell):
    """Strictly check only declared tool/output mounts, never rewrite record data."""
    expected = expected_reference(reference, cell)
    destinations = {destination for _source, destination, _mode in extra_mounts(cell)}
    actual_mounts = sorted((m for m in record['Mounts'] if m['Destination'] in destinations),
                           key=lambda m: m['Destination'])
    expected_mounts = sorted((m for m in expected['Mounts'] if m['Destination'] in destinations),
                             key=lambda m: m['Destination'])
    assert actual_mounts == expected_mounts
    actual_binds = sorted(item for item in record['HostConfig']['Binds']
                          if any(':' + destination + ':' in item for destination in destinations))
    expected_binds = sorted(item for item in expected['HostConfig']['Binds']
                            if any(':' + destination + ':' in item for destination in destinations))
    assert actual_binds == expected_binds
    return True


def configure_q8_namespace():
    """Point the returned runpy function namespace at this adapter's remote root."""
    globals_ = Q8['bind'].__globals__
    assert Q8['main'].__globals__ is globals_
    globals_.update(HERE=HERE, ROOT=ROOT, OUT=OUT, PREFIX=PREFIX)


def _load_sidecar(path):
    path = Path(path)
    if not path.exists():
        return {'requests': []}
    value = json.loads(path.read_text())
    assert isinstance(value, dict) and isinstance(value.get('requests'), list)
    return value


def collect_request(runtime, c, cell, stem, callback, kind='request', save_fn=None):
    """Collect one real request under an active unitrace resume/pause window."""
    assert cell in TRACED_CELLS
    save_fn = save_fn or _save_json
    name = c['name'](cell)
    session = session_name(cell)
    sidecar_path = Path(str(stem) + '.trace-sidecar.json')
    sidecar = _load_sidecar(sidecar_path)
    sidecar.update(cell=cell, collection='unitrace-resume-to-pause', trace_session=session)
    span = dict(kind=kind, session=session, request_stem=str(stem),
                resume_command=['docker', 'exec', name, '/tools/unitrace', '--resume', session],
                pause_command=['docker', 'exec', name, '/tools/unitrace', '--pause', session])
    failure = None
    failure_tb = None
    pause_failure = None
    resumed = False
    try:
        span['before_resume'] = now()
        runtime.command('docker', 'exec', name, '/tools/unitrace', '--resume', session, timeout=30)
        resumed = True
        span['after_resume'] = now()
        span['before_request'] = now()
        try:
            result = callback()
        except BaseException as exc:
            failure = exc
            failure_tb = exc.__traceback__
        span['after_request'] = now()
    except BaseException as exc:
        failure = exc
        failure_tb = exc.__traceback__
    finally:
        span.setdefault('after_resume', now())
        span.setdefault('before_request', now())
        span.setdefault('after_request', now())
        span['before_pause'] = now()
        try:
            # Pause is deliberately attempted even when resume/request failed;
            # the lifecycle's finally path must leave collection quiescent.
            runtime.command('docker', 'exec', name, '/tools/unitrace', '--pause', session, timeout=30)
        except BaseException as exc:
            pause_failure = exc
            span['pause_error'] = repr(exc)
        finally:
            span['after_pause'] = now()
    span['resumed'] = resumed
    span['request_succeeded'] = failure is None
    if failure is not None:
        span['request_error'] = repr(failure)
    if pause_failure is not None:
        span['pause_succeeded'] = False
    else:
        span['pause_succeeded'] = True
    sidecar['requests'].append(span)
    save_fn(sidecar_path, sidecar)
    if failure is not None:
        raise failure.with_traceback(failure_tb)
    if pause_failure is not None:
        raise pause_failure
    return result


def validate_trace_outputs(root, label):
    """Require complete, nonempty device/host/submission/Chrome output."""
    root = Path(root)
    assert root.is_dir(), (label, 'missing trace directory', str(root))
    found = {}
    for filename in TRACE_FILENAMES:
        paths = sorted(path for path in root.rglob(filename) if path.is_file())
        assert paths, (label, 'missing trace output', filename, str(root))
        assert all(path.stat().st_size > 0 for path in paths), (label, 'empty trace output', filename)
        found[filename] = [str(path.relative_to(root)) for path in paths]
    for relative in found['chrome_trace.json']:
        value = json.loads((root / relative).read_text())
        assert value not in (None, {}, []), (label, 'empty Chrome trace', relative)
    return dict(label=label, root=str(root), complete=True, files=found)


def disk_preflight(path=ROOT):
    path = Path(path)
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    record = dict(path=str(probe), free_bytes=usage.free, required_free_bytes=MIN_FREE_BYTES,
                  passed=usage.free >= MIN_FREE_BYTES)
    assert record['passed'], ('free disk floor', record)
    return record


def _duration(row):
    metrics = row.get('metrics', {})
    timings = row.get('timings', {})
    for source, values in (('elapsed_s', metrics), ('elapsed_s', timings),
                           ('predicted_ms', metrics), ('predicted_ms', timings)):
        if source in values:
            return source, values[source]
    raise AssertionError(('missing measured duration', row))


def save_overhead_pairs(c):
    """Pair clean/traced measured durations as tracing overhead evidence only."""
    result = dict(mode='profiling', performance_claim=False,
                  interpretation='paired clean/traced durations estimate tracing overhead only',
                  pairs=[], complete=True)
    for clean, traced in PAIRS:
        for point in ('short', 'long'):
            clean_path = OUT / clean / 'timing-rows.json'
            traced_path = OUT / traced / 'timing-rows.json'
            if not clean_path.exists() or not traced_path.exists():
                result['complete'] = False
                continue
            clean_rows = {int(row['ordinal']): row for row in json.loads(clean_path.read_text())
                          if row.get('point') == point and row.get('phase') == 'measured'}
            traced_rows = {int(row['ordinal']): row for row in json.loads(traced_path.read_text())
                           if row.get('point') == point and row.get('phase') == 'measured'}
            for ordinal in range(3):
                if ordinal not in clean_rows or ordinal not in traced_rows:
                    result['complete'] = False
                    continue
                clean_source, clean_duration = _duration(clean_rows[ordinal])
                traced_source, traced_duration = _duration(traced_rows[ordinal])
                result['pairs'].append(dict(clean_cell=clean, traced_cell=traced, point=point,
                                            ordinal=ordinal, clean_source=clean_source,
                                            traced_source=traced_source,
                                            clean_duration=clean_duration,
                                            traced_duration=traced_duration,
                                            delta=traced_duration - clean_duration))
    c['save'](OUT / 'paired-overhead.json', result)
    return result


def bind():
    """Bind the current Q8/IQ2 qualified maps, then add only profiling behavior."""
    configure_q8_namespace()
    prior = Q8
    runtime, analysis, c = prior['bind']()
    # The returned map and the function namespace are intentionally separate in
    # runpy.  Mutate the live contract dict and function globals, never a shadow.
    c.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX, CELLS=CELLS, ORDER=CELLS,
             TIMING_ORDER=CELLS)
    old_make = c['make_create']
    old_codes = c['adapter_code_hashes']
    old_hashes = c['source_binary_hashes']
    original_load = c['load']
    base_check = prior['B']['check_container']

    def planned_names(cells=CELLS):
        assert tuple(cells) == CELLS or set(cells) <= set(CELLS)
        result = [c['name'](cell) for cell in cells]
        assert len(result) == len(set(result))
        return result

    def make_create(cell, h):
        base = old_make(cell, h)
        return wrap_argv(base, cell) if cell in TRACED_CELLS else base

    def check_container(record, cell, h):
        before = copy.deepcopy(record)
        if cell not in TRACED_CELLS:
            result = base_check(c, record, cell, h)
        else:
            reference_path = c['qualified_dir'](c['role_for'](cell)) / 'container-before.json'
            reference = original_load(reference_path)
            check_declared_extra_mounts(record, reference, cell)
            context = dict(c)

            def load_expected(path):
                assert Path(path) == reference_path
                return expected_reference(original_load(path), cell)

            context['load'] = load_expected
            result = base_check(context, record, cell, h)
        assert record == before, 'container validation rewrote the observed record'
        return result

    def code_hashes():
        result = old_codes()
        result.update({str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES})
        return result

    def source_binary_hashes(h):
        return dict(validated_backend=old_hashes(h), profiler=tool_proof())

    c.update(planned_names=planned_names, make_create=make_create,
             check_container=check_container, adapter_code_hashes=code_hashes,
             source_binary_hashes=source_binary_hashes)

    base_runtime = runtime['Runtime']

    class ProfileRuntime(base_runtime):
        def __init__(self, h, out):
            super().__init__(h, out)
            self.mode = 'profiling'
            self.measured_rows = {}

        def preflight(self):
            # This is the first operation before inherited preflight can stop
            # the original; low space leaves the original service untouched.
            disk_preflight(ROOT)
            return super().preflight()

        def begin(self, cell):
            result = super().begin(cell)
            if cell in TRACED_CELLS:
                native_root = self.out / cell / 'profile' / 'native'
                try:
                    validation = validate_trace_outputs(native_root, cell + '-native')
                except BaseException as exc:
                    c['save'](self.out / cell / 'native-trace-validation.json',
                              dict(passed=False, error=repr(exc), preserved_partial=True))
                    raise
                c['save'](self.out / cell / 'native-trace-validation.json',
                          dict(passed=True, validation=validation,
                               traced_native_gate=True, untraced_native_gate_claim=False))
            return result

        def _trace_enabled(self, cell=None):
            return (cell or self.active) in TRACED_CELLS

        def _sidecar_stem(self, cell, stem):
            return Path(stem)

        def _collect(self, cell, stem, callback, kind):
            if not self._trace_enabled(cell):
                return callback()
            return collect_request(self, c, cell, self._sidecar_stem(cell, stem), callback, kind,
                                   save_fn=c['save'])

        def public(self, cell, job):
            if not self._trace_enabled(cell):
                return super().public(cell, job)
            logical = self.out / cell / job['stem']
            parent = super()
            return self._collect(cell, logical, lambda: parent.public(cell, job), 'tokenize')

        def _exchange(self, payload, stem):
            if not self._trace_enabled():
                return super()._exchange(payload, stem)
            parent = super()
            return self._collect(self.active, stem, lambda: parent._exchange(payload, stem), 'completion')

        def timed_request(self, cell, point, phase, ordinal):
            if not self._trace_enabled(cell):
                return super().timed_request(cell, point, phase, ordinal)
            stem = self.out / cell / point / f'{phase}-{ordinal:02}'
            parent = super()
            method = parent.timed_request
            globals_ = method.__globals__
            original_stream_chat = globals_['stream_chat']

            def traced_stream_chat(*args, **kwargs):
                return collect_request(
                    self, c, cell, stem,
                    lambda: original_stream_chat(*args, **kwargs),
                    'timed-' + phase, save_fn=c['save'])

            globals_['stream_chat'] = traced_stream_chat
            try:
                # Keep the inherited sampler, request payload, output checks,
                # and artifact schema; only the real stream call is wrapped.
                return method(cell, point, phase, ordinal)
            finally:
                globals_['stream_chat'] = original_stream_chat

        def run_timing_cell(self, cell):
            rows = super().run_timing_cell(cell)
            self.measured_rows[cell] = rows
            return rows

        def end(self, cell):
            try:
                result = super().end(cell)
            except BaseException as exc:
                if cell in TRACED_CELLS:
                    try:
                        validation = validate_trace_outputs(self.out / cell / 'profile' / 'server',
                                                            cell + '-server-partial')
                        c['save'](self.out / cell / 'server-trace-validation.json',
                                  dict(passed=False, stop_error=repr(exc), partial=validation,
                                       preserved_partial=True))
                    except BaseException as validation_exc:
                        c['save'](self.out / cell / 'server-trace-validation.json',
                                  dict(passed=False, stop_error=repr(exc),
                                       validation_error=repr(validation_exc),
                                       preserved_partial=True))
                raise
            if cell in TRACED_CELLS:
                try:
                    validation = validate_trace_outputs(self.out / cell / 'profile' / 'server',
                                                        cell + '-server')
                except BaseException as exc:
                    c['save'](self.out / cell / 'server-trace-validation.json',
                              dict(passed=False, error=repr(exc), preserved_partial=True))
                    raise
                c['save'](self.out / cell / 'server-trace-validation.json',
                          dict(passed=True, validation=validation, normal_stop_t90=True))
            return result

    runtime['Runtime'] = ProfileRuntime

    return prior, runtime, analysis, c


def _dependencies(prior, runtime):
    p, _k, h = runtime['load_dependencies']()
    source = prior['S']
    assert p['CANDIDATE'] == source['PARENT'], 'qualified ancestry must remain original'
    return p, h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-capture', action='store_true')
    action.add_argument('--lead-verify-restored', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert HERE == ROOT, (HERE, ROOT)
    prior, runtime_map, _analysis, c = bind()
    _p, h = _dependencies(prior, runtime_map)

    if args.lead_capture:
        disk_preflight(ROOT)
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        tool_proof()
        rt = runtime_map['Runtime'](h, OUT)
        try:
            summary = runtime_map['capture'](rt)
            assert summary['status'] == 'captured' and summary['restored'], summary
            assert summary['mode'] == 'profiling'
            assert not summary['timing_permitted'] and not summary['promotion_permitted']
            print('SERVING_PROFILE_CAPTURED_RESTORED_OVERHEAD_ONLY', flush=True)
        finally:
            save_overhead_pairs(c)
        return

    assert OUT.is_dir(), OUT
    rt = runtime_map['Runtime'](h, OUT)
    rt.mode = 'profiling'
    record = rt.verify_restored(OUT / 'independent-restoration-live.json')
    assert record['passed'], record
    print('SERVING_PROFILE_INDEPENDENT_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
