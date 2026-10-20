#!/usr/bin/env python3
"""Experimental full-trace Q8 ABBA adapter.

This adapter keeps the qualified Q8 runner's public exchange, native checks, source
pins, and restoration lifecycle.  Its only candidate delta is the private full-trace
capture bind/environment and the corresponding positive evidence checks.  It is a
lead-only diagnostic; trace/host clocks are not GPU timings.
"""
import argparse
import copy
import json
import re
import runpy
import socket
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter')
OUT = ROOT / 'verification-routes-20261020-r2'
PREFIX = 'bolt-lilicorr-verification-routes-20261020-r2-'
Q8_RUNNER = Path('/home/mike/b70-evals/20261019-lilicorr-q8-drafter/run-performance.py')
if not Q8_RUNNER.is_file():
    Q8_RUNNER = HERE.parent / '20261019-lilicorr-q8-drafter/run-performance.py'
REFERENCE_ROOT = ROOT / 'timing-20261020-baseline-r1'
CELLS = ('VSPF-A1', 'VSPF-W3a', 'VSPF-W3b', 'VSPF-A2')
ROLES = {'VSPF-A1': 'A', 'VSPF-W3a': 'W3', 'VSPF-W3b': 'W3', 'VSPF-A2': 'A'}
BASE_CELLS = {'A': 'VSB-A', 'W3': 'VSB-W3'}
TRACE_ENV = {'BOLT_VERIFY_CAPTURE_DIR': '/capture', 'LLAMA_TRACE': '1'}
CAPTURE_ROW_BYTES = 524288
OWN_FILES = ('README.md', 'run-performance.py', 'test-performance.py')

# These are populated by bind().  They are intentionally private to this runpy
# namespace; the qualified P namespace and every h['off'] object remain untouched.
_CURRENT_C = None
_CURRENT_RUNTIME = None
_BASE_MAKE = None
_Q8 = None


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _reference_root():
    """Use the remote clean-Q8 fixture, with the checkout fixture as CPU fallback."""
    if REFERENCE_ROOT.is_dir():
        return REFERENCE_ROOT
    local = HERE.parent / '20261019-lilicorr-q8-drafter/timing-20261020-baseline-r1'
    if local.is_dir():
        return local
    return REFERENCE_ROOT


def _reference_cell(cell):
    assert cell in CELLS, cell
    directory = _reference_root() / cell
    assert directory.is_dir(), directory
    return directory


def role_for(cell):
    assert cell in ROLES, cell
    return ROLES[cell]


def trace_name(cell):
    assert cell in CELLS, cell
    if _CURRENT_C is not None:
        return _CURRENT_C['name'](cell)
    return PREFIX + cell.lower()


def capture_path(cell, output=OUT):
    assert cell in CELLS, cell
    return Path(output) / cell / 'capture'


def _env_map(values):
    result = {}
    for value in values or []:
        key, separator, item = value.partition('=')
        assert separator and key and key not in result, key
        result[key] = item
    return result


def _bind_source(value):
    return value.split(':', 1)[0]


def _canonical_mounts(mounts):
    """Fallback canonicalization for the CPU fixture if contracts are unavailable."""
    return sorted(
        (mount.get('Type'), mount.get('Source'), mount.get('Destination'),
         mount.get('Mode'), mount.get('RW'), mount.get('Propagation'))
        for mount in mounts or [])


def _canonical(mounts):
    if _CURRENT_C is not None and 'canonical_mounts' in _CURRENT_C:
        return _CURRENT_C['canonical_mounts'](mounts)
    return _canonical_mounts(mounts)


def _base_create_from_fixture(cell):
    argv = list(load(_reference_cell(cell) / 'create-argv.json'))
    argv[argv.index('--name') + 1] = trace_name(cell)
    return argv


def base_create_argv(cell, h=None):
    """Return the clean Q8 create argv with this adapter's fresh name."""
    if h is not None:
        assert _BASE_MAKE is not None
        return list(_BASE_MAKE(cell, h))
    return _base_create_from_fixture(cell)


def _env_pairs(argv):
    pairs = []
    for index, value in enumerate(argv):
        if value == '-e':
            assert index + 1 < len(argv)
            pairs.append(argv[index + 1])
    return pairs


def _trace_server_boundary(command):
    old = 'unset BOLT_VERIFY_CAPTURE_DIR LLAMA_TRACE; export GGML_SYCL_DEBUG=0; echo BOLT_VERIFY_SERVER_START;'
    new = ('export BOLT_VERIFY_CAPTURE_DIR=/capture; export LLAMA_TRACE=1; '
           'export GGML_SYCL_DEBUG=0; echo BOLT_VERIFY_SERVER_START;')
    assert command.count(old) == 1, 'Q8 server boundary not found exactly once'
    return command.replace(old, new)


def assert_trace_argv(base, derived, cell, output=OUT):
    """Fail closed unless the argv delta is name-independent trace plumbing only."""
    assert base[:2] == ['docker', 'create']
    assert derived[:2] == ['docker', 'create']
    assert derived[derived.index('--name') + 1] == trace_name(cell)
    capture = str(capture_path(cell, output)) + ':/capture'
    assert derived.count(capture) == 1
    capture_index = derived.index(capture)
    assert capture_index > 0 and derived[capture_index - 1] == '-v'
    assert all(not value.endswith(':/capture:ro') for value in derived)

    pairs = _env_pairs(derived)
    for key, value in TRACE_ENV.items():
        assert pairs.count(key + '=' + value) == 1
    assert all('BOLT_VERIFY_CAPTURE_DIR=/tmp' != value for value in pairs)
    assert 'LLAMA_TRACE' not in _env_pairs(base)

    reverse = list(derived)
    assert reverse[-1] == _trace_server_boundary(base[-1])
    reverse[-1] = base[-1]
    del reverse[capture_index - 1:capture_index + 1]
    for key, value in TRACE_ENV.items():
        pair = key + '=' + value
        index = reverse.index(pair)
        assert index > 0 and reverse[index - 1] == '-e'
        del reverse[index - 1:index + 1]
    assert reverse == base
    assert _CURRENT_C['server_argv'](base) == _CURRENT_C['server_argv'](derived)
    assert derived[-3:-1] == ['bash', '-lc']
    return True


def make_create(cell, h=None, output=OUT):
    """Derive a Q8 launch by adding only the full-trace bind/env/boundary."""
    base = base_create_argv(cell, h)
    assert base[base.index('--name') + 1] == trace_name(cell)
    capture = str(capture_path(cell, output)) + ':/capture'
    image_index = len(base) - 3
    assert image_index > 0 and base[image_index - 1] != 'bash'
    derived = list(base[:image_index])
    derived += ['-v', capture]
    for key, value in TRACE_ENV.items():
        derived += ['-e', key + '=' + value]
    derived += base[image_index:]
    derived[-1] = _trace_server_boundary(base[-1])
    assert_trace_argv(base, derived, cell, output)
    return derived


def _expected_mounts(reference, output, cell):
    mounts = copy.deepcopy(reference.get('Mounts') or [])
    capture = str(capture_path(cell, output))
    assert not any(mount.get('Destination') == '/capture' for mount in mounts)
    mounts.append(dict(Type='bind', Source=capture, Destination='/capture',
                       Mode='', RW=True, Propagation='rprivate'))
    return mounts


def check_trace_container(record, cell, output=OUT, h=None):
    """Check the actual Docker record against the clean Q8 cell plus trace delta."""
    reference = load(_reference_cell(cell) / 'container-before.json')
    expected_argv = make_create(cell, h, output)
    assert record['Name'] == '/' + trace_name(cell)
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert record['Image'] == reference['Image']
    assert record['Config']['Cmd'] == expected_argv[-3:]
    excluded = {'Hostname', 'Env', 'Cmd'}
    assert {key: value for key, value in record['Config'].items() if key not in excluded} == {
        key: value for key, value in reference['Config'].items() if key not in excluded}
    assert _env_map(record['Config'].get('Env')) == dict(
        _env_map(reference['Config'].get('Env')), **TRACE_ENV)

    expected_host = copy.deepcopy(reference['HostConfig'])
    actual_host = copy.deepcopy(record['HostConfig'])
    capture_bind = str(capture_path(cell, output)) + ':/capture'
    expected_host['Binds'] = list(expected_host.get('Binds') or []) + [capture_bind]
    for host in (expected_host, actual_host):
        oom = host.pop('OomKillDisable', None)
        assert oom is None or oom is False, 'OOM killer must remain enabled'
        host['Binds'] = sorted(host.get('Binds') or [])
    assert actual_host == expected_host

    assert _canonical(record.get('Mounts')) == _canonical(_expected_mounts(reference, output, cell))
    capture_mounts = [mount for mount in record.get('Mounts', [])
                      if mount.get('Destination') == '/capture']
    assert len(capture_mounts) == 1
    capture_mount = capture_mounts[0]
    assert capture_mount['Type'] == 'bind'
    assert capture_mount['Source'] == str(capture_path(cell, output))
    assert capture_mount['RW'] is True
    assert 'ro' not in capture_mount.get('Mode', '').split(',')
    return True


def check_trace_pid(record, untraced):
    """Validate actual PID1 evidence without rewriting the record or env keys."""
    assert isinstance(record, dict) and isinstance(untraced, dict)
    for key in ('pid', 'origin', 'argv', 'exe', 'scoped_tmp_rows'):
        assert record[key] == untraced[key], key
    base_env = dict(untraced['diagnostics'])
    trace_env = dict(record['diagnostics'])
    assert set(base_env) == {'BOLT_TRACE_TENSORS', 'GGML_SYCL_DEBUG'}
    assert base_env == {'BOLT_TRACE_TENSORS': '0', 'GGML_SYCL_DEBUG': '0'}
    assert trace_env == dict(base_env, **TRACE_ENV)
    assert len(trace_env) == len(record['diagnostics'])
    return True


def check_trace_startup(log, cell, h):
    """Reuse native/model gates while allowing the requested trace route lines."""
    c = _CURRENT_C
    assert c is not None
    off, p, old = h['off'], h['off']['P'], h['off']['OLD']
    assert log.count(p['MARKER']) == 1
    native_log, model = log.split(p['MARKER'])
    assert 'BOLT_ROUTE route=' in native_log
    assert 'GGML_SYCL_DEBUG: 0' in model
    for forbidden in ('BOLT_TENSOR_TRACE_ENABLED', 'BOLT_PROJECTION_REPEAT',
                      'common_debug_cb_eval', 'onednn_verbose'):
        assert forbidden not in model, forbidden
    for forbidden in ('GGML_ASSERT', 'SYCL exception', 'BOLT_VERIFY error'):
        assert forbidden not in log, forbidden
    native = runpy.run_path(str(p['WIDTH1'] / 'analyze-controls.py'))
    native['parse_native_gate'](log, 'C0')
    for gate in ('iq2', 'q8', 'attention', 'attention_view'):
        p['check_' + gate + '_native_gate'](log)
    qualified = c['_public']()
    markers = (p['ATTENTION_VIEW_NATIVE_MARKER'], p['ROUTER_NATIVE_MARKER'],
               qualified['MARKER'], qualified['SINGLETON_MARKER'], p['MARKER'])
    assert all(log.count(marker) == 1 for marker in markers)
    assert [log.index(marker) for marker in markers] == sorted(log.index(marker) for marker in markers)
    segment = log.split(markers[0], 1)[1].split(markers[1], 1)[0]
    router = qualified['A']['check_router_native_output'](segment, 'router_single2')
    assert 'router_columns8=1,' not in log
    assert 'router_prefill=1,' not in log
    assert 'verify_singletons=1,' not in native_log
    width_for = qualified['C'].get('width_for', c['width_for'])
    width = width_for(role_for(cell))
    if width is None:
        result = off['N']['check_model_startup'](model)
    else:
        old['check_model_log'](model, 'C')
        assert [int(match.group(1)) for match in old['N_RS_SEQ_RE'].finditer(model)][:2] == [width, 0]
        params = old['DFLASH_PARAMS_RE'].search(model)
        block = old['DFLASH_BLOCK_RE'].search(model)
        assert params and params.groups() == (str(width), '0')
        assert block and block.group(1) == '8'
        result = dict(width=width)
    return dict(result, router_single2=router,
                evidence='unaltered raw startup and actual trace PID1')


def check_trace_server_log(log):
    """Check the raw server stream while allowing trace route/sample lines."""
    marker = _CURRENT_C['_public']()['MARKER'] if _CURRENT_C and '_public' in _CURRENT_C else None
    if marker is None:
        marker = 'BOLT_VERIFY_ROUTER_VERIFY_PHASE_NATIVE_PASSED'
    assert log.count(marker) == 1
    native, serving = log.split(marker, 1)
    assert 'BOLT_ROUTE route=' in native
    assert 'GGML_SYCL_DEBUG: 0' in serving
    for forbidden in ('GGML_ASSERT', 'SYCL exception', 'BOLT_VERIFY error',
                      'BOLT_TENSOR_TRACE_ENABLED', 'BOLT_PROJECTION_REPEAT',
                      'common_debug_cb_eval', 'onednn_verbose'):
        assert forbidden not in serving, forbidden
    assert 'BOLT_VERIFY sample ' in serving
    return True


def _save_streams(directory, prefix, streams):
    for stream, data in streams.items():
        (Path(directory) / (prefix + '.' + stream + '.log')).write_bytes(data)


def _capture_manifest(directory, require=True):
    directory = Path(directory)
    files = sorted(path for path in directory.glob('row-*')
                   if path.is_file() and not path.is_symlink())
    assert all(path.name.startswith('row-') for path in files)
    if require:
        assert files, 'no raw CPU sampler rows captured'
    sizes = [path.stat().st_size for path in files]
    assert all(size == CAPTURE_ROW_BYTES for size in sizes), sizes[:5]
    return [dict(name=path.name, path=str(path), bytes=size)
            for path, size in zip(files, sizes)]


def _save_raw_startup(directory, raw):
    for stream in ('stdout', 'stderr'):
        (Path(directory) / ('native-and-startup.raw.' + stream + '.log')).write_bytes(
            getattr(raw, stream))


def _check_runtime_evidence(c, p, h, out, cell):
    """Independent candidate checks; restoration itself remains inherited Q8 logic."""
    out = Path(out)
    directory = out / cell
    role = role_for(cell)
    assert load(directory / 'create-argv.json') == c['make_create'](cell, h)
    base = p['make_create'](c['base_cell'](cell), h)
    assert load(directory / 'argv-diff.json') == c['argv_diff'](base, c['make_create'](cell, h))
    baseline = load(_reference_cell(cell) / 'pid-before.json')
    p['C']['check_pid'](baseline, role, h)
    for phase in ('before', 'after'):
        check_trace_pid(load(directory / ('pid-' + phase + '.json')), baseline)
        check_trace_container(load(directory / ('container-' + phase + '.json')),
                              cell, out, h)
    preserved = load(directory / 'preserved-container.json')
    check_trace_container(preserved, cell, out, h)
    assert not preserved['State']['Running']
    assert c['same_container'](load(directory / 'container-before.json'),
                               load(directory / 'container-after.json'))

    raw_startup = ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                           for stream in ('stdout', 'stderr'))
    check_trace_startup(raw_startup, cell, h)
    phases = {}
    for phase in ('native-and-startup', 'server', 'restored-final'):
        phases[phase] = [(directory / (phase + '.' + stream + '.log')).read_bytes()
                         for stream in ('stdout', 'stderr')]
    check_trace_server_log(b''.join(phases['server']).decode(errors='replace'))
    check_trace_server_log(b''.join(phases['restored-final']).decode(errors='replace'))
    for index in range(2):
        assert phases['server'][index].startswith(phases['native-and-startup'][index])
        assert phases['restored-final'][index].startswith(phases['server'][index])
    for tag, filename in p['NATIVE_LOGS'].items():
        ready = (directory / ('ready-' + filename)).read_bytes()
        restored = (directory / ('restored-' + filename)).read_bytes()
        assert ready == restored
        p['check_native_log'](tag, ready.decode())
    manifest = load(directory / 'capture-files.json')
    assert manifest == _capture_manifest(directory / 'capture', require=True)
    return dict(passed=True, capture_rows=len(manifest), raw_startup_bytes=len(raw_startup.encode()))


def bind():
    """Bind the existing Q8 runner into a private namespace and add trace checks."""
    global _CURRENT_C, _CURRENT_RUNTIME, _BASE_MAKE, _Q8
    q8 = runpy.run_path(str(Q8_RUNNER))
    q8_globals = q8['bind'].__globals__
    q8_globals.update(ROOT=ROOT, OUT=OUT, PREFIX=PREFIX)
    runtime, analysis, c = q8['bind']()
    _Q8 = q8
    _CURRENT_C = c
    _CURRENT_RUNTIME = runtime
    base_make = c['make_create']
    base_code = c['adapter_code_hashes']
    _BASE_MAKE = base_make

    def trace_make_create(cell, h=None):
        base = list(base_make(cell, h)) if h is not None else _base_create_from_fixture(cell)
        base[base.index('--name') + 1] = c['name'](cell)
        return make_create(cell, h, OUT)

    def code_hashes():
        inherited = base_code()
        own = {str(HERE / name): c['sha'](HERE / name) for name in OWN_FILES}
        return dict(inherited, **own)

    def trace_container(record, cell, h):
        return check_trace_container(record, cell, OUT, h)

    c.update(make_create=trace_make_create,
             check_container=trace_container,
             adapter_code_hashes=code_hashes)
    # The inherited Runtime methods resolve C through this exact private map.
    runtime['_bind_inherited_contract_namespace'](runtime['_BOOT_K'])
    analysis['analyze'].__globals__['C'] = c
    analysis['C'] = c

    base_runtime = runtime['Runtime']

    class TraceRuntime(base_runtime):
        """Q8 Runtime with only trace argv/PID/log/capture evidence overrides."""

        def __init__(self, h, out):
            super().__init__(h, out)
            self._trace_c = c
            self._trace_p = runtime.get('P') or h['off']['P']

        def pid(self, cell, phase):
            name = self._cell_name(cell)
            record = json.loads(self.command(
                'docker', 'exec', name, 'python3', '-B', '-c',
                self._trace_p['C']['PID_PROBE']))
            c['save'](self.out / cell / ('pid-' + phase + '.json'), record)
            baseline = load(_reference_cell(cell) / 'pid-before.json')
            self._trace_p['C']['check_pid'](baseline, self._base_role(cell), self.h)
            check_trace_pid(record, baseline)
            maps = self.command('docker', 'exec', name, 'cat', '/proc/1/maps')
            (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
                '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
            return record

        def begin(self, cell):
            assert cell in c['CELLS'] and self.active is None
            self.capacity(cell)
            if not self.prefixes:
                self._stop_original_for_case()
            else:
                assert not self.inspect(self.original_name)['State']['Running']
                self.ownership(None)
            directory = self.out / cell
            directory.mkdir()
            capture_path(cell, self.out).mkdir()
            argv = c['make_create'](cell, self.h)
            base_argv = self._trace_p['make_create'](self._base_cell(cell), self.h)
            c['save'](directory / 'create-argv.json', argv)
            c['save'](directory / 'argv-diff.json', c['argv_diff'](base_argv, argv))
            self.created.append(cell)
            self.command(*argv, timeout=60)
            live = self.inspect(self._cell_name(cell))
            c['check_container'](live, cell, self.h)
            self.command('docker', 'start', self._cell_name(cell))
            c['save'](directory / 'models.json', self.ready(self.h['api'], self._cell_name(cell)))
            self.save_logs(cell, 'native-and-startup', self._cell_name(cell))
            raw = subprocess.run(['docker', 'logs', self._cell_name(cell)], capture_output=True,
                                 timeout=90, check=True)
            _save_raw_startup(directory, raw)
            startup = (raw.stdout + raw.stderr).decode('utf-8')
            contract = check_trace_startup(startup, cell, self.h)
            contract['timing_cell'] = cell
            contract['trace_role'] = role_for(cell)
            contract['evidence'] = 'unaltered raw startup and actual trace PID1'
            contract['pid_argv'] = self.pid(cell, 'before')['argv']
            c['save'](directory / 'startup.json', contract)
            self.preserve_native(cell, 'ready', required=True)
            self.observe(cell + '-ready', self._cell_name(cell))
            c['save'](directory / 'startup-buffers.json', [
                line for line in startup.splitlines()
                if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
            container = self.inspect(self._cell_name(cell))
            c['check_container'](container, cell, self.h)
            c['save'](directory / 'container-before.json', container)
            self.active = cell

        def end(self, cell):
            name = self._cell_name(cell)
            self.pid(cell, 'after')
            self.observe(cell + '-after', name)
            assert c['source_binary_hashes'](self.h) == self.hashes
            startup = (self.out / cell / 'native-and-startup.stdout.log').read_text(errors='replace') + \
                      (self.out / cell / 'native-and-startup.stderr.log').read_text(errors='replace')
            c['save'](self.out / cell / 'server-duration-stats.json', c['log_duration_stats'](startup))
            before_stop = self.inspect(name)
            c['check_container'](before_stop, cell, self.h)
            c['save'](self.out / cell / 'container-after.json', before_stop)
            self.command('docker', 'stop', '-t', '90', name)
            server = self.save_logs(cell, 'server', name)
            check_trace_server_log(server)
            c['save'](self.out / cell / 'server-duration-stats.json', c['log_duration_stats'](server))
            assert not self.inspect(name)['State']['Running']
            manifest = _capture_manifest(self.out / cell / 'capture', require=True)
            save(self.out / cell / 'capture-files.json', manifest)
            self.ownership(None)
            self.active = None

    runtime['Runtime'] = TraceRuntime

    def trace_runtime_evidence(p, h, out, cell):
        return _check_runtime_evidence(c, p, h, out, cell)

    analysis['check_runtime_evidence'] = trace_runtime_evidence
    analysis['analyze'].__globals__['check_runtime_evidence'] = trace_runtime_evidence
    analysis['analyze'].__globals__['C'] = c
    analysis['check_runtime_evidence'].__globals__['C'] = c
    return runtime, analysis, c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-timing', action='store_true')
    action.add_argument('--lead-after-timing-restore', action='store_true')
    args = parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    runtime, analysis, c = bind()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == q8_qualified_source(), 'do not rebind qualified source ancestry'
    if args.lead_run_timing:
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_text((HERE / 'README.md').read_text())
        result = runtime['capture'](runtime['Runtime'](h, OUT))
        assert result['status'] == 'captured' and result['restored'], result
        print('FULLTRACE_Q8_ABBA_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)
    else:
        runtime['Runtime'](h, OUT).verify_restored(OUT / 'analysis-live-restoration.json')
        (OUT / 'analysis-r1').mkdir(exist_ok=False)
        result = analysis['analyze'](OUT, p, h)
        assert result['status'] == 'analyzed', result['errors']
        print('FULLTRACE_Q8_ABBA_ANALYZED_RESTORED_DEVELOPMENT_ONLY', flush=True)


def q8_qualified_source():
    if _Q8 is not None:
        return _Q8['QUALIFIED_SOURCE']
    return Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2')


if __name__ == '__main__':
    main()
