#!/usr/bin/env python3
"""Lead-only short-request sampler/trace adapter for the immutable verify-phase binary.

This file deliberately reuses the frozen controller's lifecycle and public exchange.  It
adds only the trace launch delta, a same-role reproduction report, and bounded preservation
of the real request's log/capture window.  It must run on inference-host, never locally.
"""
import argparse
import json
import runpy
import socket
import subprocess
from pathlib import Path

if not __debug__:
    raise RuntimeError('trace probe requires assertions')

ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase')
HERE = Path(__file__).resolve().parent
P = runpy.run_path(str(HERE / 'public-gate.py'))
C = P['C']
K = P['K']
N = P['N']
A = P['A']

OUT = ROOT / 'trace-short55-r1'
MODE = 'router-trace-short55-r1'
CELLS = ('TRACE-A', 'TRACE-W3')
ROLES = {'TRACE-A': 'A', 'TRACE-W3': 'W3'}
BASE_CELLS = {'A': 'RVP-A', 'W3': 'RVP-W3'}
PORT = 18098
TRACE_ENV = {
    'BOLT_VERIFY_CAPTURE_DIR': '/capture',
    'LLAMA_TRACE': '1',
}
NATIVE_LOGS = P['NATIVE_LOGS']


# ---------------------------------------------------------------------------
# Pure launch/request contracts.  These are intentionally usable by the CPU
# fixture suite without constructing a Runtime or touching a live service.


def load(path):
    return C['load'](Path(path))


def save(path, value):
    C['save'](Path(path), value)


def role_for(cell):
    assert cell in ROLES, cell
    return ROLES[cell]


def trace_name(cell):
    assert cell in CELLS, cell
    return C['PREFIX'] + cell.lower()


def base_create_argv(role):
    assert role in BASE_CELLS, role
    path = ROOT / 'qualification-r1' / BASE_CELLS[role] / 'create-argv.json'
    argv = load(path)
    assert isinstance(argv, list) and argv and argv[0:2] == ['docker', 'create']
    return list(argv)


def server_boundary(command):
    old = ('unset BOLT_VERIFY_CAPTURE_DIR LLAMA_TRACE; '
           'export GGML_SYCL_DEBUG=0; echo BOLT_VERIFY_SERVER_START;')
    new = ('export BOLT_VERIFY_CAPTURE_DIR=/capture; '
           'export LLAMA_TRACE=1; '
           'export GGML_SYCL_DEBUG=0; echo BOLT_VERIFY_SERVER_START;')
    assert command.count(old) == 1, 'immutable server boundary not found exactly once'
    return command.replace(old, new)


def capture_path(cell, output=OUT):
    assert cell in CELLS, cell
    return Path(output) / cell / 'capture'


def make_create(cell, output=OUT):
    """Derive a trace argv from the completed RVP role argv.

    The reverse check below proves that the only differences are the fresh
    name, two trace environment entries, one read/write capture bind, and the
    final server environment boundary.  Native commands and all serving argv
    remain byte-for-byte inherited from the completed phase launch.
    """
    role = role_for(cell)
    base = base_create_argv(role)
    argv = list(base)
    name_index = argv.index('--name') + 1
    argv[name_index] = trace_name(cell)

    capture = str(capture_path(cell, output)) + ':/capture'
    case_mount = str(ROOT) + ':/router-verify-phase:ro'
    case_index = argv.index(case_mount)
    argv[case_index:case_index] = ['-v', capture]

    env_indices = [index for index, value in enumerate(argv) if value == '-e']
    assert env_indices
    env_index = env_indices[-1] + 2
    argv[env_index:env_index] = [
        '-e', 'BOLT_VERIFY_CAPTURE_DIR=/capture',
        '-e', 'LLAMA_TRACE=1',
    ]
    argv[-1] = server_boundary(argv[-1])
    assert_trace_argv(base, argv, cell, output)
    return argv


def _env_pairs(argv):
    pairs = []
    for index, value in enumerate(argv):
        if value == '-e':
            assert index + 1 < len(argv)
            pairs.append(argv[index + 1])
    return pairs


def assert_trace_argv(base, derived, cell, output=OUT):
    """Fail closed unless the launch delta is exactly the declared trace delta."""
    assert derived[0:2] == ['docker', 'create']
    assert derived[derived.index('--name') + 1] == trace_name(cell)
    capture = str(capture_path(cell, output)) + ':/capture'
    assert derived.count(capture) == 1
    capture_index = derived.index(capture)
    assert derived[capture_index - 1] == '-v'
    assert all(not value.endswith(':/capture:ro') for value in derived)

    pairs = _env_pairs(derived)
    assert pairs.count('BOLT_VERIFY_CAPTURE_DIR=/capture') == 1
    assert pairs.count('LLAMA_TRACE=1') == 1
    assert 'BOLT_VERIFY_CAPTURE_DIR=/tmp' not in pairs
    assert 'LLAMA_TRACE' not in _env_pairs(base)

    # Remove only the declared insertions, restore the original name and
    # boundary, and require exact equality with the source argv.
    reverse = list(derived)
    del reverse[capture_index - 1:capture_index + 1]
    for pair in ('LLAMA_TRACE=1', 'BOLT_VERIFY_CAPTURE_DIR=/capture'):
        index = reverse.index(pair)
        assert reverse[index - 1] == '-e'
        del reverse[index - 1:index + 1]
    name_index = reverse.index('--name') + 1
    reverse[name_index] = base[base.index('--name') + 1]
    reverse[-1] = base[-1]
    assert reverse == base

    expected = server_boundary(base[-1])
    assert derived[-1] == expected
    assert derived[-3:-1] == ['bash', '-lc']
    assert derived[derived.index(str(ROOT) + ':/router-verify-phase:ro') - 1] == '-v'
    assert base[-1].count('GGML_SYCL_DEBUG=0') == 1
    assert 'BOLT_TRACE_TENSORS=0' in base
    assert 'ONEDNN_VERBOSE=0' in base


def select_trace_jobs(jobs):
    """Select the frozen history through short/diagnostic, preserving order/payloads."""
    jobs = list(jobs)
    assert len(jobs) == 25
    selected = jobs[:23]
    assert len(selected) == 23
    assert [job['stem'] for job in selected[:20]] == [job['stem'] for job in jobs[:20]]
    assert [job['stem'] for job in selected[-3:]] == ['anchor', 'short/ordinary', 'short/diagnostic']
    assert all(job['stem'].startswith('quality/') for job in selected[:20])
    assert [C['qualification_kinds'](job) for job in selected] == (
        ['quality'] * 20 + ['anchor', 'ordinary', 'diagnostic'])
    return selected


def trace_jobs(h):
    selected = select_trace_jobs(P['jobs'](h))
    assert selected == P['jobs'](h)[:23]
    return selected


def check_trace_pid(record, untraced):
    """Validate raw PID evidence without passing it through the diagnostics-off checker."""
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


def compare_same_role(h, directory, role_directory, job, result):
    expected = P['read_result'](h, role_directory, job)
    kind = C['qualification_kinds'](job)
    comparison = P['compare'](kind, result['actual'], expected['actual'])
    return expected, comparison


def _trace_row(job, result, comparison, expected_role):
    kind = result['kind']
    functional = result.get('functional', True)
    row = dict(
        stem=job['stem'],
        kind=kind,
        expected_role=expected_role,
        actual=result['actual'],
        functional=functional,
        comparison=comparison,
        passed=bool(functional and comparison['passed']),
    )
    if kind == 'diagnostic':
        row['speculation'] = result.get(
            'speculation', C['check_speculation_counts'](result['actual']))
        row['actual_ids_source'] = 'choices[0].logprobs.content[*].id'
    return row


# ---------------------------------------------------------------------------
# Evidence helpers and the thin Runtime adapter.


def _base_container(role):
    return load(ROOT / 'qualification-r1' / BASE_CELLS[role] / 'container-before.json')


def _env_map(values):
    result = {}
    for value in values or []:
        key, separator, item = value.partition('=')
        assert separator and key not in result, key
        result[key] = item
    return result


def _bind_source(value):
    return value.split(':', 1)[0]


def check_trace_container(record, cell, output=OUT):
    role = role_for(cell)
    base = _base_container(role)
    argv = make_create(cell, output)
    assert record['Name'] == '/' + trace_name(cell)
    assert record['Config']['Cmd'] == argv[-3:]
    assert record['Config']['Image'] == base['Config']['Image']

    for key, value in base['Config'].items():
        if key not in ('Cmd', 'Env'):
            assert record['Config'].get(key) == value, ('Config', key)
    base_env = _env_map(base['Config'].get('Env', []))
    trace_env = _env_map(record['Config'].get('Env', []))
    assert trace_env == dict(base_env, **TRACE_ENV)

    base_host = dict(base['HostConfig'])
    trace_host = dict(record['HostConfig'])
    base_binds = list(base_host.pop('Binds', []) or [])
    trace_binds = list(trace_host.pop('Binds', []) or [])
    assert trace_host == base_host
    assert all(bind in trace_binds for bind in base_binds)
    assert sum(_bind_source(bind) == str(capture_path(cell, output)) and
               ':'.join(bind.split(':')[1:]).startswith('/capture')
               for bind in trace_binds) == 1
    assert len(trace_binds) == len(base_binds) + 1

    base_mounts = list(base.get('Mounts', []))
    trace_mounts = list(record.get('Mounts', []))
    ordinary_mounts = [mount for mount in trace_mounts
                       if mount.get('Destination') != '/capture']
    assert C['canonical_mounts'](ordinary_mounts) == C['canonical_mounts'](base_mounts)
    capture_mounts = [mount for mount in trace_mounts
                      if mount.get('Destination') == '/capture']
    assert len(capture_mounts) == 1
    mount = capture_mounts[0]
    assert mount['Type'] == 'bind'
    assert mount['Source'] == str(capture_path(cell, output))
    assert mount['RW'] is True
    assert 'ro' not in mount.get('Mode', '').split(',')
    return True


def check_trace_names(existing):
    planned = [trace_name(cell) for cell in CELLS]
    assert len(set(planned)) == len(planned)
    collisions = sorted(set(planned) & set(existing))
    assert not collisions, ('trace container name collision', collisions)
    return dict(mode=MODE, planned_cells=list(CELLS), planned_names=planned,
                collisions=collisions)


def check_trace_server_log(log):
    """Check the serving stream while allowing the requested trace lines."""
    marker = P['MARKER']
    assert log.count(marker) == 1
    native, serving = log.split(marker, 1)
    assert 'BOLT_ROUTE route=' in native
    assert 'GGML_SYCL_DEBUG: 0' in serving
    for forbidden in ('GGML_ASSERT', 'SYCL exception', 'BOLT_VERIFY error'):
        assert forbidden not in log, forbidden
    assert 'BOLT_VERIFY sample ' in serving
    return True


def _save_streams(directory, prefix, streams):
    for stream, data in streams.items():
        (directory / (prefix + '.' + stream + '.log')).write_bytes(data)


def _capture_manifest(directory, require=True):
    files = sorted(path for path in directory.glob('row-*')
                   if path.is_file() and not path.is_symlink())
    assert all(path.name.startswith('row-') for path in files)
    if require:
        assert files, 'no raw CPU sampler rows captured'
    sizes = [path.stat().st_size for path in files]
    assert all(size > 0 and size % 4 == 0 for size in sizes)
    manifest = [dict(name=path.name, path=str(path), bytes=size)
                for path, size in zip(files, sizes)]
    return manifest


class Runtime(K['Runtime']):
    """K.Runtime with trace argv/PID/container checks and no lifecycle rewrite."""

    def preflight(self):
        self.capacity('initial')
        tested = P['check_tested'](self.h)
        proof = P['candidate_proof']()
        self.hashes = C['source_binary_hashes'](self.h)
        self.before_hw = self.h['runner']['hardware']()
        frozen_hw = load(ROOT / 'qualification-r1' / 'hardware-before.json')
        assert self.before_hw == frozen_hw, 'hardware/capability snapshot differs from RVP phase'

        frozen_original = load(ROOT / 'qualification-r1' / 'original-container.json')
        live = self.inspect(self.original_name)
        assert C['same_container'](frozen_original, live)
        assert live['State']['Running']
        self.original, self.original_was_running = live, True
        health = self.h['api']['request'](self.off['OLD']['URL'], '/health')
        assert health['status'] == 'ok'

        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        planned = check_trace_names(existing)
        save(self.out / 'planned-containers.json', planned)
        save(self.out / 'original-container.json', live)
        save(self.out / 'hash-checks-before.json', self.hashes)
        save(self.out / 'hardware-before.json', self.before_hw)
        save(self.out / 'tested-code.json', tested)
        save(self.out / 'candidate-proof-before.json', proof)
        save(self.out / 'source-contracts.json', C['source_evidence'](self.h))
        save(self.out / 'job-selection.json', [
            dict(stem=job['stem'], kind=C['qualification_kinds'](job), payload=job['payload'])
            for job in trace_jobs(self.h)
        ])
        save(self.out / 'environment.json', dict(
            kernel=self.command('uname', '-a'),
            os_release=Path('/etc/os-release').read_text(),
            cpu=self.command('lscpu'),
            memory=Path('/proc/meminfo').read_text(),
            driver=self.command('modinfo', 'xe'),
            source_ids={str(source): (source / 'UPSTREAM_COMMIT').read_text()
                        for source in (N['ACCEPTED'], N['PARENT'], N['CANDIDATE'])}))
        save(self.out / 'original-live.json', live)
        self.observe('run-before', self.original_name)

    def pid(self, cell, phase):
        name = trace_name(cell)
        record = json.loads(self.command(
            'docker', 'exec', name, 'python3', '-B', '-c', C['PID_PROBE']))
        save(self.out / cell / ('pid-' + phase + '.json'), record)
        baseline = load(ROOT / 'qualification-r1' / BASE_CELLS[role_for(cell)] / 'pid-before.json')
        C['check_pid'](baseline, role_for(cell), self.h)
        # Do not call C.check_pid on the trace record: it intentionally rejects
        # the two env entries whose presence is the point of this adapter.
        check_trace_pid(record, baseline)
        maps = self.command('docker', 'exec', name, 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
            '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
        return record

    def preserve_native(self, cell, phase, required=False):
        errors = []
        name = trace_name(cell)
        for tag, filename in NATIVE_LOGS.items():
            destination = self.out / cell / (phase + '-' + filename)
            try:
                assert not destination.exists()
                self.command('docker', 'cp', name + ':/tmp/' + filename, str(destination))
            except BaseException as exc:
                errors.append(dict(tag=tag, error=repr(exc)))
        save(self.out / cell / (phase + '-native-copy.json'), dict(errors=errors))
        if required:
            assert not errors, errors
            for tag, filename in NATIVE_LOGS.items():
                A['check_router_native_output'](
                    (self.out / cell / (phase + '-' + filename)).read_text(), tag)
        return errors

    def begin(self, cell):
        assert cell in CELLS and self.active is None
        self.capacity(cell)
        if not self.prefixes:
            self._stop_original_for_case()
        else:
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
        name = trace_name(cell)
        directory = self.out / cell
        directory.mkdir()
        capture = capture_path(cell, self.out)
        capture.mkdir()
        argv = make_create(cell, self.out)
        base = base_create_argv(role_for(cell))
        save(directory / 'create-argv.json', argv)
        save(directory / 'argv-diff.json', P['positional_diff'](base, argv))

        # Once the original is stopped, the inherited published port must be
        # free before a fresh container is created.  Docker remains the final
        # authority when it installs the bind.
        sock = socket.socket()
        try:
            sock.bind(('127.0.0.1', PORT))
        except OSError as exc:
            sock.close()
            raise C['InfraFailure']('trace port is not free: ' + repr(exc)) from exc
        else:
            sock.close()
            save(directory / 'port-check.json', dict(host='127.0.0.1', port=PORT, free=True))
        self.created.append(cell)

        self.command(*argv, timeout=60)
        created = self.inspect(name)
        check_trace_container(created, cell, self.out)
        self.command('docker', 'start', name)
        save(directory / 'models.json', self.ready(self.h['api'], name))

        startup = self.save_logs(cell, 'native-and-startup', name)
        raw = subprocess.run(['docker', 'logs', name], capture_output=True,
                             timeout=90, check=True)
        for stream in ('stdout', 'stderr'):
            (directory / ('native-and-startup.raw.' + stream + '.log')).write_bytes(
                getattr(raw, stream))
        raw_startup = (raw.stdout + raw.stderr).decode('utf-8')
        baseline_cell = BASE_CELLS[role_for(cell)]
        startup_contract = P['check_startup'](raw_startup, baseline_cell, self.h)
        startup_contract['trace_role'] = role_for(cell)
        startup_contract['evidence'] = 'raw startup and actual trace PID1 below; no evidence rewriting'
        startup_contract['pid_argv'] = self.pid(cell, 'before')['argv']
        save(directory / 'startup.json', startup_contract)
        self.preserve_native(cell, 'ready', required=True)
        self.observe(cell + '-ready', name)
        save(directory / 'startup-buffers.json', [
            line for line in raw_startup.splitlines()
            if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
        check_trace_container(self.inspect(name), cell, self.out)
        save(directory / 'container-before.json', self.inspect(name))
        self.active = cell

    def end(self, cell):
        name = trace_name(cell)
        self.pid(cell, 'after')
        self.observe(cell + '-after', name)
        assert C['source_binary_hashes'](self.h) == self.hashes
        startup = (self.out / cell / 'native-and-startup.stdout.log').read_text(errors='replace') + \
            (self.out / cell / 'native-and-startup.stderr.log').read_text(errors='replace')
        save(self.out / cell / 'server-duration-stats.json', C['log_duration_stats'](startup))
        before_stop = self.inspect(name)
        check_trace_container(before_stop, cell, self.out)
        save(self.out / cell / 'container-after.json', before_stop)
        self.command('docker', 'stop', '-t', '90', name)
        server = self.save_logs(cell, 'server', name)
        check_trace_server_log(server)
        save(self.out / cell / 'server-duration-stats.json', C['log_duration_stats'](server))
        assert not self.inspect(name)['State']['Running']
        self.ownership(None)
        self.active = None

    def _preserve_restore_log(self, cell, container_name):
        super()._preserve_restore_log(cell, container_name)
        self.preserve_native(cell, 'restored', required=False)

    def verify_restored(self, destination):
        try:
            record = super().verify_restored(destination)
        except BaseException:
            if not Path(destination).is_file():
                raise
            record = load(destination)
        errors = list(record.get('errors', []))
        try:
            record['candidate_proof_after'] = P['candidate_proof']()
        except BaseException as exc:
            errors.append(dict(check='candidate-proof-after', error=repr(exc)))
        record['errors'] = errors
        record['passed'] = bool(record.get('passed')) and not errors
        save(destination, record)
        if not record['passed']:
            raise C['InfraFailure']('trace restoration/protected proof failed: ' + repr(errors))
        return record


def _snapshot_final_request(rt, cell):
    directory = rt.out / cell
    before = rt.logs(trace_name(cell))
    _save_streams(directory, 'short-diagnostic-log-prefix', before)
    return before


def _finish_final_request(rt, cell, before):
    directory = rt.out / cell
    after = rt.logs(trace_name(cell))
    fresh = C['fresh_logs'](before, after)
    _save_streams(directory, 'short-diagnostic-log-after', after)
    _save_streams(directory, 'short-diagnostic-log-fresh', fresh)
    save(directory / 'short-diagnostic-log-window.json', {
        'prefix_bytes': {key: len(value) for key, value in before.items()},
        'after_bytes': {key: len(value) for key, value in after.items()},
        'fresh_bytes': {key: len(value) for key, value in fresh.items()},
        'fresh_source': 'C.fresh_logs(prefix, after); no log normalization',
    })
    manifest = _capture_manifest(directory / 'capture', require=True)
    save(directory / 'capture-files.json', manifest)


def _report(cell):
    return dict(
        cell=cell,
        role=role_for(cell),
        expected_role=role_for(cell),
        passed=False,
        jobs=[],
        errors=[],
        fidelity_failures=[],
        exact_same_role=True,
        cross_role_comparison='not required; W3 may differ from A',
        actual_ids_source='choices[0].logprobs.content[*].id',
        retokenized_output_as_ids_claimed=False,
        raw_f32_equality_claimed=False,
        timing_permitted=False,
    )


def collect(rt, cell):
    report = _report(cell)
    jobs = trace_jobs(rt.h)
    directory = rt.out / cell
    try:
        for job in jobs:
            rt.capacity(cell + '-' + job['stem'].replace('/', '-'))
            final = job['stem'] == 'short/diagnostic'
            before = _snapshot_final_request(rt, cell) if final else None
            try:
                try:
                    result = rt.run_job(cell, job, expected_for_compare=False)
                except C['FidelityFailure']:
                    # Preserve and validate a valid mismatch; a prompt/input
                    # failure still raises from read_public and aborts closed.
                    P['read_public'](rt.h, directory, job)
                    result = P['read_result'](rt.h, directory, job)
                _, comparison = compare_same_role(
                    rt.h, directory, ROOT / 'qualification-r1' / BASE_CELLS[role_for(cell)],
                    job, result)
                row = _trace_row(job, result, comparison, role_for(cell))
                report['jobs'].append(row)
                if not row['passed']:
                    report['fidelity_failures'].append(dict(
                        stem=job['stem'], comparison=comparison,
                        functional=row['functional']))
            except BaseException as exc:
                report['errors'].append(dict(stem=job['stem'], error=repr(exc)))
                raise
            finally:
                if final:
                    _finish_final_request(rt, cell, before)
            save(directory / 'trace-report.json', report)
        assert [row['stem'] for row in report['jobs']] == [job['stem'] for job in jobs]
        report['passed'] = len(report['jobs']) == 23 and not report['errors'] and not report['fidelity_failures']
    finally:
        save(directory / 'trace-report.json', report)
    return []


def _check_window(directory):
    before = {stream: (directory / ('short-diagnostic-log-prefix.' + stream + '.log')).read_bytes()
              for stream in ('stdout', 'stderr')}
    after = {stream: (directory / ('short-diagnostic-log-after.' + stream + '.log')).read_bytes()
             for stream in ('stdout', 'stderr')}
    fresh = {stream: (directory / ('short-diagnostic-log-fresh.' + stream + '.log')).read_bytes()
             for stream in ('stdout', 'stderr')}
    assert C['fresh_logs'](before, after) == fresh
    server = {stream: (directory / ('server.' + stream + '.log')).read_bytes()
              for stream in ('stdout', 'stderr')}
    assert all(server[stream].startswith(after[stream]) for stream in ('stdout', 'stderr'))
    return True


def check_cell_evidence(h, cell):
    directory = OUT / cell
    role = role_for(cell)
    argv = make_create(cell, OUT)
    assert load(directory / 'create-argv.json') == argv
    assert load(directory / 'argv-diff.json') == P['positional_diff'](
        base_create_argv(role), argv)
    baseline = load(ROOT / 'qualification-r1' / BASE_CELLS[role] / 'pid-before.json')
    for phase in ('before', 'after'):
        check_trace_pid(load(directory / ('pid-' + phase + '.json')), baseline)
    for phase in ('before', 'after'):
        check_trace_container(load(directory / ('container-' + phase + '.json')), cell, OUT)
    preserved = load(directory / 'preserved-container.json')
    check_trace_container(preserved, cell, OUT)
    assert not preserved['State']['Running']
    assert C['same_container'](load(directory / 'container-before.json'),
                               load(directory / 'container-after.json'))

    startup_raw = ''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_text()
                          for stream in ('stdout', 'stderr'))
    P['check_startup'](startup_raw, BASE_CELLS[role], h)
    check_trace_server_log(''.join((directory / ('server.' + stream + '.log')).read_text(errors='replace')
                                   for stream in ('stdout', 'stderr')))
    check_trace_server_log(''.join((directory / ('restored-final.' + stream + '.log')).read_text(errors='replace')
                                   for stream in ('stdout', 'stderr')))
    for stream in ('stdout', 'stderr'):
        native = (directory / ('native-and-startup.' + stream + '.log')).read_bytes()
        server = (directory / ('server.' + stream + '.log')).read_bytes()
        restored = (directory / ('restored-final.' + stream + '.log')).read_bytes()
        assert server.startswith(native)
        assert restored.startswith(server)
    for tag, filename in NATIVE_LOGS.items():
        ready = (directory / ('ready-' + filename)).read_bytes()
        restored = (directory / ('restored-' + filename)).read_bytes()
        assert ready == restored
        A['check_router_native_output'](ready.decode(), tag)
    _check_window(directory)
    manifest = load(directory / 'capture-files.json')
    assert manifest == _capture_manifest(directory / 'capture', require=True)
    return True


def analyze(h):
    summary = load(OUT / 'summary.json')
    assert summary['status'] == 'captured' and summary['restored']
    assert summary['mode'] == MODE
    assert summary['order'] == summary['completed'] == list(CELLS)
    assert summary['timing_permitted'] is False and summary['measured_attempts'] == []

    # This independent check is deliberately first.  No request/result is
    # reread if exact restoration and protected hashes are not live.
    Runtime(h, OUT, MODE).verify_restored(OUT / 'independent-restoration.json')
    analysis = dict(
        status='failed',
        independently_restored=True,
        reports={},
        reproduced_cells=[],
        timing_permitted=False,
        promotion_permitted=False,
        raw_f32_equality_claimed=False,
        retokenized_output_as_ids_claimed=False,
    )
    try:
        jobs = trace_jobs(h)
        for cell in CELLS:
            check_cell_evidence(h, cell)
            report = load(OUT / cell / 'trace-report.json')
            assert [row['stem'] for row in report['jobs']] == [job['stem'] for job in jobs]
            recomputed = _report(cell)
            for job in jobs:
                result = P['read_result'](h, OUT / cell, job)
                expected = P['read_result'](
                    h, ROOT / 'qualification-r1' / BASE_CELLS[role_for(cell)], job)
                comparison = P['compare'](
                    C['qualification_kinds'](job), result['actual'], expected['actual'])
                row = _trace_row(job, result, comparison, role_for(cell))
                captured = next(item for item in report['jobs'] if item['stem'] == job['stem'])
                assert captured == row, ('capture report differs from raw evidence', job['stem'])
                recomputed['jobs'].append(row)
                if not row['passed']:
                    recomputed['fidelity_failures'].append(dict(
                        stem=job['stem'], comparison=comparison,
                        functional=row['functional']))
            recomputed['passed'] = (len(recomputed['jobs']) == 23 and
                                    not recomputed['errors'] and
                                    not recomputed['fidelity_failures'])
            assert report == recomputed
            analysis['reports'][cell] = report
            if report['passed']:
                analysis['reproduced_cells'].append(cell)
        analysis['status'] = 'analyzed'
    except BaseException as exc:
        analysis['error'] = repr(exc)
        raise
    finally:
        save(OUT / 'analysis.json', analysis)
    print('ANALYZED_AFTER_RESTORE reproduced_cells=' + repr(analysis['reproduced_cells']) +
          ' timing_permitted=False', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-trace', action='store_true')
    action.add_argument('--lead-after-restore', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'inference-host'
    assert Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    if args.lead_run_trace:
        assert not OUT.exists(), 'trace output already exists; never reuse a capture'
        OUT.mkdir(exist_ok=False)
        K['capture'](Runtime(h, OUT, MODE), CELLS, collect)
        print('CAPTURED_RESTORED analysis_pending=True timing_permitted=False', flush=True)
    else:
        analyze(h)


if __name__ == '__main__':
    main()
