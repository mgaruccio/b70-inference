#!/usr/bin/env python3
"""LEAD/REMOTE ONLY: candidate public qualification, then independent restored analysis.

Reuse the frozen width-sweep lifecycle and public exchange, not its timing path.
No helper globals, actual PID records, or runtime log bytes are rewritten.
"""
import argparse
import hashlib
import itertools
import json
import re
import runpy
import shlex
import socket
import subprocess
from pathlib import Path

if not __debug__:
    raise RuntimeError('qualification requires assertions')
ROOT = Path('/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons')
PHASE = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-verify-phase'
OLD = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-width-sweep'
N9_ROOT = ROOT / 'n9'
OLD_N9 = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-n9-control'
PARENT = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1')
CANDIDATE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r1')
for filename, digest in (
    ('contracts.py', '80547ac6ca7689e259e9a9e70f3d0f7dc0fbb8248e763e99f7bbc0d6606a877f'),
    ('controller.py', 'e8f4a537b355b5315a08fdb410682f4ac5aef13878a723f51ca4c7812bd931d8'),
):
    assert hashlib.sha256((OLD / filename).read_bytes()).hexdigest() == digest, filename
K = runpy.run_path(str(OLD / 'controller.py'))
C = K['C']  # Inherited run_job's exception identity is essential.
# These namespaces keep their original paths; use explicit new paths with generic helpers.
N = runpy.run_path(str(PHASE / 'run-trace.py'))
A = runpy.run_path(str(PHASE / 'analyze-trace.py'))
Q = runpy.run_path(str(ROOT / 'native.py'))
SINGLETON_MARKER = 'BOLT_VERIFY_SINGLETONS_NATIVE_PASSED'
OUT = ROOT / 'qualification-r1'
MODE = 'verify-singletons-qualification-r1'
ROLES = {'VSA-A': 'A', 'VSA-W3': 'W3', 'VSA-W7': 'W7'}
CELLS = tuple(ROLES)
R3_MARKER = 'BOLT_VERIFY_ROUTER_VERIFY_PHASE_NATIVE_PASSED'
MARKER = R3_MARKER
NATIVE_LOGS = {'router_single2': 'originalrouter-single2-native.log',
               'router_columns8_r3': 'router-verify-phase-native.log',
               'router_prefill': 'router-prefill-native.log',
               'q8': 'verify-singletons-q8.log', 'iq2': 'verify-singletons-iq2.log'}
FILES = ('public-gate.py', 'test-public-gate.py', 'cpu-public-gate.sh', 'qualification.md',
         'n9/run-control.py', 'n9/n9-equivalence.py', 'n9/test-control.py',
         'native.py', 'test-native.py', 'test-guards.py', 'check-source.py',
         'source-hashes.json', 'verify-singletons.patch', 'build-candidate.sh',
         'trace-probe.py', 'test-trace-probe.py', 'analyze-first-frame.py',
         'test-first-frame.py')
BUILD_FILES = ('source-diff.json', 'parent-binaries.json', 'candidate-binaries.json',
               'post-build-verification.log', 'runtime-precheck.log', 'build-attempt-r1/exit-code.txt')
PHASE_FILES = ('run-trace.py', 'analyze-trace.py', 'test-trace.py', 'build-candidate.sh', 'router-verify-phase.patch',
               'protected-source-hashes.json', 'source-hashes.json', 'accepted-binaries.json',
               'parent-binaries.json', 'columns8-binaries.json', 'candidate-binaries.json')
HTTP_SUFFIXES = ('.request.json', '.request-body', '.http-request.json', '.http.json', '.http-body')
NEW_NATIVE = (
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^router_columns8=1,(?!router_single2=1,weight=blk\\.48\\.ffn_gate_inp\\.weight,layout=0,type_a=f32,type_b=f32,m=496,n=9,)' > /tmp/router-verify-phase-native.log 2>&1; "
    "grep -Fqx '  61/61 tests passed' /tmp/router-verify-phase-native.log; "
    "count=$(grep -Fc '  61/61 tests passed' /tmp/router-verify-phase-native.log || true); test \"$count\" -eq 1; "
    "python3 -B /router-verify-phase/analyze-trace.py --tag router_columns8_r3 /tmp/router-verify-phase-native.log; "
    "GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-backend-ops test -b SYCL0 -o MUL_MAT "
    "-p '^router_prefill=1,' > /tmp/router-prefill-native.log 2>&1; "
    "grep -Fqx '  2/2 tests passed' /tmp/router-prefill-native.log; "
    "count=$(grep -Fc '  2/2 tests passed' /tmp/router-prefill-native.log || true); test \"$count\" -eq 1; "
    "python3 -B /router-verify-phase/analyze-trace.py --tag router_prefill /tmp/router-prefill-native.log; "
    "echo " + R3_MARKER + "; "
)

def singleton_native_command():
    full = Q['native_command']()
    assert full.startswith(N['NATIVE_COMMAND'])
    tail = full[len(N['NATIVE_COMMAND']):]
    tee = '2>&1 | tee "/capture/verify-singletons-$kind.log"'
    assert tail.count(tee) == 1 and tail.count('/capture/') == 3
    tail = tail.replace(tee, '> "/capture/verify-singletons-$kind.log" 2>&1')
    return tail.replace('/capture/', '/tmp/') + '\n'


def check_native_log(tag, log):
    assert tag in NATIVE_LOGS
    return Q['check'](log, tag) if tag in ('q8', 'iq2') else A['check_router_native_output'](log, tag)


def name(cell):
    assert cell in ROLES
    return C['PREFIX'] + cell.lower()


def jobs(h):
    selected = C['qualification_jobs'](h)
    assert len(selected) == len({j['stem'] for j in selected}) == 25
    assert [C['qualification_kinds'](j) for j in selected].count('quality') == 20
    return selected


def check_names(existing):
    planned = [name(cell) for cell in CELLS]
    assert len(set(planned)) == 3 and not (set(planned) & set(existing)), 'new cell collision; no reuse'
    return dict(mode=MODE, planned_cells=list(CELLS), planned_names=planned, collisions=[])


def positional_diff(base, derived):
    # Inserting a bind changes length; retain EVERY positional difference, including the tail.
    return [dict(index=i, base=a, derived=b)
            for i, (a, b) in enumerate(itertools.zip_longest(base, derived)) if a != b]


def make_create(cell, h):
    base = C['make_create'](ROLES[cell], h)
    argv = list(base)
    argv[argv.index('--name') + 1] = name(cell)
    mount = str(N['PARENT']) + ':/src:ro'  # Frozen width-sweep launch source, not this patch's parent.
    assert argv.count(mount) == 1 and argv[argv.index(mount) - 1] == '-v'
    argv[argv.index(mount)] = str(CANDIDATE) + ':/src:ro'
    p, off = h['off']['P'], h['off']
    validator = 'python3 -B -c ' + shlex.quote(
        "import pathlib,runpy; runpy.run_path('/iq2-single2-gate/run-trace.py')['check_router_native_output']("
        "pathlib.Path('/tmp/router-single2-native.log').read_text())") + '; '
    old_native = p['ROUTER_NATIVE_COMMAND']
    old_native_log = '/tmp/router-single2-native.log'
    copied_native_log = '/tmp/originalrouter-single2-native.log'
    assert old_native.count(validator) == 1
    assert "-p '^router_single2=1,' 2>&1 | tee " + old_native_log + '; ' in old_native
    updated = old_native.replace(validator,
        'python3 -B /router-verify-phase/analyze-trace.py --tag router_single2 ' + old_native_log + '; ')
    copy_original = 'cp ' + old_native_log + ' ' + copied_native_log + '; '
    assert base[-1].count(old_native + off['OFF_BOUNDARY']) == 1
    argv[-1] = base[-1].replace(old_native + off['OFF_BOUNDARY'],
                                 updated + copy_original + NEW_NATIVE + singleton_native_command() + off['OFF_BOUNDARY'])
    assert argv[-1].startswith(off['NATIVE_ON'])
    assert argv[-3:-1] == ['bash', '-lc']
    argv[-4:-4] = ['-v', str(PHASE) + ':/router-verify-phase:ro', '-v', str(ROOT) + ':/verify-singletons:ro']
    assert C['server_argv'](argv) == C['expected_pid_argv'](ROLES[cell], h)
    C['diagnostics_off_server'](argv[-1])
    # Reverse precisely the four allowed changes; no other argv/env/cap/model change is allowed.
    reverse = list(argv)
    del reverse[-8:-4]
    reverse[reverse.index('--name') + 1] = base[base.index('--name') + 1]
    reverse[reverse.index(str(CANDIDATE) + ':/src:ro')] = mount
    reverse[-1] = base[-1]
    assert reverse == base
    return argv


def environment_values(values):
    result = {}
    for entry in values:
        key, separator, value = entry.partition('=')
        assert separator and key and key not in result, 'invalid or duplicate environment key'
        result[key] = value
    return result


def check_container(record, cell, h):
    base = C['load'](C['QUALIFY_DIR'] / ROLES[cell] / 'container-before.json')
    assert record['Name'] == '/' + name(cell)
    assert record['Config']['Cmd'] == make_create(cell, h)[-3:]
    assert re.fullmatch(r'[0-9a-f]{64}', record['Id'])
    assert record['Config']['Hostname'] == record['Id'][:12]
    assert {k: v for k, v in record['Config'].items() if k not in ('Cmd', 'Env', 'Hostname')} == {
        k: v for k, v in base['Config'].items() if k not in ('Cmd', 'Env', 'Hostname')}
    assert environment_values(record['Config']['Env']) == environment_values(base['Config']['Env'])
    expected_host, actual_host = dict(base['HostConfig']), dict(record['HostConfig'])
    for host in (expected_host, actual_host):
        oom_disable = host.pop('OomKillDisable')
        assert oom_disable is None or oom_disable is False, 'OOM killer must remain enabled'
    old_mount, new_mount = str(N['PARENT']) + ':/src:ro', str(CANDIDATE) + ':/src:ro'
    binds = list(expected_host.pop('Binds'))
    assert binds.count(old_mount) == 1
    binds[binds.index(old_mount)] = new_mount
    binds.append(str(PHASE) + ':/router-verify-phase:ro')
    binds.append(str(ROOT) + ':/verify-singletons:ro')
    assert sorted(actual_host.pop('Binds')) == sorted(binds)
    assert actual_host == expected_host
    expected_mounts = [dict(m, Source=str(CANDIDATE)) if m['Destination'] == '/src' else dict(m)
                       for m in base['Mounts']]
    expected_mounts.append(dict(Type='bind', Source=str(PHASE), Destination='/router-verify-phase',
                                Mode='ro', RW=False, Propagation='rprivate'))
    expected_mounts.append(dict(Type='bind', Source=str(ROOT), Destination='/verify-singletons',
                                Mode='ro', RW=False, Propagation='rprivate'))
    assert C['canonical_mounts'](record['Mounts']) == C['canonical_mounts'](expected_mounts)
    return True


def check_startup(log, cell, h):
    # RAW docker stdout/stderr, without --timestamps. Never strip/normalize actual evidence.
    off, p, old = h['off'], h['off']['P'], h['off']['OLD']
    off['check_off_log'](log)
    native = runpy.run_path(str(p['WIDTH1'] / 'analyze-controls.py'))
    native['parse_native_gate'](log, 'C0')
    for gate in ('iq2', 'q8', 'attention', 'attention_view'):
        p['check_' + gate + '_native_gate'](log)
    assert 'verify_singletons=1,' not in log, 'singleton output must stay in its raw native files'
    markers = (p['ATTENTION_VIEW_NATIVE_MARKER'], p['ROUTER_NATIVE_MARKER'], MARKER,
               SINGLETON_MARKER, p['MARKER'])
    assert all(log.count(marker) == 1 for marker in markers)
    assert [log.index(marker) for marker in markers] == sorted(log.index(marker) for marker in markers)
    segment = log.split(markers[0], 1)[1].split(markers[1], 1)[0]
    router = A['check_router_native_output'](segment, 'router_single2')
    assert 'router_columns8=1,' not in log, '61-case output must stay in its raw native file'
    assert 'router_prefill=1,' not in log, 'prefill output must stay in its raw native file'
    assert 'call ggml_sycl_ssm_scan_prefill:' not in log.split(p['MARKER'], 1)[0]
    model = log.split(p['MARKER'], 1)[1]
    width = C['width_for'](ROLES[cell])
    if width is None:
        result = off['N']['check_model_startup'](model)
    else:
        old['check_model_log'](model, 'C')
        assert [int(m.group(1)) for m in old['N_RS_SEQ_RE'].finditer(model)][:2] == [width, 0]
        params, block = old['DFLASH_PARAMS_RE'].search(model), old['DFLASH_BLOCK_RE'].search(model)
        assert params and params.groups() == (str(width), '0')
        assert block and block.group(1) == '8'
        spec = model.split('common_speculative_impl_draft_dflash', 1)[1].split('speculative decoding context', 1)[0]
        assert 'clamp' not in spec.lower()
        assert 'R (f32)' in model and 'S (f32)' in model
        for key in ('n_outputs_max', 'n_outputs_max_per_seq'):
            match = re.search(r'llama_context: ' + key + r'\s*=\s*(\d+)', model)
            assert match and match.group(1) == str(width + 1), (key, match)
        for key, value in (('DNN', 1), ('FUSION', 1), ('ESIMD', 1), ('OPT', 1), ('VMM', 1), ('GRAPH', 0)):
            assert f'GGML_SYCL_ENABLE_{key}: {value}' in model
        result = dict(width=width, target_n_rs_seq=width, draft_n_rs_seq=0,
                      n_max=width, n_min=0, block_size=8, n_outputs_max=width + 1, n_outputs_max_per_seq=width + 1)
    assert 'BOLT_VERIFY error' not in log
    return dict(result, router_single2=router, evidence='unaltered raw startup and actual PID1')


def candidate_proof():
    control = runpy.run_path(str(N9_ROOT / 'run-control.py'))
    assert control['PARENT'] == PARENT and control['CANDIDATE'] == CANDIDATE
    return control['candidate_proof']()

def check_n9_equivalence(h):
    # Import lazily: the N9 controller may consume candidate_proof(), but the public
    # freeze must consume its NEW, read-only checker, never the old run-r3 proof.
    assert (N9_ROOT / 'n9-equivalence.py').read_bytes() == (OLD_N9 / 'n9-equivalence.py').read_bytes()
    control = runpy.run_path(str(N9_ROOT / 'run-control.py'))
    assert control['ROOT'] == N9_ROOT and control['OUT'] == N9_ROOT / 'run-r1'
    assert control['BACKENDS'] == {'VSN9-PARENT': PARENT, 'VSN9-CANDIDATE': CANDIDATE}
    return control['check_current_equivalence']()


def freeze(h):
    check_n9_equivalence(h)
    hashes = C['code_hashes'](h)
    hashes.update({str(ROOT / file): C['sha'](ROOT / file) for file in FILES + BUILD_FILES})
    hashes.update({str(PHASE / file): C['sha'](PHASE / file) for file in PHASE_FILES})
    hashes.update({str(N9_ROOT / file): C['sha'](N9_ROOT / file)
                   for file in ('run-control.py', 'test-control.py', 'n9-equivalence.py')})
    hashes.update({str(OLD_N9 / 'n9-equivalence.py'): C['sha'](OLD_N9 / 'n9-equivalence.py')})
    callsite = ROOT.parent / '20261018-nemotron35-bolt-lilicorr-router-callsite'
    for file in ('n9/run-control.py', 'n9/n9-equivalence.py', 'check-source.py',
                 'router-callsite.patch', 'source-hashes.json'):
        hashes[str(callsite / file)] = C['sha'](callsite / file)
    parser = ROOT.parent / '20261017-nemotron35-bolt-lilicorr-route-trace/analyze-trace.py'
    hashes[str(parser)] = C['sha'](parser)
    refs = C['reference_hashes'](h)
    paths = [C['QUALIFY_DIR'] / file for file in ('summary.json', 'qualification.json')]
    for role in ROLES.values():
        directory = C['QUALIFY_DIR'] / role
        paths += [directory / file for file in ('pid-before.json', 'pid-after.json', 'create-argv.json',
                                                 'container-before.json', 'native-and-startup.stdout.log',
                                                 'native-and-startup.stderr.log')]
        for job in jobs(h):
            paths += [C['artifact'](directory / job['stem'], suffix) for suffix in HTTP_SUFFIXES]
    n9_run = N9_ROOT / 'run-r1'
    paths += [n9_run / file for file in ('equivalence.json', 'analysis.json', 'inputs-before.json',
              'independent-restoration.json', 'summary.json', 'paired-argv-difference.json')]
    paths += [OLD_N9 / 'r2/candidate-binaries.json']
    for cell in ('VSN9-PARENT', 'VSN9-CANDIDATE'):
        paths += [n9_run / cell / file for file in (
            'create-argv.json', 'container-before.json', 'stdout.log', 'stderr.log',
            'exit-code.json', 'native-result.json')]
    refs.update({str(path): C['sha'](path) for path in paths})
    # N9's checker revalidates live seeded runner source/binaries and raw equality
    # against its proof on every freeze; no transferred proof or fabricated manifest.
    return dict(hashes=hashes, references=refs, source_binaries=C['source_binary_hashes'](h),
                candidate=candidate_proof())


def check_tested(h):
    candidate_proof()  # Require the exact source/build proof before every live public launch.
    tested = C['load'](ROOT / 'preparation/public-tested-code.json')
    assert tested['tests_passed'] is True and tested['freeze'] == freeze(h), 'CPU-frozen code/reference/source/binary drift'
    return tested


class Runtime(K['Runtime']):
    def preflight(self):
        self.capacity('initial')
        check_tested(self.h)
        check_n9_equivalence(self.h)
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
        C['save'](self.out / 'planned-containers.json', check_names(existing))
        C['save'](self.out / 'source-contracts.json', C['source_evidence'](self.h))
        C['save'](self.out / 'environment.json', dict(
            kernel=self.command('uname', '-a'), os_release=Path('/etc/os-release').read_text(),
            cpu=self.command('lscpu'), memory=Path('/proc/meminfo').read_text(), driver=self.command('modinfo', 'xe'),
            source_ids={str(source): (source / 'UPSTREAM_COMMIT').read_text()
                        for source in (N['ACCEPTED'], PARENT, CANDIDATE)}))
        C['save'](self.out / 'original-live.json', live)
        self.observe('run-before', self.original_name)

    def pid(self, cell, phase):
        record = json.loads(self.command('docker', 'exec', name(cell), 'python3', '-B', '-c', C['PID_PROBE']))
        C['save'](self.out / cell / ('pid-' + phase + '.json'), record)
        C['check_pid'](record, ROLES[cell], self.h)  # Pass the ACTUAL record, unchanged.
        maps = self.command('docker', 'exec', name(cell), 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
            '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
        return record

    def preserve_native(self, cell, phase, required=False):
        # Preserve raw native bytes even on partial startup or restoration failure.
        errors = []
        for tag, filename in NATIVE_LOGS.items():
            destination = self.out / cell / (phase + '-' + filename)
            try:
                assert not destination.exists()
                self.command('docker', 'cp', name(cell) + ':/tmp/' + filename, str(destination))
            except Exception as exc:
                errors.append(dict(tag=tag, error=repr(exc)))
        C['save'](self.out / cell / (phase + '-native-copy.json'), dict(errors=errors))
        if required:
            assert not errors, errors
            for tag, filename in NATIVE_LOGS.items():
                check_native_log(tag, (self.out / cell / (phase + '-' + filename)).read_bytes().decode())

    def begin(self, cell):
        assert cell in ROLES and self.active is None
        self.capacity(cell)
        if not self.prefixes:
            self._stop_original_for_case()
        else:
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
        directory = self.out / cell
        directory.mkdir()
        argv = make_create(cell, self.h)
        C['save'](directory / 'create-argv.json', argv)
        C['save'](directory / 'argv-diff.json', positional_diff(C['make_create'](ROLES[cell], self.h), argv))
        self.created.append(cell)
        self.command(*argv, timeout=60)
        check_container(self.inspect(name(cell)), cell, self.h)
        self.command('docker', 'start', name(cell))
        C['save'](directory / 'models.json', self.ready(self.h['api'], name(cell)))
        self.save_logs(cell, 'native-and-startup')  # Preserve inherited timestamped evidence too.
        raw = subprocess.run(['docker', 'logs', name(cell)], capture_output=True, timeout=90, check=True)
        for stream in ('stdout', 'stderr'):
            (directory / ('native-and-startup.raw.' + stream + '.log')).write_bytes(getattr(raw, stream))
        startup = (raw.stdout + raw.stderr).decode('utf-8')
        contract = check_startup(startup, cell, self.h)
        self.preserve_native(cell, 'ready', required=True)
        contract['pid_argv'] = self.pid(cell, 'before')['argv']
        C['save'](directory / 'startup.json', contract)
        self.observe(cell + '-ready', name(cell))
        C['save'](directory / 'startup-buffers.json', [line for line in startup.splitlines()
                  if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
        live = self.inspect(name(cell))
        check_container(live, cell, self.h)
        C['save'](directory / 'container-before.json', live)
        self.active = cell

    def end(self, cell):
        try:
            super().end(cell)
        finally:
            check_tested(self.h)

    def _preserve_restore_log(self, cell, container_name):
        try:
            super()._preserve_restore_log(cell, container_name)
        finally:
            # docker cp works on stopped containers, including failed native/startup runs.
            # Missing files are recorded, never a reason to skip original restoration.
            self.preserve_native(cell, 'restored')

    def verify_restored(self, destination):
        errors = []
        try:
            record = super().verify_restored(destination)
        except Exception as exc:
            record = C['load'](destination)
            errors.append(dict(check='legacy-restoration', error=repr(exc)))
        for key, fn in (
            ('cpu_frozen_candidate', lambda: check_tested(self.h)['freeze']['candidate']),
            ('paired_n9_equivalence', lambda: check_n9_equivalence(self.h)),
            ('preserved_cells', self.stopped_cells),
        ):
            try:
                record[key] = fn()
            except Exception as exc:
                errors.append(dict(check=key, error=repr(exc)))
        record['errors'] += errors
        record['passed'] = not record['errors']
        C['save'](destination, record)
        if not record['passed']:
            raise C['InfraFailure']('restoration/candidate proof failed: ' + repr(record['errors']))
        return record

    def stopped_cells(self):
        result = {}
        for cell in CELLS:
            directory = self.out / cell
            if not directory.exists():
                continue  # A failed preflight/begin need not have created every arm.
            live = self.inspect(name(cell))
            check_container(live, cell, self.h)
            assert not live['State']['Running']
            if (directory / 'container-before.json').exists():
                assert C['same_container'](C['load'](directory / 'container-before.json'), live)
            result[cell] = live
        return result


def read_public(h, directory, job):
    stem = directory / 'public' / job['stem']
    read_http, artifact = h['analysis']['read_http'], C['artifact']
    template = h['off']['L']['apply_request'](job['payload'])
    rendered = json.loads(read_http(artifact(stem, '.template'), template, '/apply-template'))
    assert rendered == C['load'](artifact(stem, '.template.response.json'))
    tokenize = h['off']['L']['tokenize_request'](rendered)
    response = json.loads(read_http(artifact(stem, '.tokenize'), tokenize, '/tokenize'))
    assert response == C['load'](artifact(stem, '.tokenize.response.json'))
    ids, expected = h['off']['L']['token_ids'](response), h['off']['reference_prompt']('A0', job)
    assert ids == expected, 'public prompt ID mismatch/corrupt input; abort, never retry'
    assert C['load'](artifact(stem, '.prompt-comparison.json')) == dict(
        actual_ids=ids, expected_ids=expected, comparison=C['first_difference'](ids, expected),
        retokenized_output_as_ids_claimed=False)
    return ids


def read_result(h, directory, job):
    prompt = h['off']['reference_prompt']('A0', job)
    raw = h['analysis']['read_http'](directory / job['stem'], job['payload'], '/v1/chat/completions')
    kind = C['qualification_kinds'](job)
    if kind == 'diagnostic':
        actual = h['off']['diagnostic_view'](json.loads(raw), prompt)
        return dict(kind=kind, actual=actual, speculation=C['check_speculation_counts'](actual))
    result = h['analysis']['normalize'](raw, job['payload'])
    C['check_output'](result, job, prompt_ids=prompt)
    return dict(kind=kind, actual=C['result_view'](result), result=result,
                functional=C['check_functional'](h, job, result))


def compare(kind, actual, expected):
    if kind == 'diagnostic':
        return C['diagnostic_compare'](actual, expected)
    return dict(passed=C['result_view'](actual) == C['result_view'](expected),
                actual=C['result_view'](actual), expected=C['result_view'](expected))


def new_report(cell):
    return dict(cell=cell, role=ROLES[cell], width=C['width_for'](ROLES[cell]), baseline=(cell == 'VSA-A'),
                passed=False, jobs=[], errors=[], fidelity_failures=[],
                behavior=dict(natural_stop_cases=0, fixed_length_cases=0, diagnostics_exercised=0,
                              rejected_draft_observed=0, not_exercised=[]))


def add_result(report, job, result, frozen, fresh):
    kind, actual = result['kind'], result['actual']
    comparisons = {key: compare(kind, actual, expected['actual']) for key, expected in
                   (('frozen_a', frozen), ('fresh_a', fresh))}
    functional = result.get('functional', True)
    row = dict(stem=job['stem'], kind=kind, actual=actual, functional=functional, comparisons=comparisons,
               passed=bool(functional and all(c['passed'] for c in comparisons.values())))
    behavior = report['behavior']
    if kind == 'diagnostic':
        # Count valid diagnostics/rejections BEFORE considering equality. Mismatch is not missing behavior.
        row['speculation'] = result['speculation']
        behavior['diagnostics_exercised'] += 1
        behavior['rejected_draft_observed'] += int(result['speculation']['rejected_draft_observed'])
    else:
        row['proposal_acceptance_counts'] = {key: result['result']['timings'][key]
            for key in ('draft_n', 'draft_n_accepted') if key in result['result']['timings']}
        if functional:
            behavior['natural_stop_cases' if kind in ('quality', 'anchor') else 'fixed_length_cases'] += 1
    report['jobs'].append(row)
    if not row['passed']:
        report['fidelity_failures'].append(dict(stem=job['stem'], functional=functional, comparisons=comparisons))
    return row


def finish_report(report):
    behavior = report['behavior']
    for key, expected in (('natural_stop_cases', 21), ('fixed_length_cases', 2), ('diagnostics_exercised', 2)):
        if behavior[key] != expected:
            behavior['not_exercised'].append(key)
    if report['width'] is not None and not behavior['rejected_draft_observed']:
        behavior['not_exercised'].append('real rejected draft count')
    report['passed'] = (len(report['jobs']) == 25 and not report['errors'] and not report['fidelity_failures']
                        and not behavior['not_exercised'])
    return report


def collect(rt, cell):
    report = new_report(cell)
    try:
        for job in jobs(rt.h):
            try:
                result = rt.run_job(cell, job, expected_for_compare=False)
            except C['FidelityFailure']:
                # A valid functional failure still has raw HTTP evidence. A prompt failure
                # has no response and fails closed here; neither case is retried.
                read_public(rt.h, rt.out / cell, job)
                result = read_result(rt.h, rt.out / cell, job)
            frozen = read_result(rt.h, rt.off['ROOT'] / 'run-r1/A0', job)
            fresh = result if cell == 'VSA-A' else read_result(rt.h, rt.out / 'VSA-A', job)
            row = add_result(report, job, result, frozen, fresh)
            C['save'](rt.out / cell / 'qualification.json', report)
            if report['baseline'] and not row['passed']:
                raise C['InfraFailure']('fresh no-draft A failed frozen output/functional gate')
        finish_report(report)
        if report['baseline'] and not report['passed']:
            raise C['InfraFailure']('fresh no-draft A incomplete/invalid')
    except BaseException as exc:
        report['errors'].append(dict(error=repr(exc)))
        raise C['InfraFailure']('qualification aborted: ' + repr(exc)) from exc
    finally:
        C['save'](rt.out / cell / 'qualification.json', report)
    return []  # Never measurements; inherited capture mode also forbids timing.


def check_cell_evidence(h, directory, cell):
    argv = make_create(cell, h)
    assert C['load'](directory / 'create-argv.json') == argv
    assert C['load'](directory / 'argv-diff.json') == positional_diff(C['make_create'](ROLES[cell], h), argv)
    for phase in ('before', 'after'):
        C['check_pid'](C['load'](directory / ('pid-' + phase + '.json')), ROLES[cell], h)
        check_container(C['load'](directory / ('container-' + phase + '.json')), cell, h)
    assert C['same_container'](C['load'](directory / 'container-before.json'), C['load'](directory / 'container-after.json'))
    logs = {}
    for phase in ('native-and-startup', 'server', 'restored-final'):
        logs[phase] = [(directory / (phase + '.' + stream + '.log')).read_bytes() for stream in ('stdout', 'stderr')]
        h['off']['check_off_log'](b''.join(logs[phase]).decode())
    for i in range(2):
        assert logs['server'][i].startswith(logs['native-and-startup'][i])
        assert logs['restored-final'][i].startswith(logs['server'][i])
    raw = b''.join((directory / ('native-and-startup.raw.' + stream + '.log')).read_bytes()
                   for stream in ('stdout', 'stderr')).decode()
    startup = check_startup(raw, cell, h)
    startup['pid_argv'] = C['load'](directory / 'pid-before.json')['argv']
    assert C['load'](directory / 'startup.json') == startup
    for tag, filename in NATIVE_LOGS.items():
        native = (directory / ('ready-' + filename)).read_bytes()
        assert native == (directory / ('restored-' + filename)).read_bytes()
        check_native_log(tag, native.decode())
        if tag == 'router_single2':
            assert native.decode() in raw, 'raw old-30 log absent from startup stream'
    return C['log_duration_stats'](b''.join(logs['restored-final']).decode())


def analyze(h):
    summary = C['load'](OUT / 'summary.json')
    assert summary['status'] == 'captured' and summary['restored'] and summary['mode'] == MODE
    assert summary['order'] == summary['completed'] == list(CELLS)
    assert summary['timing_permitted'] is False and summary['measured_attempts'] == []
    # A separate CLI invocation, with fresh live checks BEFORE any qualification claim.
    Runtime(h, OUT, MODE).verify_restored(OUT / 'independent-restoration.json')
    check_tested(h)
    assert C['load'](OUT / 'tested-code.json') == C['load'](ROOT / 'preparation/public-tested-code.json')
    assert C['load'](OUT / 'restoration.json')['passed']
    smoke = C['load'](OUT / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    analysis = dict(status='failed', independently_restored=True, reports={}, qualified_widths=[],
                    timing_permitted=False, promotion_permitted=False, probabilities_compared=False,
                    raw_f32_equality_claimed=False, retokenized_output_as_ids_claimed=False)
    try:
        for cell in CELLS:
            directory, report = OUT / cell, new_report(cell)
            stats = check_cell_evidence(h, directory, cell)
            captured = C['load'](directory / 'qualification.json')
            assert [row['stem'] for row in captured['jobs']] == [job['stem'] for job in jobs(h)]
            assert captured['errors'] == []
            analysis['reports'][cell] = report
            for job in jobs(h):
                read_public(h, directory, job)
                actual = read_result(h, directory, job)
                frozen = read_result(h, h['off']['ROOT'] / 'run-r1/A0', job)
                fresh = actual if cell == 'VSA-A' else read_result(h, OUT / 'VSA-A', job)
                add_result(report, job, actual, frozen, fresh)
            finish_report(report)
            assert report == captured, 'capture bookkeeping differs from independently reread raw evidence'
            report['cumulative_server_statistics'] = stats
            if cell == 'VSA-A':
                assert report['passed'], 'fresh baseline failed'
        analysis['qualified_widths'] = [r['width'] for r in analysis['reports'].values() if r['width'] and r['passed']]
        analysis['status'] = 'analyzed'
    except BaseException as exc:
        analysis['error'] = repr(exc)
        raise
    finally:
        C['save'](OUT / 'analysis.json', analysis)
    print('ANALYZED_AFTER_RESTORE qualified_widths=' + repr(analysis['qualified_widths']) + ' timing_permitted=False', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--lead-run-qualification', action='store_true')
    action.add_argument('--lead-after-restore', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'inference-host' and Path(__file__).resolve().parent == ROOT
    h = C['helpers']()
    if args.lead_run_qualification:
        tested = check_tested(h)
        OUT.mkdir(exist_ok=False)
        (OUT / 'declaration.md').write_bytes((ROOT / 'qualification.md').read_bytes())
        C['save'](OUT / 'tested-code.json', tested)
        K['capture'](Runtime(h, OUT, MODE), CELLS, collect)
        print('CAPTURED_RESTORED analysis_pending=True timing_permitted=False', flush=True)
    else:
        analyze(h)


if __name__ == '__main__':
    main()
