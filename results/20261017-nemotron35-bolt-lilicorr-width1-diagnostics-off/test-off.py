#!/usr/bin/env python3
"""REMOTE CPU ONLY. Recorded fixtures/synthetic HTTP are NOT off-path live E2E."""
import ast
import contextlib
import copy
import io
import json
import os
import runpy
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

assert socket.gethostname() == 'inference-host'
HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_OFF_CPU_SCRATCH']).resolve()
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')


def cpu_only(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.fork', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU fixture test attempted external execution/network: ' + event)


sys.addaudithook(cpu_only)
C = runpy.run_path(str(HERE / 'contracts.py'))
D = runpy.run_path(str(HERE / 'run-off.py'))
A = runpy.run_path(str(HERE / 'analyze-off.py'))
load, save = C['load'], C['save']


class Response(io.BytesIO):
    def __init__(self, body, status=200):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Type': 'application/json', 'X-Fixture': 'CPU-only'}


def fixture_raw(label, job):
    stem = C['reference_stem'](label, job['stem'])
    if not job['stem'].startswith('short/'):
        return stem.with_suffix('.http-body').read_bytes()
    if job['payload']['stream']:
        events = load(stem.with_suffix('.events.json'))
        return b''.join(b'data: ' + json.dumps(e['body']).encode() + b'\n\n' for e in events) + b'data: [DONE]\n\n'
    return stem.with_suffix('.response.json').read_bytes()


def emit_fixture(stem, endpoint, payload, raw):
    return D['exchange'](endpoint, payload, stem, opener=lambda *a, **k: Response(raw))


class OffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = C['frozen_jobs']()
        cls.refs = C['reference_hashes'](cls.jobs)
        cls.logs = {}
        for label in C['ARMS']:
            log = (C['LONG'] / 'run-r1' / label / 'native-and-startup.log').read_text()
            native, serving = log.split(C['P']['MARKER'])
            cls.logs[label] = native + C['P']['MARKER'] + '\n'.join(
                line for line in serving.splitlines() if not any(marker in line for marker in C['FORBIDDEN']))
        print('Synthetic diagnostics-off log filtering and fake HTTP over saved fixtures; NOT a fresh E2E.', flush=True)

    def rejected(self, fn, *args):
        with self.assertRaises((AssertionError, ValueError, KeyError, OSError)):
            fn(*args)

    def folder(self, name):
        path = SCRATCH / name
        path.mkdir(exist_ok=False)
        return path

    def pid(self, label='A0', env=b'BOLT_TRACE_TENSORS=0\0GGML_SYCL_DEBUG=0\0SECRET=never-retain\0'):
        argv = C['server_argv'](label)
        return C['pid_evidence'](env, b'\0'.join(a.encode() for a in argv), argv[0], [])

    def test_compile_and_no_analysis_during_capture(self):
        for name in C['FILES']:
            if name.endswith('.py'):
                compile((HERE / name).read_bytes(), str(HERE / name), 'exec')
        compile(C['PID_PROBE'], 'PID_PROBE', 'exec')
        tree = ast.parse((HERE / 'run-off.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Name) and n.func.id in ('normalize', 'analyze_responses') for n in calls))
        text = (HERE / 'analyze-off.py').read_text()
        for claim in ('raw_vector_claimed=False', 'fresh_route_equivalence_claimed=False', 'probability_fidelity_claimed=False'):
            self.assertIn(claim, text)
        self.assertIn('analysis.mkdir(exist_ok=False)', text)
        self.assertNotIn("['raw_bytes']", text)

    def test_argv_boundary_and_all_numerical_commands_unchanged(self):
        for label in C['ARMS']:
            old = C['traced_argv'](label)
            new = C['make_create'](label)
            native, server = old[-1].split(C['OLD_BOUNDARY'])
            self.assertEqual(new[-1], C['NATIVE_ON'] + native + C['OFF_BOUNDARY'] + server)
            self.assertIn(C['P']['ROUTER_NATIVE_COMMAND'], native)
            self.assertIn('bash /native-outprod.sh;', native)
            self.assertEqual(C['server_argv'](label), C['shlex'].split(server))
            # Reverse ONLY the declared changes and require the COMPLETE argv.
            restored = list(new)
            restored[restored.index('--name') + 1] = old[old.index('--name') + 1]
            for value in ('BOLT_VERIFY_CAPTURE_DIR=/capture', next(x for x in old if x.endswith(':/capture'))):
                i = old.index(value)
                restored[i-1:i-1] = old[i-1:i+1]
            restored[-1] = native + C['OLD_BOUNDARY'] + server
            self.assertEqual(restored, old)
            self.assertFalse(any('/capture' in word for word in new))
            self.assertEqual('--spec-type' in C['server_argv'](label), label == 'C0')
            self.assertIn('BOLT_TRACE_TENSORS=0', new)

    def test_pid_environment_absence_not_zero_or_config_proxy(self):
        for label in C['ARMS']:
            good = self.pid(label)
            self.assertTrue(C['check_pid'](good, label))
            self.assertNotIn('SECRET', json.dumps(good))
            self.assertNotIn('never-retain', json.dumps(good))
            for pair in (['BOLT_VERIFY_CAPTURE_DIR', '0'], ['BOLT_VERIFY_CAPTURE_DIR', ''],
                         ['BOLT_VERIFY_CAPTURE_DIR', '/tmp'], ['LLAMA_TRACE', '0'], ['LLAMA_TRACE', '1']):
                bad = copy.deepcopy(good)
                bad['diagnostics'].append(pair)
                self.rejected(C['check_pid'], bad, label)
            for key in ('BOLT_TRACE_TENSORS', 'GGML_SYCL_DEBUG'):
                for value in ('1', '', 'false'):
                    bad = copy.deepcopy(good)
                    bad['diagnostics'] = [[k, value if k == key else v] for k, v in bad['diagnostics']]
                    self.rejected(C['check_pid'], bad, label)
            for key, value in (('origin', 'container.Config.Env'), ('pid', 25), ('argv', ['bash']),
                               ('exe', '/bin/bash'), ('scoped_tmp_rows', ['row-ABC123'])):
                bad = dict(good, **{key: value})
                self.rejected(C['check_pid'], bad, label)
            duplicate = copy.deepcopy(good)
            duplicate['diagnostics'].append(['GGML_SYCL_DEBUG', '0'])
            self.rejected(C['check_pid'], duplicate, label)

    def test_actual_pid_probe_program_redacts_before_output(self):
        argv = C['server_argv']('A0')
        data = {'/proc/1/environ': b'SECRET=do-not-output\0BOLT_TRACE_TENSORS=0\0GGML_SYCL_DEBUG=0\0',
                '/proc/1/cmdline': b'\0'.join(a.encode() for a in argv)}
        seen = []
        def read_bytes(path):
            seen.append(str(path))
            return data[str(path)]
        output = io.StringIO()
        with patch.object(Path, 'read_bytes', read_bytes), patch.object(Path, 'glob', return_value=[]), \
                patch('os.readlink', return_value=argv[0]), contextlib.redirect_stdout(output):
            exec(C['PID_PROBE'], {})
        self.assertEqual(set(seen), set(data))
        self.assertNotIn('do-not-output', output.getvalue())
        C['check_pid'](json.loads(output.getvalue()), 'A0')

    def test_native_gates_kept_and_serving_trace_absence(self):
        for label, log in self.logs.items():
            C['check_startup'](log, label)
            self.assertIn('BOLT_ROUTE route=', log.split(C['P']['MARKER'])[0])
            for marker in C['FORBIDDEN']:
                self.rejected(C['check_off_log'], log + '\n' + marker)
            for marker in ('BOLT_VERIFY_ROUTER_NATIVE_PASSED', 'BOLT_VERIFY_IQ2_NATIVE_PASSED',
                           'BOLT_VERIFY_Q8_LATENT_NATIVE_PASSED', 'BOLT_VERIFY_Q8_ATTENTION_NATIVE_PASSED',
                           'BOLT_VERIFY_Q8_ATTENTION_VIEW_NATIVE_PASSED', 'BOLT_SSM_PREFILL_NUMERICAL_PASSED'):
                self.rejected(C['check_startup'], log.replace(marker, 'MISSING_GATE'), label)
            self.rejected(C['check_startup'], log.replace('30/30 tests passed', '29/30 tests passed'), label)

    def test_fixed_27_requests_public_contracts_and_traced_prompt_alignment(self):
        self.assertEqual(len(self.jobs), 27)
        self.assertEqual(sum(j['payload']['max_tokens'] for j in self.jobs), 6016)
        self.assertTrue(all(j['payload']['max_tokens'] == 256 for j in self.jobs[:20]))
        for label in C['ARMS']:
            for j in self.jobs:
                ids = C['reference_prompt'](label, j)
                template = C['L']['apply_request'](j['payload'])
                self.assertIs(template['add_generation_prompt'], True)
                self.assertEqual(template['chat_template_kwargs'], {'enable_thinking': False})
                self.assertEqual(C['L']['tokenize_request']({'prompt': 'fixture-render'}), dict(
                    content='fixture-render', add_special=True, parse_special=True, with_pieces=False))
                if j['stem'].startswith('short/'):
                    self.assertEqual(len(ids), 55)
                elif j['stem'].startswith('long/'):
                    self.assertEqual(len(ids), 8270)
                elif j['stem'] == 'anchor':
                    self.assertEqual(len(ids), 8238)
        for length in ('short', 'long'):
            jobs = [j for j in self.jobs if j['stem'].startswith(length + '/')]
            self.assertEqual(jobs[0]['payload'], jobs[1]['payload'])
            self.assertEqual(C['OLD']['diagnostic_payload'](jobs[1]['payload']), jobs[2]['payload'])
            self.assertNotIn('return_tokens', jobs[1]['payload'])
        self.rejected(C['L']['token_ids'], {'tokens': [True]})
        self.rejected(C['L']['token_ids'], {'tokens': [131072]})
        self.rejected(C['L']['tokenize_request'], {'prompt': ''})

    def test_actual_emissions_and_placeholder_probabilities(self):
        for label in C['ARMS']:
            for length in ('short', 'long'):
                job = next(j for j in self.jobs if j['stem'] == length + '/diagnostic')
                prompt = C['reference_prompt'](label, job)
                body = load(C['reference_stem'](label, job['stem']).with_suffix('.response.json'))
                good = C['diagnostic_view'](body, prompt)
                self.assertEqual(len(good['ids']), 128)
                altered = copy.deepcopy(body)
                for row in altered['choices'][0]['logprobs']['content']:
                    row['logprob'] = 777
                    row['top_logprobs'] = [{'id': 123, 'logprob': 999}]
                self.assertEqual(C['diagnostic_view'](altered, prompt), good)
                altered['choices'][0]['logprobs']['content'][0]['id'] = 131072
                self.rejected(C['diagnostic_view'], altered, prompt)
                altered = copy.deepcopy(body)
                altered['__verbose']['tokens'][0] += 1
                self.rejected(C['diagnostic_view'], altered, prompt)
                altered = copy.deepcopy(body)
                altered['choices'][0]['logprobs']['content'][0]['token'] = 'wrong'
                self.rejected(C['diagnostic_view'], altered, prompt)
                self.rejected(C['diagnostic_view'], body, prompt[:-1])
                different = list(good['ids'])
                different[73] += 1
                result = C['id_comparison'](good['ids'], different)
                self.assertFalse(result['equal'])
                self.assertEqual(result['first_difference_ordinal'], 74)
                self.assertEqual(C['id_comparison'](good['ids'], good['ids'][:-1])['first_difference_ordinal'], 128)

    def test_transport_preserves_partial_and_http_error(self):
        root = self.folder('transport')
        payload = self.jobs[0]['payload']
        path = emit_fixture(root / 'ok', '/v1/chat/completions', payload, b'{"fixture":true}')
        self.assertEqual(path.read_bytes(), b'{"fixture":true}')
        self.assertEqual(A['read_http'](root / 'ok', payload, '/v1/chat/completions'), path.read_bytes())
        def error(*a, **k):
            raise urllib.error.HTTPError('http://fixture', 503, 'fixture', {'X-Failure': 'retained'}, io.BytesIO(b'failed-body'))
        with self.assertRaises(AssertionError):
            D['exchange']('/v1/chat/completions', payload, root / 'http-error', opener=error)
        record = load(root / 'http-error.http.json')
        self.assertEqual(record['status'], 503)
        self.assertIn(['X-Failure', 'retained'], record['headers'])
        self.assertEqual((root / 'http-error.http-body').read_bytes(), b'failed-body')
        class Partial(Response):
            def read(self, size):
                if self.tell():
                    raise TimeoutError('fixture partial failure')
                return super().read(size)
        with self.assertRaises(TimeoutError):
            D['exchange']('/v1/chat/completions', payload, root / 'partial', opener=lambda *a, **k: Partial(b'partial'))
        self.assertEqual((root / 'partial.http-body').read_bytes(), b'partial')
        self.assertFalse(load(root / 'partial.http.json')['complete'])
        self.rejected(A['read_http'], root / 'partial', payload, '/v1/chat/completions')
        self.rejected(A['normalize'], b'data: {"error":"fixture"}\n\n', dict(stream=True))
        self.rejected(A['normalize'], b'data: [DONE]\n\n', dict(stream=True))

    def test_full_synthetic_http_path_and_independent_normalization(self):
        root = self.folder('synthetic-pair')
        reports = {}
        for label in C['ARMS']:
            cell = root / label
            for directory in ('quality', 'short', 'long', 'public'):
                (cell / directory).mkdir(parents=True)
            for i, job in enumerate(self.jobs):
                ids = C['reference_prompt'](label, job)
                # Synthetic HTTP contract exercise, not claimed real tokenization.
                def fake_send(endpoint, payload, stem):
                    response = {'prompt': 'SYNTHETIC CPU ONLY'} if endpoint == '/apply-template' else {'tokens': ids}
                    return emit_fixture(stem, endpoint, payload, json.dumps(response).encode())
                D['public_tokenize'](job['payload'], cell / 'public' / f'{i:02}', send=fake_send)
                emit_fixture(cell / job['stem'], '/v1/chat/completions', job['payload'], fixture_raw(label, job))
            report, responses, prompts = A['analyze_responses'](cell, label, self.jobs)
            save(cell / 'synthetic-analysis.json', report)
            self.assertTrue(report['passed'], report['errors'] or [r['stem'] for r in report['rows'] if not r['passed']])
            reports[label] = report
        for length in ('short', 'long'):
            self.assertEqual(reports['A0']['diagnostics'][length]['actual']['ids'], reports['C0']['diagnostics'][length]['actual']['ids'])
        # Corrupt an early A0 response: later quality, anchor, both ID traces and
        # the independent C0 fixture analysis must still be retained.
        (root / 'A0' / (self.jobs[0]['stem'] + '.http-body')).write_bytes(b'bad')
        report, _, _ = A['analyze_responses'](root / 'A0', 'A0', self.jobs)
        self.assertFalse(report['passed'])
        self.assertEqual(len(report['rows']), 27)
        self.assertEqual(len(report['diagnostics']), 2)
        self.assertTrue(reports['C0']['passed'])
        # Public prompt misalignment refuses actual-ID attribution.
        job = self.jobs[23]
        token_stem = root / 'A0/public/23.tokenize'
        request = load(Path(str(token_stem) + '.request.json'))
        ids = C['reference_prompt']('A0', job)
        ids[0] = (ids[0] + 1) % 131072
        raw = json.dumps({'tokens': ids}).encode()
        Path(str(token_stem) + '.http-body').write_bytes(raw)
        record = load(Path(str(token_stem) + '.http.json'))
        record['bytes'] = len(raw)
        save(Path(str(token_stem) + '.http.json'), record)
        save(Path(str(token_stem) + '.response.json'), {'tokens': ids})
        report, _, _ = A['analyze_responses'](root / 'A0', 'A0', self.jobs)
        self.assertNotIn('short', report['diagnostics'])
        self.assertFalse(report['passed'])

    def test_capacity_and_actual_pins(self):
        required = 5 * 1024**3
        self.assertTrue(C['capacity'](HERE, self.jobs, free=required)['fits'])
        self.assertFalse(C['capacity'](HERE, self.jobs, free=required - 1)['fits'])
        self.assertFalse(C['capacity'](HERE, [], free=4 * 1024**3)['fits'])
        self.rejected(C['capacity'], HERE, [dict(payload=dict(max_tokens=257))])
        hashes = C['N']['check_sources_and_binaries']()
        self.assertEqual(hashes['candidate_source_files'], 3805)
        self.assertEqual(hashes['accepted_source_files'], 3805)
        self.assertEqual(len(hashes['candidate_binaries']), 5)
        save(SCRATCH / 'hash-checks.json', hashes)
        save(SCRATCH / 'source-contracts.json', C['source_contracts']())
        record = C['capacity'](HERE, self.jobs)
        self.assertTrue(record['fits'])
        # Existing native+startup logs bound the evidence allowance rationale.
        record['traced_native_startup_bytes'] = {label: (C['LONG'] / 'run-r1' / label / 'native-and-startup.log').stat().st_size
                                               for label in C['ARMS']}
        save(SCRATCH / 'disk-budget.json', record)
        original_read = Path.read_bytes
        for target in (C['CANDIDATE'] / 'common/sampling.cpp', C['CANDIDATE'] / 'build-sycl/bin/llama-server'):
            def changed(path):
                value = original_read(path)
                return value + b'CPU-mock-drift' if path == target else value
            with patch.object(Path, 'read_bytes', changed):
                self.rejected(C['N']['check_sources_and_binaries'])

    def test_restoration_all_attempted_and_live_not_stored_proxy(self):
        root = self.folder('restoration')
        (root / 'A0').mkdir()
        original = load(C['REFERENCE'] / 'run-r1/original-container.json')
        original['State']['Running'] = True
        hardware = load(C['REFERENCE'] / 'run-r1/hardware-before.json')
        calls = []
        def command(*argv):
            calls.append(argv)
            if argv[:2] == ('docker', 'stop'):
                raise RuntimeError('CPU fixture failed candidate stop')
            if argv[:2] == ('docker', 'inspect'):
                return json.dumps([original])
            if argv[:2] == ('docker', 'ps'):
                return C['OLD']['ORIGINAL'] + '\n'
            return 'fixture-log'
        def request(url, endpoint):
            calls.append(('api', endpoint))
            return {'status': 'ok'}
        api = dict(request=request, chat=lambda *a, **k: dict(text='READY', finish_reason='stop'))
        save(root / 'original-container.json', original)
        save(root / 'hardware-before.json', hardware)
        errors = D['restore'](root, original, hardware, ['A0'], True, command, lambda *a: {}, api, lambda: hardware)
        self.assertEqual([e['action'] for e in errors], ['stop-A0'])
        self.assertIn(('docker', 'start', C['OLD']['ORIGINAL']), calls)
        self.assertIn(('api', '/health'), calls)
        self.assertTrue((root / 'hash-checks-after.json').is_file())
        self.assertTrue((root / 'hardware-after.json').is_file())
        self.assertFalse(load(root / 'restoration-checks.json')['passed'])
        live = A['verify_live_restoration'](root, command, api, lambda: hardware)
        self.assertTrue(live['independently_queried_live'])
        for field in ('Config', 'HostConfig', 'Mounts', 'Id', 'State'):
            bad = copy.deepcopy(original)
            if field == 'Mounts':
                bad[field] = []
            elif field == 'Id':
                bad[field] = 'wrong'
            elif field == 'State':
                bad[field]['Running'] = False
            else:
                bad[field]['CPU_DRIFT'] = True
            self.rejected(C['P']['check_restored'], original, bad)
        def dead_live(*argv):
            if argv[:2] == ('docker', 'inspect'):
                bad = copy.deepcopy(original)
                bad['State']['Running'] = False
                return json.dumps([bad])
            return command(*argv)
        self.rejected(A['verify_live_restoration'], root, dead_live, api, lambda: hardware)
        self.rejected(A['verify_live_restoration'], root, command, dict(request=lambda *a: {'status': 'bad'}), lambda: hardware)


if __name__ == '__main__':
    unittest.main(verbosity=2)
