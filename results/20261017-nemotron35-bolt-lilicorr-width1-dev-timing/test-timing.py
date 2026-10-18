#!/usr/bin/env python3
"""REMOTE CPU only: recorded references + fresh synthetic transport/lifecycle fixtures."""
import copy
import io
import json
import os
import runpy
import socket
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent
assert socket.gethostname() == 'inference-host'
SCRATCH = Path(os.environ['BOLT_TIMING_CPU_SCRATCH']).resolve()
assert SCRATCH.is_relative_to(ROOT / 'worker-scratch')


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU fixtures prohibit native/API/process execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if name.startswith(('/sys/', '/proc/', '/dev/dri')):
            raise AssertionError('CPU fixtures prohibit hardware/process reads: ' + name)


sys.addaudithook(audit)
C = runpy.run_path(str(ROOT / 'contracts.py'))
R = runpy.run_path(str(ROOT / 'run-timing.py'))
A = runpy.run_path(str(ROOT / 'analyze-timing.py'))
H = C['helpers']()
OFF = H['off']


class Response(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream'}


class Clock:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return float(self.n)


def sse():
    rows = [dict(choices=[dict(index=0, delta=dict(role='assistant', content=''))]),
            dict(choices=[dict(index=0, delta=dict(content='many tokens in this chunk'))]),
            dict(choices=[dict(index=0, delta={}, finish_reason='length')]),
            dict(choices=[], usage=dict(prompt_tokens=55, completion_tokens=128, total_tokens=183),
                 timings=dict(cache_n=0, predicted_n=128, predicted_ms=2000, predicted_per_second=63.5,
                              prompt_ms=500, prompt_per_second=110, draft_n=69, draft_n_accepted=57))]
    return b''.join(b'data: ' + json.dumps(row).encode() + b'\n\n' for row in rows) + b'data: [DONE]\n\n'


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def test_fixed_payloads_refs_and_candidate_ids(self):
        pre, points = C['jobs'](OFF)
        self.assertEqual(21, len(pre))
        self.assertTrue(all(j['payload']['max_tokens'] == 256 for j in pre[:20]))
        for arm in ('A0', 'C0'):
            for job in OFF['frozen_jobs']():
                raw, result = C['reference'](H, arm, job)
                ids = OFF['reference_prompt'](arm, job)
                self.assertEqual(len(ids), result['usage']['prompt_tokens'])
                if job['stem'].endswith('/diagnostic'):
                    self.assertEqual(128, len(OFF['diagnostic_view'](json.loads(raw), ids)['ids']))
            for point in points:
                ordinary = C['reference'](H, arm, points[point]['ordinary'])[1]
                diagnostic = C['reference'](H, arm, points[point]['diagnostic'])[1]
                C['check_result'](ordinary, point, diagnostic)
                self.assertFalse({'logprobs', 'response_format'} & points[point]['ordinary']['payload'].keys())

    def test_exact_candidate_config_changes_name_only(self):
        for cell in ('A1', 'C1', 'C2', 'A2'):
            actual = C['make_create'](cell, OFF)
            expected = OFF['make_create'](C['label'](cell))
            index = expected.index('--name') + 1
            self.assertEqual([index], [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b])
            self.assertEqual(C['PREFIX'] + cell.lower(), actual[index])
            self.assertNotIn(':/capture', ' '.join(actual))
            self.assertIn(OFF['NATIVE_ON'], actual[-1])
            self.assertIn(OFF['OFF_BOUNDARY'], actual[-1])
            self.assertEqual(expected[-1], actual[-1])
        with self.assertRaises(AssertionError):
            C['make_create']('O1', OFF)

    def test_distinct_native_contracts_and_fresh_original_prefix(self):
        original = C['load'](C['OFF'] / 'run-r1/original-container.json')
        contract = C['original_contract'](original, H)
        self.assertEqual(['56/56 tests passed'] * 2 + ['20/20 tests passed'] * 2 +
                         ['82/82 tests passed'] * 2 + ['34/34 tests passed'] * 2 + ['1/1 tests passed'], contract['native_counts'])
        baseline = OFF['OLD']['PROFILE'] / 'run-r1/baseline/native-and-startup.log'
        fresh = baseline.read_text()
        C['check_original_log'](fresh, contract)
        with self.assertRaises(AssertionError):
            C['check_original_log'](fresh + fresh, contract)
        before = dict(stdout=b'old native passes\n', stderr=b'old startup\n')
        after = dict(stdout=before['stdout'] + fresh.encode(), stderr=before['stderr'] + b'new stderr\n')
        self.assertEqual(fresh.encode(), C['fresh_logs'](before, after)['stdout'])
        with self.assertRaises(AssertionError):
            C['fresh_logs'](before, dict(stdout=b'truncated', stderr=after['stderr']))
        with self.assertRaises(AssertionError):
            C['check_original_log']('', contract)
        for arm in ('A0', 'C0'):
            log = (C['OFF'] / 'run-r1' / arm / 'native-and-startup.log').read_text()
            OFF['check_startup'](log, arm)
            with self.assertRaises(AssertionError):
                C['check_original_log'](log, contract)

    def test_original_immutable_config_hostconfig_mounts(self):
        original = C['load'](C['OFF'] / 'run-r1/original-container.json')
        OFF['P']['check_restored'](original, original)
        for key in ('Config', 'HostConfig', 'Mounts', 'Id'):
            bad = copy.deepcopy(original)
            bad[key] = [] if key == 'Mounts' else 'changed'
            with self.subTest(key=key), self.assertRaises((AssertionError, TypeError)):
                OFF['P']['check_restored'](original, bad)

    def test_sse_metrics_first_nonempty_not_first_event(self):
        payload = C['jobs'](OFF)[1]['short']['ordinary']['payload']
        stem = self.tmp / 'ordinary'
        result = R['stream_chat']('http://127.0.0.1:18098', payload, stem, H['analysis']['normalize'],
                                  opener=lambda *a, **k: Response(sse()), clock=Clock())
        self.assertEqual(2, result['ttfc_s'])
        self.assertEqual(5, result['elapsed_s'])
        metrics = C['timing_metrics'](result)
        self.assertEqual(2000, metrics['predicted_ms'])
        self.assertEqual(128 / 5, metrics['end_to_end_output_tps'])
        self.assertEqual(127 / 3, metrics['first_content_decode_proxy_tps'])
        self.assertEqual(payload, C['load'](R['artifact'](stem, '.request.json')))
        self.assertEqual(sse(), R['artifact'](stem, '.http-body').read_bytes())
        self.assertEqual(69, metrics['draft_n'])
        self.assertEqual(57, metrics['draft_n_accepted'])

    def test_partial_missing_usage_error_streams_are_retained(self):
        payload = C['jobs'](OFF)[1]['short']['ordinary']['payload']
        for i, raw in enumerate((sse().replace(b'data: [DONE]\n\n', b''),
                                 b'data: {"choices": []}\n\ndata: [DONE]\n\n',
                                 b'data: {"error": "failed"}\n\n', b'data: broken\n')):
            stem = self.tmp / str(i)
            with self.assertRaises((AssertionError, ValueError)):
                R['stream_chat']('http://127.0.0.1:18098', payload, stem, H['analysis']['normalize'],
                                 opener=lambda *a, raw=raw, **k: Response(raw), clock=Clock())
            self.assertEqual(raw, R['artifact'](stem, '.http-body').read_bytes())
            self.assertIn('error', C['load'](R['artifact'](stem, '.http.json')))
        class Interrupted(Response):
            def __iter__(self):
                yield b'data: {"choices": []}\n'
                raise InterruptedError('signal fixture')
        stem = self.tmp / 'interrupted'
        with self.assertRaises(InterruptedError):
            R['stream_chat']('http://127.0.0.1:18098', payload, stem, H['analysis']['normalize'],
                             opener=lambda *a, **k: Interrupted(), clock=Clock())
        self.assertFalse(C['load'](R['artifact'](stem, '.http.json'))['complete'])
        self.assertTrue(R['artifact'](stem, '.http-body').read_bytes())

    def test_fidelity_quality_and_usage_fail_closed_O_text_allowed(self):
        point = C['jobs'](OFF)[1]['short']['ordinary']
        ref = C['reference'](H, 'A0', point)[1]
        different = dict(ref, text='accepted original variation')
        C['check_result'](different, 'short')
        with self.assertRaises(AssertionError):
            C['check_result'](different, 'short', ref)
        for key, value in (('cache_n', 1), ('predicted_n', 127)):
            bad = copy.deepcopy(ref)
            bad['timings'][key] = value
            with self.assertRaises(AssertionError):
                C['check_result'](bad, 'short')
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            bad = copy.deepcopy(ref)
            bad['usage'][field] -= 1
            with self.assertRaises(AssertionError):
                C['check_result'](bad, 'short')
        self.assertFalse(OFF['L']['functional_quality']('literal', dict(text='NOT READY', finish_reason='stop', usage=ref['usage'])))
        self.assertFalse(OFF['L']['functional_anchor'](dict(text='wrong', finish_reason='stop', usage=dict(prompt_tokens=8238))))
        raw = C['reference'](H, 'A0', C['jobs'](OFF)[1]['short']['diagnostic'])[0]
        body = json.loads(raw)
        body['choices'][0]['logprobs']['content'][0]['id'] += 1
        with self.assertRaises(AssertionError):
            OFF['diagnostic_view'](body, OFF['reference_prompt']('A0', point))

    def test_sampler_always_stops_even_failed_start_and_request(self):
        for failure in ('start', 'request', None):
            sampler = Mock(samples=[])
            if failure == 'start':
                sampler.start.side_effect = RuntimeError('start failed')
            fn = Mock(return_value='ok', side_effect=RuntimeError('partial response') if failure == 'request' else None)
            stem = self.tmp / (failure or 'ok')
            if failure:
                with self.assertRaises(RuntimeError):
                    R['sampled_request'](sampler, fn, stem, clock=Clock())
            else:
                self.assertEqual('ok', R['sampled_request'](sampler, fn, stem, clock=Clock()))
            sampler.stop.assert_called_once()
            self.assertTrue(R['artifact'](stem, '.telemetry.json').exists())

    def test_counter_units_missing_reset_and_wrap_labels(self):
        samples = [dict(t=1, energy1_uj=1000000, energy2_uj=10, freq0_mhz=1000, freq1_mhz=900, temp2_mc=45000),
                   dict(t=3, energy1_uj=5000000, energy2_uj=30, freq0_mhz=1100, freq1_mhz=1000, temp2_mc=47000)]
        report = C['telemetry'](samples, 4)
        self.assertEqual(4, report['card_j'])
        self.assertEqual(2, report['card_w'])
        self.assertEqual(4, report['window_duration_s'])
        self.assertEqual(2, report['sample_span_s'])
        self.assertEqual(46000, report['temp2_mc']['summary']['median'])
        self.assertIsNone(report['peak_vram_bytes'])
        bad = copy.deepcopy(samples)
        bad[1]['energy1_uj'] = 2
        self.assertIn('reset or wrap', C['telemetry'](bad, 4)['card_status'])
        bad[1]['energy1_uj'] = None
        self.assertIn('missing', C['telemetry'](bad, 4)['card_status'])
        self.assertIsNone(C['telemetry']([], 1)['card_w'])

    def test_exact_order_two_loads_six_samples_and_no_warmup_stats(self):
        plan = C['schedule']()
        self.assertEqual(('O1', 'A1', 'C1', 'C2', 'A2', 'O2'), C['ORDER'])
        self.assertEqual(48, len(plan))
        measured = [r for r in plan if r['phase'] == 'measured']
        self.assertEqual(36, len(measured))
        self.assertEqual({6}, set(Counter((j['cell'][0], j['point']) for j in measured).values()))
        rows = []
        for i, j in enumerate(measured):
            rows.append(dict(j, passed=True, actual=dict(text='output'), equals_candidate_reference=True,
                             metrics={k: i + 1 for k in C['METRICS']}))
        stats = A['summarize'](rows, True)
        self.assertTrue(stats['descriptive_only'])
        self.assertIsNone(stats['ci'])
        self.assertEqual(6, stats['per_arm']['short']['O']['elapsed_s']['summary']['n'])
        self.assertEqual({'O-to-A', 'A-to-C', 'O-to-C'}, stats['deltas']['short'].keys())
        self.assertEqual({}, A['summarize'](rows[:-1], False)['per_arm'])
        self.assertEqual(dict(n=6, median=3.5, q1=2.25, q3=4.75, iqr=2.5), C['describe']([1, 2, 3, 4, 5, 6]))
        self.assertFalse(C['capacity'](free=C['RESERVE'] + C['ALLOWANCE'] - 1)['fits'])
        self.assertTrue(C['capacity'](free=C['RESERVE'] + C['ALLOWANCE'])['fits'])

    def test_controller_restores_on_every_phase_and_signal_no_retry(self):
        class Fake:
            def __init__(self, fail=None):
                self.fail, self.calls = fail, []
            def event(self, name):
                self.calls.append(name)
                if name == self.fail:
                    raise InterruptedError(name)
            def preflight(self): self.event('preflight')
            def begin(self, cell): self.event('begin-' + cell)
            def prevalidate(self, cell): self.event('quality-' + cell)
            def request(self, cell, point, phase, ordinal): self.event('-'.join((cell, point, phase, str(ordinal))))
            def end(self, cell): self.event('end-' + cell)
            def restore(self):
                self.event('restore')
                return []
        good = Fake()
        result = R['capture'](good, self.tmp / 'good')
        self.assertEqual('captured', result['status'])
        self.assertEqual(list(C['ORDER']), result['completed'])
        self.assertEqual(36, len(result['measured_attempts']))
        self.assertEqual('restore', good.calls[-1])
        for index, failure in enumerate(good.calls):
            fake = Fake(failure)
            report = R['capture'](fake, self.tmp / str(index))
            self.assertEqual('failed', report['status'], failure)
            self.assertEqual(1, fake.calls.count('restore'), failure)
            self.assertEqual(1, fake.calls.count(failure), failure)
            self.assertEqual(failure != 'restore', report['restored'])

    def test_real_restore_attempts_all_actions_even_when_stop_fails(self):
        for running in (True, False):
            obj = object.__new__(R['Runtime'])
            obj.out, obj.created, obj.original_name = self.tmp / str(running), ['A1', 'C1'], 'nemotron-bolt-probe'
            obj.prefixes = {}
            obj.out.mkdir()
            calls = []
            def command(*argv, **kwargs):
                calls.append(argv)
                if argv[:2] == ('docker', 'stop') and argv[-1].endswith('a1'):
                    raise RuntimeError('failed stop fixture')
                return ''
            obj.command = command
            obj.inspect = lambda name: dict(State=dict(Running=running))
            obj.save_logs = lambda cell, phase: calls.append(('logs', cell))
            obj.verify_restored = lambda path: calls.append(('independent',))
            obj.capacity = lambda tag: calls.append(('capacity',))
            obj.off = dict(OLD=dict(URL='http://unused'))
            obj.h = dict(grouped=dict(ready=lambda *a: calls.append(('ready',))),
                         api=dict(chat=lambda *a, **k: dict(text='READY', finish_reason='stop')))
            errors = obj.restore()
            self.assertEqual(1, len(errors))
            self.assertIn(('logs', 'C1'), calls)
            self.assertIn(('independent',), calls)
            self.assertEqual(not running, ('docker', 'start', 'nemotron-bolt-probe') in calls)

    def test_read_only_source_binary_pins_and_timing_contract(self):
        C['save'](SCRATCH / 'hash-checks.json', OFF['N']['check_sources_and_binaries']())
        evidence = C['source_evidence'](H)
        C['save'](SCRATCH / 'source-contracts.json', evidence)
        self.assertEqual(2, len(evidence))
        self.assertIn('t_gen_last - t_prompt_last', json.dumps(evidence))


if __name__ == '__main__':
    unittest.main(verbosity=2)
