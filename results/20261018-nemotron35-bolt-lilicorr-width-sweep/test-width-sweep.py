#!/usr/bin/env python3
"""REMOTE CPU-only contract fixtures; no Docker, inference, API, or hardware work."""
import ast
import copy
import io
import json
import os
import runpy
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_WIDTH_SWEEP_CPU_SCRATCH']).resolve()
assert socket.gethostname() == 'inference-host'
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')

C = runpy.run_path(str(HERE / 'contracts.py'))
K = runpy.run_path(str(HERE / 'controller.py'))
A = runpy.run_path(str(HERE / 'analyze-timing.py'))
H = C['helpers']()
OFF = H['off']


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU fixtures prohibit process/native/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        if name.startswith(('/sys/', '/proc/', '/dev/dri')):
            raise AssertionError('CPU fixtures prohibit hardware reads: ' + name)


sys.addaudithook(audit)


class Response(io.BytesIO):
    status = 200
    headers = {'Content-Type': 'text/event-stream'}


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return float(self.value)


def sse():
    rows = [
        dict(choices=[dict(index=0, delta=dict(role='assistant', content=''))]),
        dict(choices=[dict(index=0, delta=dict(content='fixture output'))]),
        dict(choices=[dict(index=0, delta={}, finish_reason='length')]),
        dict(choices=[], usage=dict(prompt_tokens=55, completion_tokens=128, total_tokens=183),
             timings=dict(cache_n=0, predicted_n=128, predicted_ms=2000,
                          predicted_per_second=64, prompt_ms=500,
                          prompt_per_second=110, draft_n=69, draft_n_accepted=57)),
    ]
    return b''.join(b'data: ' + json.dumps(row).encode() + b'\n\n' for row in rows) + b'data: [DONE]\n\n'


class WidthSweepFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def test_all_case_files_parse_without_running_them(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
        self.assertNotIn('--draft-max', (HERE / 'contracts.py').read_text())

    def test_width_argv_changes_only_name_and_current_max(self):
        original = copy.deepcopy(OFF['make_create']('C0'))
        for cell, width in (('A', None), ('W3', 3), ('W7', 7)):
            argv = C['make_create'](cell, H)
            base = OFF['make_create'](C['cell_label'](cell))
            diff = C['argv_diff'](base, argv)
            allowed = 1 if width is None else 2
            self.assertEqual(allowed, len(diff))
            self.assertEqual(C['PREFIX'] + cell.lower(), argv[argv.index('--name') + 1])
            C['diagnostics_off_server'](argv[-1])
            self.assertNotIn('BOLT_VERIFY_CAPTURE_DIR', C['server_argv'](argv))
            self.assertNotIn('LLAMA_TRACE', C['server_argv'](argv))
            self.assertIn('BOLT_TRACE_TENSORS=0', argv)
            self.assertNotIn('/capture', ' '.join(argv))
            if width is not None:
                self.assertIn('--spec-draft-n-max ' + str(width), argv[-1])
                self.assertNotIn('--draft-max', argv[-1])
        self.assertEqual(original, OFF['make_create']('C0'))

    def test_frozen_payloads_and_references_are_unchanged(self):
        selected = C['frozen_jobs'](H)
        jobs = C['qualification_jobs'](H)
        self.assertEqual(25, len(jobs))
        self.assertEqual(20, len(selected['quality']))
        self.assertEqual(8238, selected['anchor']['historical_prompt_tokens'])
        self.assertEqual(55, selected['short']['ordinary']['historical_prompt_tokens'])
        self.assertEqual(8270, selected['long']['ordinary']['historical_prompt_tokens'])
        before = C['reference_hashes'](H)
        for job in jobs:
            stem = OFF / 'run-r1/A0' / job['stem']
            raw = H['analysis']['read_http'](stem, job['payload'], '/v1/chat/completions')
            if C['qualification_kinds'](job) == 'diagnostic':
                view = OFF['diagnostic_view'](json.loads(raw), OFF['reference_prompt']('A0', job))
                self.assertEqual(128, len(view['ids']))
            else:
                result = H['analysis']['normalize'](raw, job['payload'])
                C['check_output'](result, job, result, OFF['reference_prompt']('A0', job))
        self.assertEqual(before, C['reference_hashes'](H))

    def test_quality_anchor_length_and_functional_gates(self):
        selected = C['frozen_jobs'](H)
        for job in C['qualification_jobs'](H):
            stem = OFF / 'run-r1/A0' / job['stem']
            raw = H['analysis']['read_http'](stem, job['payload'], '/v1/chat/completions')
            if C['qualification_kinds'](job) == 'diagnostic':
                continue
            result = H['analysis']['normalize'](raw, job['payload'])
            ids = OFF['reference_prompt']('A0', job)
            C['check_output'](result, job, result, ids)
            self.assertTrue(C['check_functional'](H, job, result))
        self.assertEqual(20, len(selected['quality']))

    def test_prompt_and_actual_id_mismatches_fail_closed_with_first_position(self):
        expected = [10, 20, 30]
        actual = [10, 21, 30]
        comparison = C['first_difference'](actual, expected)
        self.assertFalse(comparison['equal'])
        self.assertEqual(2, comparison['first_difference_ordinal'])
        left = dict(ids=actual, pieces=['a', 'b', 'c'], text='abc',
                    finish_reason='length', usage={'prompt_tokens': 1})
        right = dict(ids=expected, pieces=['a', 'b', 'c'], text='abc',
                     finish_reason='length', usage={'prompt_tokens': 1})
        diagnostic = C['diagnostic_compare'](left, right)
        self.assertFalse(diagnostic['passed'])
        self.assertFalse(diagnostic['proposal_acceptance_counts_compared'])
        self.assertFalse(diagnostic['probabilities_compared'])
        self.assertEqual(2, diagnostic['id_comparison']['first_difference_ordinal'])
        self.assertFalse(C['check_speculation_counts'](dict(proposal_acceptance_counts={}))['exercised'])
        self.assertFalse(C['check_speculation_counts'](
            dict(proposal_acceptance_counts={'draft_n': 3, 'draft_n_accepted': 3})
        )['rejected_draft_observed'])
        self.assertTrue(C['check_speculation_counts'](
            dict(proposal_acceptance_counts={'draft_n': 3, 'draft_n_accepted': 2})
        )['rejected_draft_observed'])

    def test_public_order_and_timing_filtering(self):
        self.assertEqual(('A1', 'W3a', 'W7a', 'W7b', 'W3b', 'A2'), C['eligible_order']({3, 7}))
        self.assertEqual(('A1', 'W3a', 'W3b', 'A2'), C['eligible_order']({3}))
        self.assertEqual(('A1', 'W7a', 'W7b', 'A2'), C['eligible_order']({7}))
        self.assertEqual(('A1', 'A2'), C['eligible_order'](set()))
        schedule = C['schedule']()
        self.assertEqual(48, len(schedule))
        self.assertEqual(36, len([row for row in schedule if row['phase'] == 'measured']))
        self.assertEqual(12, len([row for row in C['expected_measurements']({3, 7})
                                 if row['cell'].startswith('W3') or row['cell'].startswith('W7')]))
        self.assertEqual(24, len(C['expected_measurements']({3})))
        self.assertEqual(24, len(C['expected_measurements']({7})))
        self.assertEqual(12, len(C['expected_measurements'](set())))
        self.assertEqual(1, sum(row['phase'] == 'warmup' for row in schedule[:4]))
        self.assertEqual(12, sum(row['phase'] == 'warmup' for row in schedule))

    def test_complete_mount_objects_canonicalize_order_only(self):
        mounts = [
            {'Type': 'bind', 'Source': '/a', 'Destination': '/b', 'RW': True,
             'Propagation': 'rprivate'},
            {'Type': 'tmpfs', 'Source': '', 'Destination': '/tmp', 'RW': True},
        ]
        expected = {'Id': 'id', 'Config': {'Cmd': ['x']}, 'HostConfig': {'IpcMode': ''},
                    'Mounts': mounts}
        reordered = copy.deepcopy(expected)
        reordered['Mounts'].reverse()
        self.assertTrue(C['same_container'](expected, reordered))
        changed = copy.deepcopy(expected)
        changed['Mounts'][0]['RW'] = False
        self.assertFalse(C['same_container'](expected, changed))
        duplicate = copy.deepcopy(expected)
        duplicate['Mounts'].append(copy.deepcopy(duplicate['Mounts'][0]))
        self.assertFalse(C['same_container'](expected, duplicate))
        host_changed = copy.deepcopy(expected)
        host_changed['HostConfig']['IpcMode'] = 'host'
        self.assertFalse(C['same_container'](expected, host_changed))

    def test_stream_timing_usage_first_content_and_raw_bytes(self):
        payload = C['frozen_jobs'](H)['short']['ordinary']['payload']
        stem = self.tmp / 'stream'
        result = K['stream_chat']('http://unused', payload, stem, H['analysis']['normalize'],
                                  opener=lambda *args, **kwargs: Response(sse()), clock=Clock())
        self.assertEqual(2, result['ttfc_s'])
        self.assertEqual(5, result['elapsed_s'])
        self.assertEqual(sse(), C['artifact'](stem, '.http-body').read_bytes())
        self.assertEqual(69, result['timings']['draft_n'])
        self.assertEqual(57, result['timings']['draft_n_accepted'])
        self.assertEqual(payload, C['load'](C['artifact'](stem, '.request.json')))

    def test_partial_streams_are_retained_and_sampler_always_stops(self):
        payload = C['frozen_jobs'](H)['short']['ordinary']['payload']
        raw = sse().replace(b'data: [DONE]\n\n', b'')
        stem = self.tmp / 'partial'
        with self.assertRaises((AssertionError, ValueError)):
            K['stream_chat']('http://unused', payload, stem, H['analysis']['normalize'],
                             opener=lambda *args, **kwargs: Response(raw), clock=Clock())
        self.assertEqual(raw, C['artifact'](stem, '.http-body').read_bytes())
        sampler = Mock(samples=[])
        with self.assertRaises(RuntimeError):
            K['sampled_request'](sampler, lambda: (_ for _ in ()).throw(RuntimeError('fixture')), stem)
        sampler.stop.assert_called_once()
        self.assertTrue(C['artifact'](stem, '.telemetry.json').exists())

    def test_type7_and_warmup_exclusion(self):
        self.assertEqual(dict(n=6, median=3.5, q1=2.25, q3=4.75, iqr=2.5),
                         C['describe']([1, 2, 3, 4, 5, 6]))
        rows = []
        for ordinal in range(3):
            rows.append(dict(cell='A1', point='short', phase='measured', ordinal=ordinal,
                             passed=True, metrics={key: (None if key == 'ttfc_s' else ordinal + 1)
                                                    for key in C['METRICS']}))
        self.assertEqual(3, A['_summarize'](rows)['elapsed_s']['summary']['n'])
        self.assertEqual(0, sum(row['phase'] == 'warmup' for row in rows))
        self.assertIsNone(C['aggregate'](rows)['ttfc_s']['summary'])

    def test_source_binary_and_code_reference_pins_are_read_only(self):
        hashes_before = C['source_binary_hashes'](H)
        evidence = C['source_evidence'](H)
        self.assertTrue(hashes_before)
        self.assertTrue(evidence)
        self.assertEqual(hashes_before, C['source_binary_hashes'](H))
        references = C['reference_hashes'](H)
        self.assertTrue(references)
        self.assertEqual(references, C['reference_hashes'](H))

    def test_cpu_capture_restores_once_on_every_interruption_point(self):
        class Fake:
            mode = 'qualification'
            def __init__(self, out, fail=None, restore_fail=False):
                self.out, self.fail, self.restore_fail, self.calls = out, fail, restore_fail, []
            def event(self, name):
                self.calls.append(name)
                if name == self.fail:
                    raise InterruptedError(name)
            def preflight(self): self.event('preflight')
            def begin(self, cell): self.event('begin-' + cell)
            def run(self, cell): self.event('run-' + cell); return []
            def end(self, cell): self.event('end-' + cell)
            def restore(self):
                self.event('restore')
                if self.restore_fail:
                    return [{'action': 'fixture-restore-failure'}]
                return []
        for fail in (None, 'preflight', 'begin-A', 'run-A', 'end-A'):
            fake = Fake(self.tmp / ('run-' + str(fail)), fail=fail)
            fake.out.mkdir()
            callback = lambda rt, cell: rt.run(cell)
            if fail is None:
                result = K['capture'](fake, ('A',), callback)
                self.assertEqual('captured', result['status'])
            else:
                with self.assertRaises(InterruptedError):
                    K['capture'](fake, ('A',), callback)
                self.assertEqual('failed', C['load'](fake.out / 'summary.json')['status'])
            self.assertEqual(1, fake.calls.count('restore'))
        broken = Fake(self.tmp / 'restore-broken', restore_fail=True)
        broken.out.mkdir()
        with self.assertRaises(AssertionError):
            K['capture'](broken, ('A',), lambda rt, cell: [])
        self.assertEqual(1, broken.calls.count('restore'))

    def test_restore_attempts_all_candidates_after_one_stop_failure(self):
        obj = object.__new__(K['Runtime'])
        obj.out = self.tmp / 'restore-all'
        obj.out.mkdir()
        obj.created = ['A1', 'W3']
        obj.original_name = 'nemotron-bolt-probe'
        obj.prefixes = {}
        obj.original_was_running = False
        for cell in obj.created:
            (obj.out / cell).mkdir()
        calls = []
        def command(*argv, **kwargs):
            calls.append(argv)
            if argv[-1].endswith('a1') and argv[:2] == ('docker', 'stop'):
                raise RuntimeError('stop fixture')
            return ''
        obj.command = command
        obj.inspect = lambda name: dict(State=dict(Running=True))
        obj.logs = lambda name: dict(stdout=b'log-' + name.encode(), stderr=b'')
        obj.verify_restored = lambda path: calls.append(('independent',))
        obj.capacity = lambda tag: calls.append(('capacity', tag))
        errors = obj.restore()
        self.assertEqual(1, len(errors))
        self.assertTrue(any(call[:2] == ('docker', 'stop') and call[-1].endswith('a1') for call in calls))
        self.assertTrue(any(call[:2] == ('docker', 'stop') and call[-1].endswith('w3') for call in calls))
        self.assertIn(('independent',), calls)

    def test_no_forged_per_width_pid_evidence(self):
        for cell in ('A', 'W3', 'W7'):
            record = dict(pid=1, origin='/proc/1/environ',
                          diagnostics=[['BOLT_TRACE_TENSORS', '0'], ['GGML_SYCL_DEBUG', '0']],
                          argv=C['expected_pid_argv'](cell, H),
                          exe='/src/build-sycl/bin/llama-server', scoped_tmp_rows=[])
            self.assertTrue(C['check_pid'](record, cell, H))
        bad = dict(pid=1, origin='/proc/1/environ',
                   diagnostics=[['BOLT_TRACE_TENSORS', '0'], ['GGML_SYCL_DEBUG', '0']],
                   argv=C['expected_pid_argv']('W3', H),
                   exe='/src/build-sycl/bin/llama-server', scoped_tmp_rows=[])
        with self.assertRaises(AssertionError):
            C['check_pid'](bad, 'W7', H)


if __name__ == '__main__':
    unittest.main(verbosity=2)
