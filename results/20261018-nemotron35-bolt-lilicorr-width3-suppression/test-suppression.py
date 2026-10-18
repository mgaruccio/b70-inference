#!/usr/bin/env python3
"""REMOTE CPU fixtures only. Synthetic min=4 startup/PID is NOT live evidence."""
import ast
import copy
import os
import runpy
import signal
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ['BOLT_SUPPRESSION_CPU_SCRATCH']).resolve()
assert socket.gethostname() == 'inference-host'
assert SCRATCH.is_relative_to(HERE / 'worker-scratch')


def audit(event, args):
    if event in ('subprocess.Popen', 'os.system', 'socket.connect', 'socket.bind'):
        raise AssertionError('CPU fixtures prohibit native/process/API execution: ' + event)
    if event == 'open' and isinstance(args[0], (str, bytes)):
        assert not os.fsdecode(args[0]).startswith(('/proc/', '/sys/', '/dev/dri')), args[0]


sys.addaudithook(audit)
P = runpy.run_path(str(HERE / 'probe.py'))
C, K, CELL = P['C'], P['K'], P['CELL']
H = C['helpers']()
RECORDED = C['QUALIFY_DIR'] / 'W3'
STATS = ('#calls(b,g,a) = 4 508 0, #gen drafts = 0, #acc drafts = 0, '
         '#gen tokens = 0, #acc tokens = 0, dur(b,g,a) = 20.0, 300.0, 0.0 ms')


def changed_group(pattern, text, group, value):
    match = pattern.search(text)
    assert match
    start, end = match.span(group)
    return text[:start] + value + text[end:]


class SuppressionFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=SCRATCH))

    def test_only_three_runtime_overrides_and_syntax(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
        methods = {key for key, value in P['Runtime'].__dict__.items() if callable(value)}
        self.assertEqual({'preflight', 'begin', 'pid'}, methods)
        for name in ('_stop_original_for_case', 'end', 'restore', 'verify_restored', 'save_logs',
                     'ownership', 'telemetry_mapping', 'public', 'run_job'):
            self.assertIs(getattr(P['Runtime'], name), getattr(K['Runtime'], name))
        self.assertIs(C['FidelityFailure'], P['Runtime'].run_job.__globals__['C']['FidelityFailure'])

    def test_launch_only_name_and_server_min_change_native_identical(self):
        base = copy.deepcopy(C['make_create']('W3', H))
        argv = P['make_create'](H)
        self.assertEqual([argv.index('--name') + 1, len(argv) - 1],
                         [d['index'] for d in C['argv_diff'](base, argv)])
        self.assertEqual(P['NAME'], argv[argv.index('--name') + 1])
        self.assertEqual(base[-1].rsplit('; exec ', 1)[0], argv[-1].rsplit('; exec ', 1)[0])
        old, new = C['server_argv'](base), C['server_argv'](argv)
        index = old.index('--spec-draft-n-min') + 1
        self.assertEqual([(index, '0', '4')], [(i, a, b) for i, (a, b) in enumerate(zip(old, new)) if a != b])
        self.assertEqual('3', new[new.index('--spec-draft-n-max') + 1])
        self.assertEqual('0', new[new.index('--spec-draft-p-min') + 1])
        self.assertEqual(base, C['make_create']('W3', H))
        self.assertNotIn(CELL, C['LABELS'])  # No imported mapping/global mutation.

    def test_pid_exact_width_min_and_diagnostics(self):
        original = C['load'](RECORDED / 'pid-before.json')
        C['check_pid'](original, 'W3', H)
        record = copy.deepcopy(original)
        record['argv'] = C['server_argv'](P['make_create'](H))  # CPU synthetic only.
        P['check_pid'](record, H)
        bad_records = [original]
        for flag, value in (('--spec-draft-n-max', '7'), ('--spec-draft-n-min', '0'),
                            ('--spec-draft-p-min', '0.1')):
            bad = copy.deepcopy(record)
            bad['argv'][bad['argv'].index(flag) + 1] = value
            bad_records.append(bad)
        for key, value in (('LLAMA_TRACE', '0'), ('BOLT_VERIFY_CAPTURE_DIR', ''),
                           ('BOLT_TRACE_TENSORS', '1'), ('GGML_SYCL_DEBUG', '1'), ('UNKNOWN', '0')):
            bad = copy.deepcopy(record)
            bad['diagnostics'].append([key, value])
            bad_records.append(bad)
        for field, value in (('exe', '/wrong'), ('scoped_tmp_rows', ['row-000001']), ('pid', 2)):
            bad = copy.deepcopy(record)
            bad[field] = value
            bad_records.append(bad)
        for bad in bad_records:
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                P['check_pid'](bad, H)

    def test_recorded_native_gates_with_synthetic_min4_and_negative_startups(self):
        log = ''.join((RECORDED / ('native-and-startup.' + stream + '.log')).read_text()
                      for stream in ('stdout', 'stderr'))
        C['check_width_startup'](log, 'W3', H)
        old = H['off']['OLD']
        synthetic = changed_group(old['DFLASH_PARAMS_RE'], log, 2, '4')
        self.assertEqual(4, P['check_startup'](synthetic, H)['n_min'])
        marker = H['off']['P']['MARKER']
        bad_logs = [log,
                    changed_group(old['DFLASH_PARAMS_RE'], synthetic, 1, '7'),
                    changed_group(old['N_RS_SEQ_RE'], synthetic, 1, '7'),
                    changed_group(old['DFLASH_BLOCK_RE'], synthetic, 1, '7'),
                    synthetic.replace('common_speculative_impl_draft_dflash',
                                      'common_speculative_impl_draft_dflash clamp', 1),
                    synthetic.replace('R (f32)', 'R (f16)'),
                    synthetic.replace('GGML_SYCL_ENABLE_GRAPH: 0', 'GGML_SYCL_ENABLE_GRAPH: 1'),
                    marker + synthetic.split(marker, 1)[1], synthetic + marker]
        for key in ('n_outputs_max', 'n_outputs_max_per_seq'):
            import re
            bad_logs.append(re.sub(r'(llama_context: ' + key + r'\s*=\s*)4\b', r'\g<1>8', synthetic))
        for bad in bad_logs:
            self.assertNotEqual(synthetic, bad)
            with self.assertRaises(AssertionError):
                P['check_startup'](bad, H)
        self.assertEqual(log, ''.join((RECORDED / ('native-and-startup.' + s + '.log')).read_text()
                                     for s in ('stdout', 'stderr')))

    def test_collision_only_new_name_old_phase_names_preserved(self):
        existing = [C['ORIGINAL']] + [C['PREFIX'] + cell.lower() for cell in C['QUALIFY_ORDER']]
        before = list(existing)
        self.assertEqual([P['NAME']], P['check_names'](existing)['planned_names'])
        self.assertEqual(before, existing)
        with self.assertRaises(AssertionError):
            P['check_names'](existing + [P['NAME']])

    def test_four_frozen_requests_actual_id_references_and_hash_coverage(self):
        selected = P['jobs'](H)
        self.assertEqual(P['STEMS'], tuple(job['stem'] for job in selected))
        self.assertEqual(selected, [j for j in C['qualification_jobs'](H) if j['stem'] in P['STEMS']])
        frozen = P['freeze'](H)
        for job in selected:
            a = P['read_view'](H, H['off']['ROOT'] / 'run-r1/A0', job)
            fresh = P['read_view'](H, C['QUALIFY_DIR'] / 'A', job)
            w3 = P['read_view'](H, RECORDED, job)
            self.assertTrue(P['compare'](job, fresh, a)['passed'])
            comparison = P['compare'](job, w3, a)
            if job['stem'].endswith('/diagnostic'):
                self.assertFalse(comparison['passed'])
                self.assertEqual(116 if job['stem'].startswith('short/') else 14,
                                 comparison['id_comparison']['first_difference_ordinal'])
            for cell in ('A', 'W3'):
                path = str(C['artifact'](C['QUALIFY_DIR'] / cell / job['stem'], '.http-body'))
                self.assertIn(path, frozen['references'])
        for name in P['FILES']:
            self.assertIn(str(HERE / name), frozen['hashes'])
        self.assertEqual(C['source_binary_hashes'](H), frozen['source_binaries'])

    def test_zero_proposals_requires_final_counts_and_positive_real_draft_work(self):
        proof = P['zero_proposal_proof']('', STATS, [{}, {'draft_n': 0, 'draft_n_accepted': 0}])
        self.assertTrue(proof['proven'])
        for final in ('', '#calls(b,g,a) = 4 508 0', STATS.replace('#gen tokens = 0', '#gen tokens = 1'),
                      STATS.replace('#acc tokens = 0', '#acc tokens = 1'),
                      STATS.replace('4 508 0', '4 0 0'), STATS.replace('300.0', '0.0')):
            with self.subTest(final=final), self.assertRaises(AssertionError):
                P['zero_proposal_proof']('', final, [{}])
        with self.assertRaises(AssertionError):
            P['zero_proposal_proof'](STATS.replace('#gen tokens = 0', '#gen tokens = 1'), STATS, [{}])
        for counts in ({'draft_n': 1}, {'draft_n_accepted': 1}, {'draft_n': None}, {'draft_n': False}):
            with self.assertRaises(AssertionError):
                P['zero_proposal_proof']('', STATS, [counts])

    def test_four_valid_mismatches_continue_but_infra_aborts_and_logs_survive(self):
        rt = Mock(h=H, out=self.tmp, off={'check_off_log': Mock()})
        rt.save_logs.return_value = STATS
        results = [dict(kind=C['qualification_kinds'](job), comparison=dict(passed=False, equal=False))
                   for job in P['jobs'](H)]
        rt.run_job.side_effect = results
        P['collect'](rt, CELL)
        self.assertEqual(4, rt.run_job.call_count)
        for call in rt.run_job.call_args_list:
            self.assertEqual({'expected_for_compare': False}, call.kwargs)
        report = C['load'](self.tmp / CELL / 'capture.json')
        self.assertFalse(report['suppression_proven'])
        self.assertTrue(all(not row['matched_frozen_a'] for row in report['jobs']))
        rt.run_job.reset_mock(side_effect=True)
        rt.save_logs.reset_mock()
        rt.run_job.side_effect = [results[0], C['InfraFailure']('CPU transport fixture')]
        with self.assertRaises(C['InfraFailure']):
            P['collect'](rt, CELL)
        self.assertEqual(2, rt.run_job.call_count)
        rt.save_logs.assert_called_once_with(CELL, 'before-stop')

    def test_analysis_cannot_claim_proof_before_fresh_live_restoration(self):
        C['save'](self.tmp / 'summary.json', dict(status='captured', restored=True,
                  order=[CELL], completed=[CELL], timing_permitted=False))
        runtime = Mock()
        runtime.verify_restored.side_effect = C['InfraFailure']('restore fixture')
        proof = Mock()
        with patch.dict(P['analyze'].__globals__, OUT=self.tmp, Runtime=Mock(return_value=runtime),
                        zero_proposal_proof=proof):
            with self.assertRaises(C['InfraFailure']):
                P['analyze'](H)
        proof.assert_not_called()
        self.assertFalse((self.tmp / 'analysis.json').exists())

    def test_inherited_capture_restores_on_failures_and_all_three_signals(self):
        for failure in ('preflight', 'begin', 'run', 'end', signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            out = self.tmp / str(failure)
            out.mkdir()
            rt = Mock(out=out, mode='suppression')
            rt.restore.return_value = []
            callback = Mock(return_value=[])
            with patch.object(K['capture'].__globals__['signal'], 'signal') as install:
                if isinstance(failure, int):
                    def interrupt(rt, cell):
                        handlers = dict(call.args for call in install.call_args_list)
                        handlers[failure](failure, None)
                    callback.side_effect = interrupt
                else:
                    failing = callback if failure == 'run' else getattr(rt, failure)
                    failing.side_effect = InterruptedError('CPU fixture')
                with self.assertRaises(InterruptedError):
                    K['capture'](rt, (CELL,), callback)
            rt.restore.assert_called_once()
            self.assertEqual('failed', C['load'](out / 'summary.json')['status'])

    def test_inherited_restore_stops_and_preserves_only_control_even_if_stop_fails(self):
        rt = P['Runtime'](H, self.tmp, 'suppression')
        rt.created = [CELL]
        (self.tmp / CELL).mkdir()
        rt._stop_if_running = Mock(side_effect=RuntimeError('CPU stop fixture'))
        rt.logs = Mock(return_value={'stdout': b'raw', 'stderr': b'final'})
        rt.inspect = Mock(return_value={'fixture': 'preserved'})
        rt.verify_restored = Mock()
        rt.capacity = Mock()
        errors = rt.restore()
        self.assertEqual('stop-' + CELL, errors[0]['action'])
        rt._stop_if_running.assert_called_once_with(P['NAME'])
        rt.logs.assert_called_once_with(P['NAME'])
        rt.inspect.assert_called_once_with(P['NAME'])
        rt.verify_restored.assert_called_once()
        self.assertEqual(b'final', (self.tmp / CELL / 'restored-final.stderr.log').read_bytes())


if __name__ == '__main__':
    result = unittest.main(verbosity=2, exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
