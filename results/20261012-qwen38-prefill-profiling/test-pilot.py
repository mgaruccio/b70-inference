#!/usr/bin/env python3
import contextlib
import gzip
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PILOT = HERE / 'run-pilot.py'
M = runpy.run_path(str(PILOT), run_name='pilot_test')


class Tests(unittest.TestCase):
    def test_iteration_ledger_uses_actual_context_tokens(self):
        text = '\n'.join([
            'Engine 000: Iteration(12): 1 context requests, 6,656 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 1 ms',
            'Engine 000: Iteration(13): 1 context requests, 6,656 context tokens, 0 generation requests, 0 generation tokens',
            'Engine 000: Iteration(14): 1 context requests, 3,072 context tokens, 0 generation requests, 0 generation tokens',
            'vllm:iteration_tokens_total_sum 16384',
        ])
        rows = M['parse_iterations'](text)
        self.assertEqual([r['context_tokens'] for r in rows], [6656, 6656, 3072])
        self.assertEqual(sum(r['context_tokens'] for r in rows), 16384)

    def test_device_durations_exclude_cpu_and_launch_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                {'ph': 'X', 'cat': 'cpu_op', 'name': 'gemm', 'ts': 1, 'dur': 999999},
                {'ph': 'X', 'cat': 'xpu_runtime', 'name': 'launch', 'ts': 1, 'dur': 999999},
                {'ph': 'X', 'cat': 'kernel', 'name': 'gemm', 'ts': 1000, 'dur': 200},
                {'ph': 'X', 'cat': 'kernel', 'name': 'attention', 'ts': 1250, 'dur': 300},
            ]
            with gzip.open(root / 'rank0.pt.trace.json.gz', 'wt') as stream:
                json.dump({'traceEvents': events}, stream)
            result = M['analyze_trace'](root)
            self.assertEqual(result['device_event_count'], 2)
            self.assertAlmostEqual(result['sum_device_activity_ms'], 0.5)
            self.assertAlmostEqual(result['device_first_to_last_span_ms'], 0.55)
            self.assertEqual(result['kernels'][0]['name'], 'attention')

    def test_cpu_only_trace_is_not_device_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rank0.pt.trace.json').write_text(json.dumps({'traceEvents': [
                {'ph': 'X', 'cat': 'cpu_op', 'name': 'graph replay', 'ts': 1, 'dur': 9000}]}))
            self.assertEqual(M['analyze_trace'](root)['device_event_count'], 0)

    def test_unrequested_budget_is_rejected_before_host_actions(self):
        result = subprocess.run([sys.executable, str(PILOT), '--output', '/unused', '--budget', '9984'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)

    def run_cleanup_case(self, stop_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / 'production.sh'
            original = (HERE.parent.parent / 'scripts/start-qwen38.sh').read_text()
            production.write_text(original)
            out = root / 'cell'
            running = ['qwen38']
            launched = []
            namespace = M['main'].__globals__

            def fake_command(*argv, **kwargs):
                if argv[:2] == ('docker', 'ps'):
                    return '\n'.join(running)
                if argv[:2] == ('docker', 'stop'):
                    if stop_failure and argv[-1] == 'qwen38':
                        raise RuntimeError('stop failed before removing original')
                    running.remove(argv[-1])
                    return argv[-1]
                if argv[:2] == ('bash', '-n'):
                    return ''
                raise AssertionError(argv)

            class Process:
                def __init__(self, argv, **kwargs):
                    launcher = Path(argv[1])
                    launched.append(launcher.read_text())
                    running.append('qwen38' if launcher == production else M['NAME'])
                    kwargs['stdout'].write('B70_FIX_UNIFORM_DECODE_PREFILL: applied\n')
                    kwargs['stdout'].flush()
                def poll(self):
                    return None
                def wait(self, timeout):
                    return 0

            def fake_call(path, *args, **kwargs):
                if path == '/v1/models':
                    return {'body': {'data': [{'id': 'qwen38', 'max_model_len': 212992}]}}
                if path == '/openapi.json':
                    return {'body': {'paths': {}}}
                raise AssertionError(path)

            replacements = {'PRODUCTION': production, 'snapshot': lambda: {'boot': 'fixed'},
                            'kernel': lambda p: p.write_text('-- No entries --\n'),
                            'command': fake_command, 'call': fake_call,
                            'complete': lambda *a: {'body': {'usage': {'prompt_tokens': 5, 'completion_tokens': 1}}}}
            with patch.dict(namespace, replacements), patch.object(M['subprocess'], 'Popen', Process), \
                 patch.object(sys, 'argv', [str(PILOT), '--output', str(out)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises((AssertionError, RuntimeError)):
                    M['main']()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertTrue(summary['cleanup']['production_running'])
            self.assertEqual(running, ['qwen38'])
            self.assertEqual(production.read_text(), original)
            if not stop_failure:
                self.assertEqual(launched[-1], original)
                self.assertIn('--enable-logging-iteration-details', launched[0])
                self.assertIn('VLLM_XPU_ENABLE_XPU_GRAPH=1', launched[0])
                self.assertIn('--max-model-len 212992', launched[0])

    def test_missing_profiling_api_restores_original(self):
        self.run_cleanup_case()

    def test_failed_stop_does_not_duplicate_or_kill_original(self):
        self.run_cleanup_case(stop_failure=True)


if __name__ == '__main__':
    unittest.main()
