#!/usr/bin/env python3
"""Bounded lifecycle regression tests. No Docker/GPU/model dependency."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cell', HERE / 'run-cell.py')
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)


class Lifecycle(unittest.TestCase):
    def exercise(self, fault=None, prototype=False, weights='original', dnnl_verbose=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'src'
            (src / 'build-sycl/bin').mkdir(parents=True)
            (src / 'build-sycl/bin/llama-server').touch()
            (src / 'UPSTREAM_COMMIT').write_text('b6334e37e6ef286fc5091a41b6561b2113d915fb')
            (root / 'runtime-precheck.log').write_text('no missing libraries')
            argv = ['run-cell.py', '--name', 'test']
            if dnnl_verbose:
                argv += ['--dnnl-verbose']
            if prototype:
                candidate = src.with_name(src.name + '-nvfp4-r1')
                (candidate / 'build-sycl/bin').mkdir(parents=True)
                for binary in ('llama-server', 'test-backend-ops'):
                    (candidate / 'build-sycl/bin' / binary).touch()
                (candidate / 'UPSTREAM_COMMIT').write_text((src / 'UPSTREAM_COMMIT').read_text())
                (root / 'nvfp4-r1-runtime-precheck.log').write_text('no missing libraries')
                argv += ['--source', str(candidate), '--test-nvfp4']
            model_root = root / 'models'
            selected_model = model_root if weights == 'original' else model_root.with_name(model_root.name + '-' + weights)
            selected_model.mkdir()
            model = selected_model / 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
            model.write_bytes(b'model')
            manifest = 'model-verified.json' if weights == 'original' else f'model-verified-{weights}.json'
            (root / manifest).write_text(json.dumps({'pass_': True, 'path': str(model), 'bytes': 5}))
            argv += ['--weights', weights]
            production = root / 'production.sh'
            production.write_text('# original')
            helper = root / 'helper.py'
            helper.write_text('# helper')
            state = {'containers': {'qwen38': True}, 'restores': 0, 'ticks': 0, 'models': 0}

            def command(*argv, **kwargs):
                containers = state['containers']
                if argv[:2] == ('docker', 'ps'):
                    return '\n'.join(n for n, running in containers.items() if '-a' in argv or running)
                if argv[:3] == ('docker', 'inspect', '--format'):
                    if argv[3] == '{{.Image}}':
                        return 'sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
                    return 'true' if containers.get(argv[-1]) else 'false'
                if argv[:2] == ('docker', 'create'):
                    state['create_argv'] = argv
                    containers[cell.NAME] = False
                    return 'container-id'
                if argv[:2] == ('docker', 'stop'):
                    containers.pop(argv[-1], None)
                    return argv[-1]
                if argv[:2] == ('docker', 'start'):
                    containers[cell.NAME] = fault != 'native'
                    if fault == 'launch':
                        raise subprocess.TimeoutExpired(argv, 60)
                    return cell.NAME
                if argv[:2] == ('docker', 'logs'):
                    if fault == 'logs':
                        raise RuntimeError('logs failed')
                    return 'log'
                if argv[:2] == ('docker', 'rm'):
                    if fault == 'survivor':
                        raise RuntimeError('remove failed, owner survives')
                    containers.pop(cell.NAME, None)
                    if fault == 'remove-ack':
                        raise RuntimeError('remove ack lost, owner already removed')
                    return cell.NAME
                if argv[0] == 'python3':
                    if fault == 'smoke':
                        raise subprocess.CalledProcessError(1, argv, output='smoke failed')
                    out = Path(argv[argv.index('--out') + 1])
                    out.mkdir()
                    (out / 'summary.json').write_text('{"pass": true}')
                    return 'ok'
                raise AssertionError(argv)

            def call(path, *args, **kwargs):
                if path == '/v1/models':
                    if not state['containers'].get('qwen38'):
                        raise OSError('Qwen not ready')
                    state['models'] += 1
                    if fault == '503' and state['models'] == 1:
                        raise RuntimeError('HTTP 503')
                    return {'body': {'data': [{'id': 'qwen38', 'max_model_len': 212992}]}}
                return {'body': {'choices': [{'text': ' five', 'logprobs': {'token_logprobs': [-0.01]}}],
                                 'usage': {'prompt_tokens': 7, 'completion_tokens': 1}}}

            def start(*args, **kwargs):
                state['restores'] += 1
                if fault == 'launcher-exit' and state['restores'] == 1:
                    return SimpleNamespace(poll=lambda: 1)
                state['containers']['qwen38'] = True
                return SimpleNamespace(poll=lambda: None)

            def sleep(seconds):
                state['ticks'] += seconds

            h = {'command': command, 'save': lambda p, v: p.write_text(json.dumps(v)),
                 'snapshot': lambda: {'unchanged': True}, 'kernel': lambda p: p.write_text('-- No entries --'),
                 'call': call, 'PRODUCTION': production}
            api = {'request': lambda *a, **k: {'data': [{'id': 'nemotron-bolt'}]}}
            original_stat = __import__('posix').stat
            with patch.object(cell, 'ROOT', root), patch.object(cell, 'SRC', src), patch.object(cell, 'MODEL', model_root), patch.object(cell, 'HELPER', str(helper)), \
                 patch.object(cell.runpy, 'run_path', side_effect=[h, api]), \
                 patch.object(cell.subprocess, 'Popen', side_effect=start), \
                 patch.object(cell.os, 'stat', side_effect=lambda p, *a, **k: SimpleNamespace(st_gid=1) if str(p) == '/dev/dri/renderD128' else original_stat(p, *a, **k)), \
                 patch.object(cell.socket, 'socket'), patch.object(cell.signal, 'signal') as signals, \
                 patch.object(cell.time, 'monotonic', side_effect=lambda: state['ticks']), \
                 patch.object(cell.time, 'sleep', side_effect=sleep), \
                 patch.object(sys, 'argv', argv):
                error = None
                try:
                    cell.main()
                except BaseException as exc:
                    error = exc
                self.assertIn(((cell.signal.SIGHUP, cell.signal.SIG_IGN), {}), signals.call_args_list)
            summary = json.loads((root / 'test/summary.json').read_text())
            return state, summary, error

    def test_success_restores(self):
        state, summary, error = self.exercise()
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])
        self.assertEqual(state['restores'], 1)

    def test_start_timeout_after_daemon_start_restores(self):
        state, summary, error = self.exercise('launch')
        self.assertIsNotNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])
        self.assertEqual(state['containers'], {'qwen38': True})

    def test_smoke_failure_restores(self):
        state, summary, error = self.exercise('smoke')
        self.assertIsNotNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])

    def test_log_failure_does_not_skip_restore(self):
        state, summary, error = self.exercise('logs')
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])

    def test_surviving_owner_blocks_overlap(self):
        state, summary, error = self.exercise('survivor')
        self.assertIsNotNone(error)
        self.assertEqual(state['restores'], 0)
        self.assertIn('absent', summary['cleanup']['error'])

    def test_lost_remove_ack_rechecks_and_restores(self):
        state, summary, error = self.exercise('remove-ack')
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])

    def test_restore_retries_http_503(self):
        state, summary, error = self.exercise('503')
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])
        self.assertGreater(state['models'], 1)

    def test_restore_retries_exited_launcher(self):
        state, summary, error = self.exercise('launcher-exit')
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])
        self.assertEqual(state['restores'], 2)

    def test_onednn_verbose_is_explicit_diagnostic(self):
        state, summary, error = self.exercise(dnnl_verbose=True)
        self.assertIsNone(error)
        self.assertIn('ONEDNN_VERBOSE=1', state['create_argv'])
        self.assertTrue(summary['configuration']['dnnl_verbose'])
        self.assertTrue(summary['cleanup']['production_restored'])

    def test_requantized_model_uses_separate_readonly_mount(self):
        for weights in ('ssm-q8_0', 'ssm-q6_k'):
            with self.subTest(weights=weights):
                state, summary, error = self.exercise(weights=weights)
                self.assertIsNone(error)
                self.assertTrue(summary['cleanup']['production_restored'])
                mounts = state['create_argv']
                self.assertTrue(any(str(x).endswith(f'/models-{weights}:/models:ro') for x in mounts))
                self.assertEqual(summary['configuration']['weights'], weights)
                self.assertNotIn('NVFP4_NUMERICAL_OPT1', mounts[-1])

    def test_prototype_runs_native_checks_before_server(self):
        state, summary, error = self.exercise(prototype=True)
        self.assertIsNone(error)
        self.assertTrue(summary['cleanup']['production_restored'])
        shell = state['create_argv'][-1]
        self.assertLess(shell.index('NVFP4_NUMERICAL_OPT1'), shell.index('NVFP4_NUMERICAL_OPT0'))
        self.assertLess(shell.index('NVFP4_NUMERICAL_PASSED'), shell.index('exec /src/build-sycl/bin/llama-server'))
        self.assertIn(summary['configuration']['source'] + ':/src:ro', state['create_argv'])
        self.assertEqual(shell.count("grep -Fx '  56/56 tests passed'"), 2)
        syntax = subprocess.run(['bash', '-n'], input=shell, text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_native_process_failure_restores_without_api_checks(self):
        state, summary, error = self.exercise('native', prototype=True)
        self.assertIsNotNone(error)
        self.assertNotIn('smoke', summary)
        self.assertTrue(summary['cleanup']['production_restored'])
        self.assertEqual(state['containers'], {'qwen38': True})

    def test_requantization_requires_original_backend(self):
        candidate = cell.SRC.with_name(cell.SRC.name + '-nvfp4-r1')
        with patch.object(sys, 'argv', ['run-cell.py', '--name', 'bad', '--source', str(candidate), '--weights', 'ssm-q8_0']), \
             patch.object(cell.runpy, 'run_path') as helper:
            with self.assertRaises(AssertionError):
                cell.main()
            helper.assert_not_called()

    def test_unapproved_source_rejected_before_host_reads(self):
        with patch.object(sys, 'argv', ['run-cell.py', '--name', 'bad', '--source', '/unapproved']), \
             patch.object(cell.runpy, 'run_path') as helper:
            with self.assertRaises(AssertionError):
                cell.main()
            helper.assert_not_called()

    def test_native_checks_rejected_for_original_source(self):
        with patch.object(sys, 'argv', ['run-cell.py', '--name', 'bad', '--test-nvfp4']), \
             patch.object(cell.runpy, 'run_path') as helper:
            with self.assertRaises(AssertionError):
                cell.main()
            helper.assert_not_called()


if __name__ == '__main__':
    unittest.main()
