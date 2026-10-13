#!/usr/bin/env python3
"""Targeted lifecycle/launcher checks; not a substitute for host/API validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('diagnostic', HERE / 'run-diagnostic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = (HERE.parent / '20261012-qwen38-prefill-profiling/matrix-01/03-8192-on-32768/production-launcher.sh').read_text()


def case(fail_request=False, fail_start=False, fail_wait=False, fail_stop=False):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        out = root / 'native-01'
        (root / 'compatibility').mkdir()
        (root / 'compatibility/compatibility.json').write_text(json.dumps({'pass': True, 'graphs_enabled': True}))
        helper = root / 'helper.py'
        helper.write_text('# frozen helper fixture\n')
        production = root / 'production.sh'
        production.write_text(original)
        names = ['qwen38']
        requests = []
        profiles = []
        def command(*argv, **kwargs):
            if argv[:2] == ('docker', 'ps'):
                return '\n'.join(names)
            if argv[:2] == ('docker', 'inspect'):
                return module.IMAGE
            if argv[:2] == ('docker', 'stop'):
                assert names == [argv[-1]]
                names.clear()
                if fail_stop and argv[-1] == module.NAME:
                    raise module.subprocess.CalledProcessError(1, argv, output='container disappeared')
            return ''
        class Process:
            def __init__(self, argv, **kwargs):
                self.diagnostic = Path(argv[1]) != production
                names[:] = [module.NAME if self.diagnostic else 'qwen38']
                self.returncode = 1 if self.diagnostic and fail_start else 0
                self.wait_failed = False
                kwargs['stdout'].write('B70_FIX_UNIFORM_DECODE_PREFILL: fixture\n')
                kwargs['stdout'].flush()
            def poll(self):
                return 1 if self.diagnostic and fail_start else None
            def wait(self, timeout):
                if self.diagnostic and fail_wait and not self.wait_failed:
                    self.wait_failed = True
                    raise module.subprocess.TimeoutExpired('diagnostic docker client', timeout)
                return 0
            def terminate(self):
                pass
            def kill(self):
                pass
        def call(path, payload=None, **kwargs):
            if 'profile' in path:
                profiles.append(path)
            if path == '/v1/models':
                body = {'data': [{'id': 'qwen38', 'max_model_len': 212992}]}
            elif path == '/openapi.json':
                body = {'paths': {'/start_profile': {}, '/stop_profile': {}}}
            else:
                body = None
            return {'body': body, 'raw_text': '', 'http_status': 200}
        def complete(length, token, path):
            requests.append((length, token))
            if fail_request and token == 13101:
                raise RuntimeError('simulated diagnostic request failure')
            path.write_text('{}')
            if token == 13101:
                with (out / 'server.log').open('a') as log:
                    log.write('Iteration(1) 32,768 context tokens 0 generation tokens\n')
            return {'http_status': 200, 'elapsed_s': 1.0, 'body': {}}
        h = {'PRODUCTION': production, 'command': command,
             'save': lambda p, d: p.write_text(json.dumps(d)),
             'call': call, 'snapshot': lambda: {'boot': 'fixture'},
             'kernel': lambda p: p.write_text(''), 'complete': complete,
             'parse_iterations': lambda text: [{'context_tokens': 32768}],
             'analyze_trace': lambda p: {'device_event_count': 1}}
        with patch.object(module, 'ROOT', root), patch.object(module, 'HELPER', helper), \
             patch.object(module.runpy, 'run_path', return_value=h), \
             patch.object(module.subprocess, 'Popen', Process), \
             patch.object(module.signal, 'signal'), \
             patch('sys.argv', ['run-diagnostic.py', '--output', str(out)]):
            try:
                module.main()
            except (RuntimeError, module.subprocess.TimeoutExpired):
                assert fail_request or fail_start or fail_wait
            else:
                assert not fail_request and not fail_start
        assert names == ['qwen38'], 'Original production must be restored'
        assert production.read_text() == original
        report = json.loads((out / 'summary.json').read_text())
        assert report['cleanup']['production_running'] is True
        assert report['cleanup']['invariants_pass'] is True
        if not fail_start:
            assert profiles == ['/start_profile', '/stop_profile']
        else:
            assert not profiles
        candidate = (out / 'diagnostic-launcher.sh').read_text()
        for flag in ('--max-num-batched-tokens 8192', '--max-model-len 212992', '--gpu-memory-utilization 0.95',
                     '--kv-cache-dtype fp8', '--enable-prefix-caching', '--mamba-cache-mode align',
                     'VLLM_XPU_ENABLE_XPU_GRAPH=1'):
            assert flag in candidate
        assert requests[-1] == (5, 13102)


case()
case(fail_request=True)
case(fail_start=True)
case(fail_wait=True)
case(fail_stop=True)
print('5 lifecycle/unchanged-launcher checks passed')
