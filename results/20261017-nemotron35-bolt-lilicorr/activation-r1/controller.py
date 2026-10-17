#!/usr/bin/env python3
"""Remote-only LiLiCorr capability gate. Never runs timing benchmarks."""
import datetime
import json
from pathlib import Path
import re
import runpy
import shlex
import signal
import subprocess

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
PROFILE = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'
SPEC = ['--spec-type', 'draft-dflash', '--spec-draft-n-max', '4', '--spec-draft-n-min', '0',
        '--spec-draft-p-min', '0', '--spec-draft-sampling', 'greedy', '--no-spec-draft-backend-sampling',
        '--spec-draft-ngl', '999', '--spec-draft-model', '/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf']
ACCEPTANCE_RE = re.compile(r'accepted\s+(\d+)\s*/\s*(\d+) draft tokens')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    profile = runpy.run_path(str(PROFILE / 'run-decode-profile.py'))
    expected = profile['EXPECTED']
    source = runner['PREFILL_SSM']
    assert command('hostname').strip() == 'inference-host'
    assert binaries(source) == expected
    draft = json.loads((ROOT / 'draft-verified.json').read_text())
    assert draft['pass'] and Path(draft['path']).stat().st_size == draft['bytes']
    assert all(json.loads((ROOT / 'tokenizer-comparison.json').read_text()).values())
    before = runner['hardware']()
    old = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert old['State']['Running'] and old['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in old['Mounts'] if m['Destination'] == '/src') == str(source)
    assert 'GGML_SYCL_DEBUG=1' not in old['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    names = {phase: 'bolt-lilicorr-r1-' + phase for phase in ('reference', 'activation')}
    existing = command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    assert not any(name in existing for name in names.values())
    create_base = json.loads((OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    assert create_base[-1] == old['Config']['Cmd'][-1]
    assert str(source) + ':/src:ro' in create_base
    requests = {}
    for label, mode in [('short', 'bench'), ('long', 'long')]:
        path = PROFILE / 'run-r1/baseline' / mode / 'measured-00.request.json'
        requests[label] = json.loads(path.read_text())
    out = ROOT / 'activation-r1'
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    save(out / 'original-container.json', old)
    save(out / 'hardware-before.json', before)
    save(out / 'binaries.json', expected)
    save(out / 'draft-verified.json', draft)
    summary = {'status': 'prepared', 'tier': 'development', 'timing_permitted': False, 'phases': {}}
    created = []
    stopped = False
    reference = {}

    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', ORIGINAL)
        for phase, name in names.items():
            cell = out / phase
            cell.mkdir()
            create = list(create_base)
            create[create.index('--name') + 1] = name
            if phase == 'activation':
                index = create.index(runner['IMAGE'])
                create[index:index] = ['-v', str(ROOT / 'draft') + ':/draft:ro']
                prefix, server = create[-1].rsplit('exec ', 1)
                # Native gates use their original explicit debug settings. Enable diagnostics only for the server.
                create[-1] = prefix + 'export GGML_SYCL_DEBUG=1 LLAMA_TRACE=1; exec ' + server + ' ' + shlex.join(SPEC)
            save(cell / 'create-argv.json', create)
            command(*create, timeout=60)
            created.append((name, cell))
            command('docker', 'start', name)
            data = {}
            summary['phases'][phase] = data
            data['models'] = ready(api, name)
            native = command('docker', 'logs', name)
            (cell / 'native-and-startup.log').write_text(native)
            assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in native
            if phase == 'reference':
                log = command('python3', '-B', str(OLD / 'probe-api.py'), '--out', str(cell / 'quality'),
                              '--mode', 'quality', timeout=1200)
                (cell / 'quality.log').write_text(log)
                quality = json.loads((cell / 'quality/summary.json').read_text())
                assert quality['pass'] and len(quality['checks']) == 20
                data['quality'] = quality
            else:
                assert "adding speculative implementation 'draft-dflash'" in native
                assert 'speculative decoding not supported by this context' not in native
                assert re.search(r'n_rs_seq\s*=\s*4\b', native), 'Expected four target rollback snapshots'
            data['responses'] = {}
            data['window_since'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            for label, payload in requests.items():
                result = api['chat'](URL, payload['messages'][0]['content'], cell / label,
                                     count=128, stream=True, ignore_eos=True)
                data['responses'][label] = result
                assert json.loads((cell / (label + '.request.json')).read_text()) == payload
                profile['check_request'](result, 55 if label == 'short' else 8270)
                if phase == 'reference':
                    reference[label] = result
                else:
                    a, b = reference[label]['text'], result['text']
                    common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    comparison = {'exact_rendered_output_match': a == b, 'common_prefix_characters': common,
                                  'reference_characters': len(a), 'candidate_characters': len(b)}
                    save(cell / (label + '-comparison.json'), comparison)
                    assert a == b, f'{label}: deterministic rendered-output divergence; no timing permitted'
            if phase == 'activation':
                log = command('docker', 'logs', '--timestamps', '--since', data['window_since'], name)
                (cell / 'activation-window.log').write_text(log)
                pairs = [(int(a), int(b)) for a, b in ACCEPTANCE_RE.findall(log)]
                acceptance = {'verification_events': len(pairs), 'proposed': sum(b for a,b in pairs),
                              'accepted': sum(a for a,b in pairs), 'rejection_events': sum(a < b for a,b in pairs)}
                save(cell / 'acceptance.json', acceptance)
                assert pairs and acceptance['accepted'] > 0 and acceptance['rejection_events'] > 0
                assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
            command('docker', 'stop', '-t', '90', name)
            assert binaries(source) == expected and runner['hardware']() == before
            data['status'] = 'passed'
            save(cell / 'summary.json', data)
        summary['status'] = 'passed'
        summary['activation_passed'] = True  # Fresh non-debug quality/fidelity gates still required.
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            (out / 'failed-command-output.log').write_text(exc.output or '')
        raise
    finally:
        try:
            for name, cell in created:
                try:
                    command('docker', 'stop', '-t', '90', name)
                finally:
                    (cell / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                    save(cell / 'container.json', json.loads(command('docker', 'inspect', name))[0])
        finally:
            try:
                if stopped:
                    assert not set(names.values()).intersection(command('docker', 'ps', '--format', '{{.Names}}').splitlines())
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    smoke = api['chat'](URL, 'Reply with exactly this word and nothing else: READY', out / 'restored-smoke', count=32)
                    assert smoke['text'].strip() == 'READY'
                    restored = json.loads(command('docker', 'inspect', ORIGINAL))[0]
                    save(out / 'restored-container.json', restored)
                    assert restored['Id'] == old['Id'] and restored['Config'] == old['Config']
                    assert sorted(restored['Mounts'], key=lambda m: m['Destination']) == sorted(old['Mounts'], key=lambda m: m['Destination'])
                    assert binaries(source) == expected
                after = runner['hardware']()
                save(out / 'hardware-after.json', after)
                assert after == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save(out / 'summary.json', summary)
    print('BOLT_LILICORR_ACTIVATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
