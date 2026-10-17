#!/usr/bin/env python3
"""One fixed development cell: native gates, activation or quality-before-timing, restore SSM."""
import argparse
import datetime
import json
from pathlib import Path
import re
import runpy
import signal

ART = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
TAIL = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-grouped-tail-r1')
ORIGINAL = 'nemotron-bolt-probe'
URL = 'http://127.0.0.1:18098'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['activation', 'baseline', 'candidate'], required=True)
    args = parser.parse_args()
    helper = runpy.run_path(str(ART / 'run-grouped-diagnostic.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    assert command('hostname').strip() == 'inference-host'
    baseline = runner['PREFILL_SSM']
    source = baseline if args.phase == 'baseline' else TAIL
    out = ART / ('tail-r1-' + args.phase)
    name = 'bolt-grouped-tail-r1-' + args.phase
    debug = args.phase == 'activation'
    if source == TAIL:
        assert 'BOLT_GROUPED_TAIL_RUNTIME_PRECHECK_PASSED' in (ART / 'tail-runtime-precheck.log').read_text()
        assert (source / 'build-sycl/bin/test-grouped-iq2-sycl').is_file()
        if not debug:
            gate = json.loads((ART / 'tail-r1-activation/summary.json').read_text())
            assert gate['status'] == 'passed' and gate['restored'] and gate['source'] == str(TAIL)
            assert gate['source_hashes'] == binaries(TAIL), 'candidate changed after activation'
    assert (source / 'UPSTREAM_COMMIT').read_text().strip() == 'b6334e37e6ef286fc5091a41b6561b2113d915fb'
    old = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    assert old['State']['Running'] and old['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in old['Mounts'] if m['Destination'] == '/src') == str(baseline)
    assert 'GGML_SYCL_DEBUG=1' not in old['Config']['Env']
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    assert name not in command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
    verified = json.loads((OLD / 'model-verified.json').read_text())
    assert verified['pass_'] and Path(verified['path']).parent == runner['MODEL']
    assert Path(verified['path']).stat().st_size == verified['bytes']
    before, baseline_hashes = runner['hardware'](), binaries(baseline)
    out.mkdir(exist_ok=False)
    (out / 'controller.py').write_bytes(Path(__file__).read_bytes())
    def save(filename, obj):
        (out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    save('hardware-before.json', before)
    save('original-container.json', old)
    create = json.loads((OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    create[create.index('--name') + 1] = name
    create[create.index(str(baseline) + ':/src:ro')] = str(source) + ':/src:ro'
    assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in create[-1]
    if source == TAIL:
        prefix, server = create[-1].rsplit('exec ', 1)
        native = ('GGML_SYCL_DEBUG=1 /src/build-sycl/bin/test-grouped-iq2-sycl 2>&1 | tee /tmp/grouped-native.log; '
                  "grep -Fx 'IQ2 grouped native ORIGINAL/CANDIDATE PASS valid_cases=13 negative_cases=6' /tmp/grouped-native.log; ")
        for marker, expected in [('[SYCL][IQ2_TAIL16] path=original combo=0 ', 13),
                                 ('[SYCL][IQ2_TAIL16] path=tail16 combo=0 ', 13),
                                 ('selector=ORIGINAL returned=false output_untouched=1', 6),
                                 ('selector=CANDIDATE returned=false output_untouched=1', 6)]:
            native += f"test \"$(grep -Fc '{marker}' /tmp/grouped-native.log)\" -eq {expected}; "
        create[-1] = prefix + native + 'exec ' + server
    if debug:
        create[create.index(runner['IMAGE']):create.index(runner['IMAGE'])] = ['-e', 'GGML_SYCL_DEBUG=1']
    save('create-argv.json', create)
    summary = {'status': 'prepared', 'tier': 'development', 'phase': args.phase,
               'source': str(source), 'source_hashes': binaries(source), 'debug': debug,
               'baseline_hashes': baseline_hashes}
    stopped = created = False
    def interrupted(signum, frame):
        raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        stopped = True
        command('docker', 'stop', '-t', '90', ORIGINAL)
        summary['container_id'] = command(*create).strip()
        created = True
        command('docker', 'start', name)
        summary['models'] = ready(api, name)
        if debug:
            payload = json.loads((OLD / 'prefill-ssm-r1-baseline/long/measured-00.request.json').read_text())
            prompt = payload['messages'][0]['content']
            summary['warmup'] = api['chat'](URL, prompt, out / 'warmup', count=128, stream=True, ignore_eos=True)
            since = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds')
            summary['activation_since'] = since
            summary['activation'] = api['chat'](URL, prompt, out / 'activation', count=1, stream=True, ignore_eos=True)
            log = command('docker', 'logs', '--timestamps', '--since', since, name)
            (out / 'activation-window.log').write_text(log)
            for key, count in [('warmup', 128), ('activation', 1)]:
                value = summary[key]
                assert value['usage']['prompt_tokens'] == 8270 and value['usage']['completion_tokens'] == count
                assert value['timings']['cache_n'] == 0
            markers = [line for line in log.splitlines() if '[SYCL][IQ2_TAIL16]' in line]
            candidate = [line for line in markers if 'path=tail16 ' in line]
            assert candidate and not any('path=original ' in line for line in markers)
            narrow = [int(re.search(r'narrow_tiles=(\d+)', line).group(1)) for line in candidate]
            assert sum(narrow) > 0, 'No actual-model narrow tail activation'
            summary['candidate_calls'] = len(candidate)
            summary['candidate_narrow_tiles'] = sum(narrow)
            command('python3', '-B', str(ART / 'analyze-grouped-diagnostic.py'), str(out / 'activation-window.log'),
                    '--output', str(out / 'analysis.json'))
            analysis = json.loads((out / 'analysis.json').read_text())
            assert len(candidate) == analysis['grouped_calls']
            assert sum(narrow) == analysis['grouped_geometry']['tail_tiles_le16']
        else:
            # probe-api.py fails immediately on quality regressions; no timing until it passes.
            for mode in ('smoke', 'quality', 'bench', 'long'):
                command('python3', '-B', str(OLD / 'probe-api.py'), '--out', str(out / mode), '--mode', mode, timeout=2400)
                summary[mode] = json.loads((out / mode / 'summary.json').read_text())
                assert summary[mode]['pass'], f'{mode} gate failed'
                if mode == 'quality':
                    assert len(summary[mode]['checks']) == 20
                    assert all(check['pass'] for check in summary[mode]['checks'])
        assert runner['hardware']() == before
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        try:
            if created:
                command('docker', 'stop', '-t', '90', name)
                (out / 'server.log').write_text(command('docker', 'logs', '--timestamps', name))
                save('experiment-container.json', json.loads(command('docker', 'inspect', name))[0])
        finally:
            try:
                if stopped:
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    result = api['chat'](URL, 'Reply with exactly this word and nothing else: READY', out / 'restored-smoke', count=32)
                    assert result['text'].strip() == 'READY'
                    assert binaries(baseline) == baseline_hashes
                after = runner['hardware']()
                save('hardware-after.json', after)
                assert after == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                save('summary.json', summary)
    print('GROUPED_CELL_PASSED ' + args.phase, flush=True)


if __name__ == '__main__':
    main()
