#!/usr/bin/env python3
"""Original/replay native grouped activation gate, with Bolt restoration."""
import json
from pathlib import Path
import runpy
import signal

ART = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
OUT = ART / 'native-original-r1'
SOURCE = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-grouped-diag-r1')
NAME = 'bolt-grouped-native-original'
ORIGINAL = 'nemotron-bolt-probe'


def main():
    helper = runpy.run_path(str(ART / 'run-grouped-diagnostic.py'))
    command, ready, binaries = (helper[k] for k in ('command', 'ready', 'binaries'))
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    assert command('hostname').strip() == 'inference-host'
    original = json.loads(command('docker', 'inspect', ORIGINAL))[0]
    baseline = runner['PREFILL_SSM']
    assert original['State']['Running'] and original['Config']['Labels']['b70.experiment'] == 'bolt-fusion'
    assert next(m['Source'] for m in original['Mounts'] if m['Destination'] == '/src') == str(baseline)
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {ORIGINAL}
    assert (SOURCE / 'build-sycl/bin/test-grouped-iq2-sycl').is_file()
    before, hashes = runner['hardware'](), binaries(baseline)
    OUT.mkdir(exist_ok=False)
    (OUT / 'controller.py').write_bytes(Path(__file__).read_bytes())
    summary = {'status': 'prepared', 'source': str(SOURCE), 'hardware_before': before,
               'baseline_hashes': hashes}
    create = json.loads((OLD / 'nvfp4-prefill-r2-no-gain-recovery/create-argv.json').read_text())
    create[create.index('--name') + 1] = NAME
    create[create.index(str(baseline) + ':/src:ro')] = str(SOURCE) + ':/src:ro'
    port = create.index('-p')
    del create[port:port + 2]
    create[create.index(runner['IMAGE']):create.index(runner['IMAGE'])] = ['-e', 'GGML_SYCL_DEBUG=1']
    create[-1] = ('set -eo pipefail; source /opt/intel/oneapi/setvars.sh --force; '
                  'export LD_LIBRARY_PATH=/src/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}; '
                  'exec /src/build-sycl/bin/test-grouped-iq2-sycl')
    (OUT / 'create-argv.json').write_text(json.dumps(create, indent=2) + '\n')
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
        command('docker', 'start', NAME)
        code = command('docker', 'wait', NAME, timeout=600).strip()
        log = command('docker', 'logs', NAME)
        (OUT / 'native.log').write_text(log)
        assert code == '0', f'native exit={code}'
        assert 'IQ2 grouped native ORIGINAL/REPLAY PASS valid_cases=13 negative_cases=6' in log
        summary['status'] = 'passed'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        try:
            if created:
                command('docker', 'stop', '-t', '90', NAME)
                (OUT / 'native.log').write_text(command('docker', 'logs', NAME))
        finally:
            try:
                if stopped:
                    command('docker', 'start', ORIGINAL)
                    summary['restored_models'] = ready(api, ORIGINAL)
                    result = api['chat']('http://127.0.0.1:18098', 'Reply with exactly this word and nothing else: READY', OUT / 'restored-smoke', count=32)
                    assert result['text'].strip() == 'READY'
                    assert binaries(baseline) == hashes
                summary['hardware_after'] = runner['hardware']()
                assert summary['hardware_after'] == before
                summary['restored'] = True
            except BaseException as exc:
                summary.update(status='failed', restoration_error=repr(exc))
                raise
            finally:
                (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('GROUPED_NATIVE_ORIGINAL_PASSED', flush=True)


if __name__ == '__main__':
    main()
