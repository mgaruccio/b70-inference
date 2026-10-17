#!/usr/bin/env python3
"""Remote-only verification of the preserved failed gate and live restoration."""
import datetime
import json
from pathlib import Path
import re
import runpy

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    command, binaries = helper['command'], helper['binaries']
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    assert command('hostname').strip() == 'inference-host'
    out = ROOT / 'activation-r1'
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'failed' and summary['restored'] and not summary['timing_permitted']
    expected = [56, 56, 20, 20, 82, 82, 34, 34, 1]
    native = {}
    for phase in ('reference', 'activation'):
        log = (out / phase / 'server.log').read_text()
        assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
        rows = [(int(a), int(b)) for a,b in re.findall(r'(\d+)/(\d+) tests passed', log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED')[0])]
        assert rows == [(n,n) for n in expected for _ in range(2)]
        native[phase] = sum(expected)
    quality = json.loads((out / 'reference/quality/summary.json').read_text())
    assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
    log = (out / 'activation/server.log').read_text()
    assertion = '/src/ggml/src/ggml-sycl/outprod.cpp:12: GGML_ASSERT(ggml_is_contiguous(src0)) failed'
    assert assertion in log
    pairs = re.findall(r'accepted\s+(\d+)\s*/\s*(\d+) draft tokens', log)
    assert not pairs
    response = json.loads((out / 'activation/short.result.json').read_text())
    assert response['text'] == 'Design' and response['usage'] == {} and response['finish_reason'] is None
    candidate = json.loads((out / 'activation/container.json').read_text())
    assert candidate['State']['ExitCode'] == 139 and not candidate['State']['OOMKilled']
    original = json.loads((out / 'original-container.json').read_text())
    live = json.loads(command('docker', 'inspect', 'nemotron-bolt-probe'))[0]
    assert live['State']['Running'] and live['Id'] == original['Id'] and live['Config'] == original['Config']
    assert sorted(live['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {'nemotron-bolt-probe'}
    for name in ('bolt-lilicorr-r1-reference', 'bolt-lilicorr-r1-activation'):
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', name).strip() == 'false'
    hashes = binaries(runner['PREFILL_SSM'])
    assert hashes == json.loads((out / 'binaries.json').read_text())
    hardware = runner['hardware']()
    assert hardware == json.loads((out / 'hardware-before.json').read_text()) == json.loads((out / 'hardware-after.json').read_text())
    health = api['request']('http://127.0.0.1:18098', '/health')
    assert health['status'] == 'ok'
    ready = json.loads((out / 'restored-smoke.result.json').read_text())
    assert ready['text'].strip() == 'READY' and ready['finish_reason'] == 'stop'
    result = dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  experiment_status='blocked', native_passes=native, reference_quality='20/20',
                  candidate_failure=assertion, candidate_exit_code=139, candidate_oom_killed=False,
                  acceptance_events=0, performance_benchmark_run=False, candidate_non_debug_quality_run=False,
                  restored=True, original_id=original['Id'], live_id=live['Id'],
                  config_identical=True, all_mount_fields_identical=True, binaries=hashes, hardware=hardware,
                  health=health, restoration_ready_output=ready['text'], experimental_containers_stopped=True)
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('BOLT_LILICORR_FAILURE_AND_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
