#!/usr/bin/env python3
"""Remote CPU-only saved-run analysis plus read-only live restoration checks."""
import collections
import datetime
import json
from pathlib import Path
import re
import runpy

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-outprod')
OLD = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
GROUPED = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')


def load(path):
    return json.loads(path.read_text())


def main():
    helper = runpy.run_path(str(GROUPED / 'run-grouped-diagnostic.py'))
    command, binaries = helper['command'], helper['binaries']
    runner = runpy.run_path(str(OLD / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(OLD / 'probe-api.py'))
    assert command('hostname').strip() == 'inference-host'
    out = ROOT / 'activation-r1'
    summary = load(out / 'summary.json')
    assert summary['status'] == 'failed' and summary['restored'] and not summary['timing_permitted']
    assert 'deterministic rendered-output divergence' in summary['error']
    expected = [56, 56, 20, 20, 82, 82, 34, 34, 1]
    for phase in ('reference', 'activation'):
        log = (out / phase / 'server.log').read_text()
        assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
        rows = [tuple(map(int, p)) for p in re.findall(r'(\d+)/(\d+) tests passed', log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED')[0])]
        assert rows == [(n, n) for n in expected for _ in range(2)]
    quality = load(out / 'reference/quality/summary.json')
    assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
    log = (out / 'activation/server.log').read_text()
    assert 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED' in log
    focused = log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[1].split('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[0]
    rows = [tuple(map(int, p)) for p in re.findall(r'(\d+)/(\d+) tests passed', focused)]
    assert rows == [(80,80), (80,80), (18,18), (18,18)] * 2
    assert len(re.findall(r'not supported.*\(expected\)', focused)) == 6
    assert focused.count('(expected)') == 9
    assert '(UNEXPECTED)' not in focused and 'ERR =' not in focused
    assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
    start = datetime.datetime.fromisoformat(summary['phases']['activation']['window_since'])
    window = '\n'.join(line for line in log.splitlines() if datetime.datetime.fromisoformat(line.split()[0].replace('Z', '+00:00')) >= start)
    (out / 'activation/activation-window.log').write_text(window + '\n')
    outprod_calls = window.count('[SYCL][OP] call ggml_sycl_op_out_prod:')
    assert outprod_calls > 0
    assert window.count('[SYCL][OP] call ggml_sycl_op_out_prod done') == outprod_calls
    pairs = [tuple(map(int, p)) for p in re.findall(r'accepted\s+(\d+)\s*/\s*(\d+) draft tokens', window)]
    reference, candidate = [load(out / p / 'short.result.json') for p in ('reference', 'activation')]
    for result in (reference, candidate):
        assert result['usage']['prompt_tokens'] == 55 and result['usage']['completion_tokens'] == 128
        assert result['timings']['cache_n'] == 0 and result['finish_reason'] == 'length'
    accepted, proposed = sum(a for a,b in pairs), sum(b for a,b in pairs)
    assert accepted == candidate['timings']['draft_n_accepted'] == 86
    assert proposed == candidate['timings']['draft_n'] == 163
    assert len(pairs) == 41 and sum(a < b for a,b in pairs) == 26
    a, b = reference['text'], candidate['text']
    common = next((i for i,(x,y) in enumerate(zip(a,b)) if x != y), min(len(a),len(b)))
    assert a != b and common == 195
    historical = [
        Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr/activation-r1/reference/short'),
        Path('/home/mike/b70-evals/20261017-nemotron35-bolt-decode-profile/run-r1/baseline/bench/measured-00'),
    ]
    stability = []
    for prefix in historical:
        same_payload = load(prefix.with_suffix('.request.json')) == load(out / 'reference/short.request.json')
        same_text = load(prefix.with_suffix('.result.json'))['text'] == a
        stability.append(dict(path=str(prefix), same_payload=same_payload, same_rendered_output=same_text))
        assert same_payload and same_text
    original, live = load(out / 'original-container.json'), json.loads(command('docker', 'inspect', 'nemotron-bolt-probe'))[0]
    assert live['State']['Running'] and live['Id'] == original['Id'] and live['Config'] == original['Config']
    assert sorted(live['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {'nemotron-bolt-probe'}
    for name in ('bolt-lilicorr-outprod-r1-reference', 'bolt-lilicorr-outprod-r1-activation'):
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', name).strip() == 'false'
    hashes = binaries(runner['PREFILL_SSM'])
    assert hashes == load(out / 'binaries.json')
    candidate_path = Path('/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1')
    assert binaries(candidate_path) == load(out / 'candidate-binaries.json')
    hardware = runner['hardware']()
    assert hardware == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
    health = api['request']('http://127.0.0.1:18098', '/health')
    assert health['status'] == 'ok'
    ready = load(out / 'restored-smoke.result.json')
    assert ready['text'].strip() == 'READY' and ready['finish_reason'] == 'stop'
    result = dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        experiment_status='fidelity_failed', native385_per_launch=True, focused_numerical_checks=190,
        expected_rejections_in_test_mode=6, expected_rejections_in_support_mode=3,
        actual_model_outprod_calls_completed=outprod_calls, verification_events=len(pairs),
        proposed=proposed, accepted=accepted, acceptance_rate=accepted/proposed,
        rejection_events=26, accepted_count_histogram=dict(collections.Counter(a for a,b in pairs)),
        proposed_depth_histogram=dict(collections.Counter(b for a,b in pairs)),
        mean_accepted_draft_tokens_per_verification=accepted/len(pairs),
        mean_progress_including_target_token=(accepted+len(pairs))/len(pairs),
        exact_rendered_output_match=False, common_prefix_characters=common,
        historical_reference_stability=stability, patched_draft_off_control_run=False,
        candidate_long_request_run=False, candidate_non_debug_quality_run=False, performance_benchmark_run=False,
        restored=True, original_id=original['Id'], live_id=live['Id'], config_identical=True,
        all_mount_fields_identical=True, binaries=hashes, hardware=hardware, health=health,
        restoration_ready_output=ready['text'], experimental_containers_stopped=True)
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    print('BOLT_OUTPROD_FIDELITY_FAILURE_AND_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
