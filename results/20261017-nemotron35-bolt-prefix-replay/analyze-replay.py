#!/usr/bin/env python3
"""Saved replay analysis; no inference beyond a read-only live health check."""
import datetime
import json
from pathlib import Path
import re
import runpy

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefix-replay')


def load(path):
    return json.loads(path.read_text())


def common_length(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def score_row(body, index):
    rows = body['completion_probabilities']
    if len(rows) != len(body['tokens']) or [r['id'] for r in rows] != body['tokens']:
        return None
    row = rows[index]
    top = row['top_logprobs']
    return dict(token_id=row['id'], piece=row['token'], logprob=row['logprob'], top=top,
                top_two_logprob_gap=top[0]['logprob'] - top[1]['logprob'] if len(top) > 1 else None)


def main():
    controller = runpy.run_path(str(ROOT / 'run-replay.py'))
    prior = runpy.run_path(str(controller['PREVIOUS'] / 'run-controls.py'))
    helper = runpy.run_path(str(prior['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(prior['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(prior['OLD'] / 'probe-api.py'))
    command, binaries = helper['command'], helper['binaries']
    assert command('hostname').strip() == 'inference-host'
    out = ROOT / 'run-r1'
    summary = load(out / 'summary.json')
    assert summary['restored'] and not summary['timing_permitted'] and not summary['promotion_permitted']
    frozen = load(out / 'frozen-prefix.json')
    prefix = frozen['generated_prefix_ids']
    result = dict(matrix_status=summary['status'], matrix_error=summary.get('error'),
                  timing_permitted=False, promotion_permitted=False, cells={},
                  caveat='Refilled exact-prefix replay is not execution-equivalent to incremental decoding. Separate request families; no determinism guarantee from finite repeats.')
    sampler_keys = ['seed', 'temperature', 'top_k', 'top_p', 'min_p', 'repeat_penalty', 'presence_penalty',
                    'frequency_penalty', 'dry_multiplier', 'mirostat', 'ignore_eos', 'n_probs',
                    'post_sampling_probs', 'backend_sampling', 'samplers']
    all_prompt_ids = []
    for label, data in summary['cells'].items():
        cell = out / label
        log = (cell / 'server.log').read_text()
        item = dict(status=data['status'], fusion=data['fusion'], esimd=data['esimd'], families={})
        result['cells'][label] = item
        item['native385_passed'] = 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
        if item['native385_passed']:
            rows = [tuple(map(int, p)) for p in re.findall(r'(\d+)/(\d+) tests passed', log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[0])]
            assert rows == [(n, n) for n in [56,56,20,20,82,82,34,34,1] for _ in range(2)]
        if (cell / 'quality/summary.json').exists():
            quality = load(cell / 'quality/summary.json')
            item['quality'] = dict(passed=sum(bool(c['pass']) for c in quality['checks']), total=len(quality['checks']), gate=quality['pass'])
        if data['status'] != 'completed':
            item['completed_response_stems'] = [r['stem'] for r in data['responses']]
            continue
        assert item['native385_passed'] and item['quality'] == dict(passed=20, total=20, gate=True)
        model_log = log.split(controller['MARKER'], 1)[1]
        assert f"GGML_SYCL_ENABLE_FUSION: {data['fusion']}" in model_log
        assert f"GGML_SYCL_ENABLE_ESIMD: {data['esimd']}" in model_log
        assert 'adding speculative implementation' not in model_log and '[SYCL][OP]' not in model_log
        assert 'GGML_ASSERT' not in model_log and 'SYCL exception' not in model_log
        prompt_ids = data['prompt_ids']
        all_prompt_ids.append(prompt_ids)
        assert len(prompt_ids) == 55
        assert load(cell / 'template.response.json')['prompt'] == frozen['rendered_prompt']
        assert load(cell / 'detokenize.response.json')['content'] == frozen['rendered_prompt']
        assert load(cell / 'tokenize-plain.response.json')['tokens'] == load(cell / 'tokenize-special.response.json')['tokens'] == prompt_ids
        warm = load(cell / 'warmup.result.json')
        assert warm['usage']['prompt_tokens'] == 55 and warm['usage']['completion_tokens'] == 128
        assert warm['timings']['cache_n'] == 0 and warm['finish_reason'] == 'length'
        reference_settings = None
        for family, count in [('chat', 3), ('native', 3), ('replay', 6)]:
            bodies, records = [], []
            for i in range(count):
                stem = f'{family}-{i:02}'
                body = load(cell / (stem + '.response.json'))
                payload = load(cell / (stem + '.request.json'))
                if family == 'chat':
                    assert payload == load(controller['PREVIOUS'] / 'run-r1/A0/diagnostic.request.json')
                    body = body['__verbose']
                else:
                    ids = prompt_ids if family == 'native' else prompt_ids + prefix
                    assert payload == controller['native_payload'](ids, 128 if family == 'native' else 1)
                controller['check_native'](body, 94 if family == 'replay' else 55, 1 if family == 'replay' else 128)
                selected = {k: body['generation_settings'][k] for k in sampler_keys}
                if reference_settings is None:
                    reference_settings = selected
                assert selected == reference_settings
                bodies.append(body)
                common = common_length(body['tokens'], prefix) if family != 'replay' else None
                boundary = score_row(body, 0 if family == 'replay' else 39) if family == 'replay' or common == 39 else None
                records.append(dict(stem=stem, shared_frozen_prefix_tokens=common,
                                    boundary=boundary, generated_tokens=body['tokens']))
            token_groups = {}
            for record in records:
                token_groups.setdefault(tuple(record['generated_tokens']), []).append(record['stem'])
            scores = [json.dumps(r['boundary']['top'], sort_keys=True) for r in records if r['boundary'] is not None]
            item['families'][family] = dict(unique_token_sequences=len(token_groups),
                groups=list(token_groups.values()), distinct_boundary_score_vectors=len(set(scores)),
                boundary_available_count=len(scores), responses=records)
        item['matched_sampling_settings'] = reference_settings
    if summary['status'] == 'completed':
        assert set(result['cells']) == {label for label, _, _ in controller['CELLS']}
        assert all(ids == all_prompt_ids[0] for ids in all_prompt_ids)
    original = load(out / 'original-container.json')
    live = json.loads(command('docker', 'inspect', prior['ORIGINAL']))[0]
    assert live['State']['Running'] and live['Id'] == original['Id'] and live['Config'] == original['Config']
    assert sorted(live['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {prior['ORIGINAL']}
    for label in summary['cells']:
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', 'bolt-prefix-r1-' + label.lower()).strip() == 'false'
    assert binaries(prior['ACCEPTED']) == load(out / 'binaries.json')
    hardware = runner['hardware']()
    assert hardware == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
    health = api['request'](prior['URL'], '/health')
    assert health['status'] == 'ok'
    smoke = load(out / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    result['restoration'] = dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_id=original['Id'], live_id=live['Id'], config_identical=True, all_mount_fields_identical=True,
        binaries_identical=True, hardware=hardware, health=health, ready_output=smoke['text'], all_experiments_stopped=True)
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    for label, item in result['cells'].items():
        print(label, json.dumps({family:{k:v for k,v in data.items() if k != 'responses'} for family,data in item['families'].items()}))
    print('BOLT_PREFIX_ANALYSIS_AND_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
