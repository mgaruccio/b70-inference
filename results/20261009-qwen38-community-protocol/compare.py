#!/usr/bin/env python3
"""Validate archived HTTP measurements; retain the failed graph-evidence status.

Usage: python3 -B compare.py BASELINE_DIRECTORY CANDIDATE_DIRECTORY > comparison.json
Directories are the extracted current-serving-02/ and shared-kv-serving-01/.
"""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def load(path):
    return json.loads(path.read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)


def validate(directory, cell, prompt, output):
    result = load(directory / cell / 'results.json')
    records = result['records']
    assert len(records) == 5 and result['ignore_eos']
    assert set(result['warmups_discarded']) == {'generic', 'shape'}
    prompts = load(directory / ('prefill-prompts.json' if output == 1 else 'prompts.json'))['prompts']
    expected = [p for p in prompts if p['target_tokens'] == prompt]
    assert len(expected) == 6
    assert result['warmups_discarded']['shape']['messages_sha256'] == hashlib.sha256(
        json.dumps(expected[0]['messages'], sort_keys=True).encode()).hexdigest()
    for index, record in enumerate(records, 1):
        assert record['rep'] == index
        assert record['messages_sha256'] == hashlib.sha256(
            json.dumps(expected[index]['messages'], sort_keys=True).encode()).hexdigest()
        assert (record['prompt_tokens'], record['completion_tokens']) == (prompt, output)
        assert record['prefix_cache_hits_delta'] == 0
        assert record['requested_output_tokens'] == output and record['ignore_eos']
        rows = [json.loads(line) for line in (directory / cell / f'rep{index}.sse.jsonl').read_text().splitlines()]
        assert rows[-1]['payload'] == '[DONE]'
        times = [r['monotonic_ns'] for r in rows]
        assert times == sorted(times)
        assert record['request_start_ns'] <= times[0] <= times[-1] <= record['request_end_ns']
        first, usage, finish = None, None, None
        reasoning, content = [], []
        for row in rows[:-1]:
            event = json.loads(row['payload'])
            assert 'error' not in event
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                r = delta.get('reasoning_content') or delta.get('reasoning') or ''
                c = delta.get('content') or ''
                if (r or c) and first is None:
                    first = row['monotonic_ns']
                reasoning.append(r)
                content.append(c)
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
        assert usage and (usage['prompt_tokens'], usage['completion_tokens']) == (prompt, output)
        assert first == record['first_generated_ns']
        assert finish == record['finish_reason'] == 'length'
        assert ''.join(reasoning) == record['reasoning_text']
        assert ''.join(content) == record['content_text']
        ttft = (first - record['request_start_ns']) / 1e9
        decode = (record['request_end_ns'] - first) / 1e9
        close(record['ttft_s'], ttft)
        close(record['total_s'], (record['request_end_ns'] - record['request_start_ns']) / 1e9)
        close(record['post_first_generation_s'], decode)
        close(record['input_tokens_per_ttft_s'], prompt / ttft)
        if output > 1:
            close(record['client_post_first_tps'], (output - 1) / decode)
        else:
            assert record['client_post_first_tps'] is None
        for key, counter in [('mtp_proposed_tokens', 'mtp_draft_tokens'), ('mtp_accepted_tokens', 'mtp_accepted_tokens')]:
            close(record[key], record['counter_after'][counter] - record['counter_before'][counter])
    return records, result['summary']


def main():
    baseline, candidate = map(Path, sys.argv[1:])
    for name in ('prompts.json', 'prefill-prompts.json', 'b70-realworld-context-harness.py'):
        assert (baseline / name).read_bytes() == (candidate / name).read_bytes(), name
    b, c = [load(p / 'container.json')[0] for p in (baseline, candidate)]
    assert b['Image'] == c['Image']
    hook = '; /opt/venv/bin/python -P /experiment/qwen38_step_timing_patch.py'
    assert c['Config']['Cmd'][:-1] == b['Config']['Cmd'][:-1]
    assert c['Config']['Cmd'][-1].count(hook) == 1
    assert c['Config']['Cmd'][-1].replace(hook, '', 1) == b['Config']['Cmd'][-1]
    contract = load(candidate / 'build-contract.json')
    assert set(c['Config']['Env']) - set(b['Config']['Env']) == set(contract['environment'])
    assert not set(b['Config']['Env']) - set(c['Config']['Env'])
    extra_mounts = {f"{m['host']}:{m['container']}:ro" for m in contract['mounts']}
    assert set(c['HostConfig']['Binds']) - set(b['HostConfig']['Binds']) == extra_mounts
    assert not set(b['HostConfig']['Binds']) - set(c['HostConfig']['Binds'])
    pre = load(baseline / 'preconditions.json')
    assert pre == load(candidate / 'preconditions.json')
    for path in (baseline, candidate):
        assert pre == load(path / 'postconditions/preconditions.json')
        assert load(path / 'cleanup.json')['returncode'] == 0
        assert load(path / 'exit.json')['new_guarded_error_lines'] == []
    evidence = load(candidate / 'candidate-execution-evidence.json')
    assert evidence['eligible_dispatch_log_count'] > 0 and evidence['unsupported_q5_log_count'] == 0
    assert contract['candidate_library_sha256'] == 'e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec'
    assert any(contract['candidate_library_sha256'] in line for line in evidence['eligible_dispatch_log'])
    cells = []
    all_hashes = set()
    for prompt, output, published in ((512, 128, 112.65), (8192, 128, 103.63), (8192, 1, 1696), (130944, 128, 62.52)):
        cell = f'p{prompt}-g{output}'
        br, bs = validate(baseline, cell, prompt, output)
        cr, cs = validate(candidate, cell, prompt, output)
        hashes = [r['messages_sha256'] for r in br]
        assert hashes == [r['messages_sha256'] for r in cr]
        assert len(set(hashes)) == 5 and not all_hashes.intersection(hashes)
        all_hashes.update(hashes)
        key = 'input_tokens_per_ttft_s' if output == 1 else 'client_post_first_tps'
        summary_key = 'input_tokens_per_ttft_s' if output == 1 else 'client_post_first_decode_tps'
        bm, cm = [statistics.median(r[key] for r in records) for records in (br, cr)]
        close(bm, bs[summary_key]['median'])
        close(cm, cs[summary_key]['median'])
        cells.append({'cell': cell, 'metric': key, 'baseline': bs[summary_key], 'shared_kv': cs[summary_key],
                      'delta_percent': (cm / bm - 1) * 100, 'published_reference': published,
                      'vs_published_descriptive_percent': (cm / published - 1) * 100,
                      'identical_output_pairs': sum((x['reasoning_text'], x['content_text']) == (y['reasoning_text'], y['content_text']) for x, y in zip(br, cr)),
                      'baseline_request_seconds_median': statistics.median(r['total_s'] for r in br),
                      'shared_kv_request_seconds_median': statistics.median(r['total_s'] for r in cr),
                      'baseline_proposed_accepted': [sum(r[k] for r in br) for k in ('mtp_proposed_tokens', 'mtp_accepted_tokens')],
                      'shared_kv_proposed_accepted': [sum(r[k] for r in cr) for k in ('mtp_proposed_tokens', 'mtp_accepted_tokens')],
                      'measured_prompt_hashes': hashes})
    print(json.dumps({'tier': 'development; not standard-publication or quality qualification',
                      'validation': '40 measured HTTP records checked against archived raw SSE; inputs/config matched except candidate mounts/env/hook',
                      'baseline_runner_exit': load(baseline / 'exit.json'),
                      'candidate_runner_exit': load(candidate / 'exit.json'),
                      'candidate_execution_evidence': evidence, 'cells': cells}, indent=2))


if __name__ == '__main__':
    main()
