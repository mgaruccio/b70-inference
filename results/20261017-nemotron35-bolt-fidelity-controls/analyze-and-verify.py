#!/usr/bin/env python3
"""Remote saved-response comparisons and independent, read-only restoration checks."""
import collections
import datetime
import itertools
import json
from pathlib import Path
import re
import runpy

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls')


def load(path):
    return json.loads(path.read_text())


def difference(a, b):
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return dict(equal=a == b, common_prefix_length=index,
                first_difference_index=None if a == b else index,
                left_next=a[index:index + 32], right_next=b[index:index + 32])


def main():
    control = runpy.run_path(str(ROOT / 'run-controls.py'))
    helper = runpy.run_path(str(control['GROUPED'] / 'run-grouped-diagnostic.py'))
    runner = runpy.run_path(str(control['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    api = runpy.run_path(str(control['OLD'] / 'probe-api.py'))
    command, binaries = helper['command'], helper['binaries']
    assert command('hostname').strip() == 'inference-host'
    out = ROOT / 'run-r1'
    summary = load(out / 'summary.json')
    assert summary['restored'] and summary['timing_permitted'] is False
    result = dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  matrix_status=summary['status'], matrix_error=summary.get('error'),
                  timing_permitted=False, promotion_permitted=False,
                  diagnostic_probability_warning='Speculative probabilities are placeholders; token IDs only.',
                  cells={}, primary_output_groups=[], primary_cross_arm={}, diagnostic_comparisons=[])
    texts, diagnostics = {}, {}
    expected = [56, 56, 20, 20, 82, 82, 34, 34, 1]
    frozen = load(out / 'frozen-payload.json')
    for label, data in summary['phases'].items():
        cell = out / label
        log = (cell / 'server.log').read_text()
        item = dict(status=data.get('status', 'incomplete'))
        result['cells'][label] = item
        assert 'BOLT_SSM_PREFILL_NUMERICAL_PASSED' in log
        native = log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[0]
        rows = [tuple(map(int, p)) for p in re.findall(r'(\d+)/(\d+) tests passed', native)]
        assert rows == [(n, n) for n in expected for _ in range(2)]
        item['native385_passed'] = True
        if label[0] != 'A':
            assert 'BOLT_LILICORR_OUTPROD_NATIVE_PASSED' in log
            focused = log.split('BOLT_SSM_PREFILL_NUMERICAL_PASSED', 1)[1].split('BOLT_LILICORR_OUTPROD_NATIVE_PASSED', 1)[0]
            rows = [tuple(map(int, p)) for p in re.findall(r'(\d+)/(\d+) tests passed', focused)]
            assert rows == [(80, 80), (80, 80), (18, 18), (18, 18)] * 2
            assert len(re.findall(r'not supported.*\(expected\)', focused)) == 6
            assert focused.count('(expected)') == 9
            assert '(UNEXPECTED)' not in focused and 'ERR =' not in focused
            item.update(focused_numerical_passed=190, focused_support_contracts_passed=9)
        if (cell / 'quality/summary.json').exists():
            quality = load(cell / 'quality/summary.json')
            item['quality_passed'] = sum(bool(x['pass']) for x in quality['checks'])
            item['quality_total'] = len(quality['checks'])
            item['quality_gate'] = quality['pass']
        if item['status'] != 'completed':
            continue
        assert item['quality_gate'] and item['quality_passed'] == item['quality_total'] == 20
        assert 'GGML_ASSERT' not in log and 'SYCL exception' not in log
        responses = [load(cell / f'repeat-{i:02}.result.json') for i in range(3)]
        warmup = load(cell / 'warmup.result.json')
        for i, response in enumerate([warmup] + responses):
            stem = 'warmup' if i == 0 else f'repeat-{i - 1:02}'
            assert load(cell / (stem + '.request.json')) == frozen
            assert response['usage']['prompt_tokens'] == 55 and response['usage']['completion_tokens'] == 128
            assert response['timings']['cache_n'] == 0 and response['finish_reason'] == 'length'
        assert load(cell / 'diagnostic.request.json') == control['diagnostic_payload'](frozen)
        diag = load(cell / 'diagnostic.response.json')
        tokens = diag['__verbose']['tokens']
        assert len(tokens) == 128 and all(isinstance(t, int) for t in tokens)
        assert diag['usage']['prompt_tokens'] == 55 and diag['usage']['completion_tokens'] == 128
        assert diag['timings']['cache_n'] == 0 and diag['choices'][0]['finish_reason'] == 'length'
        diagnostic_text = diag['choices'][0]['message']['content']
        texts[label] = [r['text'] for r in responses]
        diagnostics[label] = diag
        item.update(ordinary_unique_outputs=len(set(texts[label])),
                    ordinary_same_as_warmup=[x == warmup['text'] for x in texts[label]],
                    ordinary_same_as_diagnostic=[x == diagnostic_text for x in texts[label]],
                    diagnostic_token_count=len(tokens),
                    generation_settings=diag['__verbose']['generation_settings'],
                    ordinary_vs_first=[difference(texts[label][0], x) for x in texts[label][1:]])
        since = datetime.datetime.fromisoformat(data['probe_window_since'])
        window = '\n'.join(line for line in log.splitlines()
                           if datetime.datetime.fromisoformat(line.split()[0].replace('Z', '+00:00')) >= since)
        (cell / 'probe-window.log').write_text(window + '\n')
        assert '[SYCL][OP]' not in window, 'Model probe unexpectedly enabled SYCL operator debug logging'
        pairs = [tuple(map(int, p)) for p in re.findall(r'accepted\s+(\d+)\s*/\s*(\d+) draft tokens', window)]
        counters = [r['timings'] for r in [warmup] + responses] + [diag['timings']]
        if label[0] == 'C':
            accepted, proposed = sum(a for a, b in pairs), sum(b for a, b in pairs)
            api_accepted = sum(t['draft_n_accepted'] for t in counters)
            api_proposed = sum(t['draft_n'] for t in counters)
            item['speculation'] = dict(api_accepted=api_accepted, api_proposed=api_proposed,
                window_accepted=accepted, window_proposed=proposed,
                window_totals_match_api=(accepted, proposed) == (api_accepted, api_proposed),
                window_events=len(pairs), window_rejection_events=sum(a < b for a, b in pairs),
                accepted_count_histogram=dict(collections.Counter(a for a, b in pairs)),
                proposed_depth_histogram=dict(collections.Counter(b for a, b in pairs)),
                per_request_counts=[dict(accepted=t['draft_n_accepted'], proposed=t['draft_n']) for t in counters])
        else:
            assert not pairs and all(t.get('draft_n', 0) == 0 for t in counters)
    groups = {}
    for label, values in texts.items():
        for i, text in enumerate(values):
            groups.setdefault(text, []).append(f'{label}/repeat-{i:02}')
    result['primary_output_groups'] = [dict(members=members, text=text) for text, members in groups.items()]
    for arm in 'ABC':
        labels = [x for x in texts if x.startswith(arm)]
        result.setdefault('within_arm', {})[arm] = dict(launches=labels,
            unique_outputs=len({t for label in labels for t in texts[label]}),
            matched_index_cross_launch=[difference(a, b) for a, b in zip(texts.get(arm + '0', []), texts.get(arm + '1', []))])
    for a, b in itertools.combinations('ABC', 2):
        left = [t for label in texts if label.startswith(a) for t in texts[label]]
        right = [t for label in texts if label.startswith(b) for t in texts[label]]
        comparisons = [difference(x, y) for x in left for y in right]
        result['primary_cross_arm'][a + b] = dict(pairs=len(comparisons),
            exact_matches=sum(x['equal'] for x in comparisons),
            differing_common_prefix_lengths=sorted({x['common_prefix_length'] for x in comparisons if not x['equal']}))
    for a, b in itertools.combinations(diagnostics, 2):
        x, y = diagnostics[a], diagnostics[b]
        tx, ty = x['__verbose']['tokens'], y['__verbose']['tokens']
        comparison = difference(tx, ty)
        index = comparison['first_difference_index']
        comparison.update(left=a, right=b, first_difference_token_ordinal=None if index is None else index + 1)
        for side, body, ids in [('left', x, tx), ('right', y, ty)]:
            rows = body['choices'][0].get('logprobs', {}).get('content', [])
            aligned = len(rows) == len(ids) and [r['id'] for r in rows] == ids
            comparison[side + '_first_piece'] = rows[index]['token'] if aligned and index is not None and index < len(rows) else None
        result['diagnostic_comparisons'].append(comparison)
    normal = {k: v for k, v in diagnostics.items() if k[0] in 'AB'}
    if len(normal) > 1:
        first = next(iter(normal.values()))['__verbose']['tokens']
        common = min(difference(first, b['__verbose']['tokens'])['common_prefix_length'] for b in normal.values())
        if common < len(first):
            snapshot = dict(shared_generated_prefix_tokens=common, token_ordinal=common + 1,
                            scope='Separate matched non-spec diagnostics only; not primary-stream replay or speculative confidence.', rows={})
            for label, body in normal.items():
                ids = body['__verbose']['tokens']
                rows = body['choices'][0]['logprobs']['content']
                assert ids[:common] == first[:common] and [r['id'] for r in rows] == ids
                row = rows[common]
                top = row['top_logprobs']
                snapshot['rows'][label] = dict(row=row, top_two_logprob_gap=top[0]['logprob'] - top[1]['logprob'])
            result['non_spec_probability_snapshot'] = snapshot
    if summary['status'] == 'completed':
        assert set(texts) == set(control['ORDER'])
    original = load(out / 'original-container.json')
    live = json.loads(command('docker', 'inspect', control['ORIGINAL']))[0]
    assert live['State']['Running'] and live['Id'] == original['Id'] and live['Config'] == original['Config']
    assert sorted(live['Mounts'], key=lambda m:m['Destination']) == sorted(original['Mounts'], key=lambda m:m['Destination'])
    assert set(command('docker', 'ps', '--format', '{{.Names}}').splitlines()) == {control['ORIGINAL']}
    for label in summary['phases']:
        assert command('docker', 'inspect', '-f', '{{.State.Running}}', 'bolt-fidelity-r1-' + label.lower()).strip() == 'false'
    assert binaries(control['ACCEPTED']) == load(out / 'accepted-binaries.json')
    assert binaries(control['CANDIDATE']) == load(out / 'candidate-binaries.json')
    hardware = runner['hardware']()
    assert hardware == load(out / 'hardware-before.json') == load(out / 'hardware-after.json')
    health = api['request'](control['URL'], '/health')
    assert health['status'] == 'ok'
    smoke = load(out / 'restored-smoke.result.json')
    assert smoke['text'].strip() == 'READY' and smoke['finish_reason'] == 'stop'
    result['restoration'] = dict(verified=True, original_id=original['Id'], live_id=live['Id'],
        config_identical=True, all_mount_fields_identical=True, binaries_identical=True,
        hardware=hardware, health=health, ready_output=smoke['text'], all_experiments_stopped=True)
    (ROOT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cells', 'diagnostic_comparisons', 'primary_output_groups')}, indent=2))
    print('BOLT_FIDELITY_CONTROLS_ANALYZED_AND_RESTORATION_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
