"""Summarize the fixed cold sweep; retain invalid counters rather than imputing them."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
lengths = [512, 8192, 16384, 32768, 65536, 120000, 160000, 212000]
keys = ['spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
        'spec_decode_num_accepted_tokens_total', 'request_success_total', 'generation_tokens_total']

def counters(path):
    values = {key: 0.0 for key in keys}
    positions = {}
    found = set()
    cache_disabled = False
    for line in path.read_text().splitlines():
        if not line.startswith('vllm:'):
            continue
        name = line.split('{', 1)[0][5:]
        if name in values:
            values[name] += float(line.rsplit(' ', 1)[1])
            found.add(name)
        if name == 'spec_decode_num_accepted_tokens_per_pos_total':
            pos = int(line.split('position="')[1].split('"')[0])
            assert pos not in positions
            positions[pos] = float(line.rsplit(' ', 1)[1])
        if name == 'cache_config_info':
            assert 'enable_prefix_caching="False"' in line
            cache_disabled = True
    assert cache_disabled
    return values, positions, found

def dispersion(values):
    q = statistics.quantiles(values, n=4, method='inclusive')
    return {'median': statistics.median(values), 'iqr_inclusive': q[2] - q[0]}

result = {'tier': 'development', 'heads': {}, 'identical_requests_across_heads': True,
          'comparison_order': 'stock then candidate; interrupted by candidate staging repair',
          'limitations': ['Sequential two-server comparison, not interleaved ABBA.',
                         'TTFT-derived prefill proxy, not direct engine prefill measurement.',
                         'Decode rate is (128-1)/(stream end-first nonempty chunk); MTP bursts are not per-token ITL.',
                         'Synthetic performance inputs do not test long-context answer correctness.',
                         'Maximum tested length is not a hard capacity ceiling; no transient GPU peak measurement.']}
requests = {}
for label, head in [('A', 'stock'), ('B', 'candidate')]:
    directory = root / f'long-verified-{label}-output'
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['errors'] == []
    assert [p['requested_length'] for p in summary['points']] == lengths
    run = json.loads((directory / 'run.json').read_text())
    cells = []
    requests[label] = {}
    for point in summary['points']:
        assert point['status'] == 'complete' and point['summary']['valid_count'] == 6
        n = point['requested_length']
        rows = []
        for row in [point['warmup']] + point['measurements']:
            assert row['valid'] and row['validation']['prompt_tokens'] == n
            assert row['validation']['completion_tokens'] == 128
            request = json.loads((directory / row['request_path']).read_text())
            assert len(request['prompt']) == n and request['ignore_eos'] and request['max_tokens'] == 128
            key = (n, row['kind'], row['trial'])
            digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            requests[label][key] = digest
            before, bp, bf = counters(directory / row['metrics_before']['raw_path'])
            after, ap, af = counters(directory / row['metrics_after']['raw_path'])
            delta = {k: after[k] - before[k] for k in keys}
            pos = [ap.get(i, 0) - bp.get(i, 0) for i in range(4)]
            drafts = delta['spec_decode_num_drafts_total']
            accepted = delta['spec_decode_num_accepted_tokens_total']
            valid = (bf == af == set(keys) and set(bp) == set(ap) == set(range(4))
                     and delta['request_success_total'] == 1 and delta['generation_tokens_total'] == 128
                     and drafts > 0 and delta['spec_decode_num_draft_tokens_total'] == 4 * drafts
                     and drafts >= pos[0] >= pos[1] >= pos[2] >= pos[3] >= 0
                     and accepted == sum(pos) and all(v >= 0 for v in delta.values()))
            if row['kind'] == 'measured':
                rows.append({'trial': row['trial'], 'request_sha256': digest,
                             'request_path': row['request_path'], 'actual_prompt_tokens': n, 'actual_output_tokens': 128,
                             'timing': {k: row[k] for k in point['summary']['median']},
                             'speculative_counter_valid': valid, 'counter_delta': delta, 'accepted_per_position': pos,
                             'accepted_tokens_per_draft': accepted / drafts if valid else None})
        assert len(rows) == 6
        rates = [r['accepted_tokens_per_draft'] for r in rows if r['speculative_counter_valid']]
        cells.append({'prompt_tokens': n, 'summary': point['summary'], 'measurements': rows,
                      'valid_counter_measurements': len(rates),
                      'accepted_tokens_per_draft': dispersion(rates) if len(rates) == 6 else None})
    result['heads'][head] = {'started_utc': run['started_utc'], 'finished_utc': run['finished_utc'],
                             'prefix_cache_harness_record': summary['prefix_cache'],
                             'cache_disabled_in_all_metric_snapshots': True, 'points': cells}
assert requests['A'] == requests['B'] and len(requests['A']) == 56
result['matched_requests_including_warmups'] = 56
result['measured_requests_per_head'] = 48
with (root / 'long-summary.json').open('x') as f:
    json.dump(result, f, indent=2)
print('Verified 56 matched tokenized requests/head, 48 measured/head; completed all eight lengths.')
for a, b in zip(result['heads']['stock']['points'], result['heads']['candidate']['points']):
    print(a['prompt_tokens'], 'stock', a['summary']['median'], 'candidate', b['summary']['median'],
          'counter_valid', a['valid_counter_measurements'], b['valid_counter_measurements'])
