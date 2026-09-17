import json, math, pathlib, random, statistics, sys
root = pathlib.Path(sys.argv[1])
labels = ('A1', 'B1', 'B2', 'A2')
source = {r['id']: r['source_group'] for r in map(json.loads, (root/'private-prompts/test.jsonl').read_text().splitlines())}

def counter(row, name):
    return sum(v for k, v in row['delta'].items() if k.split('{', 1)[0] == 'vllm:' + name)

def vector(row):
    return [counter(row, 'spec_decode_num_accepted_tokens_total'),
            counter(row, 'spec_decode_num_drafts_total'),
            counter(row, 'spec_decode_num_draft_tokens_total'),
            row['generated_tokens'] - 1, counter(row, 'request_decode_time_seconds_sum'),
            row['generated_tokens'], row['wall_seconds']]

def total(vectors):
    return [sum(x) for x in zip(*vectors)]

def rates(v):
    return [v[0]/v[1], v[3]/v[4], v[5]/v[6]]

runs, outputs = {}, {}
for label in labels:
    rows, tokens = {}, {}
    for path in sorted((root/f'heldout-{label}-output').glob('request-*/measurement.json')):
        row = json.loads(path.read_text())
        key = row['prompt_id']
        assert key in source and key not in rows
        response = json.loads((path.parent/'response.json').read_text())
        ids = response['choices'][0]['token_ids']
        assert len(ids) == row['generated_tokens'] == counter(row, 'request_generation_tokens_sum')
        assert all(math.isfinite(v) and v >= -1e-9 for v in row['delta'].values())
        assert counter(row, 'prefix_cache_hits_total') == 0
        rows[key], tokens[key] = row, ids
    assert set(rows) == set(source)
    runs[label], outputs[label] = rows, tokens


def summarize(rows):
    rows = list(rows)
    v = total(vector(r) for r in rows)
    speeds = [(r['generated_tokens']-1)/counter(r, 'request_decode_time_seconds_sum')
              for r in rows if r['generated_tokens']>1 and counter(r, 'request_decode_time_seconds_sum')>0]
    per_pos = [sum(sum(value for key, value in r['delta'].items()
                      if key.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{')
                      and f'position="{p}"' in key) for r in rows)/v[1] for p in range(4)]
    return dict(requests=len(rows), generated_tokens=int(v[5]), accepted_draft_tokens=int(v[0]),
                speculative_passes=int(v[1]), proposed_tokens=int(v[2]),
                accepted_per_pass=v[0]/v[1], acceptance_rate=v[0]/v[2],
                acceptance_by_position=per_pos, decode_tok_s=v[3]/v[4],
                median_request_decode_tok_s=statistics.median(speeds),
                end_to_end_tok_s=v[5]/v[6], median_wall_s=statistics.median(r['wall_seconds'] for r in rows),
                median_ttft_s=statistics.median(counter(r, 'time_to_first_token_seconds_sum') for r in rows))

stable = {key for key in source if all(outputs[label][key] == outputs['A1'][key] for label in labels)}

def cohort(keys):
    keys = sorted(keys)
    a = summarize(runs[label][key] for label in ('A1','A2') for key in keys)
    b = summarize(runs[label][key] for label in ('B1','B2') for key in keys)
    metrics = ('accepted_per_pass','decode_tok_s','end_to_end_tok_s')
    groups = {}
    for key in keys:
        pair = [total(vector(runs[label][key]) for label in side) for side in (('A1','A2'),('B1','B2'))]
        groups.setdefault(source[key], []).append(pair)
    clustered = [[total(p[side] for p in pairs) for side in (0,1)] for _, pairs in sorted(groups.items())]
    rng = random.Random(42)
    draws = [[] for _ in metrics]
    for _ in range(10000):
        sample = [clustered[rng.randrange(len(clustered))] for _ in clustered]
        ra, rb = [rates(total(p[side] for p in sample)) for side in (0,1)]
        for values, av, bv in zip(draws, ra, rb): values.append(100*(bv/av-1))
    ci = {}
    for metric, values in zip(metrics, draws):
        values.sort()
        ci[metric] = dict(delta_percent=100*(b[metric]/a[metric]-1),
                          family_bootstrap_95_percent=[values[249], values[9749]])
    return dict(prompts=len(keys), source_families=len(groups), stock=a, tuned=b, effects=ci)

result = dict(tier='development', checkpoint_step=400,
              runs={label:summarize(runs[label].values()) for label in labels},
              primary=cohort(source), identical_all_four_diagnostic=cohort(stable),
              exact_output_matches={f'{a}_vs_{b}':sum(outputs[a][key]==outputs[b][key] for key in source)
                                    for a,b in [('A1','A2'),('B1','B2'),('A1','B1'),('A2','B2')]},
              bootstrap=dict(unit='source-session family, both runs retained together',resamples=10000,seed=42),
              limitations=['Natural output lengths differ; stock also varies between repeats.',
                           'Identical-output subset is post-hoc, not the primary estimate.',
                           'Native speculative counters can include terminal candidates discarded by the API.',
                           'No functional quality score or production promotion.'])
output = root/'heldout-summary.json'
with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result,indent=2,allow_nan=False))
