#!/usr/bin/env python3
"""Offline, stdlib-only analysis of all independent ABBA measurements."""
import json
import random
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CELLS = ('a1', 'b1', 'b2', 'a2')
LIB = 'e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec'


def load(path):
    return json.loads(path.read_text())


def stats(values):
    q = st.quantiles(values, n=4, method='inclusive')
    return {'median': st.median(values), 'iqr': q[2] - q[0], 'samples': values}


def counter(path, name):
    matches = re.findall(r'^vllm:spec_decode_' + name + r'_total\{[^\n]*?\}\s+([\d.eE+-]+)$', path.read_text(), re.M)
    assert len(matches) == 1, (path, name, matches)
    return float(matches[0])


result = {'tier': 'development; quality-sensitive, not standard-publishable',
          'order': list(CELLS), 'excluded_samples': 0,
          'discovery_excluded': True, 'points': {}, 'cells': {},
          'confidence_method': '20000 paired prompt-cluster bootstrap resamples, seed42; each of six prompt indices carries both replicate cells. Conditional on one ABBA sequence (two cells per arm), not a cross-day confidence claim.'
}
metadata = {}
for cell in CELLS:
    p = ROOT / f'confirm-{cell}'
    s = load(p / 'summary.json')
    gates = load(p / 'api-checks-summary.json')
    assert s['status'] == 'passed' and s['host_unchanged'] and gates['status'] == 'passed'
    metadata[cell] = load(p / 'launch-metadata.json')
    assert metadata[cell]['confirmation_prompt_lengths'] == [512, 8192, 32768, 65536]
    result['cells'][cell] = {'status': s['status'], 'host_unchanged': s['host_unchanged'], 'api_gates': gates}
    if cell.startswith('b'):
        e = load(p / 'candidate-execution-evidence.json')
        assert e['full_graph_capture_seen'] and e['full_graph_run_seen']
        assert e['eligible_dispatch_log_count'] > 0 and e['unsupported_q5_log_count'] == 0
        assert LIB in e['eligible_dispatch_log'][0]
        result['cells'][cell]['candidate_execution'] = e
for key in ('common_contract', 'serve', 'speculation', 'target', 'image', 'stack_revision'):
    assert all(metadata[c][key] == metadata['a1'][key] for c in CELLS), key

for length in (512, 8192, 32768, 65536):
    rows, values, counts = {}, {}, {}
    for cell in CELLS:
        p = ROOT / f'confirm-{cell}' / f'length-{length}' / 'long-context'
        rows[cell] = load(p / 'summary.json')['points'][0]['measurements']
        assert len(rows[cell]) == 6
        assert all(r['valid'] and r['validation']['completion_tokens'] == 128 and r['validation']['prompt_tokens'] == length for r in rows[cell])
        values[cell] = [r['decode_tps_post_first'] for r in rows[cell]]
        counts[cell] = {}
        for name in ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens'):
            counts[cell][name] = sum(counter(p / 'points' / f'length-{length}' / f'measured-{i:02}' / 'metrics-after.raw', name) - counter(p / 'points' / f'length-{length}' / f'measured-{i:02}' / 'metrics-before.raw', name) for i in range(1, 7))
        counts[cell]['acceptance_rate'] = counts[cell]['num_accepted_tokens'] / counts[cell]['num_draft_tokens']
        counts[cell]['actual_emitted_per_step'] = 768 / counts[cell]['num_drafts']
    a, b = values['a1'] + values['a2'], values['b1'] + values['b2']
    gain = 100 * (st.median(b) / st.median(a) - 1)
    rng = random.Random(42)
    boot = []
    for _ in range(20000):
        indices = rng.choices(range(6), k=6)
        aa = [values[c][i] for c in ('a1', 'a2') for i in indices]
        bb = [values[c][i] for c in ('b1', 'b2') for i in indices]
        boot.append(100 * (st.median(bb) / st.median(aa) - 1))
    boot.sort()
    comparisons = {}
    for ac, bc in (('a1', 'b1'), ('a2', 'b2'), ('a1', 'a2'), ('b1', 'b2')):
        for i in range(1, 7):
            suffix = Path(f'length-{length}/long-context/points/length-{length}/measured-{i:02}/request.json')
            assert (ROOT / f'confirm-{ac}' / suffix).read_bytes() == (ROOT / f'confirm-{bc}' / suffix).read_bytes()
        comparisons[f'{ac}_vs_{bc}_text_matches'] = sum(x['stream']['text'] == y['stream']['text'] for x, y in zip(rows[ac], rows[bc]))
    result['points'][str(length)] = {'baseline_tps': stats(a), 'candidate_tps': stats(b), 'gain_percent': gain,
        'paired_prompt_cluster_bootstrap_95_percent': [boot[499], boot[19499]],
        'cell_tps': {c: stats(values[c]) for c in CELLS},
        'cell_median_ttft_s': {c: st.median(r['ttft_s'] for r in rows[c]) for c in CELLS},
        'speculation': counts, 'requests_identical': True, 'long_text_identity_out_of_6': comparisons,
        'long_token_identity': 'not collected; text identity is not token identity'}

result['short_output_id_matches'] = {}
for ac, bc in (('a1', 'b1'), ('a2', 'b2'), ('a1', 'a2'), ('b1', 'b2')):
    files = sorted((ROOT / f'confirm-{ac}').glob('*output-ids.json'))
    result['short_output_id_matches'][f'{ac}_vs_{bc}'] = {'matched': sum(load(p) == load(ROOT / f'confirm-{bc}' / p.name) for p in files), 'total': len(files)}
operator = load(ROOT / 'operator-03/result.json')
result['operator_cases_passed'] = sum(c['status'] == 'passed' for c in operator['cases'])
result['strict_fp32_diagnostic_failures'] = {arm: [c['name'] for c in operator['cases'] if not c.get('comparisons', {}).get(arm + '_fp32_supplemental', {}).get('strict_fp32_diagnostic', {}).get('passed', True)] for arm in ('native', 'candidate')}
result['predeclared_development_gate_passed'] = (result['points']['65536']['gain_percent'] >= 5 and all(result['points'][str(n)]['gain_percent'] >= -5 for n in (512, 8192, 32768)))
assert result['predeclared_development_gate_passed']
print(json.dumps(result, indent=2))
