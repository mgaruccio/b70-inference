#!/usr/bin/env python3
"""Validate retained real requests and summarize the completed three-bundle run."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
CELLS = ('mtp4', 'dspark', 'dflash')
LENGTHS = (512, 8192, 32768, 65536)
METRICS = ('decode_tps_post_first', 'ttft_s', 'total_time_s', 'decode_elapsed_s', 'e2e_tps')
COUNTERS = ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')


def load(path):
    return json.loads(path.read_text())


def counter(row, phase, name):
    prefix = f'vllm:spec_decode_{name}_total{{'
    values = [float(line.rsplit(' ', 1)[1]) for line in
              row[f'metrics_{phase}']['speculative_position_counter_lines'] if line.startswith(prefix)]
    assert values and all(math.isfinite(v) and v >= 0 for v in values), (phase, name)
    return sum(values)


def main():
    result = {'tier': 'development', 'baseline': 'mtp4', 'order': list(CELLS),
              'comparison': 'best configuration bundles; context/runtime/runner/batch differ',
              'decode_formula': '(completion_tokens - 1) / (stream_end - first_nonempty)',
              'warmups_excluded': True, 'sample_exclusions': [], 'cells': {}, 'identity_vs_mtp4': {}}
    points = {}
    for cell in CELLS:
        root = ROOT / f'{cell}-01'
        top = load(root / 'summary.json')
        assert top['status'] == 'passed' and top['host_unchanged'] is True
        assert top['benchmark']['gates']['status'] == 'passed'
        assert load(root / 'host-before.json') == load(root / 'host-after.json')
        config = load(root / 'effective-config.json')
        record = {'effective_config': config, 'lengths': {}}
        result['cells'][cell] = record
        for length in LENGTHS:
            base = root / f'length-{length}' / 'long-context'
            summary = load(base / 'summary.json')
            assert summary['status'] == 'completed' and not summary['errors']
            assert len(summary['points']) == 1
            point = summary['points'][0]
            assert point['requested_length'] == length and point['status'] == 'complete'
            rows = point['measurements']
            assert len(rows) == 6 and point['warmup']['valid']
            for row in [point['warmup'], *rows]:
                assert row['valid'] and not row['stream']['parse_errors']
                assert row['stream']['finish_reason'] == 'length'
                assert row['validation']['prompt_tokens'] == length
                assert row['validation']['completion_tokens'] == 128
            points[cell, length] = (base, point)
            metrics = {}
            for metric in METRICS:
                values = [row[metric] for row in rows]
                assert all(math.isfinite(v) and v > 0 for v in values)
                q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
                metrics[metric] = {'samples': values, 'median': statistics.median(values), 'iqr': q3 - q1}
                assert metrics[metric]['median'] == point['summary']['median'][metric]
            totals = {key: sum(counter(row, 'after', key) - counter(row, 'before', key) for row in rows)
                      for key in COUNTERS}
            assert 0 <= totals['num_accepted_tokens'] <= totals['num_draft_tokens']
            assert totals['num_drafts'] > 0 and totals['num_draft_tokens'] > 0
            totals['draft_acceptance_rate'] = totals['num_accepted_tokens'] / totals['num_draft_tokens']
            totals['accepted_per_round'] = totals['num_accepted_tokens'] / totals['num_drafts']
            totals['observed_proposed_per_round'] = totals['num_draft_tokens'] / totals['num_drafts']
            record['lengths'][str(length)] = {'metrics': metrics, 'measured_speculation': totals}
    for cell in CELLS:
        identity = {}
        for length in LENGTHS:
            aroot, a = points['mtp4', length]
            broot, b = points[cell, length]
            matches = []
            for ar, br in zip([a['warmup'], *a['measurements']], [b['warmup'], *b['measurements']], strict=True):
                assert load(aroot / ar['request_path']) == load(broot / br['request_path']), (cell, length, ar['trial'])
                matches.append(ar['stream']['text'] == br['stream']['text'])
            identity[str(length)] = {'payload_matches_including_warmup': 7,
                                     'warmup_text_matches': matches[0], 'measured_text_matches': matches[1:],
                                     'measured_text_match_count': sum(matches[1:])}
            value = result['cells'][cell]['lengths'][str(length)]
            baseline = result['cells']['mtp4']['lengths'][str(length)]['metrics']['decode_tps_post_first']['median']
            value['decode_delta_percent_vs_mtp4'] = 100 * (value['metrics']['decode_tps_post_first']['median'] / baseline - 1)
        short = sorted((ROOT / 'mtp4-01').glob('*-output-ids.json'))
        assert len(short) == 19
        identity['short_output_ids'] = {p.name: load(p) == load(ROOT / f'{cell}-01' / p.name) for p in short}
        result['identity_vs_mtp4'][cell] = identity
    (ROOT / 'comparison.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    for length in LENGTHS:
        print(length, {cell: round(result['cells'][cell]['lengths'][str(length)]['metrics']['decode_tps_post_first']['median'], 3)
                       for cell in CELLS})
    print('Validated all three cells, 84 cold streams, exact payload identity, metrics and cleanup')


if __name__ == '__main__':
    main()
