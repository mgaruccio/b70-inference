#!/usr/bin/env python3
"""Offline checks and descriptive A/B/B/A comparison of this campaign's archives."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile

COUNTERS = ('num_drafts', 'num_draft_tokens', 'num_accepted_tokens')
METRICS = ('decode_tps_post_first', 'ttft_s', 'total_time_s', 'e2e_tps')
LIBRARY_SHA = 'e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec'
EXTRA_ENV = ['PYTHONPATH=/experiment', 'B70_STEP_TIMING=1', 'B70_GROUPED_SERVING=1',
             'B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counter(lines, name):
    prefix = 'vllm:spec_decode_' + name + '_total{'
    values = [float(line.rsplit(' ', 1)[1]) for line in lines if line.startswith(prefix)]
    require(len(values) == 1 and math.isfinite(values[0]), 'missing/ambiguous counter ' + name)
    return values[0]


def stats(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return {'median': statistics.median(values), 'iqr': q3 - q1,
            'min': min(values), 'max': max(values), 'n': len(values)}


def load(path, arm):
    root = path.name.removesuffix('.tar.gz')
    with tarfile.open(path) as archive:
        def read(name):
            return json.load(archive.extractfile(root + '/' + name))
        result = read('cell-result.json')
        native = read('native/summary.json')
        require(result['driver_exit'] == 0 and not result['new_guarded_error_lines'], 'failed cell')
        require(result['arm'] == arm, 'wrong arm')
        require(native['status'] == 'passed' and native['host_unchanged'], 'failed native/host guard')
        gates = read('native/api-checks-summary.json')
        require(gates['status'] == 'passed' and gates['canaries'] == [True] * 3
                and gates['finite_boundaries'] == 131 and len(gates['functional']) == 8
                and all(x['pass'] for x in gates['functional']), 'failed public gates')
        point = read('native/long-context/summary.json')['points'][0]
        require(point['requested_length'] == 65536 and point['summary']['valid_count'] == 6, 'wrong cell')
        measurements = point['measurements']
        require([r['trial'] for r in measurements] == list(range(1, 7)), 'wrong trial set')
        require(point['warmup']['kind'] == 'warmup'
                and all(r['kind'] == 'measured' for r in measurements), 'wrong request kinds')
        rows = []
        for row in [point['warmup'], *measurements]:
            require(row['valid'] and row['status'] == 'ok', 'invalid request')
            require(row['validation']['prompt_tokens'] == 65536
                    and row['validation']['completion_tokens'] == 128, 'wrong tokens')
            stream = row['stream']
            require(stream['finish_reason'] == 'length' and not stream['parse_errors'], 'invalid stream')
            request = read('native/long-context/' + row['request_path'])
            require(request['seed'] == 42 and request['temperature'] == 0 and request['ignore_eos'], 'sampling changed')
            deltas = {name: counter(row['metrics_after']['speculative_position_counter_lines'], name)
                      - counter(row['metrics_before']['speculative_position_counter_lines'], name)
                      for name in COUNTERS}
            require(deltas['num_drafts'] > 0 and deltas['num_draft_tokens'] == 4 * deltas['num_drafts'], 'invalid MTP counters')
            require(0 <= deltas['num_accepted_tokens'] <= deltas['num_draft_tokens'], 'invalid accepted count')
            require(all(math.isfinite(row[m]) and row[m] > 0 for m in METRICS), 'invalid timing')
            rows.append({'trial': row['trial'], 'kind': row['kind'],
                         'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                         'output_text_sha256': hashlib.sha256(stream['text'].encode()).hexdigest(),
                         'speculative_counter_deltas': deltas, **{m: row[m] for m in METRICS}})
        launch = read('native/launch-metadata.json')
        require(launch['comparison_arm'] == arm and launch['confirmation_prompt_lengths'] == [65536], 'wrong launch arm/lengths')
        environment = launch['environment'].copy()
        execution = None
        if arm == 'candidate':
            require(environment[-4:] == EXTRA_ENV, 'candidate environment changed')
            environment = environment[:-4]
            require(launch['candidate_library']['sha256'] == LIBRARY_SHA
                    and launch['candidate_library']['read_only_mount'], 'wrong library')
            execution = read('native/candidate-execution-evidence.json')
            require(execution['eligible_dispatch_log_count'] > 0 and execution['full_graph_capture_seen']
                    and execution['full_graph_run_seen'] and execution['unsupported_q5_log_count'] == 0,
                    'candidate dispatch/FULL graph evidence missing or unsupported Q5')
        require(not any(x.startswith(('B70_GROUPED_', 'B70_STEP_TIMING=')) for x in environment), 'unexpected baseline hook')
        cache_line = next(x for x in measurements[0]['metrics_before']['speculative_position_counter_lines']
                          if x.startswith('vllm:cache_config_info{'))
        capacity = int(re.search(r'kv_cache_size_tokens="(\d+)"', cache_line).group(1))
        prefix = root + '/native/'
        short = {n[len(prefix):]: hashlib.sha256(json.dumps(json.load(archive.extractfile(n)), sort_keys=True).encode()).hexdigest()
                 for n in archive.getnames() if n.startswith(prefix) and n.endswith('-output-ids.json')
                 and '/' not in n[len(prefix):]}
        require(len(short) == 19, 'unexpected short output-ID set')
        host = read('preconditions.json')
        return {'archive': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'arm': arm, 'rows': rows, 'metrics': {m: stats([r[m] for r in rows if r['kind'] == 'measured']) for m in METRICS},
                'kv_cache_size_tokens': capacity, 'short_output_id_sha256': short,
                'candidate_execution': execution, 'public_gates': gates,
                'contract': {**{k: launch[k] for k in ('serve', 'image', 'common_contract')}, 'environment': environment},
                'host': {k: host[k] for k in ('boot_id', 'boost', 'cpu_max_khz', 'launcher_sha256', 'power_cap_microwatts')}}


def compare(a, b):
    require(a['contract'] == b['contract'] and a['host'] == b['host'], 'confounded serving/host configuration')
    require(a['kv_cache_size_tokens'] == b['kv_cache_size_tokens'], 'capacity changed')
    pairs = list(zip(a['rows'], b['rows'], strict=True))
    require(all(x['request_sha256'] == y['request_sha256'] for x, y in pairs), 'unpaired prompts')
    measured = [(x, y) for x, y in pairs if x['kind'] == 'measured']
    require(a['short_output_id_sha256'].keys() == b['short_output_id_sha256'].keys(), 'short output set changed')
    return {'same_requests_including_warmup': True,
            'measured_output_text_matches': sum(x['output_text_sha256'] == y['output_text_sha256'] for x, y in measured),
            'measured_output_text_total': len(measured),
            'short_output_id_matches': sum(v == b['short_output_id_sha256'][k] for k, v in a['short_output_id_sha256'].items()),
            'short_output_id_total': len(a['short_output_id_sha256']),
            'identical_measured_acceptance_counters': sum(x['speculative_counter_deltas'] == y['speculative_counter_deltas'] for x, y in measured),
            'median_paired_decode_delta_pct': statistics.median((y['decode_tps_post_first'] / x['decode_tps_post_first'] - 1) * 100 for x, y in measured),
            'paired_decode_delta_pct': [(y['decode_tps_post_first'] / x['decode_tps_post_first'] - 1) * 100 for x, y in measured],
            'ratio_of_medians_delta_pct': {m: (b['metrics'][m]['median'] / a['metrics'][m]['median'] - 1) * 100 for m in METRICS}}


def pooled(cells):
    rows = [r for c in cells for r in c['rows'] if r['kind'] == 'measured']
    counts = {k: sum(r['speculative_counter_deltas'][k] for r in rows) for k in COUNTERS}
    return {'metrics': {m: stats([r[m] for r in rows]) for m in METRICS},
            'counters': counts, 'draft_acceptance_fraction': counts['num_accepted_tokens'] / counts['num_draft_tokens'],
            'actual_emitted_tokens_per_draft_step': len(rows) * 128 / counts['num_drafts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('a1', 'b1', 'b2', 'a2'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    cells = {name: load(getattr(args, name), 'baseline' if name.startswith('a') else 'candidate')
             for name in ('a1', 'b1', 'b2', 'a2')}
    comparisons = {f'{b}_vs_{a}': compare(cells[a], cells[b])
                   for a, b in [('a1', 'b1'), ('a1', 'b2'), ('a2', 'b1'), ('a2', 'b2'), ('a1', 'a2'), ('b1', 'b2')]}
    pools = {arm: pooled([cells[n] for n in names])
             for arm, names in [('baseline', ('a1', 'a2')), ('candidate', ('b1', 'b2'))]}
    print(json.dumps({'tier': 'development', 'quality_sensitive': True, 'cells': cells,
                     'comparisons': comparisons, 'pooled': pools,
                     'pooled_ratio_of_medians_delta_pct': {m: (pools['candidate']['metrics'][m]['median'] /
                         pools['baseline']['metrics'][m]['median'] - 1) * 100 for m in METRICS},
                     'limitations': ['One A/B/B/A block, six paired prompt clusters; descriptive, not cross-day robustness.',
                                     'Long output comparison uses text hashes, not token IDs.',
                                     'Native nondeterminism does not establish candidate quality equivalence.',
                                     'Existing same-build HumanEval+ observation: native139/164 vs candidate136/164; no promotion.',
                                     'Maximum current serving prompt65536, not the configured212992 limit.']}, indent=2))


if __name__ == '__main__':
    main()
