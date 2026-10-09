#!/usr/bin/env python3
"""Compare three completed native serving archives; never infer kernel gains."""
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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counter(lines, name):
    prefix = 'vllm:spec_decode_' + name + '_total{'
    values = [float(line.rsplit(' ', 1)[1]) for line in lines if line.startswith(prefix)]
    require(len(values) == 1 and math.isfinite(values[0]), 'missing/ambiguous counter ' + name)
    return values[0]


def load(path):
    root = path.name.removesuffix('.tar.gz')
    with tarfile.open(path) as archive:
        def read(name):
            return json.load(archive.extractfile(root + '/' + name))
        result = read('cell-result.json')
        native = read('native/summary.json')
        require(result['driver_exit'] == 0 and not result['new_guarded_error_lines'], 'failed cell')
        require(native['status'] == 'passed' and native['host_unchanged'], 'failed native/host guard')
        point = read('native/long-context/summary.json')['points'][0]
        require(point['requested_length'] == 65536 and point['summary']['valid_count'] == 6, 'wrong cell')
        measurements = point['measurements']
        require([r['trial'] for r in measurements] == list(range(1, 7)), 'wrong trial set')
        rows = []
        for row in [point['warmup'], *measurements]:
            require(row['valid'] and row['status'] == 'ok', 'invalid request')
            require(row['validation']['prompt_tokens'] == 65536 and row['validation']['completion_tokens'] == 128, 'wrong tokens')
            stream = row['stream']
            require(stream['finish_reason'] == 'length' and not stream['parse_errors'], 'invalid stream')
            request = read('native/long-context/' + row['request_path'])
            require(request['seed'] == 42 and request['temperature'] == 0 and request['ignore_eos'], 'sampling changed')
            deltas = {name: counter(row['metrics_after']['speculative_position_counter_lines'], name)
                      - counter(row['metrics_before']['speculative_position_counter_lines'], name)
                      for name in COUNTERS}
            require(0 < deltas['num_drafts'] and deltas['num_draft_tokens'] == 4 * deltas['num_drafts'], 'invalid MTP counters')
            require(0 <= deltas['num_accepted_tokens'] <= deltas['num_draft_tokens'], 'invalid accepted count')
            for metric in METRICS:
                require(math.isfinite(row[metric]) and row[metric] > 0, 'invalid timing')
            rows.append({
                'trial': row['trial'], 'kind': row['kind'],
                'request_sha256': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                'output_text_sha256': hashlib.sha256(stream['text'].encode()).hexdigest(),
                'speculative_counter_deltas': deltas,
                **{key: row[key] for key in METRICS},
            })
        cache_line = next(x for x in measurements[0]['metrics_before']['speculative_position_counter_lines']
                          if x.startswith('vllm:cache_config_info{'))
        capacity = int(re.search(r'kv_cache_size_tokens="(\d+)"', cache_line).group(1))
        launch = read('native/launch-metadata.json')
        host = read('preconditions.json')
        return {
            'archive': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'split_patch': read('native/split-patch.json'), 'rows': rows,
            'summary': point['summary'], 'kv_cache_size_tokens': capacity,
            'contract': {k: launch[k] for k in ('serve', 'environment', 'image', 'common_contract')},
            'host': {k: host[k] for k in ('boot_id', 'boost', 'cpu_max_khz', 'launcher_sha256', 'power_cap_microwatts')},
        }


def compare(a, b):
    require(a['contract'] == b['contract'] and a['host'] == b['host'], 'confounded serving/host configuration')
    require(b['kv_cache_size_tokens'] >= a['kv_cache_size_tokens'], 'candidate capacity reduction')
    pairs = list(zip(a['rows'], b['rows'], strict=True))
    require(all(x['request_sha256'] == y['request_sha256'] for x, y in pairs), 'unpaired prompts')
    measured = [(x, y) for x, y in pairs if x['kind'] == 'measured']
    return {
        'same_requests_including_warmup': True,
        'output_text_matches_including_warmup': sum(x['output_text_sha256'] == y['output_text_sha256'] for x, y in pairs),
        'output_text_total_including_warmup': len(pairs),
        'identical_acceptance_counter_requests': sum(x['speculative_counter_deltas'] == y['speculative_counter_deltas'] for x, y in pairs),
        'median_paired_decode_rate_delta_pct': statistics.median((y['decode_tps_post_first'] / x['decode_tps_post_first'] - 1) * 100 for x, y in measured),
        'paired_decode_rate_delta_pct': [(y['decode_tps_post_first'] / x['decode_tps_post_first'] - 1) * 100 for x, y in measured],
        'median_paired_total_latency_delta_pct': statistics.median((y['total_time_s'] / x['total_time_s'] - 1) * 100 for x, y in measured),
        'ratio_of_decode_medians_delta_pct': (b['summary']['median']['decode_tps_post_first'] / a['summary']['median']['decode_tps_post_first'] - 1) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    before, candidate, after = [load(p) for p in (args.before, args.candidate, args.after)]
    require(before['split_patch']['split_choice'] == after['split_patch']['split_choice'] == 'auto', 'controls must be auto')
    require(candidate['split_patch']['split_choice'] in ('8', '16', '32'), 'explicit candidate required')
    print(json.dumps({
        'tier': 'development', 'cells': [before, candidate, after],
        'candidate_vs_before': compare(before, candidate),
        'candidate_vs_after': compare(after, candidate),
        'control_drift_after_vs_before': compare(before, after),
        'limitations': ['One A/B/A block, six measured prompts per cell; not a publishable performance result.',
                        'Output text equality is checked, not token-ID equality (API did not return token IDs).',
                        'Same FP16/FP8 types do not guarantee bitwise equality after changing reduction order.',
                        'Uninstrumented API timings do not prove native split-count dispatch.'],
    }, indent=2))


if __name__ == '__main__':
    main()
