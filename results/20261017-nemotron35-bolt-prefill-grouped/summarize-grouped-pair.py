#!/usr/bin/env python3
"""Validate and summarize the fixed grouped-tail development pair; no samples excluded."""
import json
from pathlib import Path
import statistics

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-prefill-grouped')


def load(path):
    return json.loads(path.read_text())


def normalize_create(argv):
    argv = list(argv)
    argv[argv.index('--name') + 1] = '<cell>'
    for i, value in enumerate(argv):
        if value.endswith(':/src:ro'):
            argv[i] = '<source>:/src:ro'
    # Native gates differ deliberately; the actual server launch must be identical.
    argv[-1] = argv[-1].rsplit('exec ', 1)[1]
    return argv


def summarize(root):
    summary = load(root / 'summary.json')
    assert summary['status'] == 'passed' and summary['restored'] and not summary['debug']
    quality = load(root / 'quality/summary.json')
    assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
    assert load(root / 'hardware-before.json') == load(root / 'hardware-after.json')
    result = {'quality_passed': 20, 'source': summary['source'], 'source_hashes': summary['source_hashes']}
    for mode, tokens in [('bench', 55), ('long', 8270)]:
        data = load(root / mode / 'summary.json')
        assert data['pass'] and len(data['measured']) == 3
        warm = data['warmup']
        assert warm['full_output_budget'] and warm['usage']['completion_tokens'] == 128
        assert warm['timings']['cache_n'] == 0
        for value in data['measured']:
            assert value['usage']['prompt_tokens'] == tokens
            assert value['usage']['completion_tokens'] == 128 and value['full_output_budget']
            assert value['timings']['cache_n'] == 0
        metrics = {
            'prefill_ms': [v['timings']['prompt_ms'] for v in data['measured']],
            'prefill_tokens_per_second': [v['timings']['prompt_per_second'] for v in data['measured']],
            'ttft_s': [v['ttft_s'] for v in data['measured']],
            'decode_tokens_per_second': [v['decode_tps_client'] for v in data['measured']],
        }
        result[mode] = {key: {'samples': values, 'median': statistics.median(values)} for key, values in metrics.items()}
    anchor = load(root / 'long/summary.json')['anchor']
    assert anchor['usage']['prompt_tokens'] == 8238 and anchor['text'].strip() == 'BOLT-7319-CEDAR'
    result['retrieval'] = {'prompt_tokens': 8238, 'pass': True}
    return result


def main():
    baseline = ROOT / 'tail-r1-baseline'
    candidate = ROOT / 'tail-r1-candidate'
    assert normalize_create(load(baseline / 'create-argv.json')) == normalize_create(load(candidate / 'create-argv.json'))
    assert load(baseline / 'hardware-before.json') == load(candidate / 'hardware-before.json')
    a, b = summarize(baseline), summarize(candidate)
    out = {'tier': 'development', 'design': 'matched sequential baseline/candidate launches; warmup plus three measurements',
           'baseline': a, 'candidate': b, 'excluded_samples': [], 'delta_percent': {}, 'payloads': {}, 'output_identity': {}}
    for mode in ('bench', 'long'):
        out['delta_percent'][mode] = {
            metric: 100 * (b[mode][metric]['median'] / a[mode][metric]['median'] - 1)
            for metric in a[mode]
        }
        am = load(baseline / mode / 'summary.json')['measured']
        bm = load(candidate / mode / 'summary.json')['measured']
        out['output_identity'][mode] = {'matches': sum(x['text'] == y['text'] for x, y in zip(am, bm)), 'total': 3}
    for mode in ('quality', 'bench', 'long'):
        ap = sorted((baseline / mode).glob('*.request.json'))
        bp = sorted((candidate / mode).glob('*.request.json'))
        assert ap and [p.name for p in ap] == [p.name for p in bp]
        matches = sum(x.read_bytes() == y.read_bytes() for x, y in zip(ap, bp))
        assert matches == len(ap), f'{mode}: request payload mismatch'
        out['payloads'][mode] = {'byte_identical': matches, 'total': len(ap)}
    (ROOT / 'comparison.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
