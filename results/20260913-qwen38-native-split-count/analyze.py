"""Validate retained probe evidence and reproduce the bounded no-win decision."""
import json
import math
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    cell = root / 'operator-01'
    result = json.loads((cell / 'result.json').read_text())
    assert (cell / 'exit-code.txt').read_text().strip() == '0'
    assert result['status'] == 'passed' and not result['failures']
    assert result['tolerance'] == {'rtol': 0.02, 'atol': 0.0001}
    assert result['kernels'] == '0.1.12.3'
    assert len(result['cases']) == 49 and len(result['timings']) == 288
    assert all(c['status'] == 'passed' for c in result['cases'])
    graph_cases = [c for c in result['cases'] if 'graph_reference' in c]
    assert len(graph_cases) == 42
    assert all(c['mutated_graph_eager']['max_abs'] == 0 for c in graph_cases)
    dispatch = {}
    for count in ('auto', '1', '4', '8', '16', '32'):
        events = json.loads((cell / f'dispatch-{count}.json').read_text())['traceEvents']
        calls = [e for e in events if e.get('name') == '_vllm_fa2_C::varlen_fwd']
        assert len(calls) == 1
        args = calls[0]['args']
        assert args['Input Dims'][0:3] == [[5, 24, 256], [176, 1664, 4, 256], [176, 1664, 4, 256]]
        assert args['Input Dims'][4] == [6] and args['Input Dims'][6] == [5]
        assert args['Input Dims'][8] == [5, 128]
        assert args['Concrete Inputs'][24] == ('' if count == 'auto' else count)
        assert args['Concrete Inputs'][10:12] == ['1', '65552']
        shapes = [e['args']['Concrete Inputs'][0] for e in events
                  if e.get('name') == 'aten::empty' and e.get('args', {}).get('Concrete Inputs')]
        selected = 32 if count == 'auto' else int(count)
        assert shapes.count(f'[5, 24, {selected}]') == 2
        if selected > 1:
            assert f'[5, {24 * selected}, 256]' in shapes
        kernels = [e for e in events if e.get('cat') == 'kernel']
        reductions = [e for e in kernels if 'ReduceSplitK' in e['name']]
        assert len(reductions) == (1 if selected > 1 else 0)
        assert any('XeFMHA' in e['name'] for e in kernels)
        dispatch[count] = {'observed_split_count_65541': selected,
                           'partial_output_shape': [5, 24 * selected, 256] if selected > 1 else None,
                           'reduction_kernels': len(reductions)}
    summary = {}
    for length in (517, 8197, 32773, 65541):
        row = {}
        for count in ('auto', 1, 4, 8, 16, 32):
            samples = [s for s in result['timings'] if s['length'] == length and s['count'] == count]
            assert len(samples) == 12 and sorted(s['round'] for s in samples) == list(range(12))
            times = [s['ms_per_replay'] for s in samples]
            assert all(math.isfinite(t) and t > 0 for t in times)
            quartiles = statistics.quantiles(times, n=4, method='inclusive')
            row[str(count)] = {'median_ms': statistics.median(times), 'iqr_ms': quartiles[2] - quartiles[0]}
        for stats in row.values():
            stats['latency_reduction_pct'] = 100 * (1 - stats['median_ms'] / row['auto']['median_ms'])
        summary[str(length)] = row
    winners = [str(c) for c in (1, 4, 8, 16, 32)
               if all(summary[str(n)][str(c)]['latency_reduction_pct'] >= 5 for n in (32773, 65541))
               and summary['8197'][str(c)]['latency_reduction_pct'] >= -5]
    assert not winners
    report = {'status': 'validated', 'correctness_records': 49, 'graph_cases': 42,
              'event_batches': 288, 'replays_per_batch': 4, 'dispatch': dispatch,
              'summary': summary, 'qualified_candidates': winners,
              'decision': 'No operator winner; do not run serving A/B or alter baseline.'}
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
