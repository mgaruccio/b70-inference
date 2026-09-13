"""Validate the predeclared remaining native catalog screen, retaining failures."""
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
LIB = '17b2350f5885607dba229ac87c6cd5ba723173485f37e2d75cb938e547d6cd64'
INSTALLED = '71c21e5231908cfa67f45b389de7cef6e96e564c769198a675725197234d07fc'


def validate(label, requested):
    cell = ROOT / label
    code = int((cell / 'exit-code.txt').read_text())
    row = {'cell': label, 'requested': requested, 'exit_code': code}
    if code:
        row['status'] = 'failed; retained, not qualified'
        return row
    data = json.loads((cell / 'results/result.json').read_text())
    assert data['status'] == 'passed' and not data['describe']
    assert data['catalog_index'] == str(requested)
    assert data['tolerance'] == {'rtol': 0.01, 'atol': 0.01}
    assert data['environment']['rebuilt_sha256'] == LIB
    assert data['environment']['installed_sha256'] == INSTALLED
    assert (cell / 'probe.py').read_bytes() == (ROOT / 'probe.py').read_bytes()
    assert not re.search(r' (?:dnnl_|_ZN4dnnl)', (cell / 'dynamic-symbols.txt').read_text())
    log = (cell / 'driver.log').read_text()
    assert f'group_k=128,requested={requested},entries=13' in log
    choices = re.findall(r'^b70_gemm_catalog,selected,mode=(auto|forced),index=(\d+),identity=(.+)$', log, re.M)
    assert len(choices) == 1
    mode, selected, identity = choices[0]
    assert int(selected) == (0 if requested == -1 else requested)
    assert mode == ('auto' if requested == -1 else 'forced')
    catalog = dict(re.findall(r'^b70_gemm_catalog,candidate,index=(\d+),identity=(.+)$', log, re.M))
    assert len(catalog) == 13 and catalog[selected] == identity
    row.update(status='passed', identity=identity, catalog=catalog, shapes={})
    assert len(data['cases']) == 2
    for case in data['cases']:
        name, k, n = case['name'], case['k'], case['n']
        assert (name, case['m'], k, n) in (('gate_up', 5, 5120, 34816), ('down', 5, 17408, 5120))
        assert case['weight_stride'] == [1, k // 8] and case['scales_shape'] == [k // 128, n]
        assert set(case['routes']) == {'installed', 'rebuilt'}
        shape = {}
        for route, values in case['routes'].items():
            samples = values['replay_ms']
            assert len(samples) == 12 and values['replays_per_sample'] == 16
            assert all(math.isfinite(x) and x > 0 for x in samples)
            assert samples == [v / 16 for v in values['batch_elapsed_ms']]
            assert statistics.median(samples) == values['median_ms']
            shape[route] = values['median_ms']
        rebuilt = case['routes']['rebuilt']
        if requested == -1 or name == 'down':
            assert rebuilt['exact_vs_installed'] and rebuilt['mutated_graph_max_abs'] == 0
        shape['change_vs_installed_pct'] = 100 * (shape['rebuilt'] / shape['installed'] - 1)
        row['shapes'][name] = shape
    return row


def main():
    assert int((ROOT / 'build-04/exit-code.txt').read_text()) == 0
    auto = validate('operator-auto-02', -1)
    assert auto['status'] == 'passed'
    report = {'tier': 'development', 'library_sha256': LIB, 'auto': auto, 'candidates': [], 'qualified_candidates': []}
    for index in range(3, 13):
        row = validate(f'operator-index{index}-01', index)
        report['candidates'].append(row)
        if row['status'] != 'passed':
            continue
        assert row['catalog'] == auto['catalog']
        for name, shape in row['shapes'].items():
            baseline = auto['shapes'][name]
            ratio = shape['rebuilt'] / shape['installed']
            baseline_ratio = baseline['rebuilt'] / baseline['installed']
            shape['normalized_change_vs_auto_pct'] = 100 * (ratio / baseline_ratio - 1)
        shapes = row['shapes']
        if (shapes['gate_up']['change_vs_installed_pct'] <= -5
                and shapes['gate_up']['normalized_change_vs_auto_pct'] <= -5
                and shapes['down']['change_vs_installed_pct'] <= 5):
            report['qualified_candidates'].append(index)
    (ROOT / 'analysis-remaining.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'qualified_candidates': report['qualified_candidates'], 'candidates': [
        {k: v for k, v in row.items() if k not in ('catalog', 'identity')} for row in report['candidates']
    ]}, indent=2))


if __name__ == '__main__':
    main()
