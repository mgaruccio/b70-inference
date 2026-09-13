"""Reproduce this bounded catalog trial's evidence checks and no-win decision."""
import json
import math
from pathlib import Path
import re
import statistics


def main():
    root = Path(__file__).resolve().parent
    lib_hash = 'f69aed9aec33459b4484095ce4e03899856ce892f6451d719dcb5f42a0e38f3a'
    installed_hash = '71c21e5231908cfa67f45b389de7cef6e96e564c769198a675725197234d07fc'
    report = {'tier': 'development', 'status': 'validated', 'library_sha256': lib_hash,
              'build_exit_codes': {}, 'operators': {}, 'qualified_candidates': []}
    for attempt, expected in ((1, 1), (2, 1), (3, 0)):
        code = int((root / f'build-{attempt:02}/exit-code.txt').read_text())
        assert code == expected
        report['build_exit_codes'][str(attempt)] = code
    cache = (root / 'build-03/onednn-CMakeCache.txt').read_text()
    for setting in ('CMAKE_BUILD_TYPE:STRING=Release', 'DNNL_CPU_RUNTIME:STRING=NONE',
                    'DNNL_GPU_RUNTIME:STRING=SYCL', 'DNNL_DEV_MODE:BOOL=OFF',
                    'DNNL_LIBRARY_TYPE:STRING=STATIC', 'DNNL_ENABLE_PRIMITIVE:STRING=ALL',
                    'DNNL_ENABLE_PRIMITIVE_GPU_ISA:STRING=XE2'):
        assert setting in cache
    batches = 0
    catalog_identities = {}
    for label, requested, selected in (('auto', -1, 0), ('index1', 1, 1), ('index2', 2, 2)):
        cell = root / f'operator-{label}-01'
        data = json.loads((cell / 'results/result.json').read_text())
        assert int((cell / 'exit-code.txt').read_text()) == 0 and data['status'] == 'passed'
        assert data['catalog_index'] == str(requested) and not data['describe']
        assert data['tolerance'] == {'rtol': 0.01, 'atol': 0.01}
        assert data['environment']['rebuilt_sha256'] == lib_hash
        assert data['environment']['installed_sha256'] == installed_hash
        assert (cell / 'probe.py').read_bytes() == (root / 'probe.py').read_bytes()
        symbols = (cell / 'dynamic-symbols.txt').read_text()
        assert not re.search(r' (?:dnnl_|_ZN4dnnl)', symbols)
        log = (cell / 'driver.log').read_text()
        assert f'group_k=128,requested={requested},entries=13' in log
        choices = re.findall(r'^b70_gemm_catalog,selected,mode=(auto|forced),index=(\d+),identity=(.+)$', log, re.M)
        assert len(choices) == 1
        mode, index, identity = choices[0]
        assert int(index) == selected and mode == ('auto' if label == 'auto' else 'forced')
        choices_list = dict(re.findall(r'^b70_gemm_catalog,candidate,index=(\d+),identity=(.+)$', log, re.M))
        assert identity == choices_list[str(selected)]
        if catalog_identities:
            assert choices_list == catalog_identities
        catalog_identities = choices_list
        entry = {'requested': requested, 'selected': selected, 'identity': identity, 'shapes': {}}
        assert len(data['cases']) == 2
        for case in data['cases']:
            name, k, n = case['name'], case['k'], case['n']
            assert (name, case['m'], k, n) in (('gate_up', 5, 5120, 34816), ('down', 5, 17408, 5120))
            assert case['weight_stride'] == [1, k // 8]
            assert case['scales_shape'] == [k // 128, n]
            assert set(case['routes']) == {'installed', 'rebuilt'}
            shape = {}
            for route, values in case['routes'].items():
                samples = values['replay_ms']
                assert len(samples) == 12 and values['replays_per_sample'] == 16
                assert all(math.isfinite(x) and x > 0 for x in samples)
                assert samples == [v / 16 for v in values['batch_elapsed_ms']]
                assert statistics.median(samples) == values['median_ms']
                shape[route] = {key: values[key] for key in (
                    'median_ms', 'iqr_ms', 'max_abs_vs_installed',
                    'exact_vs_installed', 'sampled_fp32_max_abs', 'mutated_graph_max_abs')}
                batches += len(samples)
            if label == 'auto' or name == 'down':
                assert shape['rebuilt']['exact_vs_installed']
                assert shape['rebuilt']['mutated_graph_max_abs'] == 0
            shape['latency_change_vs_installed_pct'] = 100 * (
                shape['rebuilt']['median_ms'] / shape['installed']['median_ms'] - 1)
            entry['shapes'][name] = shape
        report['operators'][label] = entry
    for label in ('index1', 'index2'):
        shapes = report['operators'][label]['shapes']
        for name, shape in shapes.items():
            auto = report['operators']['auto']['shapes'][name]
            ratio = shape['rebuilt']['median_ms'] / shape['installed']['median_ms']
            auto_ratio = auto['rebuilt']['median_ms'] / auto['installed']['median_ms']
            shape['latency_change_vs_rebuilt_auto_pct'] = 100 * (shape['rebuilt']['median_ms'] / auto['rebuilt']['median_ms'] - 1)
            shape['installed_normalized_change_vs_rebuilt_auto_pct'] = 100 * (ratio / auto_ratio - 1)
        if (shapes['gate_up']['latency_change_vs_installed_pct'] <= -5
                and shapes['gate_up']['installed_normalized_change_vs_rebuilt_auto_pct'] <= -5
                and shapes['down']['latency_change_vs_installed_pct'] <= 5):
            report['qualified_candidates'].append(label)
    assert batches == 144 and not report['qualified_candidates']
    report.update(event_batches=batches, timed_graph_replays=batches * 16,
                  decision='Keep installed native GEMM; no serving A/B, no promotion.')
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'operators'}, indent=2))


if __name__ == '__main__':
    main()
