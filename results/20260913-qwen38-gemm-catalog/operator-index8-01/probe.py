"""Installed versus rebuilt native W4A16; run only on inference-host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'status': 'running', 'tier': 'development', 'argv': sys.orig_argv,
              'catalog_index': os.environ.get('B70_M5_GATEUP_CATALOG_INDEX', '-1'),
              'describe': args.describe, 'tolerance': {'rtol': 0.01, 'atol': 0.01},
              'cases': []}
    def save():
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        import torch
        import vllm_xpu_kernels._xpu_C
        torch.set_num_threads(8)
        assert torch.xpu.is_available()
        if not args.describe:
            assert os.environ.get('ONEDNN_VERBOSE', '0') == '0'
        torch.ops.load_library(str(args.library))
        result['environment'] = {'torch': torch.__version__,
            'device': str(torch.xpu.get_device_properties(0)),
            'rebuilt_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'installed_sha256': hashlib.sha256(Path(vllm_xpu_kernels._xpu_C.__file__).read_bytes()).hexdigest(),
            'installed_schema': str(torch.ops._xpu_C.int4_gemm_w4a16.default._schema),
            'rebuilt_schema': str(torch.ops.b70_gemm_catalog.int4_gemm_w4a16.default._schema)}
        for label, k, n in (('gate_up', 5120, 34816), ('down', 17408, 5120)):
            torch.manual_seed(20260912)
            x = torch.randn((5, k), dtype=torch.float16, device='xpu') * 0.5
            weight = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int32, device='xpu').t()
            scales = torch.rand((k // 128, n), dtype=torch.float16, device='xpu') * 0.02 + 0.01
            zp = torch.tensor([8], dtype=torch.int8, device='xpu')
            def installed():
                return torch.ops._xpu_C.int4_gemm_w4a16(x, weight, None, scales, zp, 128, None)
            def rebuilt():
                return torch.ops.b70_gemm_catalog.int4_gemm_w4a16(x, weight, None, scales, zp, 128, None)
            routes = {'installed': installed, 'rebuilt': rebuilt}
            case = {'name': label, 'm': 5, 'k': k, 'n': n,
                    'weight_shape': list(weight.shape), 'weight_stride': list(weight.stride()),
                    'scales_shape': list(scales.shape), 'routes': {}}
            result['cases'].append(case)
            save()
            cols = torch.linspace(0, n - 1, 32).long().to('xpu')
            words = weight.index_select(1, cols).cpu().long()
            shifts = torch.arange(8).view(1, 8, 1) * 4
            unpacked = ((words[:, None, :] >> shifts) & 15).reshape(k, 32)
            s = scales.index_select(1, cols).cpu().float().repeat_interleave(128, dim=0)
            dequant = ((unpacked.float() - 8) * s).half().float()
            ref = x.cpu().float() @ dequant
            baseline = installed().clone()
            for name, fn in routes.items():
                print(f'GEMM_CATALOG_ROUTE {label} {name}', flush=True)
                y = fn()
                torch.xpu.synchronize()
                entry = {'max_abs_vs_installed': (y.float() - baseline.float()).abs().max().item(),
                         'exact_vs_installed': torch.equal(y, baseline)}
                case['routes'][name] = entry
                actual = y.index_select(1, cols).cpu().float()
                entry['sampled_fp32_max_abs'] = (actual - ref).abs().max().item()
                save()
                assert torch.isfinite(y).all().item()
                torch.testing.assert_close(y, baseline, rtol=0.01, atol=0.01)
                torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.01)
            if args.describe:
                continue
            graphs = {}
            for name, fn in routes.items():
                for _ in range(3):
                    fn()
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    output = fn()
                for _ in range(3):
                    graph.replay()
                torch.xpu.synchronize()
                torch.testing.assert_close(output, baseline, rtol=0.01, atol=0.01)
                graphs[name] = (graph, output)
            x.mul_(-0.75).add_(0.125)
            expected = installed().clone()
            changed_ref = x.cpu().float() @ dequant
            for name, (graph, output) in graphs.items():
                graph.replay()
                torch.xpu.synchronize()
                entry = case['routes'][name]
                entry['mutated_graph_max_abs'] = (output.float() - expected.float()).abs().max().item()
                save()
                assert torch.isfinite(output).all().item()
                torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
                torch.testing.assert_close(output.index_select(1, cols).cpu().float(), changed_ref,
                                           rtol=0.01, atol=0.01)
            events = []
            names = list(graphs)
            for repeat in range(12):
                order = names if repeat % 2 == 0 else names[::-1]
                for name in order:
                    start, end = (torch.xpu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(16):
                        graphs[name][0].replay()
                    end.record()
                    events.append((repeat, name, start, end))
            torch.xpu.synchronize()
            for name in names:
                durations = [start.elapsed_time(end) for _, route, start, end in events if route == name]
                samples = [duration / 16 for duration in durations]
                quartiles = statistics.quantiles(samples, n=4, method='inclusive')
                case['routes'][name].update(batch_elapsed_ms=durations, replay_ms=samples,
                    replays_per_sample=16, median_ms=statistics.median(samples), iqr_ms=quartiles[2] - quartiles[0])
            case['timing_order'] = [[r, name] for r, name, _, _ in events]
            print(json.dumps({'case': label, 'routes': case['routes']}), flush=True)
            save()
            del graphs
        result['status'] = 'passed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
    save()
    print(json.dumps({'status': result['status'], 'out': str(args.out)}), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
