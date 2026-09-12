"""Development-only native oneDNN W4A16 dispatch probe; no serving patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--describe', action='store_true', help='verbose dispatch only, no timing claims')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'tier': 'development native operator investigation', 'status': 'running',
              'command': shlex.join(sys.orig_argv), 'describe_only': args.describe,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'contract': 'FP16 compute, symmetric packed INT4, G128, scalar int8 zero point 8; no activation quantization',
              'cases': []}
    path = args.out / 'results.json'

    def save():
        path.write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        import torch
        import vllm_xpu_kernels._xpu_C  # Register GEMM, not only package-level attention ops.
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        assert torch.xpu.is_available()
        verbose = os.environ.get('ONEDNN_VERBOSE', '0')
        if not args.describe:
            assert verbose == '0', 'Verbose profiling must be off for timing'
        result['environment'] = {'torch': torch.__version__, 'device': torch.xpu.get_device_name(),
                                 'platform': platform.platform(), 'ONEDNN_VERBOSE': verbose,
                                 'native_schema': str(torch.ops._xpu_C.int4_gemm_w4a16.default._schema)}
        for label, k, n in (('gate_up', 5120, 34816), ('down', 17408, 5120)):
            torch.manual_seed(20260912)
            x = torch.randn((5, k), dtype=torch.float16, device='xpu') * 0.5
            # Native expects packed K contiguous within each output column.
            weight = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int32, device='xpu').t()
            scales = torch.rand((k // 128, n), dtype=torch.float16, device='xpu') * 0.02 + 0.01
            zp = torch.tensor([8], dtype=torch.int8, device='xpu')

            def native(a):
                return torch.ops._xpu_C.int4_gemm_w4a16(a, weight, None, scales, zp, 128, None)

            def padded(rows):
                a = torch.zeros((rows, k), dtype=x.dtype, device=x.device)
                a[:5].copy_(x)
                return native(a)[:5]

            routes = {'native5': lambda: native(x), 'pad8': lambda: padded(8),
                      'pad16': lambda: padded(16),
                      'five_native1': lambda: torch.cat([native(x[i:i+1]) for i in range(5)])}
            case = {'name': label, 'm': 5, 'n': n, 'k': k, 'weight_shape': list(weight.shape),
                    'weight_stride': list(weight.stride()), 'scales_shape': list(scales.shape),
                    'routes': {}}
            result['cases'].append(case)
            baseline = None
            for name, fn in routes.items():
                print(f'GEMM_ROUTE {label} {name} M5 K{k} N{n}', flush=True)
                y = fn()
                torch.xpu.synchronize()
                assert torch.isfinite(y).all().item()
                if baseline is None:
                    baseline = y.clone()
                entry = {'max_abs_vs_native5': (y.float() - baseline.float()).abs().max().item(),
                         'exact_vs_native5': torch.equal(y, baseline)}
                case['routes'][name] = entry
                torch.testing.assert_close(y, baseline, rtol=0.01, atol=0.01)
                if args.describe:
                    continue
                # Independent unpack/dequantize and CPU FP32 accumulation on
                # 32 evenly spaced output columns; compare all output elements
                # between routes above. Tolerances follow upstream W4A16 tests.
                cols = torch.linspace(0, n - 1, 32).long().to('xpu')
                words = weight.index_select(1, cols).cpu().long()
                shifts = torch.arange(8).view(1, 8, 1) * 4
                q = ((words[:, None, :] >> shifts) & 15).reshape(k, 32)
                s = scales.index_select(1, cols).cpu().float().repeat_interleave(128, dim=0)
                dequant = ((q.float() - 8) * s).half().float()
                ref = x.cpu().float() @ dequant
                actual = y.index_select(1, cols).cpu().float()
                entry['sampled_fp32_max_abs'] = (actual - ref).abs().max().item()
                torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.01)
            save()
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
            # Mutated inputs must be read at replay; padding/copies are inside
            # the timed graph, never precomputed on behalf of a candidate.
            x.mul_(-0.75).add_(0.125)
            expected = native(x).clone()
            for name, (graph, output) in graphs.items():
                graph.replay()
                torch.xpu.synchronize()
                torch.testing.assert_close(output, expected, rtol=0.01, atol=0.01)
                case['routes'][name]['mutated_graph_max_abs'] = (output.float() - expected.float()).abs().max().item()
            names = list(graphs)
            events = []
            start_wall = time.perf_counter()
            for repeat in range(12):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in order:
                    start = torch.xpu.Event(enable_timing=True)
                    end = torch.xpu.Event(enable_timing=True)
                    start.record()
                    for _ in range(16):
                        graphs[name][0].replay()
                    end.record()
                    events.append((name, start, end))
            torch.xpu.synchronize()
            case['timing_wall_seconds'] = time.perf_counter() - start_wall
            for name in names:
                durations = [s.elapsed_time(e) for route, s, e in events if route == name]
                samples = [duration / 16 for duration in durations]
                case['routes'][name].update(batch_elapsed_ms=durations, replays_per_sample=16)
                quartiles = statistics.quantiles(samples, n=4, method='inclusive')
                case['routes'][name].update(replay_ms=samples, median_ms=statistics.median(samples),
                                          iqr_ms=quartiles[2] - quartiles[0])
            case['timing_contract'] = '12 rotated/interleaved samples per route, 16 graph replays per event interval divided by16; warm calls3/replays3; verbose off; includes route padding and concatenation'
            print(json.dumps({'case': label, 'medians': {n: v['median_ms'] for n, v in case['routes'].items()}}), flush=True)
            save()
            del graphs
        result['status'] = 'passed'
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
    save()
    print(f"{result['status']}: {path}", flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
