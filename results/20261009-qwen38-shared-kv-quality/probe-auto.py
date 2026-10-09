"""Compare the unchanged installed auto helper, forced-32 control and candidate."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

LENGTHS = (1, 5, 65, 139, 149, 280, 512, 1024, 1664, 1984, 1985,
           1988, 1989, 2048, 2049, 2052, 2053, 4097, 8197, 32773, 65541)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('prior_probe', Path(__file__).with_name('prior-probe.py'))
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    assert torch.__version__ == '2.13.0+xpu'
    assert hashlib.sha256(Path(prior.fa.__file__).read_bytes()).hexdigest() == '2a8ce07e2839232bc9f0e9cc9969a4410099c0c75ce616e72d4583e950864942'
    torch.set_num_threads(12)
    candidate = prior.load('auto_probe_candidate', Path(__file__).with_name('grouped_verify.py'))
    checks = prior.load('auto_probe_checks', Path(__file__).with_name('check-grouped-split-k.py'))
    checks.torch = torch
    torch.ops.load_library(str(args.library))
    result = {'status': 'running', 'scope': 'operator diagnosis, not model quality qualification',
              'rtol': 0.02, 'atol': 1e-4, 'cases': [], 'graph_replays': [],
              'routes': {'auto': 'unchanged installed helper; num_splits=None',
                         'native': 'prior forced-32 diagnostic control', 'candidate': 'unchanged build-06'}}
    routes = prior.Routes(candidate, args.output)

    def save():
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def compare(a, b):
        difference = a.view(torch.int16) != b.view(torch.int16)
        return {'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                'bitwise_equal': not bool(difference.any()),
                'different_elements_per_token': difference.sum(dim=(1, 2)).cpu().tolist(),
                'max_abs': float((a.float() - b.float()).abs().max()),
                'within_original_tolerance': bool(torch.allclose(a, b, rtol=0.02, atol=1e-4))}

    def record(collection, metadata, outputs):
        metadata.update(candidate_vs_auto=compare(outputs['candidate'], outputs['auto']),
                        forced32_vs_auto=compare(outputs['native'], outputs['auto']),
                        candidate_vs_forced32=compare(outputs['candidate'], outputs['native']))
        collection.append(metadata)
        save()
        assert all(metadata[key]['finite'] and metadata[key]['within_original_tolerance']
                   for key in ('candidate_vs_auto', 'forced32_vs_auto', 'candidate_vs_forced32'))

    try:
        save()
        for seed in (42, 43):
            torch.manual_seed(seed)
            kv = torch.randn((152, 1664, 4, 512), dtype=torch.float16, device='xpu').mul_(0.5).to(torch.float8_e4m3fn)
            k, v = kv[..., :256], kv[..., 256:]
            for length in LENGTHS:
                c = checks.Case(
                    torch.randn((5, 24, 512), dtype=torch.float16, device='xpu')[..., :256], k, v,
                    torch.empty((5, 24, 256), dtype=torch.float16, device='xpu'),
                    torch.tensor([0, 5], dtype=torch.int32, device='xpu'),
                    torch.tensor([length], dtype=torch.int32, device='xpu'),
                    torch.randperm(152, device='xpu')[:128].to(torch.int32)[None, :],
                    torch.tensor(0.75, device='xpu').expand(1, 4),
                    torch.tensor(1.25, device='xpu').expand(1, 4), 212992)
                for maximum in (length, 212992):
                    case = replace(c, maximum=maximum)
                    outputs = {route: routes.forward(case, route).clone() for route in ('auto', 'native', 'candidate')}
                    record(result['cases'], {'seed': seed, 'length': length, 'max_seqlen_k': maximum}, outputs)
        # Keep metadata/grid fixed during replay; mutate only the device length.
        c = replace(c, maximum=212992)
        graphs = {route: prior.capture(replace(c, out=torch.empty_like(c.out)), routes, route)
                  for route in ('auto', 'native', 'candidate')}
        for length in (139, 149, 280, 1985, 2049, 65541):
            c.used.fill_(length)
            for graph, _ in graphs.values():
                graph.replay()
            torch.xpu.synchronize()
            record(result['graph_replays'], {'capture_length': 65541, 'length': length, 'max_seqlen_k': 212992},
                   {route: output.clone() for route, (_, output) in graphs.items()})
        result['status'] = 'passed_original_tolerance'
    except Exception:
        result['status'] = 'failed'
        raise
    finally:
        routes.close()
        save()
    print(json.dumps({'status': result['status'], 'eager_cases': len(result['cases']),
                      'graph_cases': len(result['graph_replays'])}))


if __name__ == '__main__':
    main()
