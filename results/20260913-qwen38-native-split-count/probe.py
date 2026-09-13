"""Bounded native verifier probe. Run only in the pinned image on inference-host."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import statistics
import traceback

import torch
from vllm_xpu_kernels import flash_attn_interface as fa

OUT = Path('/output')
COUNTS = (None, 1, 4, 8, 16, 32)
RTOL, ATOL = 0.02, 0.0001
result = {'tier': 'development', 'tolerance': {'rtol': RTOL, 'atol': ATOL},
          'cases': [], 'timings': [], 'failures': []}


def save():
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def metrics(y, ref):
    delta = y.float() - ref.float()
    values = {'max_abs': delta.abs().max().item(),
              'rmse': delta.square().mean().sqrt().item()}
    torch.testing.assert_close(y.float(), ref.float(), rtol=RTOL, atol=ATOL)
    assert torch.isfinite(y).all().item()
    return values


def reference(q, k, v, table, length, ks, vs):
    # Independent FP32 implementation; one KV head at a time limits scratch.
    pages = table[0, :(length + 1663) // 1664].long()
    kk = k[pages].reshape(-1, 4, 256)[:length].float() * ks
    vv = v[pages].reshape(-1, 4, 256)[:length].float() * vs
    ans = torch.empty_like(q, dtype=torch.float32)
    positions = torch.arange(length, device='xpu')
    live = length - 4 + torch.arange(5, device='xpu')
    for head in range(4):
        qq = q[:, head * 6:(head + 1) * 6].float()
        scores = torch.einsum('qhd,kd->qhk', qq, kk[:, head]) / 16
        scores.masked_fill_(positions[None, None, :] >= live[:, None, None], -torch.inf)
        ans[:, head * 6:(head + 1) * 6] = torch.einsum(
            'qhk,kd->qhd', scores.softmax(-1), vv[:, head])
    return ans


def main():
    torch.manual_seed(42)
    source = inspect.getsource(fa._spec_decode_varlen_fwd)
    marker = 'None,  # num_splits (let the kernel pick via get_num_splits)'
    assert source.count(marker) == 1
    original = fa._spec_decode_varlen_fwd
    funcs = {None: original}
    for count in COUNTS[1:]:
        scope = dict(fa.__dict__)
        exec(compile(source.replace(marker, f'{count},  # explicit probe count'),
                     f'<native-split-{count}>', 'exec'), scope)
        funcs[count] = scope['_spec_decode_varlen_fwd']
    def no_fallback(*args, **kwargs):
        raise RuntimeError('Reference fallback forbidden')
    fa._fallback_varlen_attn = no_fallback
    package = Path(fa.__file__).parent
    result.update(torch=torch.__version__, kernels=importlib.metadata.version('vllm-xpu-kernels'),
                  device=str(torch.xpu.get_device_properties(0)),
                  schema=str(torch.ops._vllm_fa2_C.varlen_fwd.default._schema),
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(fa.__file__), package / '_vllm_fa2_C.abi3.so',
                                    package / 'libattn_kernels_xe_2.so']})
    (OUT / 'installed-spec-helper.py').write_text(source)
    save()
    # Full allocation count matches recorded production verifier; random physical pages.
    k = torch.randn(176, 1664, 4, 256, dtype=torch.float16, device='xpu').to(torch.float8_e4m3fn)
    v = torch.randn(176, 1664, 4, 256, dtype=torch.float16, device='xpu').to(torch.float8_e4m3fn)
    q_storage = torch.randn(5, 48, 256, dtype=torch.float16, device='xpu')
    q = q_storage[:, :24]  # last dimension contiguous, pitched token stride
    cu = torch.tensor([0, 5], dtype=torch.int32, device='xpu')
    lengths = torch.tensor([65541], dtype=torch.int32, device='xpu')
    table = torch.randperm(176, device='xpu', dtype=torch.int64)[:128].to(torch.int32)[None, :]
    ks, vs = 0.75, 1.25
    k_scale = torch.tensor(ks, dtype=torch.float32, device='xpu').expand(1, 4)
    v_scale = torch.tensor(vs, dtype=torch.float32, device='xpu').expand(1, 4)
    result['geometry'] = {name: {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype)}
                          for name, t in [('q', q), ('k', k), ('v', v), ('table', table)]}

    def forward(count, max_k, output):
        fa._spec_decode_varlen_fwd = funcs[count]
        y = fa.flash_attn_varlen_func(q=q, k=k, v=v, out=output,
            cu_seqlens_q=cu, seqused_k=lengths, max_seqlen_q=5,
            max_seqlen_k=max_k, block_table=table, softmax_scale=0.0625,
            causal=True, k_descale=k_scale, v_descale=v_scale)
        assert y.data_ptr() == output.data_ptr()
        return y

    # Each case retains its failure; baseline failure blocks timing for that case.
    for length in (517, 1663, 1664, 1665, 8197, 32773, 65541):
        lengths.fill_(length)
        max_k = ((length + 15) // 16) * 16
        ref = reference(q, k, v, table, length, ks, vs)
        baseline = forward(None, max_k, torch.empty_like(q)).clone()
        base_case = {'length': length, 'count': 'auto'}
        try:
            base_case['reference'] = metrics(baseline, ref)
            base_case['status'] = 'passed'
        except Exception:
            base_case.update(status='failed', error=traceback.format_exc())
            result['failures'].append(base_case)
        result['cases'].append(base_case)
        save()
        if base_case['status'] != 'passed':
            continue
        graphs = {}
        for count in COUNTS:
            case = {'length': length, 'count': count if count is not None else 'auto'}
            try:
                output = torch.empty_like(q)
                for _ in range(3):
                    forward(count, max_k, output)
                torch.xpu.synchronize()
                case['vs_auto'] = metrics(output, baseline)
                case['reference'] = metrics(output, ref)
                if length == 65541:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.XPU], record_shapes=True,
                            profile_memory=True) as prof:
                        forward(count, max_k, output)
                        torch.xpu.synchronize()
                    prof.export_chrome_trace(str(OUT / f'dispatch-{case["count"]}.json'))
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    forward(count, max_k, output)
                for _ in range(3):
                    graph.replay()
                torch.xpu.synchronize()
                case['graph_reference'] = metrics(output, ref)
                # Same graph, device metadata updated; public max bound unchanged.
                old_table = table.clone()
                table.copy_(table.flip(1))
                lengths.fill_(length - 3)
                changed_ref = reference(q, k, v, table, length - 3, ks, vs)
                graph.replay()
                torch.xpu.synchronize()
                case['mutated_graph_reference'] = metrics(output, changed_ref)
                eager = forward(count, max_k, torch.empty_like(q))
                case['mutated_graph_eager'] = metrics(output, eager)
                table.copy_(old_table)
                lengths.fill_(length)
                graph.replay()
                torch.xpu.synchronize()
                case['restored_graph_reference'] = metrics(output, ref)
                graphs[count] = (graph, output)
                case['status'] = 'passed'
            except Exception:
                case.update(status='failed', error=traceback.format_exc())
                result['failures'].append(case)
                # Restore metadata even if a mutation gate failed.
                if 'old_table' in locals():
                    table.copy_(old_table)
                lengths.fill_(length)
            result['cases'].append(case)
            save()
        if length not in (517, 8197, 32773, 65541) or None not in graphs:
            continue
        samples = {count: [] for count in graphs}
        order = list(graphs)
        for round_id in range(12):
            for count in (order if round_id % 2 == 0 else order[::-1]):
                graph, _ = graphs[count]
                start, end = (torch.xpu.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(4):
                    graph.replay()
                end.record()
                torch.xpu.synchronize()
                ms = start.elapsed_time(end) / 4
                samples[count].append(ms)
                result['timings'].append({'length': length, 'count': count if count is not None else 'auto',
                                          'round': round_id, 'replays': 4, 'ms_per_replay': ms})
        result.setdefault('summary', {})[str(length)] = {
            str(count): {'median_ms': statistics.median(times),
                         'iqr_ms': statistics.quantiles(times, n=4, method='inclusive')[2]
                                   - statistics.quantiles(times, n=4, method='inclusive')[0]}
            for count, times in samples.items()}
        save()
        print(json.dumps({'length': length, 'summary': result['summary'][str(length)]}), flush=True)
    fa._spec_decode_varlen_fwd = original
    result['status'] = 'failed' if result['failures'] else 'passed'
    save()


try:
    main()
except Exception:
    result.update(status='failed', fatal=traceback.format_exc())
    save()
    raise
finally:
    print(json.dumps({'status': result.get('status'), 'failures': len(result['failures'])}), flush=True)
if result['status'] != 'passed':
    raise SystemExit(1)
