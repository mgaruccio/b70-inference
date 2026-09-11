"""Native operator preflight only; serving performance is a separate API gate."""
import json
import statistics
import traceback
from pathlib import Path
import torch
from vllm_xpu_kernels import flash_attn_interface as fa

out = Path('/output')
out.mkdir(exist_ok=True)
torch.manual_seed(42)
q = torch.randn(5, 24, 256, dtype=torch.float16, device='xpu')
k = torch.randn(40, 1664, 4, 256, dtype=torch.float16, device='xpu').to(torch.float8_e4m3fn)
v = torch.randn_like(k.to(torch.float16)).to(torch.float8_e4m3fn)
cu = torch.tensor([0, 5], dtype=torch.int32, device='xpu')
lengths = torch.tensor([65541], dtype=torch.int32, device='xpu')
blocks = torch.zeros((1, 128), dtype=torch.int32, device='xpu')
blocks[0, :40] = torch.arange(40, dtype=torch.int32, device='xpu')
scale = torch.ones((), dtype=torch.float32, device='xpu').expand(1, 4)

def reject_fallback(*args, **kwargs):
    raise RuntimeError('REJECTED: reference-attention fallback is not a native candidate')

fa._fallback_varlen_attn = reject_fallback

def forward():
    return fa.flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu,
        seqused_k=lengths, max_seqlen_q=5, max_seqlen_k=65552,
        block_table=blocks, softmax_scale=0.0625, causal=True,
        k_descale=scale, v_descale=scale)

result = {'tier': 'development operator preflight', 'torch': torch.__version__,
          'device': torch.xpu.get_device_name(), 'q_shape': list(q.shape),
          'kv_shape': list(k.shape), 'kv_dtype': str(k.dtype), 'used_kv': 65541,
          'routes': {}, 'fallback_forbidden': True}
reference = None
for threshold in (16, 4):
    route = {'threshold': threshold}
    try:
        fa._SPEC_DECODE_MAX_QLEN = threshold
        for _ in range(3):
            y = forward()
        torch.xpu.synchronize()
        assert torch.isfinite(y).all().item()
        if reference is None:
            reference = y.clone()
        else:
            diff = y.float() - reference.float()
            route['max_abs_error'] = diff.abs().max().item()
            route['rmse'] = diff.square().mean().sqrt().item()
            route['cosine'] = torch.nn.functional.cosine_similarity(y.float().flatten(), reference.float().flatten(), dim=0).item()
            torch.testing.assert_close(y, reference, rtol=0.02, atol=0.0001)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            y = forward()
        for _ in range(3):
            graph.replay()
        torch.xpu.synchronize()
        starts = [torch.xpu.Event(enable_timing=True) for _ in range(12)]
        ends = [torch.xpu.Event(enable_timing=True) for _ in range(12)]
        for start, end in zip(starts, ends):
            start.record(); graph.replay(); end.record()
        torch.xpu.synchronize()
        times = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        route.update(status='passed', replay_ms=times, median_replay_ms=statistics.median(times))
    except Exception:
        route.update(status='failed', error=traceback.format_exc())
    result['routes'][str(threshold)] = route
(out / 'attention-paths.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
