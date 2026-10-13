#!/usr/bin/env python3
"""Tiny synthetic native-op graph compatibility probe; NOT a model benchmark."""
import importlib
import json
from pathlib import Path

import torch

OUT = Path('/results')
for name in ('vllm_xpu_kernels._xpu_C', 'vllm._xpu_ops', 'vllm._C'):
    if hasattr(torch.ops._xpu_C, 'int4_gemm_w4a16'):
        break
    try:
        importlib.import_module(name)
    except ImportError:
        pass
op = torch.ops._xpu_C.int4_gemm_w4a16
assert torch.xpu.is_available()
x = torch.ones((32, 128), dtype=torch.float16, device='xpu')
# Physical [N,K/8] storage, noncontiguous [K/8,N] native view.
qweight = torch.zeros((128, 16), dtype=torch.int32, device='xpu').t()
scales = torch.full((1, 128), 0.01, dtype=torch.float16, device='xpu')
zero = torch.tensor([8], dtype=torch.int8, device='xpu')

def native():
    return op(x, qweight, None, scales, zero, 128, None)

warm = native()
assert torch.isfinite(warm).all().item()
graph = torch.xpu.XPUGraph()
trace_path = OUT / 'compatibility.trace.json'
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU],
    record_shapes=True, with_stack=False, profile_memory=False, with_flops=False,
) as profiler:
    eager = native()
    with torch.xpu.graph(graph):
        captured = native()
    graph.replay()
profiler.export_chrome_trace(str(trace_path))
assert torch.isfinite(captured).all().item()
assert torch.equal(eager, captured)
trace = json.loads(trace_path.read_text())
events = trace['traceEvents']
ops = [e for e in events if e.get('name') == '_xpu_C::int4_gemm_w4a16']
kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
assert ops and kernels, 'Native CPU op / real device activity absent'
assert any(e.get('args', {}).get('Input Dims', [])[:2] == [[32, 128], [16, 128]] for e in ops), 'Native op dimensions absent'
report = {'pass': True, 'tier': 'development compatibility only',
          'synthetic_weights': True, 'torch': torch.__version__,
          'schema': str(op.default._schema), 'graphs_enabled': True,
          'finite_and_equal': True, 'native_op_count': len(ops),
          'kernel_count': len(kernels), 'input_dims': [e.get('args', {}).get('Input Dims') for e in ops]}
(OUT / 'compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
