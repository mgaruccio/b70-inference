"""Actual pinned XPU grouped RMSNorm vs independent per-layer native calls."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from vllm import _custom_ops as ops


def measure(a, b):
    d = a.float() - b.float()
    return {"exact": bool(torch.equal(a, b)),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(d.abs().max()), "rmse": float(d.square().mean().sqrt())}


def probe(x, weights):
    x, weights = x.to('xpu').contiguous(), weights.to('xpu').contiguous()
    grouped, separate, row0 = [torch.empty_like(x) for _ in range(3)]
    ops.rms_norm(grouped, x, weights, 1e-6)
    for i in range(x.shape[0]):
        ops.rms_norm(separate[i], x[i], weights[i], 1e-6)
        ops.rms_norm(row0[i], x[i], weights[0], 1e-6)
    torch.xpu.synchronize()
    result = {"shape": list(x.shape), "dtype": str(x.dtype),
              "grouped_vs_per_layer": measure(grouped, separate),
              "grouped_vs_row0": measure(grouped, row0),
              "layers": [{"grouped_vs_per_layer": measure(grouped[i], separate[i]),
                          "grouped_vs_row0": measure(grouped[i], row0[i]),
                          "means": [float(y[i].float().mean()) for y in (grouped, separate, row0)]}
                         for i in range(x.shape[0])]}
    return result, grouped.cpu()


def main():
    data = torch.load('/parity/capture-02/capture.pt', map_location='cpu', weights_only=True)
    with safe_open('/draft/model.safetensors', framework='pt', device='cpu') as f:
        weights = torch.stack([f.get_tensor(f'layers.{i}.self_attn.k_norm.weight') for i in range(5)])
    real, grouped = probe(data['context_k_raw'], weights)
    real['grouped_vs_captured'] = measure(grouped, data['context_k_norm'])
    results = {'device': torch.xpu.get_device_name(), 'torch': torch.__version__, 'captured': real}
    for dtype in (torch.bfloat16, torch.float16):
        x = torch.ones((5, 3, 2, 128), dtype=dtype)
        w = torch.arange(1, 6, dtype=dtype)[:, None].expand(5, 128).contiguous()
        results[str(dtype)] = probe(x, w)[0]
    Path('/output/grouped-rmsnorm-probe.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
