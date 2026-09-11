"""Actual XPU cache dispatch probe; no model weights or target configuration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
from vllm import _custom_ops as ops

root = Path(importlib.util.find_spec("vllm").origin).parent
assert str(root).startswith("/opt/venv/")
assert hashlib.sha256((root / "_custom_ops.py").read_bytes()).hexdigest() == "eb439f4656903c11c8789cf8fd0eb86300f69540cf07e29a0564a9476aedecb3"
assert torch.xpu.is_available()
device = torch.device("xpu:0")
key = ((torch.arange(4 * 8 * 128).reshape(4, 8, 128) % 31) / 32).to(device=device, dtype=torch.bfloat16)
value = -key
slots = torch.tensor([0, 15, 16, -1], device=device, dtype=torch.int64)
scale = torch.ones(1, device=device, dtype=torch.float32)
rows = []
for mode in ("bfloat16", "auto"):
    kc = torch.full((2, 16, 8, 128), -13, device=device, dtype=torch.bfloat16)
    vc = torch.full_like(kc, -13)
    expected_k, expected_v = kc.cpu(), vc.cpu()
    for index, slot in enumerate((0, 15, 16)):
        expected_k[slot // 16, slot % 16] = key[index].cpu()
        expected_v[slot // 16, slot % 16] = value[index].cpu()
    try:
        ops.reshape_and_cache_flash(key, value, kc, vc, slots, mode, scale, scale)
        torch.xpu.synchronize()
        exact = torch.equal(kc.cpu(), expected_k) and torch.equal(vc.cpu(), expected_v)
        assert exact
        rows.append({"mode": mode, "storage_dtype": str(kc.dtype), "exact_cache_and_untouched_slots": exact})
    except RuntimeError as exc:
        if mode != "bfloat16" or "Unsupported data type of kv cache: bfloat16" not in str(exc):
            raise
        rows.append({"mode": mode, "storage_dtype": str(kc.dtype), "expected_dispatch_rejection": str(exc)})
assert rows[0].get("expected_dispatch_rejection")
assert rows[1].get("exact_cache_and_untouched_slots")
print(json.dumps({"torch": torch.__version__, "device": torch.xpu.get_device_name(), "cases": rows}, indent=2))
