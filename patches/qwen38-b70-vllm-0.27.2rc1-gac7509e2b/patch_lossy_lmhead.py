#!/usr/bin/env python3
"""Research-only target-head capture/GPTQ overlay; never rewrites checkpoint weights."""
from __future__ import annotations

from pathlib import Path

MARKER = "B70_LOSSY_LMHEAD_RESEARCH"
OLD = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
'''
NEW = '''    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # B70_LOSSY_LMHEAD_RESEARCH: opt-in disposable experiment only.
        from vllm.model_executor.models.b70_lossy_lmhead import head_logits
        return head_logits(self, hidden_states)
'''
HELPER = '''\
"""Opt-in B70 target-head experiment. Dense target remains the default."""
import json
import os
from pathlib import Path

import torch


class PackedHeadMethod:
    def __init__(self, path, weight):
        packed = torch.load(path, map_location="cpu", weights_only=True)
        n, k = weight.shape
        q = packed["qweight"]
        s = packed["scales"]
        z = packed["qzeros"]
        group = packed["group_size"]
        if group != 128 or q.shape != (k // 8, n) or s.shape != (k // group, n):
            raise ValueError("calibrated head has incompatible dimensions/groups")
        if q.dtype != torch.int32 or s.dtype != torch.float16:
            raise ValueError("calibrated head has incompatible storage dtypes")
        if z.dtype != torch.int8 or z.numel() != 1 or z.item() != 8:
            raise ValueError("calibrated head requires symmetric zero 8")
        if not torch.isfinite(s).all() or not (s > 0).all():
            raise ValueError("calibrated head has invalid scales")
        self.qweight = q.t().contiguous().t().to(weight.device)
        if self.qweight.stride() != (1, k // 8):
            raise ValueError("calibrated head lost the XPU NT packing stride")
        self.scales = s.contiguous().to(weight.device)
        self.qzeros = z.to(weight.device)
        self.group_size = group
        print("B70_CALIBRATED_TARGET_HEAD_READY=1", flush=True)

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("research head does not support a bias")
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        logits = torch.ops._xpu_C.int4_gemm_w4a16(
            flat, self.qweight, None, self.scales, self.qzeros, self.group_size, None
        )
        return logits.reshape(*x.shape[:-1], self.qweight.shape[1])


def capture(model, hidden_states, root):
    control = Path(root) / "capture-request.json"
    if not control.is_file():
        return
    request = json.loads(control.read_text())
    split = request["split"]
    name = request["name"]
    if split not in ("calibration", "heldout") or not name.replace("-", "").isalnum():
        raise ValueError("invalid capture request")
    counts = getattr(model, "_b70_capture_counts", {})
    count = counts.get(split, 0)
    limit = 16384 if split == "calibration" else 4096
    rows = hidden_states.reshape(-1, hidden_states.shape[-1])[:max(0, limit - count)]
    if not rows.numel():
        return
    cpu = rows.detach().to("cpu")
    if not torch.isfinite(cpu).all():
        raise ValueError("nonfinite real hidden state")
    directory = Path(root) / split
    directory.mkdir(exist_ok=True)
    torch.save({"hidden_states": cpu, "request": name}, directory / f"{name}-{count:06d}.pt")
    counts[split] = count + len(cpu)
    model._b70_capture_counts = counts


def head_logits(model, hidden_states):
    mode = os.environ.get("B70_TARGET_HEAD_MODE", "dense")
    if mode not in ("dense", "capture", "gptq"):
        raise ValueError("unknown research target-head mode")
    if mode == "gptq" and not isinstance(model.lm_head.quant_method, PackedHeadMethod):
        model.lm_head.quant_method = PackedHeadMethod(
            os.environ["B70_TARGET_HEAD_FILE"], model.lm_head.weight
        )
    logits = model.logits_processor(model.lm_head, hidden_states)
    if mode == "capture":
        capture(model, hidden_states, os.environ["B70_HEAD_CAPTURE_ROOT"])
    return logits
'''


def patch(root: Path) -> None:
    path = root / "model_executor/models/qwen3_5.py"
    text = path.read_text()
    if MARKER not in text:
        if text.count(OLD) != 1:
            raise RuntimeError("pinned target compute_logits anchor mismatch")
        text = text.replace(OLD, NEW, 1)
    compile(text, str(path), "exec")
    compile(HELPER, "b70_lossy_lmhead.py", "exec")
    (path.parent / "b70_lossy_lmhead.py").write_text(HELPER)
    path.write_text(text)
    print(f"{MARKER}: {path}", flush=True)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
