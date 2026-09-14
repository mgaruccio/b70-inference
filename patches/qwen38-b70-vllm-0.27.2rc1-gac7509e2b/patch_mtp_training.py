#!/usr/bin/env python3
"""Install the opt-in MTP adapter into a disposable pinned vLLM container only.

Run AFTER the live five patches (either RTN patch order also remains valid):
    python /patches/patch_mtp_training.py
No model files, target forwards, shared weights, or production launchers change.
"""
from __future__ import annotations

from pathlib import Path


OVERLAY_MARKER = "B70_MTP_TRAINING_OVERLAY"
CAPTURE_MARKER = "B70_MTP_TRAINING_CAPTURE"
OVERLAY_OLD = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names(weights):
'''
OVERLAY_NEW = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # B70_MTP_TRAINING_OVERLAY: before HF remapping, packing, and draft RTN.
        from vllm.model_executor.models.b70_mtp_training import overlay_weights
        weights = overlay_weights(weights)
        def remap_weight_names(weights):
'''
CAPTURE_OLD = '''        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
'''
CAPTURE_NEW = '''        # B70_MTP_TRAINING_CAPTURE: outside model compilation / XPU graph execution.
        from vllm.model_executor.models.b70_mtp_training import capture_replay_step
        capture_replay_step(self, scheduler_output, model_output)

        with record_function_or_nullcontext("gpu_model_runner: postprocess"):
'''


def patch(root: Path) -> None:
    changes = []
    for relative, marker, old, new in (
        ("model_executor/models/qwen3_5_mtp.py", OVERLAY_MARKER, OVERLAY_OLD, OVERLAY_NEW),
        ("v1/worker/gpu_model_runner.py", CAPTURE_MARKER, CAPTURE_OLD, CAPTURE_NEW),
    ):
        path = root / relative
        source = path.read_text()
        if marker in source:
            if source.count(new) != 1:
                raise RuntimeError(f"incompatible existing {marker}: {path}")
        else:
            if source.count(old) != 1:
                raise RuntimeError(f"pinned MTP training anchor mismatch: {path}")
            source = source.replace(old, new, 1)
        compile(source, str(path), "exec")
        changes.append((path, source))

    helper = Path(__file__).with_name("b70_mtp_training.py").read_text()
    compile(helper, "b70_mtp_training.py", "exec")
    # All anchors and syntax are checked before any installation writes.
    (root / "model_executor/models/b70_mtp_training.py").write_text(helper)
    for path, source in changes:
        path.write_text(source)
    print(f"{OVERLAY_MARKER}, {CAPTURE_MARKER}: {root}", flush=True)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
