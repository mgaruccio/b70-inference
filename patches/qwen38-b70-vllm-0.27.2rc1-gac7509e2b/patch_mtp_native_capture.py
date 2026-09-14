#!/usr/bin/env python3
"""Install one opt-in native MTP capture seam in the disposable pinned image.

Apply after deployed patches. No target/model weights or persistent launchers
change; B70_MTP_NATIVE_CAPTURE_DIR unset means no collector or tensor copies.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "B70_MTP_NATIVE_CAPTURE_SEAM"
OLD = '''        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
'''
NEW = OLD + '''        # B70_MTP_NATIVE_CAPTURE_SEAM: before proposer/scratch reuse; current GPU sample.
        from vllm.model_executor.models.b70_mtp_native_capture import capture_native_step
        capture_native_step(
            self, scheduler_output, spec_decode_metadata, hidden_states, sampler_output
        )
'''


def patched_source(source):
    if source.count(OLD) != 1:
        raise RuntimeError("pinned native MTP sampler anchor must occur exactly once")
    if MARKER in source:
        if source.count(MARKER) != 1 or source.count(NEW) != 1:
            raise RuntimeError("incompatible existing native MTP capture seam")
    else:
        source = source.replace(OLD, NEW, 1)
    compile(source, "gpu_model_runner.py", "exec")
    return source


def patch(root):
    root = Path(root)
    target = root / "v1/worker/gpu_model_runner.py"
    source = patched_source(target.read_text())
    helper = Path(__file__).with_name("b70_mtp_native_capture.py").read_text()
    compile(helper, "b70_mtp_native_capture.py", "exec")
    # Validate both complete sources before either installation write.
    (root / "model_executor/models/b70_mtp_native_capture.py").write_text(helper)
    target.write_text(source)
    print(MARKER + ": installed", flush=True)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
