#!/usr/bin/env python3
"""Pinned, opt-in fixed verification prefixes for the legacy async DFlash2 path.

Apply AFTER patch-vllm-qwen38-dflash2-bf16.py in a disposable 73029d424 image.
B70_DFLASH2_VERIFY_CAP must be absent (off) or exactly 1, 3, or 7. Generation,
SchedulerOutput.num_spec_tokens_to_schedule, lookahead allocation, GPU draft
storage, and GDN rollback slots remain K=7. Only the NEXT step's read-only
placeholder list is shortened; current-step counters use its actual schedule.
Cap 7 preserves the original scheduling outputs. No adaptive controller here.

Fresh upstream references (73029d42441321b631779db3475031f5ec26dd6c):
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/core/sched/async_scheduler.py
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/gpu_model_runner.py
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/spec_decode/llm_base_proposer.py
The GPU scatter uses actual scheduled draft_len but the previous generation
width as its stride. LLMB.propose assigns its K argument, so reducing the
SchedulerOutput generation count would incorrectly change generation too.

This patch only runs in AsyncScheduler. The experiment harness MUST explicitly
select --async-scheduling (no custom scheduler); a synchronous scheduler never
executes this opt-in. Source dependencies are validated before any write.
The existing GDN boundary overlay and native alternating-prefix/state-history
oracle are still required before public API testing. CPU tests are not proof
of native recurrent-state correctness. No persistent launcher changes.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path


MARKER = "B70_DFLASH2_VERIFY_PREFIX"
RELATIVE_PATH = "v1/core/sched/async_scheduler.py"
ORIGINAL_SHA256 = "e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508"
# Whole source pins: unchanged scheduler/scatter, and the reviewed BF16 overlay's
# config guards/proposers. These files are validated, NEVER rewritten here.
DEPENDENCY_SHA256 = {
    "v1/core/sched/scheduler.py": "35758b60df936ee004b22a5faa0c40ea42e8ce3c628343a40bda080d74f1c203",
    "v1/worker/gpu_model_runner.py": "d620deb484fee968aeefefcf8cc901cf2665118e1a2415ee1bb906fd697b3054",
    "config/speculative.py": "b2d69c5dcfd0de5e66f1b12b3739095b220a0f2bebf96efb686e33c391cf9e78",
    "v1/spec_decode/llm_base_proposer.py": "768947f4102302374f9b19dad1f08818247e88b099ebdfbd86b506990f8eebcf",
    "v1/spec_decode/dflash.py": "39dfccdc124139530d3a1e798457bfea50743496763b665d0887259bdc5236be",
}

INIT = '''        # B70_DFLASH2_VERIFY_PREFIX: fixed next-step verification, NOT generation.
        self._b70_dflash2_verify_cap = None
        import os as _b70_os

        cap = _b70_os.environ.get("B70_DFLASH2_VERIFY_CAP")
        if cap is not None:
            if cap not in ("1", "3", "7"):
                raise ValueError("B70_DFLASH2_VERIFY_CAP must be exactly 1, 3, or 7")
            from vllm.config.speculative import (
                _B70_DFLASH2_BF16,
                _b70_dflash2_enabled,
                _b70_dflash2_validate,
            )

            if (_b70_os.environ.get("B70_DFLASH2_BF16") != "1"
                    or not _B70_DFLASH2_BF16):
                raise ValueError("B70_DFLASH2_VERIFY_CAP requires B70_DFLASH2_BF16=1")
            try:
                config = self.vllm_config
                spec = config.speculative_config
                pc = self.parallel_config
                if (self.scheduler_config.async_scheduling is not True
                        or type(self.scheduler_config.max_num_seqs) is not int
                        or self.scheduler_config.max_num_seqs != 1
                        or self.use_v2_model_runner is not False
                        or config.use_v2_model_runner is not False
                        or type(self.num_spec_tokens) is not int
                        or self.num_spec_tokens != 7
                        or self.num_sampled_tokens_per_step != 1
                        or self.dynamic_sd_lookup is not None
                        or spec is None
                        or type(spec.num_speculative_tokens) is not int
                        or spec.num_speculative_tokens != 7
                        or spec.num_speculative_tokens_per_batch_size is not None
                        or spec.parallel_drafting is not False
                        or pc.tensor_parallel_size != 1
                        or pc.pipeline_parallel_size != 1
                        or pc.data_parallel_size != 1
                        or pc.decode_context_parallel_size != 1
                        or pc.prefill_context_parallel_size != 1
                        or config.lora_config is not None
                        or config.kv_transfer_config is not None
                        or config.ec_transfer_config is not None):
                    raise ValueError("B70 verification cap requires legacy async C1, "
                                     "fixed DFlash K7, single XPU, no LoRA/transfers")
                if not _b70_dflash2_enabled(spec):
                    raise ValueError("B70 verification cap requires the BF16 DFlash2 overlay")
                # Reuse its exact checkpoint, dtype, XPU, greedy/standard and
                # partial-INT4 guards rather than maintain a weaker duplicate.
                _b70_dflash2_validate(spec)
            except (AttributeError, KeyError, TypeError) as exc:
                raise ValueError("malformed B70 DFlash2 verification configuration") from exc
            self._b70_dflash2_verify_cap = int(cap)
'''

STEP_GUARD = '''        if self._b70_dflash2_verify_cap is not None:
            if (type(scheduler_output.num_spec_tokens_to_schedule) is not int
                    or scheduler_output.num_spec_tokens_to_schedule != 7
                    or self.dynamic_sd_lookup is not None):
                # Refuse before the parent's counter updates; never pass a
                # reduced generation width to the proposer, even on empty steps.
                raise RuntimeError("B70 verification cap requires generation K7 without dynamic SD")
'''

PLACEHOLDERS = '''        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
'''
CAPPED_PLACEHOLDERS = '''        self._spec_token_placeholders = [-1] * (
            scheduler_output.num_spec_tokens_to_schedule
            if self._b70_dflash2_verify_cap is None
            else self._b70_dflash2_verify_cap
        )
'''


def transformations() -> tuple[tuple[str, str], ...]:
    return (
        ("        # reusable read-only placeholder list for speculative decoding.\n",
         INIT + "        # reusable read-only placeholder list for speculative decoding.\n"),
        ("        super()._update_after_schedule(scheduler_output)\n",
         STEP_GUARD + "        super()._update_after_schedule(scheduler_output)\n"),
        (PLACEHOLDERS, CAPPED_PLACEHOLDERS),
    )


def patch_text(source: str) -> str:
    """Fingerprint the entire source; exact reconstruction makes replay strict."""
    original = source
    is_patched = MARKER in source
    if is_patched:
        if source.count(MARKER) != 1:
            raise RuntimeError("duplicate/moved verification overlay marker")
        for old, new in reversed(transformations()):
            if original.count(new) != 1:
                raise RuntimeError("partial/changed verification overlay")
            original = original.replace(new, old, 1)
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("async scheduler differs from pinned 73029d424 source")
    result = original
    for old, new in transformations():
        if result.count(old) != 1:
            raise RuntimeError("pinned verification insertion anchor changed")
        result = result.replace(old, new, 1)
    if is_patched and result != source:
        raise RuntimeError("verification overlay replay differs")
    compile(result, RELATIVE_PATH, "exec")
    return result


def apply(root: Path) -> bool:
    """Validate every dependency and compile the result before the only write."""
    for relative, expected in DEPENDENCY_SHA256.items():
        source = (root / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise RuntimeError(f"{relative}: changed/missing pinned BF16 handoff dependency")
    path = root / RELATIVE_PATH
    before = path.read_bytes().decode("utf-8")
    after = patch_text(before)
    if after == before:
        return False
    path.write_bytes(after.encode("utf-8"))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="vllm package root; omit inside serving image")
    args = parser.parse_args()
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vllm package not found")
        root = Path(spec.origin).parent
    changed = apply(root)
    print(f"{MARKER}: {'patched' if changed else 'already patched'} {root / RELATIVE_PATH}",
          flush=True)


if __name__ == "__main__":
    main()
