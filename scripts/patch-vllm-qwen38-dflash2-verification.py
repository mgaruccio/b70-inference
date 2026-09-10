#!/usr/bin/env python3
"""Pinned, opt-in verification prefixes for the legacy async DFlash2 path.

Apply after the deployed prefill guard and BF16 overlay in a disposable
73029d424 image (guard -> xpu_prefill -> BF16 -> xpu_boundary -> cache_groups
-> verification). The original unguarded GPU runner is deliberately rejected.
B70_DFLASH2_VERIFY_CAP must be absent (off) or exactly 1, 3, 7, or adaptive.
SchedulerOutput.num_spec_tokens_to_schedule, lookahead allocation, GPU draft
storage, and GDN rollback slots remain K=7. Only the NEXT step's read-only
placeholder list is shortened; current-step counters use its actual schedule.
Cap 7 preserves the original scheduling outputs. Adaptive compares 7 and 3
using completed useful tokens / CPU completion intervals, NOT GPU timings or
acceptance rates. It settles for two same-depth intervals, scores eight rounds
per arm, requires a 3% win, and re-probes after 24 valid exploitation rounds.
State is scheduler-local and transient; one summary is logged upon removal.

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
# Whole source pins: unchanged scheduler, guarded GPU runner, and the reviewed
# BF16 config/proposers. These files are validated, NEVER rewritten here.
# GPU prefill guard script SHA256:
# baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad
DEPENDENCY_SHA256 = {
    "v1/core/sched/scheduler.py": "35758b60df936ee004b22a5faa0c40ea42e8ce3c628343a40bda080d74f1c203",
    "v1/worker/gpu_model_runner.py": "00f22cb5fe8bc2f05cc93b0500faaa55d8b3a77754fde3e23767648f621f4dd4",
    "config/speculative.py": "b2d69c5dcfd0de5e66f1b12b3739095b220a0f2bebf96efb686e33c391cf9e78",
    "v1/spec_decode/llm_base_proposer.py": "768947f4102302374f9b19dad1f08818247e88b099ebdfbd86b506990f8eebcf",
    "v1/spec_decode/dflash.py": "39dfccdc124139530d3a1e798457bfea50743496763b665d0887259bdc5236be",
}

INIT = '''        # B70_DFLASH2_VERIFY_PREFIX: next-step verification, NOT generation.
        self._b70_dflash2_verify_cap = None
        import os as _b70_os

        cap = _b70_os.environ.get("B70_DFLASH2_VERIFY_CAP")
        if cap is not None:
            if cap not in ("1", "3", "7", "adaptive"):
                raise ValueError("B70_DFLASH2_VERIFY_CAP must be exactly 1, 3, or 7, or adaptive")
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
                        # Pinned SpeculativeConfig normalizes DFlash to True.
                        or spec.parallel_drafting is not True
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
            self._b70_dflash2_verify_cap = cap if cap == "adaptive" else int(cap)
            if cap == "adaptive":
                self._b70_dflash2_verify_states = {}
                self._b70_dflash2_verify_pending = {}
'''

# These helpers are inserted into the pinned module and covered by exact replay.
# Pending metadata stays on the scheduler, never on the worker-bound output.
ADAPTIVE_HELPERS = '''import json as _b70_json
import math as _b70_math
from time import perf_counter as _b70_perf_counter
from weakref import ref as _b70_ref


class _B70DFlash2VerifyState:
    def __init__(self, request):
        self.request = request
        self.num_preemptions = request.num_preemptions
        self.pending_cap = 7
        self.current_k = None
        self.seen_decode = False
        self.incumbent = 7
        self.phase = "baseline"
        self.last_completion = None
        self.settle = 2
        self.window = [0, 0, 0.0]  # rounds, useful emitted tokens, CPU seconds
        self.reference = None
        self.totals = {k: [0, 0, 0.0] for k in (3, 7)}
        self.scheduled = {}
        self.selections = {3: 0, 7: 0}
        self.switches = {"7->3": 0, "3->7": 0}
        self.decisions = {3: 0, 7: 0}
        self.ignored = {}

    def ignore(self, reason):
        self.ignored[reason] = self.ignored.get(reason, 0) + 1
        # Never charge prefill, a stale/mixed-depth interval, or a gap to an arm.
        self.last_completion = None
        self.settle = 2

    def select(self, cap):
        if cap != self.pending_cap:
            self.switches[f"{self.pending_cap}->{cap}"] += 1
            self.pending_cap = cap
            self.last_completion = None
            self.settle = 2

    def observe(self, depth, tokens, now):
        if depth not in (3, 7):
            self.ignore("clipped_depth")
            return
        if depth != self.pending_cap:
            self.ignore("depth_mismatch")
            return
        if not 1 <= tokens <= depth + 1:
            self.ignore("token_bounds")
            return
        if not _b70_math.isfinite(now):
            self.ignore("invalid_interval")
            return
        previous = self.last_completion
        self.last_completion = now
        if previous is None:
            self.ignored["interval_start"] = self.ignored.get("interval_start", 0) + 1
            return
        elapsed = now - previous
        if (not _b70_math.isfinite(elapsed) or elapsed <= 0
                or not _b70_math.isfinite(self.totals[depth][2] + elapsed)):
            self.ignore("invalid_interval")
            return
        if self.settle:
            self.settle -= 1
            self.ignored["settle"] = self.ignored.get("settle", 0) + 1
            return
        for counters in (self.window, self.totals[depth]):
            counters[0] += 1
            counters[1] += tokens  # Includes the bonus token, not acceptance %.
            counters[2] += elapsed
        target_rounds = 24 if self.phase == "exploit" else 8
        if self.window[0] < target_rounds:
            return
        if self.phase in ("baseline", "exploit"):
            self.reference = self.window
            self.phase = "probe"
            self.select(3 if self.incumbent == 7 else 7)
        else:
            # Compare fresh windows with hysteresis against the incumbent, not
            # against the arm currently being probed. Cross-multiply rates.
            if (self.window[1] * self.reference[2]
                    > 1.03 * self.reference[1] * self.window[2]):
                self.incumbent = self.pending_cap
            self.decisions[self.incumbent] += 1
            self.phase = "exploit"
            self.select(self.incumbent)
        self.window = [0, 0, 0.0]

    def summary(self, reason):
        return {
            "request_id": self.request.request_id, "reason": reason,
            "metric": "useful_tokens_per_cpu_completion_second",
            "scheduled_depths": self.scheduled, "next_cap_selections": self.selections,
            "pending_cap": self.pending_cap, "current_k": self.current_k,
            "incumbent": self.incumbent, "phase": self.phase,
            "switches": self.switches, "decisions": self.decisions,
            "scored": {k: dict(zip(("rounds", "tokens", "seconds"), values))
                       for k, values in self.totals.items()},
            "ignored": self.ignored,
        }
'''

ADAPTIVE_METHODS = '''    def _b70_dflash2_reap(self):
        states = self._b70_dflash2_verify_states
        for req_id, state in list(states.items()):
            request = self.requests.get(req_id)
            reason = None
            if request is not state.request:
                reason = "removed" if request is None else "replaced"
            elif request.is_finished():
                reason = "finished"
            elif request.num_preemptions != state.num_preemptions:
                reason = "preempted"
            elif request.status != RequestStatus.RUNNING:
                reason = "inactive"
            if reason is not None:
                del states[req_id]
                logger.info("B70_DFLASH2_ADAPTIVE %s", _b70_json.dumps(
                    state.summary(reason), separators=(",", ":"), allow_nan=False))
        pending = self._b70_dflash2_verify_pending
        for key, (output_ref, samples) in list(pending.items()):
            live = [sample for sample in samples if states.get(sample[0]) is sample[1]]
            if output_ref() is None:
                for _, state, _, _, _ in live:
                    state.ignore("abandoned_output")
                live = []
            if not live:
                del pending[key]
            elif len(live) != len(samples):
                pending[key] = (output_ref, live)

    def _b70_dflash2_prepare(self, scheduler_output):
        self._b70_dflash2_reap()
        samples = []
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            state = self._b70_dflash2_verify_states.get(req_id)
            if state is None:
                state = _B70DFlash2VerifyState(request)
                self._b70_dflash2_verify_states[req_id] = state
            depth = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            state.current_k = depth
            state.scheduled[depth] = state.scheduled.get(depth, 0) + 1
            # Snapshot BEFORE the parent advances counters; is_prefill_chunk at
            # completion can describe a newer, already queued schedule instead.
            prefill = request.num_computed_tokens < request.num_prompt_tokens
            first_decode = not prefill and not state.seen_decode
            if not prefill:
                state.seen_decode = True
            samples.append((req_id, state, depth, prefill, first_decode))
        if samples:
            self._b70_dflash2_verify_pending[id(scheduler_output)] = (
                _b70_ref(scheduler_output), samples)

    def update_from_output(self, scheduler_output, model_runner_output):
        if self._b70_dflash2_verify_cap != "adaptive":
            return super().update_from_output(scheduler_output, model_runner_output)
        self._b70_dflash2_reap()
        pending = self._b70_dflash2_verify_pending.pop(id(scheduler_output), None)
        samples = []
        now = _b70_perf_counter()
        if pending is not None and pending[0]() is scheduler_output:
            for req_id, state, depth, prefill, first_decode in pending[1]:
                request = state.request
                reason = None
                if request.num_stale_output_tokens > 0:
                    reason = "stale_output"
                elif prefill:
                    reason = "prefill"
                elif first_decode:
                    reason = "first_decode"
                index = model_runner_output.req_id_to_index.get(req_id)
                ids = model_runner_output.sampled_token_ids
                # Snapshot only lengths: the unchanged parent may truncate the
                # actual list at EOS/length stop. No token values are retained.
                count = len(ids[index]) if ids and index is not None else 0
                samples.append((req_id, state, depth, count, reason))
        result = super().update_from_output(scheduler_output, model_runner_output)
        for req_id, state, depth, count, reason in samples:
            request = state.request
            if request.is_finished() or self.requests.get(req_id) is not request:
                state.ignore("terminal")
            elif (request.num_preemptions != state.num_preemptions
                    or request.status != RequestStatus.RUNNING):
                state.ignore("inactive")
            elif reason is not None:
                state.ignore(reason)
            else:
                state.observe(depth, count, now)
        self._b70_dflash2_reap()
        return result

'''

STEP_GUARD = '''        if self._b70_dflash2_verify_cap is not None:
            if (type(scheduler_output.num_spec_tokens_to_schedule) is not int
                    or scheduler_output.num_spec_tokens_to_schedule != 7
                    or self.dynamic_sd_lookup is not None):
                # Refuse before the parent's counter updates; never pass a
                # reduced generation width to the proposer, even on empty steps.
                raise RuntimeError("B70 verification cap requires generation K7 without dynamic SD")
        if self._b70_dflash2_verify_cap == "adaptive":
            self._b70_dflash2_prepare(scheduler_output)
'''

PLACEHOLDERS = '''        self._spec_token_placeholders = [
            -1
        ] * scheduler_output.num_spec_tokens_to_schedule
'''
CAPPED_PLACEHOLDERS = '''        self._spec_token_placeholders = [-1] * (
            scheduler_output.num_spec_tokens_to_schedule
            if self._b70_dflash2_verify_cap in (None, "adaptive")
            else self._b70_dflash2_verify_cap
        )
'''

REQUEST_PLACEHOLDERS = '''            if self._b70_dflash2_verify_cap == "adaptive":
                state = self._b70_dflash2_verify_states[req_id]
                state.selections[state.pending_cap] += 1
                # Never resize a list referenced by an in-flight schedule or a
                # different request. Current-step counters still use actual K.
                request.spec_token_ids = [-1] * state.pending_cap
            else:
                request.spec_token_ids = self._spec_token_placeholders
'''


def transformations() -> tuple[tuple[str, str], ...]:
    return (
        ("class AsyncScheduler(Scheduler):\n",
         ADAPTIVE_HELPERS + "\n\nclass AsyncScheduler(Scheduler):\n"),
        ("        # reusable read-only placeholder list for speculative decoding.\n",
         INIT + "        # reusable read-only placeholder list for speculative decoding.\n"),
        ("        super()._update_after_schedule(scheduler_output)\n",
         STEP_GUARD + "        super()._update_after_schedule(scheduler_output)\n"),
        (PLACEHOLDERS, CAPPED_PLACEHOLDERS),
        ("            request.spec_token_ids = self._spec_token_placeholders\n",
         REQUEST_PLACEHOLDERS),
        ("    def _update_request_with_output(\n",
         ADAPTIVE_METHODS + "    def _update_request_with_output(\n"),
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
