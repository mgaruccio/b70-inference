"""Public-API checks shared by the target control and DSpark smoke cell."""
import functools
from types import SimpleNamespace

import qwen38_lossy_probe as probe


def run_checks(out, *, speculative):
    summary = {"status": "running", "speculative": speculative}
    cell = SimpleNamespace(out=out, rows=[], summary=summary)
    cell.chat = functools.partial(probe.Cell.chat, cell)
    try:
        probe.Cell.gates(cell)
        probe.Cell.quality(cell)
        # Fixed prompt families and request labels across control/candidate.
        # Four repeats expose baseline repeatability before assigning divergence
        # to speculation. These are smoke outputs, not a performance benchmark.
        for family, prompt in (("code", probe.CODE), ("prose", probe.PROSE)):
            for repeat in range(4):
                cell.chat(f"parity-{family}-{repeat}", prompt, 128)
        parity_rows = [r for r in cell.rows if r["label"].startswith("parity-")]
        summary["repeatability"] = {
            family: len({tuple(r["token_ids"]) for r in parity_rows if r["label"].startswith(f"parity-{family}-")})
            for family in ("code", "prose")
        }
        def total(name):
            return sum(v for r in parity_rows for k, v in r["metric_deltas"].items()
                       if k.startswith("vllm:spec_decode_num_" + name + "_total{"))
        steps, accepted, drafted = total("drafts"), total("accepted_tokens"), total("draft_tokens")
        summary["acceptance"] = {"steps": steps, "accepted": accepted, "drafted": drafted,
                                  "mean_emitted_per_step": 1 + accepted / steps if steps else None}
        if speculative and not (steps > 0 and accepted > 0 and drafted > 0):
            raise RuntimeError("DSpark must show nonzero measured proposals and acceptance")
        if not speculative and (steps or accepted or drafted):
            raise RuntimeError("target-only control unexpectedly speculated")
        summary["status"] = "passed"
        return summary
    except Exception:
        summary["status"] = "failed"
        raise
    finally:
        probe.save(out / "api-checks-summary.json", summary)
