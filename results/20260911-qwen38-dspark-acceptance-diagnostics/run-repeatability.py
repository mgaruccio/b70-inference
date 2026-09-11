#!/usr/bin/env python3
"""Fixed-seed public API repetition using the unchanged diagnostic server driver."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def repeat_matrix(client, probe):
    rows = []
    for family, prompt, thinking in (("code", probe.CODE, False), ("prose", probe.PROSE, True)):
        for repeat in range(4):
            # Labels differ only for artifact filenames; payloads are identical.
            row = client.chat(
                f"repeat-{family}-{repeat}", prompt, 512, True,
                temperature=0.0, top_p=1.0, top_k=-1, seed=42,
                thinking=thinking, cache_salt=f"fixed-repeat-{family}",
            )
            rows.append(row)
    summary = {"expected_requests": 8, "completed_requests": len(rows),
               "all_transport_pass": all(r["transport_pass"] for r in rows),
               "seed": 42, "temperature": 0, "forced_output_tokens": 512,
               "families": {}}
    for family in ("code", "prose"):
        group = [r for r in rows if r["label"].startswith(f"repeat-{family}-")]
        outputs = [r["token_ids"] for r in group]
        summary["families"][family] = {
            "distinct_outputs": len({tuple(ids) for ids in outputs}),
            "first_divergence_from_first": [next((i for i, (a, b) in enumerate(zip(outputs[0], ids)) if a != b), None) for ids in outputs],
            "prompt_ids_identical": all(r["prompt_token_ids"] == group[0]["prompt_token_ids"] for r in group),
            "output_sha256": [r["output_sha256"] for r in group],
        }
        assert summary["families"][family]["prompt_ids_identical"]
    (client.out / "repeatability.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    driver = runpy.run_path(str(ROOT / "run-acceptance-diagnostics.py"))
    driver["run"].__globals__["run_matrix"] = repeat_matrix
    driver["run"].__globals__["MATRIX_REQUESTS"] = 8
    sys.exit(driver["main"]())
