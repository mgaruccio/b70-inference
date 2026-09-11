"""Reproduce paired short-smoke input/output checks (not a quality benchmark)."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
control = root / "target-v2-control"
candidate = root / "dspark-v2-cache-fix-smoke"
rows = []
for request in sorted(control.glob("*-request.json")):
    name = request.name.removesuffix("-request.json")
    a = json.loads((control / f"{name}-result.json").read_text())
    b = json.loads((candidate / f"{name}-result.json").read_text())
    inputs_equal = json.loads(request.read_text()) == json.loads((candidate / request.name).read_text())
    prompt_ids_equal = a["prompt_token_ids"] == b["prompt_token_ids"]
    output_ids_equal = a["token_ids"] == b["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
                 min(len(a["token_ids"]), len(b["token_ids"])))
    rows.append({"label": name, "payload_equal": inputs_equal, "prompt_ids_equal": prompt_ids_equal,
                 "output_ids_equal": output_ids_equal, "control_tokens": len(a["token_ids"]),
                 "candidate_tokens": len(b["token_ids"]),
                 "first_divergence_zero_based": None if output_ids_equal else first,
                 "both_transport_pass": a["transport_pass"] and b["transport_pass"]})
assert len(rows) == 19
assert all(r["payload_equal"] and r["prompt_ids_equal"] and r["both_transport_pass"] for r in rows)
parity = [r for r in rows if r["label"].startswith("parity-")]
assert len(parity) == 8
checks = {name: json.loads((directory / "api-checks-summary.json").read_text())
          for name, directory in (("control", control), ("candidate", candidate))}
assert all(c["status"] == "passed" and c["finite_boundaries"] == 131 for c in checks.values())
assert all(all(r["pass"] for r in c["functional"]) and len(c["functional"]) == 8 for c in checks.values())
result = {"paired_requests": len(rows), "exact_output_matches": sum(r["output_ids_equal"] for r in rows),
          "parity_streams": len(parity), "exact_parity_matches": sum(r["output_ids_equal"] for r in parity),
          "repeatability": {k: c["repeatability"] for k, c in checks.items()},
          "candidate_acceptance": checks["candidate"]["acceptance"], "rows": rows}
print(json.dumps(result, indent=2))
if not all(r["output_ids_equal"] for r in rows):
    raise SystemExit(1)
