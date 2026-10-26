#!/usr/bin/env python3
"""Diagnostic trace after all unprofiled candidate measurements have completed."""
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
root = Path(sys.argv[1])
out = root / "draft-graphs"
spec = importlib.util.spec_from_file_location("probe", root / "control/eagle-source/tests/intel15_http_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
body = probe._request_body("Qwen3.8-27B", json.loads((root / "fixture.json").read_text())["prompts"]["32768"])
body["max_tokens"] = 128
raw = bytearray()
pending = b""
started = False
base = "http://127.0.0.1:18110"
def post(path):
    with urllib.request.urlopen(urllib.request.Request(base + path, data=b"", method="POST"), timeout=120) as r:
        assert r.status == 200
        r.read()
req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as response:
        while chunk := response.read1(65536):
            raw.extend(chunk)
            pending += chunk
            while b"\n\n" in pending:
                block, pending = pending.split(b"\n\n", 1)
                for line in block.splitlines():
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]": continue
                    event = json.loads(line[5:])
                    if not started and any(c.get("token_ids") for c in event.get("choices", [])):
                        post("/start_profile")
                        started = True
    result = probe.parse_stream([bytes(raw)])
    assert started and len(result["token_ids"]) == 128 and result["finish_reason"] == "length"
    assert result["usage"]["prompt_tokens"] == 32768
    (out / "profile-request.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Candidate profile completed after all unprofiled measurements", flush=True)
finally:
    (out / "profile-request.sse").write_bytes(raw)
    if started: post("/stop_profile")
