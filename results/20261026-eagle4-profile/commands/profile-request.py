#!/usr/bin/env python3
"""Profile a bounded warmed real HTTP decode, after observing its first token."""
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
control = root / "control"
spec = importlib.util.spec_from_file_location("probe", control / "eagle-source/tests/intel15_http_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
base = "http://127.0.0.1:18110"
fixture = json.loads((root / "fixture.json").read_text())

def post(path):
    req = urllib.request.Request(base + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return {"status": r.status, "body": r.read().decode()}

# Excluded warmup: compile both public input classes before enabling tracing.
for length in (8192, 32768):
    body = probe._request_body("Qwen3.8-27B", fixture["prompts"][str(length)])
    body["max_tokens"] = 128
    result, _, started, finished = probe.stream_completion(
        base, body, 600, control / f"profile-warmup-{length}.sse")
    print(json.dumps({"warmup": length, "seconds": finished - started}), flush=True)

for length in (8192, 32768):
    body = probe._request_body("Qwen3.8-27B", fixture["prompts"][str(length)])
    body["max_tokens"] = 128
    raw = bytearray()
    pending = b""
    triggered = False
    started = time.monotonic()
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            while chunk := response.read1(65536):
                raw.extend(chunk)
                pending += chunk
                while b"\n\n" in pending:
                    block, pending = pending.split(b"\n\n", 1)
                    for line in block.splitlines():
                        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                            continue
                        event = json.loads(line[5:])
                        has_tokens = any(c.get("token_ids") for c in event.get("choices", []))
                        if has_tokens and not triggered:
                            triggered = True
                            print(json.dumps({"length": length, "first_token_seconds": time.monotonic() - started,
                                              "start_profile": post("/start_profile")}), flush=True)
        result = probe.parse_stream([bytes(raw)])
        assert len(result["token_ids"]) == 128 and result["finish_reason"] == "length"
        assert result["usage"]["prompt_tokens"] == length
        assert triggered, "No token-ID event triggered profiling"
        print(json.dumps({"length": length, "stop_profile": post("/stop_profile")}), flush=True)
        (control / f"profile-request-{length}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        (control / f"profile-request-{length}.sse").write_bytes(raw)
