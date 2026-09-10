#!/usr/bin/env python3
"""Bounded no-op discrepancy diagnosis; same retained 512-token requests, A/B/A."""
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/scripts/experiments"))
import qwen38_dflash2_probe as runner
import qwen38_long_context_bench as cold


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, stop)
    original = json.loads((ROOT / "baseline/long-context/summary.json").read_text())["points"][0]
    candidate = json.loads((ROOT / "cap7/long-context/summary.json").read_text())["points"][0]
    requests = []
    for a, b in zip([original["warmup"], *original["measurements"]],
                    [candidate["warmup"], *candidate["measurements"]]):
        if a["stream"]["text"] != b["stream"]["text"]:
            payload = json.loads((ROOT / "baseline/long-context" / a["request_path"]).read_text())
            assert len(payload["prompt"]) == 512 and payload["seed"] == 42 and payload["temperature"] == 0
            requests.append((a["request_path"], payload))
    assert len(requests) == 2, "diagnostic selection changed"
    for name, cap in (("baseline-a", None), ("cap7", 7), ("baseline-b", None)):
        out = ROOT / "repeatability" / name
        if out.exists():
            raise RuntimeError(f"refusing to overwrite {out}")
        args = SimpleNamespace(out=out, mode="dflash2", context=180224, graph=True, audit=False,
            suite="standard", draft_int4=Path("/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-quant/rtn-int4-g128"),
            guard=Path("/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2/runner/patch_uniform_decode_prefill.py"),
            patch=ROOT / "code/scripts/patch-vllm-qwen38-dflash2-bf16.py",
            prefill_patch=ROOT / "code/scripts/patch-vllm-qwen38-xpu-prefill.py",
            cache_group_size=8, max_num_batched_tokens=2048, verification_cap=cap)
        cell = runner.DFlashCell(args)
        try:
            cell.start()
            cell.gates()
            cell.quality()
            code = '''import hashlib, importlib.util, json
from pathlib import Path
r=Path(importlib.util.find_spec("vllm").origin).parent
paths=["v1/core/sched/async_scheduler.py","v1/core/sched/scheduler.py","v1/worker/gpu_model_runner.py","v1/spec_decode/llm_base_proposer.py","v1/spec_decode/dflash.py","config/speculative.py","_xpu_ops.py"]
print(json.dumps({p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in paths}))
'''
            command = ["docker", "exec", "qwen38", "python", "-c", code]
            save(out / "source-capture-argv.json", command)
            save(out / "effective-source-sha256.json", json.loads(subprocess.check_output(command, text=True)))
            client = cold.PublicAPI("http://127.0.0.1:8000")
            rows = []
            for request_index, (source, payload) in enumerate(requests):
                for repeat in range(6):
                    dest = out / f"request-{request_index}/repeat-{repeat}"
                    save(dest / "request.json", payload)
                    events = []
                    try:
                        for timestamp, raw in client.stream("/v1/completions", payload):
                            events.append((timestamp, raw))
                    finally:
                        save(dest / "sse.json", [{"monotonic_s": t, "raw": r.decode("utf-8")} for t, r in events])
                    parsed = cold.parse_sse_events(events)
                    validation = cold.validate_token_counts(parsed.get("usage"), 512)
                    save(dest / "response.json", parsed)
                    if not validation["valid"] or parsed["parse_errors"]:
                        raise RuntimeError(f"invalid stream {dest}")
                    rows.append({"source": source, "request_index": request_index,
                                 "repeat": repeat, "text": parsed["text"]})
                    save(out / "replays.json", rows)
            cell.summary["status"] = "repeatability_completed"
        except BaseException as error:
            cell.summary.update(status="failed", error=repr(error))
            raise
        finally:
            cell.close()
    print("REPEATABILITY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
