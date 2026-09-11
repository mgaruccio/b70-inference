#!/usr/bin/env python3
"""Bounded XPU V2 target prerequisite, executed only on inference-host."""
import functools
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import qwen38_lossy_probe as probe

IMAGE = "vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4"
TARGET = "/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
NAME = "qwen38-dspark-v2-prerequisite"
OUT = Path(__file__).resolve().parent / "dspark-v2-cache-fix-smoke"
LAUNCHER = Path("/home/mike/inference/launchers/start-qwen38.sh")
POWER = Path("/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap")


def command(*argv):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=30)


def invariants():
    return {
        "launcher_sha256": hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(),
        "power_cap": POWER.read_text().strip(),
        "running_containers": command("docker", "ps", "--format", "{{.Names}}").splitlines(),
        "glimmer_running": command("docker", "inspect", "-f", "{{.State.Running}}", "glimmer-tb21-prefix-c8").strip(),
    }


def main():
    OUT.mkdir(exist_ok=False)
    before = invariants()
    probe.save(OUT / "host-before.json", before)
    if before != {"launcher_sha256": probe.LAUNCHER_SHA, "power_cap": "275000000", "running_containers": [], "glimmer_running": "false"}:
        raise RuntimeError(f"host prerequisites failed: {before}")
    serve = [
        "vllm", "serve", "/model", "--quantization", "gptq", "--dtype", "float16",
        "--max-model-len", "8192", "--gpu-memory-utilization", "0.95",
        "--kv-cache-dtype", "fp8", "--port", "8000", "--max-num-seqs", "1",
        "--max-num-batched-tokens", "2048", "--no-enable-prefix-caching",
        "--mamba-cache-mode", "align", "--performance-mode", "balanced",
        "--chat-template-content-format", "openai",
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
        "--reasoning-parser", "qwen3", "--served-model-name", "qwen38",
        "--language-model-only", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
        "--enforce-eager",
    ]
    spec_config = {"method": "dspark", "model": "/draft", "num_speculative_tokens": 7,
                   "kv_cache_dtype": "bfloat16", "quantization": None,
                   "rejection_sample_method": "standard", "draft_sample_method": "greedy",
                   "enable_adaptive_verification": False}
    serve += ["--speculative-config", json.dumps(spec_config)]
    argv = [
        "docker", "run", "--rm", "--name", NAME, "--ipc=host",
        "-p", "127.0.0.1:8000:8000", "--device", "/dev/dri",
        "--group-add", str(Path("/dev/dri/renderD128").stat().st_gid),
        "-v", "/dev/dri:/dev/dri:ro", "-v", f"{TARGET}:/model:ro",
        "-e", "VLLM_USE_V2_MODEL_RUNNER=1", "-e", "VLLM_TARGET_DEVICE=xpu",
        "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", "-e", "ZE_AFFINITY_MASK=0",
        "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True", "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=0",
        "-v", f"{Path(__file__).resolve().parent}:/experiment:ro", "-v", f"{OUT}:/output",
        "-v", f"{Path(__file__).resolve().parent / 'draft'}:/draft:ro",
        "-e", "B70_DSPARK_BF16=1",
        "--entrypoint", "bash", IMAGE, "-lc",
        "set -e; /opt/venv/bin/python -P /experiment/apply-prefill.py; "
        "/opt/venv/bin/python -P /experiment/patch_dspark_bf16.py; exec " + shlex.join(serve),
    ]
    probe.save(OUT / "launch-argv.json", argv)
    (OUT / "launcher.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(argv) + "\n")
    (OUT / "image-inspect.json").write_text(command("docker", "image", "inspect", IMAGE))
    summary = {"tier": "development", "status": "starting", "image": IMAGE,
               "context": 8192, "speculation": True, "runner": "v2",
               "patches": ["existing-xpu-short-prefill", "dspark-bf16"],
               "speculative_config": spec_config,
               "purpose": "DSpark V2 mixed-dtype correctness smoke, not a performance benchmark"}
    cell = SimpleNamespace(out=OUT, rows=[], summary=summary)
    cell.chat = functools.partial(probe.Cell.chat, cell)
    proc = None
    failure = None
    try:
        with (OUT / "server.log").open("w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {proc.returncode}")
                try:
                    probe.get("/health")
                    break
                except OSError:
                    time.sleep(3)
            else:
                raise TimeoutError("startup exceeded 900 seconds")
            models = json.loads(probe.get("/v1/models"))
            probe.save(OUT / "models.json", models)
            if not any(m["id"] == "qwen38" and m.get("max_model_len") == 8192 for m in models["data"]):
                raise RuntimeError("wrong served model/context")
            source_check = '''import hashlib, importlib.util, json, runpy
from pathlib import Path
root = Path(importlib.util.find_spec("vllm").origin).parent
assert str(root).startswith("/opt/venv/")
overlay = runpy.run_path("/experiment/patch_dspark_bf16.py")
sources = {p: (root / p).read_bytes().decode() for p in overlay["PINNED_SHA256"]}
assert overlay["prepare"](sources) == sources
assert hashlib.sha256((root / "v1/attention/backends/gdn_attn.py").read_bytes()).hexdigest() == "5173f3394c1385d215bd99f0d12290e8336da844b96da612954da01d62a0b062"
print(json.dumps({"root": str(root), "exact_replay": True, "sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}}))
'''
            source_argv = ["docker", "exec", NAME, "/opt/venv/bin/python", "-P", "-c", source_check]
            probe.save(OUT / "source-check-argv.json", source_argv)
            probe.save(OUT / "effective-dspark-source.json", json.loads(command(*source_argv)))
            import runpy
            checks = runpy.run_path(str(Path(__file__).resolve().parent / "dspark-smoke-checks.py"))
            summary["api_checks"] = checks["run_checks"](OUT, speculative=True)
            summary["status"] = "passed"
    except Exception as exc:
        if hasattr(exc, "read"):
            (OUT / "http-error-body.txt").write_bytes(exc.read())
        failure = traceback.format_exc()
        (OUT / "failure.txt").write_text(failure)
        summary["status"] = "failed"
        summary["error"] = failure
    finally:
        # Remove only the container owned by this experiment. Never stop another workload.
        cleanup = subprocess.run(["docker", "rm", "-f", NAME], text=True, capture_output=True, timeout=60)
        probe.save(OUT / "container-cleanup.json", {"returncode": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr})
        if proc is not None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        after = invariants()
        probe.save(OUT / "host-after.json", after)
        summary["host_unchanged"] = after == before
        probe.save(OUT / "summary.json", summary)
        if after != before:
            raise RuntimeError("host invariants changed")
    print(json.dumps(summary), flush=True)
    if failure:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
