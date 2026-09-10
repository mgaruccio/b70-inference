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
OUT = Path(__file__).resolve().parent / "target-v2-diagnostic"
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
    argv = [
        "docker", "run", "--rm", "--name", NAME, "--ipc=host",
        "-p", "127.0.0.1:8000:8000", "--device", "/dev/dri",
        "--group-add", str(Path("/dev/dri/renderD128").stat().st_gid),
        "-v", "/dev/dri:/dev/dri:ro", "-v", f"{TARGET}:/model:ro",
        "-e", "VLLM_USE_V2_MODEL_RUNNER=1", "-e", "VLLM_TARGET_DEVICE=xpu",
        "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", "-e", "ZE_AFFINITY_MASK=0",
        "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True", "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=0",
        "--entrypoint", "bash", IMAGE, "-lc", "exec " + shlex.join(serve),
    ]
    probe.save(OUT / "launch-argv.json", argv)
    (OUT / "launcher.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(argv) + "\n")
    (OUT / "image-inspect.json").write_text(command("docker", "image", "inspect", IMAGE))
    summary = {"tier": "development", "status": "starting", "image": IMAGE,
               "context": 8192, "speculation": False, "runner": "v2", "patches": [],
               "purpose": "XPU V2 target prerequisite, not a DSpark benchmark"}
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
            probe.Cell.gates(cell)
            # Exercise multiple decode steps, not just one-token boundary probes.
            cell.chat("decode-128", "Explain why a Python function should avoid mutable default arguments.", 128)
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
