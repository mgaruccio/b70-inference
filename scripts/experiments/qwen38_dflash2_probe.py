#!/usr/bin/env python3
"""Disposable strict DFlash2/target-only XPU cells; reuse the existing API probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
import urllib.error
from pathlib import Path

import qwen38_lossy_probe as probe

IMAGE = "vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4"
ROOT = Path("/home/mike/inference")
TARGET = ROOT / "models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
DRAFT = ROOT / "models/Qwen3.8-27B-DFlash2"


def server_command(args, out):
    """Return argv, never interpolate experiment paths into shell syntax."""
    argv = ["docker", "run", "--rm", "--name", "qwen38", "--ipc=host",
            "-p", "127.0.0.1:8000:8000", "--device", "/dev/dri",
            "--group-add", str(Path("/dev/dri/renderD128").stat().st_gid),
            "-v", "/dev/dri:/dev/dri:ro", "-v", f"{TARGET}:/model:ro",
            "-v", f"{out}:/profile",
            "-v", f"{out / 'patch_uniform_decode_prefill.py'}:/prefill_guard.py:ro",
            "-v", f"{out / 'patch_xpu_prefill.py'}:/gdn_prefill.py:ro",
            "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
            "-e", "VLLM_TARGET_DEVICE=xpu", "-e", "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
            "-e", "ZE_AFFINITY_MASK=0", "-e", "PYTORCH_ALLOC_CONF=expandable_segments:True",
            "-e", "VLLM_XPU_ENABLE_XPU_GRAPH=1"]
    serve = ["vllm", "serve", "/model", "--quantization", "gptq", "--dtype", "float16",
             "--max-model-len", str(args.context), "--gpu-memory-utilization", "0.95",
             "--kv-cache-dtype", "fp8", "--port", "8000", "--max-num-seqs", "1",
             "--max-num-batched-tokens", "8192", "--no-enable-prefix-caching",
             "--mamba-cache-mode", "align", "--performance-mode", "balanced",
             "--chat-template-content-format", "openai",
             "--default-chat-template-kwargs", '{"enable_thinking":false}',
             "--reasoning-parser", "qwen3", "--served-model-name", "qwen38",
             "--language-model-only", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"]
    if args.graph:
        serve += ["--compilation-config", '{"cudagraph_capture_sizes":[1,2,4,8]}', "--cudagraph-metrics"]
    else:
        serve += ["--enforce-eager"]
    setup = "python /prefill_guard.py; python /gdn_prefill.py; "
    if args.mode == "dflash2":
        argv += ["-v", f"{DRAFT}:/draft:ro", "-v", f"{out / 'patch_dflash2.py'}:/overlay.py:ro",
                 "-e", "B70_DFLASH2_BF16=1", "-e", f"B70_DFLASH2_AUDIT={int(args.audit)}"]
        setup += "python /overlay.py; "
        serve += ["--speculative-config", json.dumps({"method": "dflash", "model": "/draft",
                                                       "kv_cache_dtype": "auto", "num_speculative_tokens": 7})]
    argv += ["--entrypoint", "bash", IMAGE, "-lc", "set -e; " + setup + "exec " + shlex.join(serve)]
    return argv


class DFlashCell(probe.Cell):
    def __init__(self, args):
        # Preserve all existing input/transport/functional tests unchanged.
        args.head, args.cascade_patch, args.alpha = "dense", None, 0
        super().__init__(args)
        self.summary.update(mode=args.mode, image=IMAGE, context=args.context, model_runner="legacy",
                            graph=args.graph, audit=args.audit, target_dtype="float16",
                            draft_dtype="bfloat16" if args.mode == "dflash2" else None,
                            speculative_tokens=7 if args.mode == "dflash2" else 0)

    def start(self):
        if hashlib.sha256(self.original).hexdigest() != probe.LAUNCHER_SHA:
            raise RuntimeError("persistent Qwen launcher changed")
        if self.power.read_text().strip() != "275000000":
            raise RuntimeError("power cap is not 275 W")
        running = probe.command("docker", "ps", "--format", "{{.Names}}").splitlines()
        if running:
            raise RuntimeError(f"requires idle host; running containers: {running}")
        self.summary["glimmer_running_before"] = probe.command(
            "docker", "inspect", "-f", "{{.State.Running}}", "glimmer-tb21-prefix-c8").strip()
        if self.summary["glimmer_running_before"] != "false":
            raise RuntimeError("Glimmer must stay stopped")
        guard = self.args.guard.resolve()
        if hashlib.sha256(guard.read_bytes()).hexdigest() != probe.GUARD_SHA:
            raise RuntimeError("prefill guard mismatch")
        (self.out / "patch_uniform_decode_prefill.py").write_bytes(guard.read_bytes())
        self.summary["prefill_guard_sha256"] = probe.GUARD_SHA
        prefill = self.args.prefill_patch.resolve().read_bytes()
        (self.out / "patch_xpu_prefill.py").write_bytes(prefill)
        self.summary["gdn_prefill_patch_sha256"] = hashlib.sha256(prefill).hexdigest()
        for source in (Path(__file__), Path(probe.__file__)):
            (self.out / source.name).write_bytes(source.read_bytes())
        if self.args.mode == "dflash2":
            (self.out / "patch_dflash2.py").write_bytes(self.args.patch.resolve().read_bytes())
            self.summary["draft_config"] = json.loads((DRAFT / "config.json").read_text())
        argv = server_command(self.args, self.out)
        probe.save(self.out / "launch-argv.json", argv)
        (self.out / "launcher.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(argv) + "\n")
        self.log = (self.out / "server.log").open("w")
        self.proc = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Qwen exited during startup: {self.proc.returncode}")
            try:
                probe.get("/health")
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("startup exceeded 900 seconds")
        models = json.loads(probe.get("/v1/models"))
        probe.save(self.out / "models.json", models)
        if not any(m["id"] == "qwen38" and m["max_model_len"] == self.args.context for m in models["data"]):
            raise RuntimeError("wrong served model/context")
        self.summary["models"] = models
        self.summary["status"] = "running"
        print("CELL_READY=" + str(self.out), flush=True)

    def check_acceptance(self):
        # Restrict to requests in this cell, excluding warmup and startup counters.
        rows = [r for r in self.rows if r["label"].startswith(("code-", "prose-"))
                and not r["label"].endswith("-warmup")]
        def total(name):
            return sum(v for r in rows for k, v in r["metric_deltas"].items()
                       if k.startswith("vllm:spec_decode_num_" + name + "_total{"))
        steps, accepted, drafted = total("drafts"), total("accepted_tokens"), total("draft_tokens")
        self.summary["acceptance"] = {"steps": steps, "accepted": accepted, "drafted": drafted,
                                       "mean_emitted_per_step": 1 + accepted / steps if steps else None}
        if self.args.mode == "dflash2" and not (steps > 0 and accepted > 0 and drafted > 0):
            raise RuntimeError("DFlash2 must have measured nonzero draft acceptance")

    def close(self):
        super().close()
        self.summary["glimmer_running_after"] = probe.command(
            "docker", "inspect", "-f", "{{.State.Running}}", "glimmer-tb21-prefix-c8").strip()
        probe.save(self.out / "summary.json", self.summary)
        if not (self.summary["launcher_unchanged"] and self.summary["power_unchanged"]
                and self.summary["glimmer_running_after"] == "false"):
            raise RuntimeError("host invariants changed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mode", choices=("control", "dflash2"), required=True)
    p.add_argument("--patch", type=Path)
    p.add_argument("--guard", type=Path, default=Path("/tmp/qwen-b70-patch-uniform-decode-prefill.py"))
    p.add_argument("--prefill-patch", type=Path, required=True)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--suite", choices=("smoke", "compare", "long"), default="compare")
    a = p.parse_args()
    if a.mode == "dflash2" and not a.patch:
        p.error("DFlash2 requires its explicit isolated overlay")
    if a.audit and (a.mode != "dflash2" or a.graph):
        p.error("numerical audit is DFlash2 eager-only; do not rank its timings")
    if not 512 <= a.context <= 212992:
        p.error("context must be between 512 and 212992")
    if a.suite == "long" and a.context < 200032:
        p.error("long suite needs room for 200000 prompt + 32 output tokens")
    cell = DFlashCell(a)
    print("ARTIFACT_DIR=" + str(cell.out), flush=True)
    try:
        cell.start()
        cell.gates()
        cell.quality()
        if a.suite == "smoke":
            cell.chat("code-smoke", probe.CODE, 256)
            cell.chat("prose-smoke", probe.PROSE, 256)
        else:
            cell.speed()
        cell.check_acceptance()
        if a.suite == "long":
            cell.long_context()
        cell.summary["status"] = "completed"
    except Exception as error:
        cell.summary.update(status="failed", error=repr(error))
        if isinstance(error, urllib.error.HTTPError):
            cell.summary["http_error_body"] = error.read().decode(errors="replace")
        raise
    finally:
        cell.close()


if __name__ == "__main__":
    main()
