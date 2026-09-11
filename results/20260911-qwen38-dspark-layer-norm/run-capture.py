#!/usr/bin/env python3
"""Two real API requests: unarmed baseline, then one armed first-proposal capture.

Run on inference-host ONLY AFTER lead review. Reuse invariant-safe launch and
SSE/metric accounting; do not run the old matrix or broad smoke suite.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import traceback
import uuid

from parity_common import CODE, MAX_CONTEXT, OVERLAY_SHA, sha

ROOT = Path(__file__).resolve().parent


def launch(driver, out, previous, overlay):
    ns = driver["dependency_mounts"].__globals__
    ns["PREVIOUS"] = previous
    argv, metadata = driver["dependency_mounts"](out, previous / "draft", "dspark", "fp8")
    old_name = metadata["container_name"]
    name = "qwen38-dspark-reference-" + uuid.uuid4().hex[:12]
    argv[argv.index(old_name)] = name
    metadata["container_name"] = name
    index = argv.index("--entrypoint")
    argv[index:index] = ["-v", f"{ROOT}:/parity:ro", "-v", f"{overlay}:/experiment/patch_dspark_bf16.py:ro",
                         "-e", "PYTHONPATH=/parity", "-e", "DSPARK_PARITY_OUTPUT=/output"]
    prefix, serve = argv[-1].rsplit("exec ", 1)
    argv[-1] = prefix + "/opt/venv/bin/python -P /parity/patch-capture.py > /output/hook-install.json; exec " + serve
    metadata["patch_order"].append("capture")
    metadata["canonical_overlay_sha256"] = sha(overlay)
    return argv, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True, help="existing run-acceptance-diagnostics.py")
    parser.add_argument("--previous", type=Path, required=True, help="immutable 20260910-dspark-v2-feasibility")
    parser.add_argument("--overlay", type=Path, required=True, help="canonical corrected overlay from scripts/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-tokens", type=int, choices=(16, 32), default=16)
    parser.add_argument("--approve-gpu-launch", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    if not args.approve_gpu_launch:
        parser.error("no automatic GPU launch; review assets then pass --approve-gpu-launch")
    out, previous, overlay = args.out.resolve(), args.previous.resolve(), args.overlay.resolve()
    if out.parent != ROOT or out.exists():
        parser.error(f"--out must be a NEW immediate child of {ROOT}")
    if sha(overlay) != OVERLAY_SHA:
        parser.error("requires commit 8a19b58's layer-specific norm canonical overlay")
    driver = runpy.run_path(str(args.driver.resolve()))
    driver["validate_assets"].__globals__["PREVIOUS"] = previous
    dependencies = driver["validate_assets"](previous / "draft", require_draft=True)
    probe = runpy.run_path(str(previous / "qwen38_lossy_probe.py"))
    if probe["CODE"] != CODE:
        raise ValueError("existing code prompt changed")
    # Check before creating/launching anything. Never stop an unrelated container.
    before = driver["host_invariants"]()
    if before != driver["expected_host_invariants"]():
        raise RuntimeError(f"host prerequisites failed: {before}")
    out.mkdir()
    save = driver["save"]
    save(out / "host-before.json", before)
    argv, metadata = launch(driver, out, previous, overlay)
    dependencies["files"]["draft_overlay"] = {"path": str(overlay), "sha256": sha(overlay)}
    dependencies["driver"] = {"path": str(args.driver), "sha256": sha(args.driver)}
    dependencies["hook_files"] = {p.name: sha(p) for p in ROOT.glob("*.py")}
    save(out / "dependencies.json", dependencies)
    save(out / "launch-argv.json", argv)
    save(out / "launch-metadata.json", metadata)
    save(out / "driver-args.json", {"argv": sys.argv, "cwd": os.getcwd()})
    (out / "launcher.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(argv) + "\n")
    (out / "image-inspect.json").write_text(driver["command"]("docker", "image", "inspect", driver["IMAGE"]).stdout)
    summary = {"tier": "development", "status": "starting", "purpose": "same-input first draft numerical comparison",
               "configuration": "corrected greedy eager C1 8192; FP16 GPTQ target/FP8 KV; BF16 draft K7",
               "intentional_difference": "one request with synchronous read-only tensor capture",
               "not_acceptance_proof": "finite outputs and final-output identity alone are insufficient"}
    proc = None
    try:
        with (out / "server.log").open("w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
            client = driver["DiagnosticClient"](out, request_timeout=args.request_timeout)
            driver["wait_for_server"](proc, client, args.startup_timeout)
            check_argv = ["docker", "exec", metadata["container_name"], "/opt/venv/bin/python", "-P", "/parity/patch-capture.py", "--check"]
            save(out / "source-check-argv.json", check_argv)
            save(out / "effective-source.json", json.loads(driver["command"](*check_argv, timeout=120).stdout))
            off = client.chat("capture-off", CODE, args.output_tokens, True, cache_salt="reference-parity-code")
            ids = json.loads((out / "capture-off-rendered-prompt-ids.json").read_text())
            if not 0 < len(ids) <= MAX_CONTEXT:
                raise ValueError("known API prompt must fit the one-prefill capture bound")
            save(out / "arm.json", {"prompt_token_ids": ids, "request_label": "capture-on",
                                     "prompt": CODE, "first_step_only": True, "max_context": MAX_CONTEXT})
            on = client.chat("capture-on", CODE, args.output_tokens, True, cache_salt="reference-parity-code")
            captured = json.loads((out / "capture.json").read_text())
            if not captured["complete"] or captured["errors"] or not (out / "capture.pt").is_file():
                raise RuntimeError(f"first proposal capture failed: {captured.get('errors')}")
            on_ids = json.loads((out / "capture-on-rendered-prompt-ids.json").read_text())
            off_output = json.loads((out / "capture-off-output-ids.json").read_text())
            on_output = json.loads((out / "capture-on-output-ids.json").read_text())
            summary["same_rendered_prompt"] = ids == on_ids == captured["armed"]["prompt_token_ids"]
            summary["capture_on_off_output_ids_equal"] = off_output == on_output
            summary["stream_transport_pass"] = off["transport_pass"] and on["transport_pass"]
            summary["api_metrics"] = [off, on]
            summary["status"] = "captured-awaiting-official-replay"
            if not summary["same_rendered_prompt"] or not summary["capture_on_off_output_ids_equal"]:
                raise RuntimeError("capture-on/off API identity differs; retain artifacts and investigate")
    except Exception:
        summary["status"] = "failed"
        summary["error"] = traceback.format_exc()
        (out / "failure.txt").write_text(summary["error"])
    finally:
        (out / "arm.json").unlink(missing_ok=True)
        if proc is not None:
            cleanup = subprocess.run(["docker", "rm", "-f", metadata["container_name"]], capture_output=True, text=True, timeout=60)
            save(out / "cleanup.json", {"returncode": cleanup.returncode, "stdout": cleanup.stdout, "stderr": cleanup.stderr})
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        after = driver["host_invariants"]()
        save(out / "host-after.json", after)
        summary["host_unchanged"] = after == before
        if after != before:
            summary["status"] = "failed"
            summary["cleanup_error"] = "host invariants changed"
        save(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "captured-awaiting-official-replay" else 1


if __name__ == "__main__":
    sys.exit(main())
