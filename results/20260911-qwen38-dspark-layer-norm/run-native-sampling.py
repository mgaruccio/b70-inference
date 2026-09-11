#!/usr/bin/env python3
"""Compare native Markov precision and draft sampling without changing old assets."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"), required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--long-context", action="store_true")
    options, remaining = parser.parse_known_args()
    overlay = ROOT / "patch-dspark-native.py"
    if not overlay.is_file():
        raise FileNotFoundError(overlay)
    driver = runpy.run_path(str(ROOT / "run-acceptance-diagnostics.py"))
    namespace = driver["run"].__globals__
    namespace["SPEC_CONFIG"] = dict(namespace["SPEC_CONFIG"], draft_sample_method=options.draft_sample_method)
    if options.long_context:
        namespace["CONTEXT"] = 65664
        namespace["MATRIX_REQUESTS"] = 7
        namespace["FORCED_OUTPUT"] = 128

        def long_context(client, probe):
            argv = [sys.executable, str(namespace["PREVIOUS"] / "qwen38_long_context_bench.py"),
                    "--base-url", client.base, "--model", "qwen38", "--out", str(client.out / "long-context"),
                    "--lengths", "65536", "--near-limit", "65536", "--confirm-prefix-cache-disabled"]
            namespace["save"](client.out / "long-client-argv.json", argv)
            with (client.out / "long-client-output.txt").open("w") as log:
                subprocess.run(argv, check=True, timeout=1200, stdout=log, stderr=subprocess.STDOUT)
            summary = json.loads((client.out / "long-context/summary.json").read_text())
            assert summary["status"] == "completed" and not summary["errors"]
            return summary

        namespace["run_matrix"] = long_context
    original_mounts = namespace["dependency_mounts"]
    original_validate = namespace["validate_assets"]

    def mounts(out, draft, cell, kv_cache_dtype):
        if cell != "dspark":
            raise ValueError("native sampling comparison requires --cell dspark")
        argv, metadata = original_mounts(out, draft, cell, kv_cache_dtype)
        if options.graphs:
            serve = metadata["serve"]
            serve.remove("--enforce-eager")
            serve += ["--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                                        "cudagraph_capture_sizes": [7, 8]}), "--cudagraph-metrics"]
            argv[argv.index("VLLM_XPU_ENABLE_XPU_GRAPH=0")] = "VLLM_XPU_ENABLE_XPU_GRAPH=1"
            argv[-1] = argv[-1].rsplit("exec ", 1)[0] + "exec " + shlex.join(serve)
        # A nested read-only bind overrides only this file in the old campaign mount.
        index = argv.index("--entrypoint")
        argv[index:index] = ["-v", f"{overlay}:/experiment/patch_dspark_bf16.py:ro"]
        metadata["candidate_overlay"] = {"path": str(overlay), "sha256": namespace["sha256"](overlay)}
        return argv, metadata

    def validate(draft, *, require_draft):
        manifest = original_validate(draft, require_draft=require_draft)
        manifest["files"]["draft_overlay"] = {
            "path": str(overlay), "sha256": namespace["sha256"](overlay),
        }
        if options.long_context:
            client = namespace["PREVIOUS"] / "qwen38_long_context_bench.py"
            manifest["files"]["long_client"] = {"path": str(client), "sha256": namespace["sha256"](client)}
        return manifest

    namespace["dependency_mounts"] = mounts
    namespace["validate_assets"] = validate
    sys.argv = [sys.argv[0], *remaining]
    return driver["main"]()


if __name__ == "__main__":
    sys.exit(main())
