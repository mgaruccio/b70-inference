#!/usr/bin/env python3
"""Compare native Markov precision and draft sampling without changing old assets."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--draft-sample-method", choices=("greedy", "probabilistic"), required=True)
    options, remaining = parser.parse_known_args()
    overlay = ROOT / "patch-dspark-native.py"
    if not overlay.is_file():
        raise FileNotFoundError(overlay)
    driver = runpy.run_path(str(ROOT / "run-acceptance-diagnostics.py"))
    namespace = driver["run"].__globals__
    namespace["SPEC_CONFIG"] = dict(namespace["SPEC_CONFIG"], draft_sample_method=options.draft_sample_method)
    original_mounts = namespace["dependency_mounts"]
    original_validate = namespace["validate_assets"]

    def mounts(out, draft, cell, kv_cache_dtype):
        if cell != "dspark":
            raise ValueError("native sampling comparison requires --cell dspark")
        argv, metadata = original_mounts(out, draft, cell, kv_cache_dtype)
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
        return manifest

    namespace["dependency_mounts"] = mounts
    namespace["validate_assets"] = validate
    sys.argv = [sys.argv[0], *remaining]
    return driver["main"]()


if __name__ == "__main__":
    sys.exit(main())
