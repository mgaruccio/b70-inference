#!/usr/bin/env python3
"""Run the bounded MTP4 public-boundary A/B trial for grouped serving.

This is a disposable wrapper around the existing step-profile lifecycle.  It
keeps the pinned MTP4 stack and real HTTP gates, and adds only the candidate's
read-only operator library plus worker import shim.  It is not a profile mode
or a production launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import runpy
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROFILE = ROOT.parent / "20260911-qwen38-step-profile-64k"
DIAGNOSTICS = ROOT.parent / "20260911-qwen38-dspark-layer-norm"
CANONICAL_PATCH = (
    ROOT.parent
    / "20260913-qwen38-mtp4-draft-attribution"
    / "timing"
    / "qwen38_step_timing_patch.py"
)
DEFAULT_LIBRARY = Path(
    "/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/"
    "20260913-qwen38-native-grouped-verify/build-05/build/"
    "libb70_grouped_verify.so"
)
EXPECTED_LIBRARY_SHA256 = (
    "4630ef2db027c3443ff63b16a611699c0db250c2cc53ed67aaad8a1a318f4490"
)
EXPECTED_CANONICAL_PATCH_SHA256 = "dc51dded6d9848b4cc504f273785a1d92af32f7526213b94ffce55549c4ef4bd"


class ServingContractError(RuntimeError):
    """A candidate did not prove real eligible graph-mode integration."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_profile_mode(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    for index, value in enumerate(argv):
        if value == "--profile" or value.startswith("--profile="):
            parser.error("serving A/B does not support --profile; use the benchmark mode")
        if value == "--mode" and index + 1 < len(argv) and argv[index + 1] == "profile":
            parser.error("serving A/B rejects --mode profile; use the benchmark mode")
        if value == "--mode=profile":
            parser.error("serving A/B rejects --mode profile; use the benchmark mode")
        if value == "--cell" or value.startswith("--cell="):
            parser.error("serving A/B is fixed to the MTP4 cell")


def _candidate_mounts(
    module: Path,
    patch: Path,
    seam: Path,
    library: Path,
) -> list[dict[str, str]]:
    return [
        {
            "host": str(module),
            "container": "/experiment/qwen38_step_timing_overlay.py",
            "mode": "ro",
            "role": "grouped_serving_overlay",
        },
        {
            "host": str(patch),
            "container": "/experiment/qwen38_step_timing_patch.py",
            "mode": "ro",
            "role": "worker_import_shim",
        },
        {
            "host": str(seam),
            "container": "/experiment/grouped_verify.py",
            "mode": "ro",
            "role": "grouped_verify_seam",
        },
        {
            "host": str(library),
            "container": "/candidate/libb70_grouped_verify.so",
            "mode": "ro",
            "role": "qualified_custom_operator",
        },
    ]


def _candidate_execution_evidence(out: Path) -> dict[str, Any]:
    log_path = out / "server.log"
    if not log_path.is_file():
        raise ServingContractError(f"candidate server log is missing: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    eligible_lines = [
        line.strip()
        for line in text.splitlines()
        if "[B70_GROUPED_SERVING]" in line and '"event":"eligible-dispatch"' in line
    ]
    unsupported_lines = [
        line.strip()
        for line in text.splitlines()
        if "[B70_GROUPED_SERVING]" in line and '"event":"unsupported-q5"' in line
    ]
    evidence = {
        "server_log": str(log_path),
        "eligible_dispatch_log_count": len(eligible_lines),
        "eligible_dispatch_log": eligible_lines[:1],
        "unsupported_q5_log_count": len(unsupported_lines),
        "unsupported_q5_log": unsupported_lines[:1],
        "full_graph_capture_seen": "Capturing CUDA graphs (decode, FULL)" in text,
        "full_graph_run_seen": bool(re.search(r"\|\s*FULL\s*\|", text)),
        "required_public_boundary": "real HTTP benchmark completed before this check",
    }
    (out / "candidate-execution-evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not evidence["eligible_dispatch_log_count"]:
        raise ServingContractError(
            "candidate produced no eligible grouped-serving dispatch log; "
            f"unsupported-q5 evidence={evidence['unsupported_q5_log']!r}"
        )
    if not evidence["full_graph_capture_seen"] or not evidence["full_graph_run_seen"]:
        raise ServingContractError(
            "candidate did not prove a FULL graph capture and run: "
            f"{evidence}"
        )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    options, remaining = parser.parse_known_args()
    _reject_profile_mode(parser, remaining)

    driver = runpy.run_path(str(PROFILE / "run-step-profile.py"))
    namespace = driver["main"].__globals__
    namespace["CAMPAIGN"] = ROOT.name
    namespace["CONTEXT"] = 212992
    namespace["BATCHED_TOKENS"] = 8192
    variant = "candidate" if options.candidate else "baseline"
    namespace["cell_name"] = lambda cell: f"b70-grouped-serving-{cell}-{variant}"

    original_build = namespace["build_launch"]
    original_benchmark = namespace["run_benchmark"]

    def build(cell: str, out: Path, args: argparse.Namespace):
        if cell != "mtp4":
            raise ValueError("serving A/B is fixed MTP4 only")
        argv, metadata = original_build(cell, out, args)
        metadata["experiment"] = (
            f"qualified grouped serving {variant}; fixed MTP4, 212992/8192"
        )
        metadata["comparison_arm"] = variant
        metadata["serving_adapter"] = {
            "enabled": bool(options.candidate),
            "operator": "b70_grouped_verify.forward",
            "helper": "_spec_decode_varlen_fwd",
            "geometry": "HND FP8e4m3fn KV[176,1664,4,256]",
        }
        metadata["intentional_comparison_difference"] = (
            "candidate-only worker import hook and qualified custom operator; "
            "baseline has neither"
        )
        if not options.candidate:
            return argv, metadata

        module = ROOT / "serving-overlay.py"
        seam = ROOT / "grouped_verify.py"
        for label, path in {
            "serving overlay": module,
            "canonical worker patch": CANONICAL_PATCH,
            "grouped verify seam": seam,
            "custom operator library": options.library,
        }.items():
            if not path.is_file():
                raise ServingContractError(f"required candidate asset is missing: {label}={path}")
        observed_library_sha256 = sha256_file(options.library)
        if observed_library_sha256 != EXPECTED_LIBRARY_SHA256:
            raise ServingContractError(
                "candidate library hash mismatch: "
                f"expected {EXPECTED_LIBRARY_SHA256}, observed {observed_library_sha256}, "
                f"path={options.library}"
        )
        observed_patch_sha256 = sha256_file(CANONICAL_PATCH)
        if observed_patch_sha256 != EXPECTED_CANONICAL_PATCH_SHA256:
            raise ServingContractError(
                "canonical worker patch hash mismatch: "
                f"expected {EXPECTED_CANONICAL_PATCH_SHA256}, observed {observed_patch_sha256}, "
                f"path={CANONICAL_PATCH}"
            )

        mounts = _candidate_mounts(module, CANONICAL_PATCH, seam, options.library)
        environment = [
            "PYTHONPATH=/experiment",
            "B70_STEP_TIMING=1",
            "B70_GROUPED_SERVING=1",
            "B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so",
        ]
        extra: list[str] = []
        for mount in mounts:
            extra.extend(["-v", f"{mount['host']}:{mount['container']}:ro"])
        for value in environment:
            extra.extend(["-e", value])
        entrypoint_index = argv.index("--entrypoint")
        argv[entrypoint_index:entrypoint_index] = extra
        prefix, serve = argv[-1].rsplit("; exec ", 1)
        argv[-1] = (
            prefix
            + "; /opt/venv/bin/python -P /experiment/qwen38_step_timing_patch.py; exec "
            + serve
        )
        metadata["mounts"].extend(mounts)
        metadata["environment"].extend(environment)
        metadata["candidate_sources"] = {
            str(path): sha256_file(path)
            for path in (module, CANONICAL_PATCH, seam)
        }
        metadata["candidate_library"] = {
            "host": str(options.library),
            "container": "/candidate/libb70_grouped_verify.so",
            "sha256": observed_library_sha256,
            "read_only_mount": True,
        }
        return argv, metadata

    def benchmark(out: Path, args: argparse.Namespace, long_client: Path):
        diagnostics = runpy.run_path(
            str(DIAGNOSTICS / "run-acceptance-diagnostics.py")
        )
        diagnostic_namespace = diagnostics["run"].__globals__
        probe, checks = diagnostic_namespace["load_previous_modules"]()
        probe.IMAGE = namespace["DEFAULT_MTP_IMAGE"]
        client = diagnostic_namespace["DiagnosticClient"](
            out, base=args.base_url, request_timeout=900
        )
        gates = diagnostic_namespace["run_shared_gates"](
            client, out, probe, checks, True
        )
        if gates.get("status") != "passed":
            raise ServingContractError(f"public API gates failed: {gates}")
        result = original_benchmark(out, args, long_client)
        if options.candidate:
            result["candidate_execution"] = _candidate_execution_evidence(out)
        return result

    namespace["build_launch"] = build
    namespace["run_benchmark"] = benchmark
    sys.argv = [sys.argv[0], "--cell", "mtp4", *remaining]
    return driver["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
