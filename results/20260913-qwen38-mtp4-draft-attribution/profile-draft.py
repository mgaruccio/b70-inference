#!/usr/bin/env python3
"""Run one graph-enabled HTTP MTP4 draft-attribution profile.

The existing step-profile driver owns Docker, public HTTP, metrics, and cleanup.
This wrapper changes only its campaign constants and mounts the bounded
post-start instrumentation; it deliberately leaves the MTP4 graph configuration
and five-patch stack intact.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import shlex
import sys


ROOT = Path(__file__).resolve().parent
BASE_DRIVER = ROOT.parent / "20260911-qwen38-step-profile-64k" / "run-step-profile.py"
REPO_ROOT = ROOT.parents[1]


def resolve_timing_dir(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            ROOT / "timing",
            ROOT.parent / "20260911-qwen38-step-profile-64k" / "timing",
            REPO_ROOT / "scripts" / "experiments",
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / name).is_file() for name in (
            "qwen38_step_timing_overlay.py",
            "qwen38_step_timing_patch.py",
        )):
            return candidate
    choices = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"canonical timing files not found; checked: {choices}")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timing-dir", type=Path, help="directory containing canonical timing modules")
    options, remaining = parser.parse_known_args()
    if not BASE_DRIVER.is_file():
        raise RuntimeError(f"existing lifecycle driver is unavailable: {BASE_DRIVER}")
    timing = resolve_timing_dir(options.timing_dir)
    driver = runpy.run_path(str(BASE_DRIVER))
    namespace = driver["main"].__globals__
    namespace.update(
        CAMPAIGN=ROOT.name,
        CONTEXT=212_992,
        BATCHED_TOKENS=8_192,
        cell_name=lambda cell: "b70-mtp4-draft-attribution",
    )
    original = namespace["build_launch"]

    def build(cell, out, args):
        if cell != "mtp4":
            raise RuntimeError("this campaign is limited to the pinned MTP4 cell")
        argv, metadata = original(cell, out, args)
        module = ROOT / "draft-annotations.py"
        patch = ROOT / "draft-patch.py"
        source_files = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                module,
                patch,
                timing / "qwen38_step_timing_overlay.py",
                timing / "qwen38_step_timing_patch.py",
            )
        }
        index = argv.index("--entrypoint")
        extra = [
            "-v",
            f"{timing}:/timing:ro",
            "-v",
            f"{module}:/draft/draft_annotations.py:ro",
            "-v",
            f"{patch}:/draft/draft-patch.py:ro",
            "-e",
            "PYTHONPATH=/timing:/draft",
            "-e",
            "B70_STEP_TIMING=1",
            "-e",
            "B70_STEP_TIMING_DIR=/output/step-timing",
            "-e",
            "B70_STEP_TIMING_MAX_SAMPLES=64",
            "-e",
            "B70_DRAFT_ATTRIBUTION=1",
            "-e",
            "B70_DRAFT_ATTRIBUTION_DIR=/output/draft-attribution",
            "-e",
            "B70_DRAFT_ATTRIBUTION_MAX_SCOPES=256",
            "-e",
            "B70_DRAFT_ATTRIBUTION_MAX_DISPATCHES=256",
        ]
        argv[index:index] = extra
        prefix = argv[-1].rsplit("; exec ", 1)[0]
        argv[-1] = (
            prefix
            + "; /opt/venv/bin/python -P /draft/draft-patch.py; exec "
            + shlex.join(metadata["serve"])
        )
        metadata["environment"].extend(
            [
                "PYTHONPATH=/timing:/draft",
                "B70_STEP_TIMING=1",
                "B70_STEP_TIMING_DIR=/output/step-timing",
                "B70_STEP_TIMING_MAX_SAMPLES=64",
                "B70_DRAFT_ATTRIBUTION=1",
                "B70_DRAFT_ATTRIBUTION_DIR=/output/draft-attribution",
                "B70_DRAFT_ATTRIBUTION_MAX_SCOPES=256",
                "B70_DRAFT_ATTRIBUTION_MAX_DISPATCHES=256",
            ]
        )
        metadata["mounts"].extend(
            [
                {"host": str(timing), "container": "/timing", "mode": "ro", "role": "canonical_step_timing"},
                {"host": str(module), "container": "/draft/draft_annotations.py", "mode": "ro", "role": "draft_annotations"},
                {"host": str(patch), "container": "/draft/draft-patch.py", "mode": "ro", "role": "draft_patch_wrapper"},
            ]
        )
        metadata["annotation_sources"] = source_files
        metadata["attribution_mode"] = (
            "graph-enabled MTP4; post-profile CPU phase scopes and dispatcher metadata; "
            "canonical deferred current-stream XPU graph events and complete draft spans"
        )
        metadata["attribution_contract"] = {
            "scope_activation": "after /start_profile returns",
            "scope_stop": "on /stop_profile return or error cleanup",
            "graph_mode_unchanged": True,
            "eager_fallback_allowed": True,
            "whole_model_eager": False,
            "cpu_scope_time_is_inclusive": True,
            "cpu_scope_time_is_not_added_to_graph_event_time": True,
            "draft_span_timing": "deferred current-stream XPU events around propose; resolved at stop",
            "dispatcher_stage_labels": {"5": "first_five_tokens", "1": "later_one_token"},
        }
        return argv, metadata

    namespace["build_launch"] = build
    sys.argv = [
        sys.argv[0],
        *remaining,
        # Keep the campaign's finite profile window and MTP4 cell authoritative
        # even if a caller supplied duplicate base-driver flags.  argparse still
        # handles --help before validating the required run arguments.
        "--cell",
        "mtp4",
        "--profile",
        "--profile-delay-iterations",
        "3",
        "--profile-max-iterations",
        "5",
        "--profile-stop-after-events",
        "24",
    ]
    return driver["main"]()


if __name__ == "__main__":
    raise SystemExit(main())
