#!/usr/bin/env python3
"""Stage or run the bounded Qwen 3.8 MTP4 GDN-locality Phase0 harness.

Direct execution is for ``inference-host`` and reuses the existing
``run-step-profile.py`` lifecycle through ``runpy``.  The optional staging
entrypoint copies only this campaign's scripts and the canonical timing helper
to a fresh remote campaign directory; it never starts Docker from the staging
host.  Baseline and diagnostic outputs are required to be new directories.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
BASE_DRIVER = ROOT.parent / "20260911-qwen38-step-profile-64k" / "run-step-profile.py"
REPO_ROOT = ROOT.parents[1]
TIMING_SOURCE = REPO_ROOT / "scripts" / "experiments"
DEFAULT_REMOTE_HOST = "inference-host"
DEFAULT_REMOTE_DIR = "/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality"
DEFAULT_OUT_NAMES = {
    "baseline": "baseline-01",
    "profile": "diagnostic-graph-01",
    "eager-profile": "diagnostic-eager-01",
    "unitrace-profile": "diagnostic-unitrace-01",
}
BASELINE_LENGTHS = (512, 65_536)
PROFILE_DELAY_ITERATIONS = 3
PROFILE_MAX_ITERATIONS = 5
PROFILE_STOP_AFTER_EVENTS = 24
UNITRACE_SESSION = "b70gdnlocality02"
UNITRACE_CONTAINER_ROOT = "/unitrace"
UNITRACE_SOURCE_SHA = "887bba6e28ce84cc0d3813ef876e24add107c318"
UNITRACE_OPTIONS = (
    "--start-paused",
    "--device-timing",
    "--chrome-kernel-logging",
    "--chrome-device-logging",
)

EXPECTED_LAUNCHER_SHA256 = "63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4"
EXPECTED_IMAGE_DIGEST = "f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"

# This is the contract copied from the archived MTP4 golden launch.  The
# wrapper validates the effective argv before a remote server is started.
GOLDEN_CONTRACT: dict[str, Any] = {
    "image_digest": EXPECTED_IMAGE_DIGEST,
    "vllm_revision": "ac7509e2b",
    "method": "mtp",
    "num_speculative_tokens": 4,
    "quantization": "gptq",
    "dtype": "float16",
    "kv_cache_dtype": "fp8",
    "max_model_len": 212_992,
    "max_num_batched_tokens": 8_192,
    "max_num_seqs": 1,
    "gpu_memory_utilization": "0.95",
    "cudagraph_capture_sizes": [1, 2, 4, 8],
    "prefix_caching": False,
    "thinking": False,
    "temperature": 0.0,
    "seed": 42,
    "ignore_eos": True,
    "completion_tokens": 128,
}


class HarnessError(RuntimeError):
    """A local staging, contract, or lifecycle setup error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_after(argv: Sequence[str], flag: str) -> str:
    try:
        return str(argv[list(argv).index(flag) + 1])
    except (ValueError, IndexError) as exc:
        raise HarnessError(f"golden launch is missing {flag}") from exc


def golden_launch_summary(serve: Sequence[str], *, eager: bool = False) -> dict[str, Any]:
    """Validate the bounded launch contract and return inspectable facts."""
    if _value_after(serve, "--quantization") != GOLDEN_CONTRACT["quantization"]:
        raise HarnessError("Phase0 requires GPTQ target quantization")
    if _value_after(serve, "--dtype") != GOLDEN_CONTRACT["dtype"]:
        raise HarnessError("Phase0 requires float16 target compute")
    if _value_after(serve, "--kv-cache-dtype") != GOLDEN_CONTRACT["kv_cache_dtype"]:
        raise HarnessError("Phase0 requires FP8 KV cache")
    if int(_value_after(serve, "--max-model-len")) != GOLDEN_CONTRACT["max_model_len"]:
        raise HarnessError("Phase0 requires max-model-len 212992")
    if int(_value_after(serve, "--max-num-batched-tokens")) != GOLDEN_CONTRACT["max_num_batched_tokens"]:
        raise HarnessError("Phase0 requires max-num-batched-tokens 8192")
    if int(_value_after(serve, "--max-num-seqs")) != GOLDEN_CONTRACT["max_num_seqs"]:
        raise HarnessError("Phase0 requires C1")
    if _value_after(serve, "--gpu-memory-utilization") != GOLDEN_CONTRACT["gpu_memory_utilization"]:
        raise HarnessError("Phase0 requires gpu-memory-utilization 0.95")
    if "--no-enable-prefix-caching" not in serve:
        raise HarnessError("Phase0 requires prefix caching disabled")
    if "--enforce-eager" in serve and not eager:
        raise HarnessError("graph-enabled baseline/profile must not use --enforce-eager")
    if not eager and "--compilation-config" not in serve:
        raise HarnessError("graph-enabled baseline/profile must retain compilation config")
    speculation = json.loads(_value_after(serve, "--speculative-config"))
    if speculation.get("method") != GOLDEN_CONTRACT["method"] or speculation.get("num_speculative_tokens") != GOLDEN_CONTRACT["num_speculative_tokens"]:
        raise HarnessError(f"Phase0 requires native MTP K4, observed {speculation!r}")
    chat_kwargs = json.loads(_value_after(serve, "--default-chat-template-kwargs"))
    if chat_kwargs.get("enable_thinking") is not False:
        raise HarnessError("Phase0 requires thinking disabled")
    compilation = None
    if "--compilation-config" in serve:
        compilation = json.loads(_value_after(serve, "--compilation-config"))
        if compilation.get("cudagraph_capture_sizes") != GOLDEN_CONTRACT["cudagraph_capture_sizes"]:
            raise HarnessError(f"unexpected graph capture sizes: {compilation!r}")
    return {
        "image_digest": EXPECTED_IMAGE_DIGEST,
        "vllm_revision": GOLDEN_CONTRACT["vllm_revision"],
        "native_mtp_k": GOLDEN_CONTRACT["num_speculative_tokens"],
        "target_quantization": GOLDEN_CONTRACT["quantization"],
        "target_dtype": GOLDEN_CONTRACT["dtype"],
        "kv_cache_dtype": GOLDEN_CONTRACT["kv_cache_dtype"],
        "max_model_len": GOLDEN_CONTRACT["max_model_len"],
        "max_num_batched_tokens": GOLDEN_CONTRACT["max_num_batched_tokens"],
        "max_num_seqs": GOLDEN_CONTRACT["max_num_seqs"],
        "gpu_memory_utilization": GOLDEN_CONTRACT["gpu_memory_utilization"],
        "prefix_caching": False,
        "thinking": False,
        "xpu_graphs": not eager,
        "cudagraph_capture_sizes": compilation.get("cudagraph_capture_sizes") if compilation else None,
        "intentional_difference": "--enforce-eager and graph flags only" if eager else None,
    }


def _container_name(mode: str) -> str:
    return {
        "baseline": "b70-gdn-locality-baseline",
        "profile": "b70-gdn-locality-graph-profile",
        "eager-profile": "b70-gdn-locality-eager-profile",
        "unitrace-profile": "b70-gdn-locality-unitrace-profile",
    }[mode]


def _resolve_timing_dir(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend((ROOT / "timing", TIMING_SOURCE))
    for candidate in candidates:
        candidate = candidate.resolve()
        if all((candidate / name).is_file() for name in ("qwen38_step_timing_overlay.py", "qwen38_step_timing_patch.py")):
            return candidate
    raise HarnessError("canonical qwen38 step-timing overlay/patch not found")


def _replace_profiler_args(namespace: Mapping[str, Any], serve: list[str], config: dict[str, Any]) -> list[str]:
    profiler_argv = namespace["profiler_argv"]
    return [token for token in serve if not token.startswith("--profiler-config.")] + profiler_argv(config)


def _prepare_eager(serve: list[str], environment: list[str]) -> None:
    if "--compilation-config" in serve:
        index = serve.index("--compilation-config")
        del serve[index:index + 2]
    while "--cudagraph-metrics" in serve:
        serve.remove("--cudagraph-metrics")
    if "--enforce-eager" not in serve:
        serve.append("--enforce-eager")
    for index, value in enumerate(environment):
        if value == "VLLM_XPU_ENABLE_XPU_GRAPH=1":
            environment[index] = "VLLM_XPU_ENABLE_XPU_GRAPH=0"


def _append_instrumentation(
    argv: list[str],
    metadata: dict[str, Any],
    *,
    mode: str,
    timing: Path,
    unitrace_install: Path | None = None,
) -> None:
    module = ROOT / "gdn-annotations.py"
    patch = ROOT / "gdn-patch.py"
    for path in (module, patch, timing / "qwen38_step_timing_overlay.py", timing / "qwen38_step_timing_patch.py"):
        if not path.is_file():
            raise HarnessError(f"diagnostic source is missing: {path}")

    original_serve = list(metadata["serve"])
    serve = list(original_serve)
    unitrace_binary = None
    if mode == "unitrace-profile":
        if unitrace_install is None:
            raise HarnessError("unitrace-profile requires --unitrace-install")
        unitrace_install = unitrace_install.resolve()
        unitrace_binary = unitrace_install / "bin" / "unitrace"
        if not unitrace_binary.is_file():
            raise HarnessError(f"unitrace binary is missing: {unitrace_binary}")
        serve = [
            f"{UNITRACE_CONTAINER_ROOT}/bin/unitrace",
            *UNITRACE_OPTIONS,
            "--session",
            UNITRACE_SESSION,
            "--output-dir-path",
            "/output/unitrace",
            *serve,
        ]

    # _configure_launch has already replaced profiler args and, when selected,
    # removed graph flags.  This helper adds only read-only diagnostic mounts
    # and rebuilds the shell command.  The original vLLM argv remains in
    # metadata['serve']; unitrace's wrapper is recorded separately.
    index = argv.index("--entrypoint")
    extra_argv = [
        "-v",
        f"{timing}:/timing:ro",
        "-v",
        f"{module}:/gdn/gdn_annotations.py:ro",
        "-v",
        f"{patch}:/gdn/gdn-patch.py:ro",
        "-e",
        "PYTHONPATH=/timing:/gdn",
        "-e",
        "B70_STEP_TIMING=1",
        "-e",
        "B70_STEP_TIMING_DIR=/output/step-timing",
        "-e",
        "B70_STEP_TIMING_MAX_SAMPLES=512",
        "-e",
        "B70_GDN_LOCALITY=1",
        "-e",
        "B70_GDN_LOCALITY_MAX_SAMPLES=128",
    ]
    mounts = [
        {"host": str(timing), "container": "/timing", "mode": "ro", "role": "canonical_step_timing"},
        {"host": str(module), "container": "/gdn/gdn_annotations.py", "mode": "ro", "role": "gdn_annotations"},
        {"host": str(patch), "container": "/gdn/gdn-patch.py", "mode": "ro", "role": "gdn_patch_wrapper"},
    ]
    environment = [
        "PYTHONPATH=/timing:/gdn",
        "B70_STEP_TIMING=1",
        "B70_STEP_TIMING_DIR=/output/step-timing",
        "B70_STEP_TIMING_MAX_SAMPLES=512",
        "B70_GDN_LOCALITY=1",
        "B70_GDN_LOCALITY_MAX_SAMPLES=128",
    ]
    if unitrace_binary is not None:
        extra_argv.extend([
            "-v",
            f"{unitrace_install}:{UNITRACE_CONTAINER_ROOT}:ro",
            "-e",
            f"B70_UNITRACE_CONTROL={UNITRACE_CONTAINER_ROOT}/bin/unitrace",
            "-e",
            f"B70_UNITRACE_SESSION={UNITRACE_SESSION}",
            "-e",
            "B70_GDN_LOCALITY_XPU_EVENTS=0",
        ])
        mounts.append({"host": str(unitrace_install), "container": UNITRACE_CONTAINER_ROOT, "mode": "ro", "role": "unitrace_install"})
        environment.extend([
            f"B70_UNITRACE_CONTROL={UNITRACE_CONTAINER_ROOT}/bin/unitrace",
            f"B70_UNITRACE_SESSION={UNITRACE_SESSION}",
            "B70_GDN_LOCALITY_XPU_EVENTS=0",
        ])
        metadata["unitrace"] = {
            "source_sha": UNITRACE_SOURCE_SHA,
            "binary": str(unitrace_binary),
            "container_binary": f"{UNITRACE_CONTAINER_ROOT}/bin/unitrace",
            "session": UNITRACE_SESSION,
            "options": [*UNITRACE_OPTIONS, "--session", UNITRACE_SESSION, "--output-dir-path", "/output/unitrace"],
            "capture_control": "worker /start_profile resumes and /stop_profile pauses the unitrace session",
            "per_gdn_xpu_event_markers": False,
            "trace_directory": "unitrace",
        }
    argv[index:index] = extra_argv
    prefix = argv[-1].rsplit("; exec ", 1)[0]
    argv[-1] = prefix + "; /opt/venv/bin/python -P /gdn/gdn-patch.py; exec " + shlex.join(serve)
    metadata["mounts"].extend(mounts)
    metadata["environment"].extend(environment)
    metadata["annotation_sources"] = {
        str(path): sha256_file(path)
        for path in (module, patch, timing / "qwen38_step_timing_overlay.py", timing / "qwen38_step_timing_patch.py")
    }
    metadata["attribution_only"] = True
    metadata["attribution_contract"] = {
        "operator": "torch.ops._xpu_C.gdn_attention",
        "metadata_source": "bounded server-log B70_GDN_LOCALITY_METADATA lines",
        "timing_source": "deferred XPU Event pairs around complete graph replay; unitrace Level Zero kernel/device trace for inner instances",
        "native_trace_source": "vLLM torch profiler with record_shapes=true",
        "graph_inner_ops_may_be_hidden": True,
        "eager_or_partial_graph_cannot_establish_gdn_share": True,
        "fusion_candidate_gate": "at least 5% of reconciled target graph replay device time",
        "fusion_selection_requires": [
            "normal graph-enabled MTP4 serving",
            "complete target graph kernel coverage from unitrace",
            "explicit target-root GDN provenance",
            "whole-replay timing reconciliation",
            "native-relative output/state parity",
        ],
        "unsupported_measurement": "no in-graph PyTorch event markers or inferred eager-to-graph shares",
        "eager_mode_is_diagnostic_not_baseline": True,
        "scope_activation": "after native /start_profile returns",
        "scope_stop": "on native /stop_profile return or cleanup",
    }
    if mode == "eager-profile":
        metadata["attribution_contract"].update(
            graph_mode=False,
            decision_status="inconclusive",
            purpose="diagnostic fallback only; cannot select or reject fusion",
        )
    elif mode == "unitrace-profile":
        metadata["attribution_contract"].update(
            graph_mode=True,
            decision_status="inconclusive_until_unitrace_coverage_reconciled",
            purpose="normal graph-enabled serving attribution only",
        )
    else:
        metadata["attribution_contract"].update(
            graph_mode=True,
            decision_status="inconclusive_until_complete_graph_coverage_reconciled",
            purpose="graph torch trace is diagnostic evidence, never a fusion decision",
        )


def _configure_launch(
    namespace: dict[str, Any],
    original_build: Any,
    mode: str,
    timing: Path | None,
    unitrace_install: Path | None,
):
    def build(cell: str, out: Path, args: argparse.Namespace):
        if cell != "mtp4":
            raise HarnessError("Phase0 is limited to the pinned native MTP4 cell")
        argv, metadata = original_build(cell, out, args)
        eager = mode == "eager-profile"
        expected_image = namespace["DEFAULT_MTP_IMAGE"]
        if metadata.get("image") != expected_image:
            raise HarnessError(f"Phase0 requires pinned MTP4 image {expected_image!r}, observed {metadata.get('image')!r}")
        if any(mount.get("container") != "/output" and mount.get("mode") != "ro" for mount in metadata.get("mounts", [])):
            raise HarnessError("Phase0 requires every source mount except /output to be read-only")
        golden = golden_launch_summary(metadata["serve"], eager=False)
        if mode != "baseline":
            config = dict(metadata.get("profiler_config") or {})
            config["torch_profiler_record_shapes"] = True
            metadata["profiler_config"] = config
            metadata["serve"] = _replace_profiler_args(namespace, list(metadata["serve"]), config)
            if eager:
                _prepare_eager(metadata["serve"], metadata["environment"])
                golden = golden_launch_summary(metadata["serve"], eager=True)
            # The original argv's final shell command contains the original
            # serve list; instrumentation rebuilds it after these changes.
            prefix = argv[-1].rsplit("; exec ", 1)[0]
            argv[-1] = prefix + "; exec " + shlex.join(metadata["serve"])
            _append_instrumentation(
                argv,
                metadata,
                mode=mode,
                timing=timing or _resolve_timing_dir(None),
                unitrace_install=unitrace_install,
            )
        metadata["phase0_mode"] = mode
        metadata["golden_launch_contract"] = golden
        metadata["container_name"] = _container_name(mode)
        metadata["common_contract"]["prompt_lengths"] = list(BASELINE_LENGTHS if mode == "baseline" else (65_536,))
        metadata["common_contract"]["output_tokens"] = GOLDEN_CONTRACT["completion_tokens"]
        metadata["common_contract"]["seed"] = GOLDEN_CONTRACT["seed"]
        metadata["common_contract"]["temperature"] = GOLDEN_CONTRACT["temperature"]
        metadata["common_contract"]["ignore_eos"] = GOLDEN_CONTRACT["ignore_eos"]
        metadata["source_mount_policy"] = "archived inputs and diagnostic sources read-only; only /output writable"
        metadata["intentional_changes"] = [
            "new unique owned container name",
            "new fail-if-existing output directory",
            "Phase0 mode-specific profiling only" if mode != "baseline" else "no profiler or diagnostic instrumentation in baseline",
        ]
        return argv, metadata

    return build


def _configure_benchmark(namespace: dict[str, Any], original_benchmark: Any):
    def benchmark(out: Path, args: argparse.Namespace, long_client: Path) -> dict[str, Any]:
        points: dict[str, Any] = {}
        original_prompt = namespace["PROMPT_TOKENS"]
        try:
            for length in BASELINE_LENGTHS:
                namespace["PROMPT_TOKENS"] = length
                point_out = out / f"length-{length}"
                if point_out.exists():
                    raise HarnessError(f"baseline point output already exists: {point_out}")
                print(f"BEGIN phase0 baseline input={length}", flush=True)
                points[str(length)] = original_benchmark(point_out, args, long_client)
                print(f"PASS phase0 baseline input={length}", flush=True)
        finally:
            namespace["PROMPT_TOKENS"] = original_prompt
        return {
            "status": "completed",
            "purpose": "representative uninstrumented baseline only",
            "prompt_lengths": list(BASELINE_LENGTHS),
            "warmup_per_point": 1,
            "measured_per_point": 6,
            "points": points,
            "profiler_enabled": False,
            "full_8k_32k_sweep": False,
        }

    return benchmark


def _reject_overrides(remaining: Sequence[str]) -> None:
    forbidden = {"--cell", "--profile", "--mode"}
    for token in remaining:
        if token in forbidden or any(token.startswith(flag + "=") for flag in forbidden):
            raise HarnessError(f"Phase0 owns {token}; use --mode and the pinned MTP4 cell")


def _unitrace_artifact_check(out: Path) -> int:
    """Record trace visibility and session-control failures after server exit."""
    trace_dir = out / "unitrace"
    files = sorted(path for path in trace_dir.rglob("*") if path.is_file()) if trace_dir.is_dir() else []
    json_files = [path for path in files if path.suffix in {".json", ".pftrace", ".jsonl"}]
    control_events: list[dict[str, Any]] = []
    control_failures: list[str] = []
    marker = "B70_GDN_LOCALITY_SUMMARY:"
    for log_path in sorted(out.rglob("*.log")):
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if marker not in line:
                continue
            try:
                payload = json.loads(line.split(marker, 1)[1].strip())
            except json.JSONDecodeError:
                control_failures.append(f"malformed GDN summary in {log_path.name}")
                continue
            events = payload.get("unitrace_control_events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                continue
            control_events.extend(event for event in events if isinstance(event, dict))
            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("returncode") != 0:
                    control_failures.append(
                        f"{event.get('action', 'unknown')} returncode={event.get('returncode')!r}"
                    )

    valid_json_files = []
    invalid_trace_files = []
    for path in json_files:
        if path.suffix != ".json":
            invalid_trace_files.append(str(path.relative_to(out)))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
            if not isinstance(events, list) or not any(
                isinstance(event, dict) and event.get("ph") == "X"
                and isinstance(event.get("dur"), (int, float)) and event["dur"] > 0
                for event in events
            ):
                raise ValueError("no completed trace activities")
            valid_json_files.append(path)
        except (OSError, ValueError, TypeError):
            invalid_trace_files.append(str(path.relative_to(out)))
    successful_actions = {event.get("action") for event in control_events if event.get("returncode") == 0}
    if not {"resume", "pause", "stop"}.issubset(successful_actions):
        control_failures.append("capture requires successful resume, pause and stop/flush controls")
    observed = bool(valid_json_files)
    blocked = not observed or bool(control_failures)
    if not observed:
        reason = "unitrace emitted no trace; no graph-aware attribution is available"
    elif control_failures:
        reason = "unitrace session control failed; graph capture coverage is not trustworthy"
    else:
        reason = None
    result = {
        "status": "blocked" if blocked else "observed",
        "trace_directory": str(trace_dir),
        "files": [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size} for path in files],
        "json_trace_files": [str(path.relative_to(out)) for path in json_files],
        "valid_json_trace_files": [str(path.relative_to(out)) for path in valid_json_files],
        "invalid_or_empty_trace_files": invalid_trace_files,
        "unitrace_control_events": control_events,
        "unitrace_control_failures": control_failures,
        "decision_status": "inconclusive_until_graph_coverage_reconciled",
        "reason": reason,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "unitrace-artifact-check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = out / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["unitrace_artifact_check"] = result
        summary["attribution_status"] = result["decision_status"]
        if blocked and summary.get("status") == "passed":
            summary["status"] = "failed"
            summary["error"] = {
                "type": "UnitraceTraceMissing" if not observed else "UnitraceControlFailed",
                "message": reason,
            }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if blocked else 0

def run_on_current_host(
    mode: str,
    out: Path,
    remaining: Sequence[str],
    timing: Path | None,
    unitrace_install: Path | None,
) -> int:
    if not BASE_DRIVER.is_file():
        raise HarnessError(f"existing lifecycle driver is unavailable: {BASE_DRIVER}")
    if mode == "unitrace-profile" and unitrace_install is None:
        raise HarnessError("unitrace-profile requires --unitrace-install")
    if mode == "baseline":
        _reject_overrides(remaining)
    driver = runpy.run_path(str(BASE_DRIVER))
    namespace: dict[str, Any] = driver["main"].__globals__
    namespace.update(
        CAMPAIGN=ROOT.name,
        CONTEXT=212_992,
        BATCHED_TOKENS=8_192,
        PROMPT_TOKENS=65_536,
        cell_name=lambda _cell: _container_name(mode),
    )
    namespace["build_launch"] = _configure_launch(
        namespace,
        namespace["build_launch"],
        mode,
        timing,
        unitrace_install,
    )
    if mode == "baseline":
        namespace["run_benchmark"] = _configure_benchmark(namespace, namespace["run_benchmark"])
    else:
        # Keep a diagnostic profile finite and separate from the uninstrumented
        # baseline.  The values are appended last so callers cannot broaden it.
        remaining = [*remaining, "--profile", "--profile-delay-iterations", str(PROFILE_DELAY_ITERATIONS),
                     "--profile-max-iterations", str(PROFILE_MAX_ITERATIONS),
                     "--profile-stop-after-events", str(PROFILE_STOP_AFTER_EVENTS)]
    sys.argv = [str(ROOT / "run-phase0.py"), *remaining, "--cell", "mtp4", "--out", str(out)]
    status = int(driver["main"]())
    if mode == "unitrace-profile" and status == 0:
        return _unitrace_artifact_check(out)
    if mode == "unitrace-profile":
        _unitrace_artifact_check(out)
    return status


def _tar_payload() -> bytes:
    files = [
        "run-phase0.py",
        "gdn-annotations.py",
        "gdn-patch.py",
        "summarize-gdn.py",
        "README.md",
        "test-phase0.py",
        "build-unitrace.sh",
    ]
    timing_files = [
        TIMING_SOURCE / "qwen38_step_timing_overlay.py",
        TIMING_SOURCE / "qwen38_step_timing_patch.py",
    ]
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in files:
            path = ROOT / name
            if not path.is_file():
                raise HarnessError(f"staging source is missing: {path}")
            archive.add(path, arcname=name, recursive=False)
        for path in timing_files:
            if not path.is_file():
                raise HarnessError(f"canonical timing source is missing: {path}")
            archive.add(path, arcname=f"timing/{path.name}", recursive=False)
    return stream.getvalue()


def _remote_path(value: str) -> str:
    if not value.startswith("/") or "\n" in value or "\x00" in value:
        raise HarnessError(f"remote path must be absolute and shell-safe: {value!r}")
    return value.rstrip("/")


def stage_campaign(remote_host: str, remote_dir: str) -> None:
    remote_dir = _remote_path(remote_dir)
    check = f"test ! -e {shlex.quote(remote_dir)} && test ! -L {shlex.quote(remote_dir)} && mkdir -p {shlex.quote(remote_dir)}"
    result = subprocess.run(["ssh", remote_host, check], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise HarnessError(f"remote campaign directory is not fresh or could not be created: {remote_dir}\n{result.stderr}")
    payload = _tar_payload()
    result = subprocess.run(
        ["ssh", remote_host, f"tar -xf - -C {shlex.quote(remote_dir)}"],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise HarnessError(f"remote staging failed for {remote_dir}: {result.stderr.decode('utf-8', 'replace')}")
    print(json.dumps({
        "status": "staged",
        "remote_host": remote_host,
        "remote_dir": remote_dir,
        "files": ["run-phase0.py", "gdn-annotations.py", "gdn-patch.py", "summarize-gdn.py", "README.md", "test-phase0.py", "build-unitrace.sh", "timing/qwen38_step_timing_overlay.py", "timing/qwen38_step_timing_patch.py"],
    }, sort_keys=True), flush=True)


def run_remote(remote_host: str, remote_dir: str, mode: str, out_name: str, extra: Sequence[str]) -> int:
    remote_dir = _remote_path(remote_dir)
    if "/" in out_name or out_name in {"", ".", ".."}:
        raise HarnessError(f"--out-name must be a fresh child name, got {out_name!r}")
    remote_script = f"{remote_dir}/run-phase0.py"
    remote_out = f"{remote_dir}/{out_name}"
    extra = list(extra)
    if mode == "unitrace-profile" and "--unitrace-install" not in extra:
        extra.extend(["--unitrace-install", f"{remote_dir}/unitrace-install"])
    command = ["python3", "-u", remote_script, "--mode", mode, "--out", remote_out, *extra]
    print(f"REMOTE_COMMAND ssh {remote_host} {shlex.join(command)}", flush=True)
    result = subprocess.run(["ssh", remote_host, shlex.join(command)], check=False)
    return result.returncode


def parse_cli(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(DEFAULT_OUT_NAMES), help="direct remote mode")
    parser.add_argument("--out", type=Path, help="new output directory for direct inference-host execution")
    parser.add_argument("--timing-dir", type=Path, help="canonical timing source directory for direct execution")
    parser.add_argument("--unitrace-install", type=Path, help="read-only unitrace install root containing bin/unitrace")
    parser.add_argument("--stage-only", action="store_true", help="stage source files to a fresh remote directory and stop")
    parser.add_argument("--stage", action="store_true", help="stage to a fresh remote directory, then run the selected mode")
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--out-name", help="fresh child directory under --remote-dir")
    parser.add_argument("--remote-arg", action="append", default=[], help="extra argument passed to the remote direct run; repeatable")
    args, remaining = parser.parse_known_args(argv)
    if args.stage and args.stage_only:
        parser.error("use only one of --stage and --stage-only")
    if args.stage or args.stage_only:
        if args.mode is None and not args.stage_only:
            parser.error("--stage requires --mode")
        if args.out is not None:
            parser.error("--out is for direct execution; use --out-name for staged execution")
        if remaining:
            parser.error(f"unrecognized staging arguments: {remaining}")
    else:
        if args.mode is None:
            parser.error("direct execution requires --mode")
        if args.out is None:
            parser.error("direct execution requires --out")
    return args, remaining


def main(argv: Sequence[str] | None = None) -> int:
    args, remaining = parse_cli(argv)
    if args.stage or args.stage_only:
        stage_campaign(args.remote_host, args.remote_dir)
        if args.stage_only:
            return 0
        out_name = args.out_name or DEFAULT_OUT_NAMES[args.mode]
        return run_remote(args.remote_host, args.remote_dir, args.mode, out_name, args.remote_arg)
    return run_on_current_host(args.mode, args.out, remaining, args.timing_dir, args.unitrace_install)


if __name__ == "__main__":
    raise SystemExit(main())
