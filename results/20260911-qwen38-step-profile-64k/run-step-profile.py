#!/usr/bin/env python3
"""Run the bounded Qwen 64K profiling/throughput campaign on inference-host.

This is a development-only driver.  It owns a disposable container, but never
changes the persistent launcher or any source/model snapshot.  The normal mode
runs the existing cold long-context client; ``--profile`` runs one streamed
64K completion and brackets a small CPU+XPU Torch profile after the first
non-empty output event.
"""
from __future__ import annotations

import argparse
import datetime as datetime_module
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Mapping, Sequence


CAMPAIGN = "20260911-qwen38-step-profile-64k"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "qwen38"
CONTEXT = 65_664
PROMPT_TOKENS = 65_536
BATCHED_TOKENS = 2_048
OUTPUT_TOKENS = 128
SEED = 42
TEMPERATURE = 0.0
EXPECTED_LAUNCHER_SHA256 = "63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4"
EXPECTED_DSPARK_OVERLAY_SHA256 = "0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643"
EXPECTED_LONG_CLIENT_SHA256 = "a01a99b21f36ef446d220df66a4e739a02b3ab18dfe99dec8beb333719276907"

DEFAULT_TARGET = Path("/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16")
DEFAULT_MTP_ROOT = Path(
    "/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard/mtp4-long"
)
DEFAULT_DSPARK_ROOT = Path(
    "/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility"
)
DEFAULT_DSPARK_CAMPAIGN = Path(
    "/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm"
)
DEFAULT_PRODUCTION_LAUNCHER = Path("/home/mike/inference/launchers/start-qwen38.sh")
DEFAULT_POWER_PATH = Path("/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap")
DEFAULT_GLIMMER_NAME = "glimmer-tb21-prefix-c8"
DEFAULT_MTP_IMAGE = (
    "vllm/vllm-openai-xpu@sha256:"
    "f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
)
DEFAULT_DSPARK_IMAGE = (
    "vllm/vllm-openai-xpu@sha256:"
    "7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4"
)


class CampaignError(RuntimeError):
    """A setup, public API, or evidence validation failure."""


class CampaignInterrupted(CampaignError):
    """The operator interrupted this disposable run."""


class HTTPFailure(CampaignError):
    """An HTTP failure retaining the endpoint and response body."""

    def __init__(self, path: str, status: int, body: bytes) -> None:
        self.path = path
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status} from {path}")


class HTTPResponse:
    def __init__(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
        self.status = status
        self.headers = dict(headers)
        self.body = body

    def json(self, path: str) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CampaignError(f"invalid JSON from {path}: {exc}") from exc


class PublicClient:
    """Small stdlib-only client for the public HTTP boundary."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> HTTPResponse:
        if not path.startswith("/"):
            path = "/" + path
        body = (
            None
            if payload is None
            else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        )
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return HTTPResponse(response.status, response.headers, response.read())
        except urllib.error.HTTPError as exc:
            return HTTPResponse(exc.code, exc.headers, exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CampaignError(f"connection failed for {path}: {exc}") from exc

    def open_stream(self, path: str, payload: Mapping[str, Any]):
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as exc:
            raise HTTPFailure(path, exc.code, exc.read()) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CampaignError(f"stream connection failed for {path}: {exc}") from exc


def utc_now() -> str:
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def command_result(*argv: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def checked_stdout(*argv: str, timeout: float = 60.0) -> str:
    result = command_result(*argv, timeout=timeout)
    if result.returncode != 0:
        raise CampaignError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout


def host_invariants(
    production_launcher: Path,
    power_path: Path,
    glimmer_name: str,
) -> dict[str, Any]:
    if not production_launcher.is_file():
        raise CampaignError(f"production launcher is missing: {production_launcher}")
    if not power_path.is_file():
        raise CampaignError(f"power-cap file is missing: {power_path}")
    running = checked_stdout("docker", "ps", "--format", "{{.Names}}").splitlines()
    all_containers = checked_stdout("docker", "ps", "-a", "--format", "{{.Names}}").splitlines()
    glimmer = checked_stdout(
        "docker", "inspect", "-f", "{{.State.Running}}", glimmer_name
    ).strip()
    return {
        "launcher_sha256": sha256_file(production_launcher),
        "power_cap": power_path.read_text(encoding="utf-8").strip(),
        "running_containers": running,
        "all_containers": all_containers,
        "glimmer_running": glimmer,
    }


def assert_idle_host(before: Mapping[str, Any], container_name: str) -> None:
    expected = {
        "launcher_sha256": EXPECTED_LAUNCHER_SHA256,
        "power_cap": "275000000",
        "running_containers": [],
        "glimmer_running": "false",
    }
    for key, value in expected.items():
        if before.get(key) != value:
            raise CampaignError(
                f"host preflight invariant failed for {key}: "
                f"expected {value!r}, observed {before.get(key)!r}"
            )
    if container_name in before.get("all_containers", []):
        raise CampaignError(
            f"owned container name already exists (including stopped containers): {container_name}"
        )


def assert_host_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    for key in ("launcher_sha256", "power_cap", "running_containers", "glimmer_running"):
        if before.get(key) != after.get(key):
            raise CampaignError(
                f"host invariant changed for {key}: "
                f"before={before.get(key)!r}, after={after.get(key)!r}"
            )


def shell_script(argv: Sequence[str]) -> str:
    return "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(list(argv)) + "\n"


def json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def cell_name(cell: str) -> str:
    return f"b70-step-profile-{cell}"


def profiler_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profiler": "torch",
        "torch_profiler_dir": "/output/profile",
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "ignore_frontend": True,
        "delay_iterations": args.profile_delay_iterations,
        "max_iterations": args.profile_max_iterations,
        "warmup_iterations": 0,
        "active_iterations": 5,
        "wait_iterations": 0,
        "capture_torch_profiler": False,
    }


def profiler_argv(config: Mapping[str, Any]) -> list[str]:
    return [
        "--profiler-config.profiler=torch",
        f"--profiler-config.torch_profiler_dir={config['torch_profiler_dir']}",
        f"--profiler-config.torch_profiler_with_stack={str(config['torch_profiler_with_stack']).lower()}",
        f"--profiler-config.torch_profiler_with_flops={str(config['torch_profiler_with_flops']).lower()}",
        f"--profiler-config.torch_profiler_use_gzip={str(config['torch_profiler_use_gzip']).lower()}",
        f"--profiler-config.torch_profiler_dump_cuda_time_total={str(config['torch_profiler_dump_cuda_time_total']).lower()}",
        f"--profiler-config.torch_profiler_record_shapes={str(config['torch_profiler_record_shapes']).lower()}",
        f"--profiler-config.torch_profiler_with_memory={str(config['torch_profiler_with_memory']).lower()}",
        f"--profiler-config.ignore_frontend={str(config['ignore_frontend']).lower()}",
        f"--profiler-config.delay_iterations={config['delay_iterations']}",
        f"--profiler-config.max_iterations={config['max_iterations']}",
        "--profiler-config.warmup_iterations=0",
        f"--profiler-config.active_iterations={config['active_iterations']}",
        "--profiler-config.wait_iterations=0",
    ]


def build_launch(
    cell: str,
    out: Path,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any]]:
    render_device = Path("/dev/dri/renderD128")
    if not render_device.exists():
        raise CampaignError(f"{render_device} is required on inference-host")
    group_id = render_device.stat().st_gid
    name = cell_name(cell)
    profile = args.profile_mode
    profile_config = profiler_config(args) if profile else None

    if cell == "mtp4":
        root = args.mtp4_root.resolve()
        image = DEFAULT_MTP_IMAGE
        target = args.target_dir.resolve()
        serve: list[str] = [
            "vllm",
            "serve",
            "/model",
            "--performance-mode",
            "balanced",
            "--compilation-config",
            json_compact({"cudagraph_capture_sizes": [1, 2, 4, 8]}),
            "--cudagraph-metrics",
            "--quantization",
            "gptq",
            "--dtype",
            "float16",
            "--max-model-len",
            str(CONTEXT),
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--port",
            "8000",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            str(BATCHED_TOKENS),
            "--no-enable-prefix-caching",
            "--mamba-cache-mode",
            "align",
            "--chat-template-content-format",
            "openai",
            "--default-chat-template-kwargs",
            json_compact({"enable_thinking": False}),
            "--reasoning-parser",
            "qwen3",
            "--override-generation-config",
            json_compact({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0}),
            "--served-model-name",
            DEFAULT_MODEL,
            "--language-model-only",
            "--speculative-config",
            json_compact({"method": "mtp", "num_speculative_tokens": 4}),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
        ]
        mounts: list[dict[str, Any]] = [
            {"host": str(root), "container": "/profile", "mode": "ro", "role": "archived_mtp4_root"},
            {"host": str(target), "container": "/model", "mode": "ro", "role": "target_model"},
            {
                "host": str(root / "patch_uniform_decode_prefill.py"),
                "container": "/prefill_guard.py",
                "mode": "ro",
                "role": "prefill_guard",
            },
        ]
        patch_mounts = [
            (root / "reference-source/patches/patch_mtp_nightly.py", "/patch_mtp.py", "mtp_nightly"),
            (root / "reference-source/patches/patch_mtp_boundary.py", "/patch_boundary.py", "mtp_boundary"),
            (root / "reference-source/patches/patch_gdn_mixed_split_v5.py", "/patch_v5.py", "gdn_mixed_split_v5"),
            (root / "reference-source/patches/patch_draft_lmhead_int4.py", "/patch_s.py", "draft_lmhead_int4"),
            (root / "reference-source/patches/patch_draft_mtp_int4.py", "/patch_m1.py", "draft_mtp_int4"),
        ]
        mounts.extend(
            {"host": str(host), "container": container, "mode": "ro", "role": role}
            for host, container, role in patch_mounts
        )
        env = [
            "VLLM_TARGET_DEVICE=xpu",
            "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
            "ZE_AFFINITY_MASK=0",
            "B70_MTP_BF16_DRAFT=1",
            "B70_DRAFT_LMHEAD_INT4=1",
            "B70_DRAFT_MTP_INT4=1",
            "VLLM_XPU_ENABLE_XPU_GRAPH=1",
            "PYTORCH_ALLOC_CONF=expandable_segments:True",
        ]
        patch_commands = [
            "python /patch_mtp.py",
            "python /patch_boundary.py",
            "python /patch_v5.py",
            "python /patch_s.py",
            "python /patch_m1.py",
            "python /prefill_guard.py",
        ]
        patch_order = [role for _, _, role in patch_mounts] + ["prefill_guard"]
        stack_revision = "vLLM ac7509e2b / pinned image f01e24f6"
    elif cell == "dspark":
        root = args.dspark_root.resolve()
        campaign_root = args.dspark_campaign.resolve()
        image = DEFAULT_DSPARK_IMAGE
        target = args.target_dir.resolve()
        draft = args.draft_dir.resolve()
        overlay = campaign_root / "patch-dspark-native.py"
        serve = [
            "vllm",
            "serve",
            "/model",
            "--quantization",
            "gptq",
            "--dtype",
            "float16",
            "--max-model-len",
            str(CONTEXT),
            "--gpu-memory-utilization",
            "0.95",
            "--kv-cache-dtype",
            "fp8",
            "--port",
            "8000",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            str(BATCHED_TOKENS),
            "--no-enable-prefix-caching",
            "--mamba-cache-mode",
            "align",
            "--performance-mode",
            "balanced",
            "--chat-template-content-format",
            "openai",
            "--default-chat-template-kwargs",
            json_compact({"enable_thinking": False}),
            "--reasoning-parser",
            "qwen3",
            "--served-model-name",
            DEFAULT_MODEL,
            "--language-model-only",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            "--speculative-config",
            json_compact(
                {
                    "method": "dspark",
                    "model": "/draft",
                    "num_speculative_tokens": 7,
                    "kv_cache_dtype": "bfloat16",
                    "quantization": None,
                    "rejection_sample_method": "standard",
                    "draft_sample_method": "greedy",
                    "enable_adaptive_verification": False,
                }
            ),
            "--compilation-config",
            json_compact(
                {
                    "mode": 0,
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [7, 8],
                }
            ),
            "--cudagraph-metrics",
        ]
        mounts = [
            {"host": str(target), "container": "/model", "mode": "ro", "role": "target_model"},
            {"host": str(root), "container": "/experiment", "mode": "ro", "role": "dspark_previous_campaign"},
            {"host": str(draft), "container": "/draft", "mode": "ro", "role": "frozen_draft"},
            {
                "host": str(overlay),
                "container": "/experiment/patch_dspark_bf16.py",
                "mode": "ro",
                "role": "corrected_dspark_overlay",
            },
        ]
        env = [
            "VLLM_USE_V2_MODEL_RUNNER=1",
            "VLLM_TARGET_DEVICE=xpu",
            "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
            "ZE_AFFINITY_MASK=0",
            "PYTORCH_ALLOC_CONF=expandable_segments:True",
            "VLLM_XPU_ENABLE_XPU_GRAPH=1",
            "B70_DSPARK_BF16=1",
        ]
        patch_commands = [
            "/opt/venv/bin/python -P /experiment/apply-prefill.py",
            "/opt/venv/bin/python -P /experiment/patch_dspark_bf16.py",
            "/opt/venv/bin/python -P /experiment/patch-vllm-qwen38-xpu-boundary.py",
        ]
        patch_order = ["apply_prefill", "corrected_dspark_overlay", "xpu_boundary"]
        stack_revision = "vLLM 73029d424 / pinned image 7a558f63"
    else:
        raise CampaignError(f"unknown cell: {cell}")

    if profile_config is not None:
        serve.extend(profiler_argv(profile_config))

    argv: list[str] = [
        "docker",
        "run",
        "--pull=never",
        "--rm",
        "--name",
        name,
        "--ipc=host",
        "-p",
        "127.0.0.1:8000:8000",
        "--device",
        "/dev/dri",
        "--group-add",
        str(group_id),
        "-v",
        "/dev/dri:/dev/dri:ro",
    ]
    for mount in mounts:
        suffix = ":ro" if mount["mode"] == "ro" else ""
        argv.extend(["-v", f"{mount['host']}:{mount['container']}{suffix}"])
    argv.extend(["-v", f"{out.resolve()}:/output"])
    for item in env:
        argv.extend(["-e", item])
    argv.extend(["--entrypoint", "bash", image, "-lc"])
    command = "set -e; " + "; ".join(patch_commands) + "; exec " + shlex.join(serve)
    argv.append(command)

    metadata: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "cell": cell,
        "mode": "profile" if profile else "benchmark",
        "container_name": name,
        "image": image,
        "stack_revision": stack_revision,
        "target": str(target),
        "serve": serve,
        "common_contract": {
            "base_url": args.base_url,
            "served_model_name": DEFAULT_MODEL,
            "max_model_len": CONTEXT,
            "max_num_batched_tokens": BATCHED_TOKENS,
            "max_num_seqs": 1,
            "target_dtype": "float16 compute / GPTQ Int4 symmetric G128",
            "kv_cache_dtype": "fp8",
            "prefix_caching": False,
            "cpu_offload": False,
            "xpu_graphs": True,
            "thinking": False,
        },
        "speculation": (
            {"method": "mtp", "num_speculative_tokens": 4}
            if cell == "mtp4"
            else {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_dtype": "bfloat16",
                "draft_sample_method": "greedy",
                "rejection_sample_method": "standard",
                "enable_adaptive_verification": False,
            }
        ),
        "mounts": mounts + [{"host": str(out.resolve()), "container": "/output", "mode": "rw", "role": "new_run_output"}],
        "environment": env,
        "patch_order": patch_order,
        "profiler_config": profile_config,
        "intentional_comparison_difference": "whole runtime/image/patch-stack BUNDLE; do not attribute a cross-cell delta to one patch",
    }
    return argv, metadata


def required_assets(cell: str, args: argparse.Namespace) -> dict[str, Path]:
    if cell == "mtp4":
        root = args.mtp4_root
        assets = {
            "mtp4_root": root,
            "prefill_guard": root / "patch_uniform_decode_prefill.py",
            "mtp_nightly": root / "reference-source/patches/patch_mtp_nightly.py",
            "mtp_boundary": root / "reference-source/patches/patch_mtp_boundary.py",
            "gdn_mixed_split_v5": root / "reference-source/patches/patch_gdn_mixed_split_v5.py",
            "draft_lmhead_int4": root / "reference-source/patches/patch_draft_lmhead_int4.py",
            "draft_mtp_int4": root / "reference-source/patches/patch_draft_mtp_int4.py",
        }
    else:
        assets = {
            "dspark_root": args.dspark_root,
            "dspark_campaign": args.dspark_campaign,
            "apply_prefill": args.dspark_root / "apply-prefill.py",
            "dspark_overlay": args.dspark_campaign / "patch-dspark-native.py",
            "xpu_boundary": args.dspark_root / "patch-vllm-qwen38-xpu-boundary.py",
            "draft": args.draft_dir,
        }
    assets["target_model"] = args.target_dir
    for label, path in assets.items():
        if not path.exists():
            raise CampaignError(f"required {cell} asset is missing: {label}={path}")
    for label, path in assets.items():
        if label not in {"mtp4_root", "dspark_root", "dspark_campaign", "draft", "target_model"} and not path.is_file():
            raise CampaignError(f"required {cell} asset is not a file: {label}={path}")
    if cell == "dspark":
        observed = sha256_file(assets["dspark_overlay"])
        if observed != EXPECTED_DSPARK_OVERLAY_SHA256:
            raise CampaignError(
                f"corrected DSpark overlay hash mismatch: expected {EXPECTED_DSPARK_OVERLAY_SHA256}, observed {observed}"
            )
    return assets


def resolve_long_client(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.long_client is not None:
        candidates.append(args.long_client)
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / "scripts/experiments/qwen38_long_context_bench.py",
            args.mtp4_root / "qwen38_long_context_bench.py",
            args.dspark_root / "qwen38_long_context_bench.py",
        ]
    )
    for path in candidates:
        if path.is_file():
            observed = sha256_file(path)
            if observed != EXPECTED_LONG_CLIENT_SHA256:
                raise CampaignError(
                    f"long client must be reused unchanged ({EXPECTED_LONG_CLIENT_SHA256}); "
                    f"{path} has {observed}"
                )
            return path.resolve()
    raise CampaignError(
        "qwen38_long_context_bench.py not found; pass --long-client with the unchanged archived/source copy"
    )


def save_http_response(out: Path, stem: str, response: HTTPResponse) -> Any | None:
    write_bytes(out / f"{stem}.raw", response.body)
    record: dict[str, Any] = {
        "status": response.status,
        "headers": response.headers,
        "raw_path": f"{stem}.raw",
    }
    try:
        value = response.json(stem)
    except CampaignError as exc:
        record["json_error"] = str(exc)
        write_json(out / f"{stem}.json", record)
        return None
    write_json(out / f"{stem}.json", value)
    return value


def wait_for_server(
    proc: subprocess.Popen[str],
    client: PublicClient,
    out: Path,
    model: str,
    startup_timeout: float,
) -> Mapping[str, Any]:
    deadline = time.monotonic() + startup_timeout
    attempts: list[dict[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            write_json(out / "startup-attempts.json", {"attempts": attempts, "server_exit_code": proc.returncode})
            raise CampaignError(f"server container exited during startup with code {proc.returncode}")
        health: HTTPResponse | None = None
        models_response: HTTPResponse | None = None
        errors: list[str] = []
        try:
            health = client.request("/health", timeout_s=15)
        except CampaignError as exc:
            errors.append(f"health: {exc}")
        try:
            models_response = client.request("/v1/models", timeout_s=15)
        except CampaignError as exc:
            errors.append(f"models: {exc}")
        attempt = {
            "at_utc": utc_now(),
            "health_status": health.status if health is not None else None,
            "models_status": models_response.status if models_response is not None else None,
            "errors": errors,
        }
        attempts.append(attempt)
        if models_response is not None and models_response.status == 200:
            models = models_response.json("/v1/models")
            write_json(out / "models.json", models)
            entries = models.get("data") if isinstance(models, Mapping) else None
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, Mapping) and entry.get("id") == model:
                        selected = entry
                        break
            if selected is not None:
                max_lengths = [
                    value
                    for key, value in selected.items()
                    if str(key).lower().replace("-", "_") in {"max_model_len", "max_context_len", "max_context_length"}
                ]
                if not max_lengths:
                    raise CampaignError("/v1/models did not expose max_model_len for the selected model")
                if int(max_lengths[0]) != CONTEXT:
                    raise CampaignError(
                        f"served model context mismatch: expected {CONTEXT}, observed {max_lengths[0]}"
                    )
                break
        time.sleep(3)
    write_json(out / "startup-attempts.json", {"attempts": attempts})
    if selected is None:
        raise CampaignError(f"server did not expose {model!r} at the expected context within {startup_timeout}s")
    health = client.request("/health", timeout_s=30)
    save_http_response(out, "health-start", health)
    write_json(out / "model-selection.json", {"requested_model": model, "selected_model": selected})
    info = client.request("/server_info", timeout_s=30)
    info_value = save_http_response(out, "server-info", info)
    if isinstance(info_value, Mapping):
        for key in ("enable_prefix_caching", "enable_prefix_cache"):
            if info_value.get(key) is True:
                raise CampaignError(f"/server_info reports {key}=true")
    return selected


def container_inspect(name: str) -> tuple[bool, Any | None]:
    result = command_result("docker", "inspect", name, timeout=30)
    if result.returncode != 0:
        return False, None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return True, {"raw": result.stdout, "parse_error": "invalid docker inspect JSON"}
    if isinstance(value, list) and value:
        return True, value[0]
    return True, value


def observe_owned_container(name: str, image: str, timeout_s: float = 30.0) -> tuple[bool, Any | None]:
    deadline = time.monotonic() + timeout_s
    last: Any | None = None
    while time.monotonic() < deadline:
        exists, value = container_inspect(name)
        if exists:
            last = value
            configured = value.get("Config", {}).get("Image") if isinstance(value, Mapping) else None
            if configured == image:
                return True, value
        time.sleep(0.25)
    return False, last


def capture_runtime_identity(out: Path, name: str) -> None:
    inspect = command_result("docker", "inspect", name, timeout=60)
    write_text(out / "container-inspect.raw", inspect.stdout + inspect.stderr)
    try:
        write_json(out / "container-inspect.json", json.loads(inspect.stdout))
    except json.JSONDecodeError:
        write_json(out / "container-inspect.json", {"returncode": inspect.returncode, "raw": inspect.stdout, "stderr": inspect.stderr})

    identity_code = (
        "import json, torch, vllm; "
        "print(json.dumps({'vllm': getattr(vllm, '__version__', 'unknown'), "
        "'torch': getattr(torch, '__version__', 'unknown'), "
        "'xpu_available': bool(getattr(torch, 'xpu', None) and torch.xpu.is_available())}))"
    )
    identity = command_result(
        "docker", "exec", name, "/opt/venv/bin/python", "-P", "-c", identity_code, timeout=120
    )
    write_json(
        out / "runtime-identity.json",
        {
            "argv": ["docker", "exec", name, "/opt/venv/bin/python", "-P", "-c", identity_code],
            "returncode": identity.returncode,
            "stdout": identity.stdout,
            "stderr": identity.stderr,
        },
    )
    collect = command_result("docker", "exec", name, "python", "-m", "vllm.collect_env", timeout=180)
    write_text(out / "collect-env.txt", collect.stdout + ("\nSTDERR:\n" + collect.stderr if collect.stderr else ""))
    write_text(out / "collect-env.command.txt", shlex.join(["docker", "exec", name, "python", "-m", "vllm.collect_env"]) + "\n")
    write_json(
        out / "collect-env-result.json",
        {"returncode": collect.returncode, "stdout_bytes": len(collect.stdout), "stderr": collect.stderr},
    )


def parse_prometheus(raw: str) -> tuple[dict[str, float], list[str]]:
    values: dict[str, float] = {}
    errors: list[str] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line or line.startswith("#") or not line.startswith("vllm:"):
            continue
        try:
            key, value_text = line.rsplit(None, 1)
            value = float(value_text)
        except (ValueError, IndexError):
            errors.append(f"line {line_number}: not a metric: {line[:200]}")
            continue
        if not math.isfinite(value):
            errors.append(f"line {line_number}: non-finite metric value")
            continue
        values[key] = value
    return values, errors


def validate_metric_snapshots(root: Path, *, require_speculation: bool = True) -> dict[str, Any]:
    raw_files = sorted(root.rglob("metrics-*.raw"))
    if not raw_files:
        raise CampaignError(f"no raw /metrics snapshots under {root}")
    snapshots: dict[Path, dict[str, float]] = {}
    parse_errors: dict[str, list[str]] = {}
    for path in raw_files:
        values, errors = parse_prometheus(path.read_text(encoding="utf-8", errors="replace"))
        snapshots[path] = values
        if errors:
            parse_errors[str(path.relative_to(root))] = errors
    required_bases = {
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    }
    observed_bases = {key.split("{", 1)[0] for values in snapshots.values() for key in values}
    missing = sorted(required_bases - observed_bases) if require_speculation else []
    if parse_errors or missing:
        raise CampaignError(
            f"metrics validation failed: parse_errors={parse_errors}, missing_spec_counters={missing}"
        )

    counter_errors: list[str] = []
    for before_path in sorted(root.rglob("metrics-before.raw")):
        after_path = before_path.with_name("metrics-after.raw")
        if after_path not in snapshots:
            counter_errors.append(f"missing after snapshot for {before_path.relative_to(root)}")
            continue
        before = snapshots[before_path]
        after = snapshots[after_path]
        for key, before_value in before.items():
            if "_total" not in key:
                continue
            if key in after and after[key] + 1e-6 < before_value:
                counter_errors.append(
                    f"counter decreased for {key}: {before_value} -> {after[key]}"
                )
    if counter_errors:
        raise CampaignError(f"metrics counter validation failed: {counter_errors}")
    return {
        "raw_snapshot_count": len(raw_files),
        "spec_counter_bases": sorted(observed_bases & required_bases),
        "counter_errors": [],
        "parse_errors": {},
    }


def validate_long_result(root: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError("long-context summary is not an object")
    if value.get("status") != "completed":
        raise CampaignError(f"long-context summary status is {value.get('status')!r}")
    if value.get("errors"):
        raise CampaignError(f"long-context summary contains errors: {value['errors']}")
    points = value.get("points")
    if not isinstance(points, list) or len(points) != 1:
        raise CampaignError(f"expected exactly one selected 64K point, observed {points!r}")
    point = points[0]
    if point.get("requested_length") != PROMPT_TOKENS or point.get("status") != "complete":
        raise CampaignError(f"64K point is not complete: {point!r}")
    attempts = [point.get("warmup")] + list(point.get("measurements") or [])
    if point.get("warmup", {}).get("valid") is not True:
        raise CampaignError("64K warmup is invalid")
    if len(point.get("measurements") or []) != 6:
        raise CampaignError("64K point does not contain six measured trials")
    for index, row in enumerate(attempts):
        if not isinstance(row, Mapping) or row.get("valid") is not True:
            raise CampaignError(f"invalid 64K attempt {index}: {row!r}")
        validation = row.get("validation") or {}
        if validation.get("prompt_tokens") != PROMPT_TOKENS or validation.get("completion_tokens") != OUTPUT_TOKENS:
            raise CampaignError(f"wrong token counts in 64K attempt {index}: {validation!r}")
        stream = row.get("stream") or {}
        if stream.get("parse_errors") or stream.get("finish_reason") != "length":
            raise CampaignError(f"bad stream in 64K attempt {index}: {stream!r}")
    metrics = validate_metric_snapshots(root, require_speculation=True)
    return {
        "point_count": 1,
        "warmup_count": 1,
        "measured_count": 6,
        "expected_prompt_tokens": PROMPT_TOKENS,
        "expected_completion_tokens": OUTPUT_TOKENS,
        "metrics": metrics,
    }


def resolve_prompt_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen38_step_profile_long_client", path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot load long client: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = ("content_segments", "compose_prompt", "completion_payload", "IM_END_ID")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise CampaignError(f"long client lacks prompt helpers {missing}: {path}")
    return module


def tokenize(client: PublicClient, out: Path, stem: str, payload: Mapping[str, Any]) -> list[int]:
    write_json(out / f"{stem}-request.json", payload)
    response = client.request("/tokenize", payload)
    write_bytes(out / f"{stem}-response.raw", response.body)
    if response.status >= 400:
        raise HTTPFailure("/tokenize", response.status, response.body)
    value = response.json(f"/tokenize {stem}")
    write_json(out / f"{stem}-response.json", value)
    tokens = value.get("tokens") if isinstance(value, Mapping) else None
    if tokens is None and isinstance(value, Mapping):
        tokens = value.get("prompt_token_ids")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise CampaignError(f"/tokenize {stem} did not return an integer token list")
    return list(tokens)


def render_profile_prompt(client: PublicClient, out: Path, module: Any) -> tuple[list[int], str]:
    rendering = out / "rendering"
    template_payload = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": ""}],
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    template = tokenize(client, rendering, "profile-template", template_payload)
    if not any(token == module.IM_END_ID for token in template):
        raise CampaignError(f"profile template lacks Qwen im_end token {module.IM_END_ID}")
    _, body_text, _, _ = module.content_segments(512, "body")
    body = tokenize(
        client,
        rendering,
        "profile-content-body",
        {"model": DEFAULT_MODEL, "prompt": body_text, "add_special_tokens": False},
    )
    prefix_text, _, tail_text, nonce = module.content_segments(PROMPT_TOKENS, "profile")
    prefix = tokenize(
        client,
        rendering,
        "profile-content-prefix",
        {"model": DEFAULT_MODEL, "prompt": prefix_text, "add_special_tokens": False},
    )
    tail = tokenize(
        client,
        rendering,
        "profile-content-tail",
        {"model": DEFAULT_MODEL, "prompt": tail_text, "add_special_tokens": False},
    )
    prompt = module.compose_prompt(template, prefix, body, tail, PROMPT_TOKENS, module.IM_END_ID)
    write_json(
        out / "rendering/profile-rendered-prompt.json",
        {
            "requested_length": PROMPT_TOKENS,
            "trial": "profile",
            "nonce": nonce,
            "prompt_token_count": len(prompt),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
            "prompt": prompt,
            "source": {"prefix": prefix_text, "body": body_text, "tail": tail_text},
        },
    )
    return prompt, nonce


def completion_text(chunk: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        text = choice.get("text")
        if isinstance(text, str):
            pieces.append(text)
            continue
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str):
                pieces.append(content)
    return "".join(pieces)


def post_profile_control(client: PublicClient, out: Path, label: str, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    request_path = out / f"profile-{label}-request.json"
    write_json(request_path, {})
    try:
        response = client.request(f"/{label}_profile", {}, timeout_s=timeout_s)
        write_bytes(out / f"profile-{label}-response.raw", response.body)
        result: dict[str, Any] = {
            "label": label,
            "status": response.status,
            "started_monotonic": started,
            "finished_monotonic": time.monotonic(),
            "response_bytes": len(response.body),
        }
        write_json(out / f"profile-{label}.json", result)
        if response.status >= 400:
            raise HTTPFailure(f"/{label}_profile", response.status, response.body)
        return result
    except BaseException as exc:
        result = {
            "label": label,
            "status": "error",
            "started_monotonic": started,
            "finished_monotonic": time.monotonic(),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        write_json(out / f"profile-{label}.json", result)
        raise


def run_profile_request(
    client: PublicClient,
    out: Path,
    module: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt, nonce = render_profile_prompt(client, out, module)
    payload = module.completion_payload(DEFAULT_MODEL, prompt)
    write_json(out / "profile-request.json", payload)
    write_bytes(
        out / "profile-request.raw",
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
    )
    before = client.request("/metrics")
    write_bytes(out / "metrics-before.raw", before.body)
    before_values, before_errors = parse_prometheus(before.body.decode("utf-8", "replace"))
    write_json(
        out / "metrics-before.json",
        {"status": before.status, "values": before_values, "parse_errors": before_errors},
    )

    control_timeout = args.control_timeout
    states: dict[str, Any] = {"start": None, "stop": None}
    start_thread: threading.Thread | None = None
    stop_thread: threading.Thread | None = None
    first_output_event: int | None = None
    first_output_elapsed: float | None = None
    nonempty_events = 0
    done = False
    finish_reason: Any = None
    usage: Mapping[str, Any] | None = None
    parse_errors: list[str] = []
    error_objects: list[Any] = []
    output_parts: list[str] = []
    event_count = 0
    started = time.monotonic()

    def start_worker() -> None:
        try:
            states["start"] = post_profile_control(client, out, "start", control_timeout)
        except BaseException as exc:
            states["start"] = {"status": "error", "error": str(exc)}

    def stop_worker() -> None:
        if start_thread is not None:
            start_thread.join(timeout=control_timeout)
        try:
            states["stop"] = post_profile_control(client, out, "stop", control_timeout)
        except BaseException as exc:
            states["stop"] = {"status": "error", "error": str(exc)}

    raw_path = out / "profile-sse.raw"
    jsonl_path = out / "profile-sse.jsonl"
    try:
        with client.open_stream("/v1/completions", payload) as response, raw_path.open("wb") as raw_file, jsonl_path.open("w", encoding="utf-8") as parsed_file:
            for raw_line in response:
                now = time.monotonic()
                elapsed = now - started
                raw_file.write(raw_line)
                raw_file.flush()
                line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                event_count += 1
                parsed_event: dict[str, Any] = {"elapsed_s": elapsed, "raw": line}
                if data == "[DONE]":
                    done = True
                    parsed_event["done"] = True
                    parsed_file.write(json.dumps(parsed_event, ensure_ascii=False) + "\n")
                    parsed_file.flush()
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    parse_errors.append(f"event {event_count}: {exc}")
                    parsed_event["parse_error"] = str(exc)
                    parsed_file.write(json.dumps(parsed_event, ensure_ascii=False) + "\n")
                    parsed_file.flush()
                    continue
                if not isinstance(chunk, Mapping):
                    parse_errors.append(f"event {event_count}: SSE JSON is not an object")
                    parsed_event["parse_error"] = "not an object"
                else:
                    text = completion_text(chunk)
                    parsed_event["text"] = text
                    if chunk.get("error") is not None:
                        error_objects.append(chunk.get("error"))
                    if isinstance(chunk.get("usage"), Mapping):
                        usage = chunk["usage"]
                    choices = chunk.get("choices")
                    if isinstance(choices, list):
                        for choice in choices:
                            if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                                finish_reason = choice.get("finish_reason")
                    if text:
                        output_parts.append(text)
                        nonempty_events += 1
                        if first_output_event is None:
                            first_output_event = event_count
                            first_output_elapsed = elapsed
                            start_thread = threading.Thread(target=start_worker, name="profile-start", daemon=True)
                            start_thread.start()
                            write_json(
                                out / "profile-boundary.json",
                                {
                                    "activation": "after_first_nonempty_sse_output",
                                    "first_output_event": first_output_event,
                                    "first_output_elapsed_s": first_output_elapsed,
                                    "delay_iterations": args.profile_delay_iterations,
                                },
                            )
                        if nonempty_events >= args.profile_stop_after_events and stop_thread is None:
                            stop_thread = threading.Thread(target=stop_worker, name="profile-stop", daemon=True)
                            stop_thread.start()
                parsed_file.write(json.dumps(parsed_event, ensure_ascii=False) + "\n")
                parsed_file.flush()
    except BaseException as exc:
        write_json(out / "profile-stream-error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if start_thread is not None:
            start_thread.join(timeout=control_timeout)
        if start_thread is not None and stop_thread is None:
            stop_thread = threading.Thread(target=stop_worker, name="profile-stop-finally", daemon=True)
            stop_thread.start()
        if stop_thread is not None:
            stop_thread.join(timeout=control_timeout * 2)
        write_json(
            out / "profile-control-state.json",
            {
                "start": states.get("start"),
                "stop": states.get("stop"),
                "start_thread_alive": start_thread.is_alive() if start_thread is not None else False,
                "stop_thread_alive": stop_thread.is_alive() if stop_thread is not None else False,
            },
        )


    if first_output_event is None:
        raise CampaignError("profile request produced no non-empty output event; profiler was not started")
    if start_thread is not None and start_thread.is_alive():
        raise CampaignError("/start_profile control request did not finish within the control timeout")
    if stop_thread is not None and stop_thread.is_alive():
        raise CampaignError("/stop_profile control request did not finish within the control timeout")
    if states.get("start", {}).get("status") == "error":
        raise CampaignError(f"/start_profile failed: {states['start']}")
    if states.get("stop", {}).get("status") == "error":
        raise CampaignError(f"/stop_profile failed: {states['stop']}")
    if not done or finish_reason != "length" or parse_errors or error_objects:
        raise CampaignError(
            f"profile stream validation failed: done={done}, finish_reason={finish_reason!r}, "
            f"parse_errors={parse_errors}, errors={error_objects}"
        )
    if not isinstance(usage, Mapping) or usage.get("prompt_tokens") != PROMPT_TOKENS or usage.get("completion_tokens") != OUTPUT_TOKENS:
        raise CampaignError(f"profile usage counts are wrong: {usage!r}")

    write_text(out / "profile-output.txt", "".join(output_parts))
    after = client.request("/metrics")
    write_bytes(out / "metrics-after.raw", after.body)
    after_values, after_errors = parse_prometheus(after.body.decode("utf-8", "replace"))
    write_json(
        out / "metrics-after.json",
        {"status": after.status, "values": after_values, "parse_errors": after_errors},
    )
    metric_validation = validate_metric_snapshots(out, require_speculation=True)
    stream_summary = {
        "requested_length": PROMPT_TOKENS,
        "nonce": nonce,
        "event_count": event_count,
        "nonempty_events": nonempty_events,
        "first_output_event": first_output_event,
        "first_output_elapsed_s": first_output_elapsed,
        "done": done,
        "finish_reason": finish_reason,
        "usage": dict(usage),
        "parse_errors": parse_errors,
        "error_objects": error_objects,
        "controls": states,
        "activation_contract": "start_profile launched only after first non-empty SSE output event; delay applies only to later worker steps",
        "metric_validation": metric_validation,
    }
    write_json(out / "profile-stream-summary.json", stream_summary)
    return stream_summary


def wait_for_profile_artifacts(out: Path, timeout_s: float) -> dict[str, Any]:
    profile_dir = out / "profile"
    deadline = time.monotonic() + timeout_s
    files: list[Path] = []
    while time.monotonic() < deadline:
        files = sorted(path for path in profile_dir.rglob("*") if path.is_file()) if profile_dir.is_dir() else []
        if files:
            break
        time.sleep(1)
    if not files:
        raise CampaignError(f"Torch profiler produced no files under {profile_dir} within {timeout_s}s")
    records = [
        {
            "path": str(path.relative_to(out)),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    result = {
        "directory": "profile",
        "files": records,
        "trace_files_are_host_local": True,
        "trace_interpretation": "inspect event annotations and reject any prefill/context-contaminated sample before ranking kernels",
    }
    write_json(out / "profile-artifacts.json", result)
    return result


def run_benchmark(
    out: Path,
    args: argparse.Namespace,
    long_client: Path,
) -> dict[str, Any]:
    client_argv = [
        sys.executable,
        "-B",
        str(long_client),
        "--base-url",
        args.base_url,
        "--model",
        DEFAULT_MODEL,
        "--out",
        str(out / "long-context"),
        "--lengths",
        str(PROMPT_TOKENS),
        "--near-limit",
        str(PROMPT_TOKENS),
        "--confirm-prefix-cache-disabled",
    ]
    write_json(out / "long-client-argv.json", client_argv)
    write_text(out / "long-client-command.txt", shlex.join(client_argv) + "\n")
    try:
        result = subprocess.run(
            client_argv,
            capture_output=True,
            text=True,
            timeout=args.benchmark_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        write_text(out / "long-client-output.txt", (exc.stdout or "") + "\nTIMEOUT\n" + (exc.stderr or ""))
        raise CampaignError(f"long client exceeded {args.benchmark_timeout}s") from exc
    write_text(out / "long-client-output.txt", result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""))
    write_json(
        out / "long-client-result.json",
        {"returncode": result.returncode, "stdout_bytes": len(result.stdout), "stderr": result.stderr},
    )
    summary_path = out / "long-context/summary.json"
    if not summary_path.is_file():
        raise CampaignError("long client did not produce long-context/summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validation = validate_long_result(out / "long-context", summary)
    return {"summary": summary, "validation": validation}


def cleanup_owned_container(name: str, owned: bool) -> dict[str, Any]:
    if not owned:
        return {
            "container_name": name,
            "owned_container_observed": False,
            "status": "skipped",
            "reason": "container was never observed with the expected image; no removal attempted",
        }
    result = command_result("docker", "rm", "-f", name, timeout=60)
    return {
        "argv": ["docker", "rm", "-f", name],
        "container_name": name,
        "owned_container_observed": True,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "removed" if result.returncode == 0 else "remove_failed_or_already_gone",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("mtp4", "dspark"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory for this one cell")
    parser.add_argument("--profile", dest="profile_mode", action="store_true", help="run one profiled 64K completion")
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--long-client", type=Path)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--mtp4-root", type=Path, default=DEFAULT_MTP_ROOT)
    parser.add_argument("--dspark-root", type=Path, default=DEFAULT_DSPARK_ROOT)
    parser.add_argument("--dspark-campaign", type=Path, default=DEFAULT_DSPARK_CAMPAIGN)
    parser.add_argument("--draft-dir", type=Path, default=DEFAULT_DSPARK_ROOT / "draft")
    parser.add_argument("--production-launcher", type=Path, default=DEFAULT_PRODUCTION_LAUNCHER)
    parser.add_argument("--power-path", type=Path, default=DEFAULT_POWER_PATH)
    parser.add_argument("--glimmer-name", default=DEFAULT_GLIMMER_NAME)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--benchmark-timeout", type=float, default=10_800.0)
    parser.add_argument("--control-timeout", type=float, default=120.0)
    parser.add_argument("--profile-delay-iterations", type=int, default=0)
    parser.add_argument("--profile-max-iterations", type=int, default=5)
    parser.add_argument("--profile-stop-after-events", type=int, default=8)
    parser.add_argument("--profile-artifact-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.profile_mode and args.mode != "benchmark":
        parser.error("use either --profile or --mode profile, not both")
    args.profile_mode = args.profile_mode or args.mode == "profile"
    for name in (
        "startup_timeout",
        "request_timeout",
        "benchmark_timeout",
        "control_timeout",
        "profile_artifact_timeout",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.profile_delay_iterations < 0 or args.profile_max_iterations <= 0 or args.profile_stop_after_events <= 0:
        parser.error("profile iteration/event limits must be non-negative/positive")
    if args.profile_delay_iterations == 2:
        parser.error("delay_iterations=2 is intentionally rejected; it is the prefill-contaminated proposal")
    if args.out.exists():
        parser.error(f"--out must name a new directory: {args.out}")
    return args


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    name = cell_name(args.cell)
    image = DEFAULT_MTP_IMAGE if args.cell == "mtp4" else DEFAULT_DSPARK_IMAGE
    summary: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "tier": "development",
        "status": "starting",
        "cell": args.cell,
        "mode": "profile" if args.profile_mode else "benchmark",
        "purpose": "64K profiling/paired throughput evidence; not a publishable result or production promotion",
        "host_contract": {
            "production_launcher_sha256": EXPECTED_LAUNCHER_SHA256,
            "power_cap_microwatts": "275000000",
            "no_running_containers": True,
            "glimmer_stopped": True,
        },
        "workload_contract": {
            "prompt_tokens": PROMPT_TOKENS,
            "completion_tokens": OUTPUT_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "ignore_eos": True,
            "prefix_cache": False,
        },
    }
    write_json(
        out / "driver-args.json",
        {"argv": list(sys.argv), "cwd": os.getcwd(), "started_utc": utc_now()},
    )
    proc: subprocess.Popen[str] | None = None
    log_handle: Any = None
    owned = False
    before: dict[str, Any] | None = None
    failure: str | None = None
    old_handlers: dict[int, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise CampaignInterrupted(f"received signal {signum}")

    try:
        old_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, interrupt)
        old_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, interrupt)
        assets = required_assets(args.cell, args)
        write_json(
            out / "asset-manifest.json",
            {label: {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None} for label, path in assets.items()},
        )
        long_client = resolve_long_client(args)
        write_json(
            out / "long-client-source.json",
            {"path": str(long_client), "sha256": sha256_file(long_client), "unchanged_expected_sha256": EXPECTED_LONG_CLIENT_SHA256},
        )
        argv, metadata = build_launch(args.cell, out, args)
        write_json(out / "launch-argv.json", argv)
        write_json(out / "launch-metadata.json", metadata)
        if metadata.get("profiler_config") is not None:
            write_json(out / "profiler-config.json", metadata["profiler_config"])
        write_text(out / "launcher.sh", shell_script(argv))
        (out / "launcher.sh").chmod(0o755)
        image_inspect = command_result("docker", "image", "inspect", image, timeout=120)
        write_text(out / "image-inspect.raw", image_inspect.stdout + image_inspect.stderr)
        try:
            write_json(out / "image-inspect.json", json.loads(image_inspect.stdout))
        except json.JSONDecodeError:
            write_json(out / "image-inspect.json", {"returncode": image_inspect.returncode, "stderr": image_inspect.stderr})
        if image_inspect.returncode != 0:
            raise CampaignError(f"required image is not installed: {image}")

        write_text(
            out / "host-command.txt",
            "\n".join(
                [
                    shlex.join(["sha256sum", str(args.production_launcher)]),
                    shlex.join(["cat", str(args.power_path)]),
                    "docker ps --format '{{.Names}}'",
                    "docker ps -a --format '{{.Names}}'",
                    shlex.join(["docker", "inspect", "-f", "{{.State.Running}}", args.glimmer_name]),
                ]
            )
            + "\n",
        )
        before = host_invariants(args.production_launcher, args.power_path, args.glimmer_name)
        write_json(out / "host-before.json", before)
        assert_idle_host(before, name)
        log_handle = (out / "server.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        owned, observed = observe_owned_container(name, image)
        if observed is not None:
            write_json(out / "container-observed.json", observed)
        if not owned:
            raise CampaignError("docker run started but the expected owned container was never observed")
        selected_model = wait_for_server(
            proc,
            PublicClient(args.base_url, args.request_timeout),
            out,
            DEFAULT_MODEL,
            args.startup_timeout,
        )
        summary["served_model"] = selected_model
        write_json(
            out / "effective-config.json",
            {
                "requested_server_arguments": metadata["serve"],
                "common_contract": metadata["common_contract"],
                "speculation": metadata["speculation"],
                "profiler_config": metadata.get("profiler_config"),
                "served_model": selected_model,
            },
        )
        capture_runtime_identity(out, name)
        client = PublicClient(args.base_url, args.request_timeout)
        if args.profile_mode:
            module = resolve_prompt_module(long_client)
            summary["profile"] = run_profile_request(client, out, module, args)
            summary["profile_artifacts"] = wait_for_profile_artifacts(out, args.profile_artifact_timeout)
        else:
            summary["benchmark"] = run_benchmark(out, args, long_client)
        summary["status"] = "passed"
    except BaseException as exc:
        failure = traceback.format_exc()
        write_text(out / "failure.txt", failure)
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        cleanup_error: str | None = None
        try:
            cleanup = cleanup_owned_container(name, owned)
            write_json(out / "container-cleanup.json", cleanup)
            if owned and cleanup.get("returncode") not in (0, None):
                # --rm may have removed the container before this explicit cleanup.
                exists, _ = container_inspect(name)
                cleanup["container_present_after_cleanup"] = exists
                write_json(out / "container-cleanup.json", cleanup)
                if exists:
                    cleanup_error = f"owned container cleanup failed: {cleanup}"
        except BaseException as exc:
            cleanup_error = traceback.format_exc()
            write_text(out / "cleanup-error.txt", cleanup_error)
        if proc is not None:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()
        try:
            after = host_invariants(args.production_launcher, args.power_path, args.glimmer_name)
            write_json(out / "host-after.json", after)
            summary["host_unchanged"] = before is not None and after == before
            if before is not None:
                try:
                    assert_host_unchanged(before, after)
                except CampaignError as exc:
                    if failure is None:
                        failure = str(exc)
                        write_text(out / "failure.txt", failure + "\n")
                    summary["status"] = "failed"
                    summary["host_invariant_error"] = str(exc)
        except BaseException as exc:
            summary["host_after_error"] = {"type": type(exc).__name__, "message": str(exc)}
            if failure is None:
                failure = traceback.format_exc()
                write_text(out / "failure.txt", failure)
            summary["status"] = "failed"
        if cleanup_error is not None:
            summary["cleanup_error"] = cleanup_error
            if failure is None:
                failure = cleanup_error
                write_text(out / "failure.txt", cleanup_error + "\n")
            summary["status"] = "failed"
        summary["finished_utc"] = utc_now()
        write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary.get("status") == "passed" else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
