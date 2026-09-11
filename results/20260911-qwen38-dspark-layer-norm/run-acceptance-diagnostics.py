#!/usr/bin/env python3
"""Bounded, disposable DSpark acceptance diagnostic for the B70 public API.

This is an experiment driver, not a launcher or runtime patch.  It mounts the
already-frozen overlays and draft snapshot from the preceding campaign into a
new disposable container, runs the existing correctness gates, and records a
small fixed matrix of streamed requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def previous_campaign() -> Path:
    candidates = (
        ROOT.parent / "20260910-qwen38-dspark-v2-feasibility",
        ROOT.parent / "20260910-dspark-v2-feasibility",
    )
    for candidate in candidates:
        if (candidate / "qwen38_lossy_probe.py").is_file() and (candidate / "draft").is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / "qwen38_lossy_probe.py").is_file():
            return candidate
    return candidates[0]


PREVIOUS = previous_campaign()


IMAGE = "vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4"
TARGET = "/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
LAUNCHER = Path("/home/mike/inference/launchers/start-qwen38.sh")
POWER = Path("/sys/class/drm/card0/device/hwmon/hwmon2/power1_cap")
BASE = "http://127.0.0.1:8000"
CONTEXT = 8192
LAUNCHER_SHA = "63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4"
DRAFT_REVISION = "b9a5dbdf03bc999c6c73c426b19c2d9041cea393"
DRAFT_CONFIG_SHA = "dd65fb1b01c2adea69512ff2990a79d58eb7fe2c7ea97375aa66f657a29a5bfd"
DRAFT_MODEL_SHA = "2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7"

MATH = (
    "Solve this deterministic arithmetic problem exactly. Let a=137, b=29, "
    "and c=11. Compute (a*b + c**3) - (a-b)*c, show the intermediate "
    "integer arithmetic, and state the final answer clearly. Do not use code."
)
PROMPT_NAMES = ("code", "math", "prose")
SAMPLING = (
    ("temp0", 0.0, 1.0, -1),
    ("temp1", 1.0, 0.95, 20),
)
SEEDS = (42, 43, 44)
FORCED_OUTPUT = 512
MATRIX_REQUESTS = 36

SPEC_CONFIG = {
    "method": "dspark",
    "model": "/draft",
    "num_speculative_tokens": 7,
    "kv_cache_dtype": "bfloat16",
    "quantization": None,
    "rejection_sample_method": "standard",
    "draft_sample_method": "greedy",
    "enable_adaptive_verification": False,
}

METRIC_PREFIXES = (
    "vllm:spec_decode_",
    "vllm:prefix_cache_",
    "vllm:external_prefix_cache_",
)
POSITION_RE = re.compile(
    r"vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position=\"([^\"]+)\""
)


def save(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*argv: str, timeout: int = 60, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        **kwargs,
    )


def command_stdout(*argv: str, timeout: int = 60) -> str:
    return command(*argv, timeout=timeout).stdout


def host_invariants() -> dict[str, Any]:
    return {
        "launcher_sha256": sha256(LAUNCHER),
        "power_cap": POWER.read_text().strip(),
        "running_containers": command_stdout(
            "docker", "ps", "--format", "{{.Names}}"
        ).splitlines(),
        "glimmer_running": command_stdout(
            "docker", "inspect", "-f", "{{.State.Running}}", "glimmer-tb21-prefix-c8"
        ).strip(),
    }


def expected_host_invariants() -> dict[str, Any]:
    return {
        "launcher_sha256": LAUNCHER_SHA,
        "power_cap": "275000000",
        "running_containers": [],
        "glimmer_running": "false",
    }


def load_previous_modules() -> tuple[Any, dict[str, Any]]:
    if not PREVIOUS.is_dir():
        raise RuntimeError(f"previous campaign is missing: {PREVIOUS}")
    sys.path.insert(0, str(PREVIOUS))
    import importlib
    import runpy

    probe = importlib.import_module("qwen38_lossy_probe")
    checks = runpy.run_path(str(PREVIOUS / "dspark-smoke-checks.py"))
    return probe, checks


def validate_assets(draft: Path, *, require_draft: bool) -> dict[str, Any]:
    files = {
        "probe": PREVIOUS / "qwen38_lossy_probe.py",
        "checks": PREVIOUS / "dspark-smoke-checks.py",
        "prefill_runner": PREVIOUS / "apply-prefill.py",
        "prefill_overlay": PREVIOUS / "patch_xpu_prefill.py",
        "draft_overlay": PREVIOUS / "patch_dspark_bf16.py",
        "boundary_overlay": PREVIOUS / "patch-vllm-qwen38-xpu-boundary.py",
    }
    for name, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen dependency {name}: {path}")
    draft_record: dict[str, Any] | None = None
    if require_draft:
        if not draft.is_dir():
            raise RuntimeError(f"frozen draft directory is missing: {draft}")
        config = draft / "config.json"
        weights = draft / "model.safetensors"
        if not config.is_file() or not weights.is_file():
            raise RuntimeError(f"frozen draft must contain config.json and model.safetensors: {draft}")
        if sha256(config) != DRAFT_CONFIG_SHA:
            raise RuntimeError("draft config hash differs from the frozen remote snapshot")
        if weights.stat().st_size <= 0:
            raise RuntimeError("frozen draft weights are empty")
        if PREVIOUS.resolve() not in draft.resolve().parents:
            raise RuntimeError("draft must remain the previous campaign's remote snapshot")
        draft_record = {
            "mount": str(draft),
            "revision": DRAFT_REVISION,
            "config_sha256": DRAFT_CONFIG_SHA,
            "model_sha256_expected": DRAFT_MODEL_SHA,
            "model_bytes": weights.stat().st_size,
            "read_only": True,
        }
    return {
        "previous_campaign": str(PREVIOUS),
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
        "draft": draft_record,
    }


def metric_snapshot(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            key, value_text = line.rsplit(None, 1)
            value = float(value_text)
        except (ValueError, IndexError):
            continue
        if any(key.startswith(prefix) for prefix in METRIC_PREFIXES) and math.isfinite(value):
            values[key] = value
    return values


def all_metrics_finite(raw: str) -> bool:
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            value = float(line.rsplit(None, 1)[1])
        except (ValueError, IndexError):
            continue
        if not math.isfinite(value):
            return False
    return True


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys}


def exact_metric_total(delta: dict[str, float], family: str) -> float:
    prefix = f"vllm:spec_decode_num_{family}_total{{"
    return sum(value for key, value in delta.items() if key.startswith(prefix))


def position_counters(values: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        match = POSITION_RE.match(key)
        if match:
            result[match.group(1)] = value
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def dependency_mounts(out: Path, draft: Path, cell: str, kv_cache_dtype: str) -> tuple[list[str], dict[str, Any]]:
    if not Path("/dev/dri/renderD128").exists():
        raise RuntimeError("/dev/dri/renderD128 is required on the inference host")
    name = "qwen38-dspark-acceptance-" + ("target" if cell == "target" else "dspark")
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
        kv_cache_dtype,
        "--port",
        "8000",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
        "--no-enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--performance-mode",
        "balanced",
        "--chat-template-content-format",
        "openai",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
        "--reasoning-parser",
        "qwen3",
        "--served-model-name",
        "qwen38",
        "--language-model-only",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--enforce-eager",
    ]
    if cell == "dspark":
        serve += ["--speculative-config", json.dumps(SPEC_CONFIG, separators=(",", ":"))]
    patch_commands = ["/opt/venv/bin/python -P /experiment/apply-prefill.py"]
    if cell == "dspark":
        patch_commands += [
            "/opt/venv/bin/python -P /experiment/patch_dspark_bf16.py",
            "/opt/venv/bin/python -P /experiment/patch-vllm-qwen38-xpu-boundary.py",
        ]
    patch_commands.append("exec " + shlex.join(serve))
    argv = [
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
        str(Path("/dev/dri/renderD128").stat().st_gid),
        "-v",
        "/dev/dri:/dev/dri:ro",
        "-v",
        f"{TARGET}:/model:ro",
        "-v",
        f"{PREVIOUS}:/experiment:ro",
        "-v",
        f"{draft}:/draft:ro",
        "-v",
        f"{out}:/output",
        "-e",
        "VLLM_USE_V2_MODEL_RUNNER=1",
        "-e",
        "VLLM_TARGET_DEVICE=xpu",
        "-e",
        "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
        "-e",
        "ZE_AFFINITY_MASK=0",
        "-e",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "-e",
        "VLLM_XPU_ENABLE_XPU_GRAPH=0",
    ]
    if cell == "dspark":
        argv += ["-e", "B70_DSPARK_BF16=1"]
    argv += ["--entrypoint", "bash", IMAGE, "-lc", "set -e; " + "; ".join(patch_commands)]
    metadata = {
        "container_name": name,
        "serve": serve,
        "speculative_config": SPEC_CONFIG if cell == "dspark" else None,
        "mounts": {
            "target": f"{TARGET}:/model:ro",
            "previous_campaign": f"{PREVIOUS}:/experiment:ro",
            "frozen_draft": f"{draft}:/draft:ro",
            "output": f"{out}:/output",
        },
        "patch_order": ["prefill", "draft", "boundary"] if cell == "dspark" else ["prefill"],
    }
    return argv, metadata


class DiagnosticClient:
    """Small stdlib-only SSE client with per-request counter accounting."""

    def __init__(self, out: Path, base: str = BASE, request_timeout: int = 900) -> None:
        self.out = out
        self.base = base.rstrip("/")
        self.request_timeout = request_timeout

    def get(self, path: str, timeout: int = 30) -> str:
        with urllib.request.urlopen(self.base + path, timeout=timeout) as response:
            return response.read().decode()

    def metrics(self, label: str, phase: str) -> tuple[str, dict[str, float]]:
        raw = self.get("/metrics")
        (self.out / f"{label}-metrics-{phase}.raw").write_text(raw)
        snapshot = metric_snapshot(raw)
        save(self.out / f"{label}-metrics-{phase}.json", snapshot)
        return raw, snapshot

    def chat(
        self,
        label: str,
        prompt: str,
        count: int = FORCED_OUTPUT,
        forced: bool = True,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int = 42,
        thinking: bool = False,
        cache_salt: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": "qwen38",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "max_tokens": count,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking},
            "cache_salt": cache_salt or label,
            "return_token_ids": True,
        }
        if forced:
            payload["ignore_eos"] = True
        save(self.out / f"{label}-request.json", payload)
        request_bytes = json.dumps(payload).encode()
        (self.out / f"{label}-request.raw").write_bytes(request_bytes)
        before_raw, before = self.metrics(label, "before")
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            request_bytes,
            {"Content-Type": "application/json"},
        )
        started = time.monotonic()
        first_token_at: float | None = None
        text = ""
        reasoning = ""
        token_ids: list[int] = []
        prompt_ids: list[int] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        done = False
        error_object: Any = None
        sse_events = 0
        raw_sse_path = self.out / f"{label}-sse.raw"
        parsed_sse_path = self.out / f"{label}-sse.jsonl"
        with (
            urllib.request.urlopen(request, timeout=self.request_timeout) as response,
            raw_sse_path.open("wb") as raw_sse,
            parsed_sse_path.open("w") as parsed_sse,
        ):
            for raw_line in response:
                raw_sse.write(raw_line)
                raw_sse.flush()
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                now = time.monotonic()
                parsed_sse.write(json.dumps({"elapsed": now - started, "data": data}) + "\n")
                parsed_sse.flush()
                sse_events += 1
                if data == "[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                if obj.get("error"):
                    error_object = obj["error"]
                    raise RuntimeError(f"server returned an error for {label}: {error_object}")
                if obj.get("usage"):
                    usage = obj["usage"]
                if obj.get("prompt_token_ids") is not None:
                    prompt_ids = obj["prompt_token_ids"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (piece or reasoning_piece) and first_token_at is None:
                        first_token_at = now
                    text += piece
                    reasoning += reasoning_piece
                    token_ids.extend(choice.get("token_ids") or [])
                    finish_reason = choice.get("finish_reason") or finish_reason
        finished = time.monotonic()
        after_raw, after = self.metrics(label, "after")
        deltas = metric_delta(before, after)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        draft_steps = exact_metric_total(deltas, "drafts")
        proposals = exact_metric_total(deltas, "draft_tokens")
        accepts = exact_metric_total(deltas, "accepted_tokens")
        per_position_before = position_counters(before)
        per_position_after = position_counters(after)
        per_position = position_counters(deltas)
        transport_checks = {
            "done_marker": done,
            "sse_event_count_positive": sse_events > 0,
            "no_error_object": error_object is None,
            "usage_present": bool(usage),
            "prompt_ids_present": bool(prompt_ids),
            "prompt_ids_match_usage": len(prompt_ids) == prompt_tokens,
            "output_ids_match_usage": len(token_ids) == completion_tokens,
            "completion_tokens_positive": completion_tokens > 0,
            "visible_or_reasoning_text_nonempty": bool(text.strip() or reasoning.strip()),
            "finish_reason_present": bool(finish_reason),
            "metrics_finite": all_metrics_finite(before_raw) and all_metrics_finite(after_raw),
        }
        if forced:
            transport_checks.update(
                {
                    "forced_count": completion_tokens == count,
                    "forced_length_finish": finish_reason == "length",
                }
            )
        transport_pass = all(transport_checks.values())
        row: dict[str, Any] = {
            "label": label,
            "prompt": prompt,
            "cache_salt": payload["cache_salt"],
            "sampling": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "thinking": thinking,
            },
            "content": text,
            "reasoning_content": reasoning,
            "token_ids": token_ids,
            "output_token_ids": token_ids,
            "prompt_token_ids": prompt_ids,
            "rendered_prompt_token_ids": prompt_ids,
            "usage": usage,
            "finish_reason": finish_reason,
            "transport_checks": transport_checks,
            "transport_pass": transport_pass,
            "output_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            "ttft_s": None if first_token_at is None else first_token_at - started,
            "decode_tps": None
            if first_token_at is None or completion_tokens < 2
            else (completion_tokens - 1) / (finished - first_token_at),
            "elapsed_s": finished - started,
            "metric_deltas": deltas,
            "per_position_accepted_counters": per_position,
            "per_position_accepted_before": per_position_before,
            "per_position_accepted_after": per_position_after,
            "draftsteps_delta": draft_steps,
            "proposals_delta": proposals,
            "accepts_delta": accepts,
            "emitted_per_step": 1 + accepts / draft_steps if draft_steps else None,
        }
        save(self.out / f"{label}-rendered-prompt-ids.json", prompt_ids)
        save(self.out / f"{label}-output-ids.json", token_ids)
        save(self.out / f"{label}-usage-metrics.json", {
            "usage": usage,
            "metric_deltas": deltas,
            "per_position_accepted_before": per_position_before,
            "per_position_accepted_after": per_position_after,
            "per_position_accepted_counters": per_position,
            "draftsteps_delta": draft_steps,
            "proposals_delta": proposals,
            "accepts_delta": accepts,
            "emitted_per_step": row["emitted_per_step"],
        })
        save(self.out / f"{label}-result.json", row)
        print(
            json.dumps(
                {
                    "label": label,
                    "usage": usage,
                    "transport_pass": transport_pass,
                    "draftsteps_delta": draft_steps,
                    "proposals_delta": proposals,
                    "accepts_delta": accepts,
                }
            ),
            flush=True,
        )
        if not transport_pass:
            raise RuntimeError(f"transport checks failed for {label}: {transport_checks}")
        return row


def run_shared_gates(client: DiagnosticClient, out: Path, probe: Any, checks: dict[str, Any], speculative: bool) -> dict[str, Any]:
    """Run the previous campaign's 3/131/8 gates plus its 19-request smoke."""
    original_chat = probe.Cell.chat

    def compatible_chat(cell: Any, label: str, prompt: str, count: int = 512, forced: bool = True) -> dict[str, Any]:
        row = client.chat(label, prompt, count, forced)
        cell.rows.append(row)
        return row

    probe.Cell.chat = compatible_chat
    try:
        # The shared function constructs the existing Cell and calls its gates,
        # functional sandbox, and four-repeat code/prose parity smoke.
        return checks["run_checks"](out, speculative=speculative)
    finally:
        probe.Cell.chat = original_chat


def run_matrix(client: DiagnosticClient, probe: Any) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prompts = tuple(zip(PROMPT_NAMES, (probe.CODE, MATH, probe.PROSE)))
    for prompt_name, prompt in prompts:
        for thinking_name, thinking in (("think0", False), ("think1", True)):
            for sampling_name, temperature, top_p, top_k in SAMPLING:
                for seed in SEEDS:
                    label = f"experiment-{prompt_name}-{thinking_name}-{sampling_name}-seed{seed}"
                    cache_salt = f"acceptance-{prompt_name}-{thinking_name}-{sampling_name}-seed{seed}"
                    rows.append(
                        client.chat(
                            label,
                            prompt,
                            FORCED_OUTPUT,
                            True,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            seed=seed,
                            thinking=thinking,
                            cache_salt=cache_salt,
                        )
                    )
    if len(rows) != 36:
        raise AssertionError(f"matrix emitted {len(rows)} requests, expected 36")
    per_position: dict[str, float] = {}
    for row in rows:
        for position, value in row["per_position_accepted_counters"].items():
            per_position[position] = per_position.get(position, 0.0) + value
    draftsteps = sum(row["draftsteps_delta"] for row in rows)
    proposals = sum(row["proposals_delta"] for row in rows)
    accepts = sum(row["accepts_delta"] for row in rows)
    return {
        "expected_requests": 36,
        "completed_requests": len(rows),
        "transport_pass_count": sum(bool(row["transport_pass"]) for row in rows),
        "all_transport_pass": all(row["transport_pass"] for row in rows),
        "forced_output_tokens": FORCED_OUTPUT,
        "prompts": [name for name, _ in prompts],
        "thinking_values": [False, True],
        "sampling": [
            {"name": name, "temperature": temperature, "top_p": top_p, "top_k": top_k}
            for name, temperature, top_p, top_k in SAMPLING
        ],
        "seeds": list(SEEDS),
        "draftsteps_delta": draftsteps,
        "proposals_delta": proposals,
        "accepts_delta": accepts,
        "emitted_per_step": 1 + accepts / draftsteps if draftsteps else None,
        "per_position_accepted_counters": dict(sorted(per_position.items(), key=lambda item: int(item[0]))),
        "stochastic_token_parity_compared": False,
        "rows": [{
            "label": row["label"],
            "sampling": row["sampling"],
            "usage": row["usage"],
            "transport_pass": row["transport_pass"],
            "draftsteps_delta": row["draftsteps_delta"],
            "proposals_delta": row["proposals_delta"],
            "accepts_delta": row["accepts_delta"],
            "emitted_per_step": row["emitted_per_step"],
            "output_sha256": row["output_sha256"],
        } for row in rows],
    }


def source_check(out: Path, container_name: str, cell: str) -> None:
    if cell == "target":
        if not (out / "effective-prefill-source.json").is_file():
            raise RuntimeError("prefill overlay did not record its effective source")
        return
    source_check_code = r'''import hashlib, importlib.util, json, runpy
from pathlib import Path
root = Path(importlib.util.find_spec("vllm").origin).parent
assert str(root).startswith("/opt/venv/")
overlay = runpy.run_path("/experiment/patch_dspark_bf16.py")
sources = {p: (root / p).read_bytes().decode() for p in overlay["PINNED_SHA256"]}
assert overlay["prepare"](sources) == sources
boundary = runpy.run_path("/experiment/patch-vllm-qwen38-xpu-boundary.py")
xpu_source = (root / "_xpu_ops.py").read_text()
assert boundary["patch_text"](xpu_source) == xpu_source
sources["_xpu_ops.py"] = xpu_source
assert hashlib.sha256((root / "v1/attention/backends/gdn_attn.py").read_bytes()).hexdigest() == "5173f3394c1385d215bd99f0d12290e8336da844b96da612954da01d62a0b062"
print(json.dumps({"root": str(root), "exact_replay": True, "sha256": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}}))
'''
    argv = ["docker", "exec", container_name, "/opt/venv/bin/python", "-P", "-c", source_check_code]
    save(out / "source-check-argv.json", argv)
    result = command(*argv, timeout=120).stdout
    save(out / "effective-dspark-source.json", json.loads(result))


def prepare_output(args: argparse.Namespace) -> Path:
    default = ROOT / ("target-only" if args.cell == "target" else "current-dspark")
    out = (args.out or default).resolve()
    if not out.is_relative_to(ROOT) or out == ROOT:
        raise RuntimeError(f"output must be a new child of {ROOT}: {out}")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    if not out.parent.is_dir():
        raise RuntimeError(f"output parent does not exist: {out.parent}")
    out.mkdir()
    return out


def wait_for_server(proc: subprocess.Popen[str], client: DiagnosticClient, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        try:
            client.get("/health")
            break
        except (OSError, urllib.error.URLError):
            time.sleep(3)
    else:
        raise TimeoutError(f"startup exceeded {timeout} seconds")
    models = json.loads(client.get("/v1/models"))
    save(client.out / "models.json", models)
    if not any(model.get("id") == "qwen38" and model.get("max_model_len") == CONTEXT for model in models.get("data", [])):
        raise RuntimeError("wrong served model or context")
    return models


def run(args: argparse.Namespace) -> int:
    out = prepare_output(args)
    probe, checks = load_previous_modules()
    # Keep the shared functional sandbox on the same already-present image;
    # no helper invocation may trigger an implicit image pull.
    probe.IMAGE = IMAGE
    draft = (Path(args.draft_dir) if args.draft_dir else PREVIOUS / "draft").resolve()
    dependency_manifest = validate_assets(draft, require_draft=True)
    launch_argv, launch_metadata = dependency_mounts(out, draft, args.cell, args.kv_cache_dtype)
    save(out / "driver-args.json", {"argv": sys.argv, "cwd": os.getcwd(), "cell": args.cell})
    save(out / "dependencies.json", dependency_manifest)
    save(out / "launch-metadata.json", launch_metadata)
    save(out / "launch-argv.json", launch_argv)
    (out / "launcher.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(launch_argv) + "\n")
    (out / "launcher.sh").chmod(0o755)
    image_inspect = command("docker", "image", "inspect", IMAGE, timeout=60).stdout
    (out / "image-inspect.json").write_text(image_inspect)

    summary: dict[str, Any] = {
        "tier": "development",
        "status": "starting",
        "purpose": "bounded EXPERIMENT ONLY acceptance diagnostics; no production claim",
        "cell": args.cell,
        "image": IMAGE,
        "context": CONTEXT,
        "target": TARGET,
        "kv_cache_dtype": args.kv_cache_dtype,
        "runner": "v2-eager-C1" if "--enforce-eager" in launch_metadata["serve"] else "v2-graph-configured-C1",
        "target_dtype": "FP16 compute, GPTQ Int4 symmetric G128",
        "draft_dtype": "BF16",
        "speculative_config": SPEC_CONFIG if args.cell == "dspark" else None,
        "matrix": {"expected_requests": MATRIX_REQUESTS, "forced_output_tokens": FORCED_OUTPUT},
        "stochastic_token_parity_compared": False,
        "shared_gate_contract": "3 canaries + 131 finite boundaries + 8 functional + previous 19-request greedy parity smoke",
    }
    proc: subprocess.Popen[str] | None = None
    log_handle: Any = None
    before: dict[str, Any] | None = None
    failure: str | None = None
    try:
        before = host_invariants()
        save(out / "host-before.json", before)
        if before != expected_host_invariants():
            raise RuntimeError(f"host prerequisites failed: {before}")
        log_handle = (out / "server.log").open("w")
        proc = subprocess.Popen(launch_argv, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        client = DiagnosticClient(out, request_timeout=args.request_timeout)
        models = wait_for_server(proc, client, args.startup_timeout)
        summary["models"] = models
        source_check(out, launch_metadata["container_name"], args.cell)
        summary["api_checks"] = run_shared_gates(client, out, probe, checks, args.cell == "dspark")
        summary["diagnostic"] = run_matrix(client, probe)
        summary["status"] = "passed"
    except Exception:
        failure = traceback.format_exc()
        (out / "failure.txt").write_text(failure)
        summary["status"] = "failed"
        summary["error"] = failure
    finally:
        cleanup = subprocess.run(
            ["docker", "rm", "-f", launch_metadata["container_name"]],
            capture_output=True,
            text=True,
            timeout=60,
        ) if proc is not None else subprocess.CompletedProcess([], 0, "cleanup skipped: no owned launch", "")
        save(out / "container-cleanup.json", {
            "argv": ["docker", "rm", "-f", launch_metadata["container_name"]],
            "returncode": cleanup.returncode,
            "stdout": cleanup.stdout,
            "stderr": cleanup.stderr,
        })
        if proc is not None:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()
        after = host_invariants()
        save(out / "host-after.json", after)
        summary["host_unchanged"] = before is not None and after == before
        if before is not None and after != before and failure is None:
            failure = "host invariants changed\n"
            (out / "failure.txt").write_text(failure)
            summary["status"] = "failed"
            summary["error"] = failure
        save(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 1 if failure or summary.get("status") != "passed" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=("target", "dspark"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--draft-dir", type=Path)
    parser.add_argument("--kv-cache-dtype", choices=("fp8", "auto"), default="fp8")
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
