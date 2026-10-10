#!/usr/bin/env python3
"""Pinned-container microbenchmark for the experiment-scoped INT4 prefill GEMM.

This program is intentionally a standalone supplemental probe.  It reads real
GPTQ projection tensors from ``--model`` with safetensors, keeps one projection
on the XPU at a time, compares the installed native W4A16 operator with the
Triton candidate, and writes a JSON record.  It does not load a vLLM model or
patch a serving launcher.  Run it inside the pinned vLLM XPU container, for
example::

    python bench-prefill-int4.py --model /model --out /results/prefill-int4.json

``--help`` is deliberately torch-free so the CLI contract can be checked on a
host.  GPU results are never reported as passed without observed XPU calls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GROUP_SIZE = 128
PACKING = 8
M_VALUES = (8192, 32)
DEFAULT_REPS = 5
DEFAULT_WARMUP = 2
DEFAULT_REFERENCE_COLUMNS = 32


@dataclass(frozen=True)
class ProjectionSpec:
    """A paired qweight/scales record discovered from the model files."""

    base_name: str
    qweight_key: str
    scales_key: str
    qzeros_key: str | None
    g_idx_key: str | None
    qweight_file: Path
    scales_file: Path
    qzeros_file: Path | None
    qweight_shape: tuple[int, ...]
    scales_shape: tuple[int, ...]
    disk_layout: str
    role: str

    @property
    def k(self) -> int:
        return int(self.scales_shape[0]) * GROUP_SIZE

    @property
    def n(self) -> int:
        return int(self.scales_shape[1])

    @property
    def shape_key(self) -> tuple[int, int]:
        return self.k, self.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_name": self.base_name,
            "role": self.role,
            "qweight": {
                "key": self.qweight_key,
                "file": str(self.qweight_file),
                "shape": list(self.qweight_shape),
            },
            "scales": {
                "key": self.scales_key,
                "file": str(self.scales_file),
                "shape": list(self.scales_shape),
            },
            "qzeros": {
                "key": self.qzeros_key,
                "file": str(self.qzeros_file) if self.qzeros_file else None,
            },
            "g_idx": {"key": self.g_idx_key},
            "logical_shape": {"K": self.k, "N": self.n},
            "disk_layout": self.disk_layout,
        }


def _safe_text(exc: BaseException, limit: int = 1800) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _tensor_meta(tensor: Any) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
    }


def _scalar_float(tensor: Any) -> float:
    return float(tensor.detach().cpu().item())


def _true_value(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def _load_quantization_config(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        raise RuntimeError("config.json has no quantization_config object")
    bits = qconfig.get("bits")
    group_size = qconfig.get("group_size")
    try:
        normalized_bits = int(bits)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"model is not 4-bit GPTQ: bits={bits!r}") from exc
    if normalized_bits != 4:
        raise RuntimeError(f"model is not 4-bit GPTQ: bits={bits!r}")
    try:
        normalized_group = int(group_size)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"model group_size is not 128: {group_size!r}") from exc
    if normalized_group != GROUP_SIZE:
        raise RuntimeError(f"model is not GPTQ-G128: group_size={group_size!r}")
    if qconfig.get("sym") is not True:
        raise RuntimeError(f"model is not explicitly symmetric GPTQ: sym={qconfig.get('sym')!r}")
    act_order_keys = ("desc_act", "act_order", "actorder")
    active_act_order = [
        key for key in act_order_keys if _true_value(qconfig.get(key))
    ]
    if active_act_order:
        raise RuntimeError(
            "act-order GPTQ is unsupported; config enables " + ", ".join(active_act_order)
        )
    return {
        "quantization_config": qconfig,
        "bits": 4,
        "group_size": GROUP_SIZE,
        "sym": True,
        "act_order_flags": {key: qconfig.get(key) for key in act_order_keys},
    }


def _resolve_weight_files(model: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Discover the index and verify its referenced safetensor keys."""

    index_path = model / "model.safetensors.index.json"
    key_to_file: dict[str, Path] = {}
    index_info: dict[str, Any] = {"path": str(index_path), "present": index_path.is_file()}
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("model.safetensors.index.json has no weight_map")
        for key, relative in weight_map.items():
            path = (model / str(relative)).resolve()
            if not path.is_file():
                raise RuntimeError(f"weight_map file is missing: {path}")
            key_to_file[str(key)] = path
        index_info["entries"] = len(weight_map)
    else:
        files = sorted(model.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"no safetensors files under {model}")
        index_info["entries"] = None
        for path in files:
            path = path.resolve()
            # The actual keys below are authoritative when there is no index.
            key_to_file.update({})

    # Open each file only for its key list.  This does not materialize model
    # tensors and catches stale/incomplete index entries before a GPU run.
    from safetensors import safe_open

    files = sorted({path for path in key_to_file.values()})
    if not files:
        files = sorted(path.resolve() for path in model.glob("*.safetensors"))
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                key_to_file.setdefault(str(key), path)
    if not key_to_file:
        raise RuntimeError(f"no tensor keys discovered under {model}")
    index_info["files"] = [str(path) for path in files]
    index_info["discovered_keys"] = len(key_to_file)
    return key_to_file, index_info


def _slice_shape(key_to_file: dict[str, Path], key: str) -> tuple[int, ...]:
    from safetensors import safe_open

    path = key_to_file[key]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        shape = handle.get_slice(key).get_shape()
    return tuple(int(value) for value in shape)


def _projection_role(name: str) -> tuple[int, str]:
    lowered = name.lower()
    if "qkv" in lowered or "query_key_value" in lowered:
        return 0, "qkv"
    if "gate_up" in lowered or "gateproj" in lowered or "gate_proj" in lowered:
        return 1, "gate_up"
    if "down_proj" in lowered or "downproj" in lowered:
        return 2, "down"
    if "o_proj" in lowered or "out_proj" in lowered or "outproj" in lowered:
        return 3, "output"
    if "up_proj" in lowered or "upproj" in lowered:
        return 4, "gate_up"
    if "q_proj" in lowered or "query_proj" in lowered:
        return 5, "qkv"
    if "k_proj" in lowered or "key_proj" in lowered:
        return 6, "qkv"
    if "v_proj" in lowered or "value_proj" in lowered:
        return 7, "qkv"
    return 20, "other"


def _discover_projections(
    key_to_file: dict[str, Path],
    max_projections: int,
) -> tuple[list[ProjectionSpec], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    records: list[ProjectionSpec] = []
    qweight_keys = sorted(key for key in key_to_file if key.endswith(".qweight"))
    if not qweight_keys:
        raise RuntimeError("no *.qweight tensors found in the model safetensors")
    for qweight_key in qweight_keys:
        base = qweight_key[: -len(".qweight")]
        scales_key = base + ".scales"
        if scales_key not in key_to_file:
            errors.append({"key": qweight_key, "error": "paired .scales key is missing"})
            continue
        qshape = _slice_shape(key_to_file, qweight_key)
        sshape = _slice_shape(key_to_file, scales_key)
        try:
            if len(sshape) != 2 or sshape[0] <= 0 or sshape[1] <= 0:
                raise ValueError(f"invalid scales shape {sshape}")
            k = sshape[0] * GROUP_SIZE
            n = sshape[1]
            expected_k_n = (k // PACKING, n)
            expected_n_k = (n, k // PACKING)
            if qshape == expected_k_n:
                disk_layout = "[K/8,N]"
            elif qshape == expected_n_k:
                disk_layout = "[N,K/8]"
            else:
                raise ValueError(
                    f"qweight shape {qshape} does not match scales {sshape}; "
                    f"expected {expected_k_n} or {expected_n_k}"
                )
            _role_priority, role = _projection_role(base)
            qzeros_key = base + ".qzeros"
            g_idx_key = base + ".g_idx"
            records.append(
                ProjectionSpec(
                    base_name=base,
                    qweight_key=qweight_key,
                    scales_key=scales_key,
                    qzeros_key=qzeros_key if qzeros_key in key_to_file else None,
                    g_idx_key=g_idx_key if g_idx_key in key_to_file else None,
                    qweight_file=key_to_file[qweight_key],
                    scales_file=key_to_file[scales_key],
                    qzeros_file=key_to_file.get(qzeros_key),
                    qweight_shape=qshape,
                    scales_shape=sshape,
                    disk_layout=disk_layout,
                    role=role,
                )
            )
        except Exception as exc:
            errors.append({"key": qweight_key, "error": _safe_text(exc)})

    act_order_records = [record.base_name for record in records if record.g_idx_key]
    if act_order_records:
        raise RuntimeError(
            "act-order g_idx tensors are present; unsupported records: "
            + ", ".join(act_order_records[:8])
        )
    if not records:
        raise RuntimeError("no shape-valid qweight/scales projection pairs found")

    # Prefer requested representative roles, then retain one actual tensor per
    # unique K/N shape so a large model is never resident on the XPU at once.
    records.sort(key=lambda record: (_projection_role(record.base_name), record.base_name))
    chosen: list[ProjectionSpec] = []
    seen_shapes: set[tuple[int, int]] = set()
    for record in records:
        if record.shape_key in seen_shapes:
            continue
        chosen.append(record)
        seen_shapes.add(record.shape_key)
        if len(chosen) >= max_projections:
            break
    if not chosen:
        raise RuntimeError("projection selection produced no records")
    return chosen, errors


def _load_tensor(key_to_file: dict[str, Path], key: str, torch: Any) -> Any:
    from safetensors import safe_open

    with safe_open(str(key_to_file[key]), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _load_projection_tensors(
    record: ProjectionSpec,
    key_to_file: dict[str, Path],
    torch: Any,
    device: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one real projection, transpose its disk qweight, then move it."""

    qweight_disk = _load_tensor(key_to_file, record.qweight_key, torch)
    scales_cpu = _load_tensor(key_to_file, record.scales_key, torch)
    if qweight_disk.dtype != torch.int32:
        raise RuntimeError(f"{record.qweight_key} dtype is {qweight_disk.dtype}, expected int32")
    if scales_cpu.dtype != torch.float16:
        raise RuntimeError(f"{record.scales_key} dtype is {scales_cpu.dtype}, expected float16")
    if not qweight_disk.is_contiguous() or not scales_cpu.is_contiguous():
        raise RuntimeError("on-disk qweight/scales must be contiguous safetensor tensors")
    if tuple(qweight_disk.shape) != record.qweight_shape:
        raise RuntimeError("qweight shape changed after discovery")
    if tuple(scales_cpu.shape) != record.scales_shape:
        raise RuntimeError("scales shape changed after discovery")

    if record.qzeros_key is not None:
        qzeros_cpu = _load_tensor(key_to_file, record.qzeros_key, torch)
    else:
        qzeros_cpu = torch.tensor([8], dtype=torch.int8)
    if qzeros_cpu.dtype != torch.int8 or qzeros_cpu.numel() != 1:
        raise RuntimeError(f"{record.base_name} qzeros must be one int8 scalar")
    if int(qzeros_cpu.reshape(-1)[0].item()) != 8:
        raise RuntimeError(f"{record.base_name} requires symmetric qzeros=[8]")

    if record.disk_layout == "[K/8,N]":
        qweight_storage_cpu = qweight_disk.transpose(0, 1).contiguous()
    else:
        qweight_storage_cpu = qweight_disk.contiguous()
    if tuple(qweight_storage_cpu.shape) != (record.n, record.k // PACKING):
        raise RuntimeError("candidate qweight transpose produced an unexpected shape")
    qweight_device = qweight_storage_cpu.to(device=device)
    # This is the exact non-contiguous [K/8,N] view consumed by the installed
    # oneDNN operator; its physical storage remains the candidate's [N,K/8].
    native_qweight = qweight_device.transpose(0, 1)
    scales_device = scales_cpu.to(device=device)
    qzeros_device = qzeros_cpu.to(device=device)
    if not qweight_device.is_contiguous() or not scales_device.is_contiguous():
        raise RuntimeError("device qweight/scales lost contiguous layout")
    if tuple(native_qweight.stride()) != (1, record.k // PACKING):
        raise RuntimeError(f"native qweight view has unexpected stride {native_qweight.stride()}")

    metadata = {
        "spec": record.as_dict(),
        "device_tensors": {
            "qweight_candidate": _tensor_meta(qweight_device),
            "qweight_installed_view": _tensor_meta(native_qweight),
            "scales": _tensor_meta(scales_device),
            "qzeros": _tensor_meta(qzeros_device),
        },
        "qzeros_source": "model" if record.qzeros_key else "symmetric_scalar_created",
    }
    return {
        "qweight_candidate": qweight_device,
        "qweight_native": native_qweight,
        "scales": scales_device,
        "qzeros": qzeros_device,
    }, metadata


def _ensure_native_operator(torch: Any) -> Any:
    errors: list[str] = []
    for module_name in ("vllm._xpu_ops", "vllm._C", "vllm._xpu_ops.xpu_ops"):
        if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
            break
        try:
            __import__(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {_safe_text(exc)}")
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        raise RuntimeError(
            "torch.ops._xpu_C.int4_gemm_w4a16 unavailable; registration="
            + "; ".join(errors)
        )
    return torch.ops._xpu_C.int4_gemm_w4a16


def _load_candidate_module() -> Any:
    path = Path(__file__).with_name("prefill_int4.py")
    spec = importlib.util.spec_from_file_location("qwen_prefill_int4_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import candidate module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_info(torch: Any) -> dict[str, Any]:
    available = bool(torch.xpu.is_available())
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "triton": _distribution_version("triton", "triton-xpu"),
        "vllm": _distribution_version("vllm"),
        "vllm_xpu_kernels": _distribution_version("vllm-xpu-kernels"),
        "safetensors": _distribution_version("safetensors"),
        "xpu_available": available,
        "ONEDNN_VERBOSE": __import__("os").environ.get("ONEDNN_VERBOSE", "0"),
    }
    if available:
        result["device_name"] = torch.xpu.get_device_name(0)
        try:
            props = torch.xpu.get_device_properties(0)
            result["device_properties"] = {
                key: getattr(props, key)
                for key in (
                    "name",
                    "total_memory",
                    "driver_version",
                    "major",
                    "minor",
                    "gpu_slices",
                    "gpu_subslices_per_slice",
                    "max_compute_units",
                )
                if isinstance(getattr(props, key, None), (str, int, float, bool))
            }
        except Exception as exc:
            result["device_properties_error"] = _safe_text(exc)
    return result


def _memory_info(torch: Any, device: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for label, function_name in (
        ("allocated_bytes", "memory_allocated"),
        ("reserved_bytes", "memory_reserved"),
        ("peak_allocated_bytes", "max_memory_allocated"),
        ("peak_reserved_bytes", "max_memory_reserved"),
    ):
        try:
            values[label] = int(getattr(torch.xpu, function_name)(device))
        except Exception as exc:
            values[label] = None
            values[label + "_error"] = _safe_text(exc)
    return values


def _reset_peak(torch: Any, device: Any) -> None:
    try:
        torch.xpu.reset_peak_memory_stats(device)
    except Exception:
        pass


def _sample_columns(n: int, count: int) -> list[int]:
    count = min(max(0, int(count)), n)
    if count == 0:
        return []
    priority = [0, n - 1, 1, n - 2, 63, 64, 127, 128, n // 2]
    columns: list[int] = []
    for value in priority:
        if 0 <= value < n and value not in columns:
            columns.append(value)
        if len(columns) >= count:
            return sorted(columns[:count])
    denominator = max(1, count - 1)
    for index in range(count):
        value = round(index * (n - 1) / denominator)
        if value not in columns:
            columns.append(value)
    for value in range(n):
        if len(columns) >= count:
            break
        if value not in columns:
            columns.append(value)
    return sorted(columns[:count])


def _compare(torch: Any, candidate: Any, installed: Any, rtol: float, atol: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate_shape": [int(value) for value in candidate.shape],
        "installed_shape": [int(value) for value in installed.shape],
        "rtol": rtol,
        "atol": atol,
    }
    if tuple(candidate.shape) != tuple(installed.shape):
        result.update({"finite": False, "allclose": False, "error": "shape mismatch"})
        return result
    candidate_finite = bool(torch.isfinite(candidate).all().item())
    installed_finite = bool(torch.isfinite(installed).all().item())
    result["candidate_finite"] = candidate_finite
    result["installed_finite"] = installed_finite
    result["finite"] = candidate_finite and installed_finite
    if not result["finite"]:
        result["allclose"] = False
        return result
    candidate_f = candidate.float()
    installed_f = installed.float()
    absolute = (candidate_f - installed_f).abs()
    relative = absolute / installed_f.abs().clamp_min(1.0e-6)
    rmse = torch.sqrt(torch.mean((candidate_f - installed_f) ** 2))
    norm = torch.sqrt(torch.mean(installed_f**2)).clamp_min(1.0e-12)
    cosine = torch.sum(candidate_f * installed_f) / (
        torch.linalg.vector_norm(candidate_f) * torch.linalg.vector_norm(installed_f)
    ).clamp_min(1.0e-12)
    result.update(
        {
            "max_abs": _scalar_float(absolute.max()),
            "max_relative": _scalar_float(relative.max()),
            "normalized_rmse": _scalar_float(rmse / norm),
            "cosine": _scalar_float(cosine),
            "allclose": bool(torch.allclose(candidate_f, installed_f, rtol=rtol, atol=atol)),
            "torch_equal": bool(torch.equal(candidate, installed)),
        }
    )
    return result


def _fp32_reference(
    torch: Any,
    x: Any,
    qweight_candidate: Any,
    scales: Any,
    columns: list[int],
) -> tuple[Any, list[int]]:
    if not columns:
        return None, []
    indices = torch.tensor(columns, dtype=torch.long, device=x.device)
    packed = qweight_candidate.index_select(0, indices).detach().cpu()
    scale = scales.index_select(1, indices).detach().cpu()
    activations = x.detach().cpu().float()
    words = packed.t().contiguous().to(torch.int64)
    shifts = (torch.arange(PACKING, dtype=torch.int64).view(1, 1, -1) * 4)
    codes = ((words[:, :, None] >> shifts) & 0xF).permute(0, 2, 1)
    codes = codes.reshape(qweight_candidate.shape[1] * PACKING, len(columns)).float()
    weights = (codes - 8.0) * scale.float().repeat_interleave(GROUP_SIZE, dim=0)
    return activations @ weights, columns


def _timed_matrix(
    torch: Any,
    arms: dict[str, Any],
    warmup: int,
    repetitions: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Warm each arm, then measure rotated full-call wall durations."""

    errors: dict[str, str] = {}
    active = list(arms)
    for name in list(active):
        for _ in range(warmup):
            try:
                torch.xpu.synchronize()
                value = arms[name]()
                torch.xpu.synchronize()
                del value
            except Exception as exc:
                errors[name] = _safe_text(exc)
                active.remove(name)
                break
    if "installed" not in active:
        raise RuntimeError("installed baseline warmup failed: " + errors.get("installed", "unknown error"))

    samples: dict[str, list[float]] = {name: [] for name in active}
    for repetition in range(repetitions):
        if not active:
            break
        offset = repetition % len(active)
        order = active[offset:] + active[:offset]
        for name in order:
            try:
                torch.xpu.synchronize()
                started = time.perf_counter()
                value = arms[name]()
                torch.xpu.synchronize()
                samples[name].append((time.perf_counter() - started) * 1000.0)
                del value
            except Exception as exc:
                errors[name] = _safe_text(exc)
                active.remove(name)
                if name == "installed":
                    raise RuntimeError("installed baseline timing failed: " + errors[name]) from exc

    timing: dict[str, dict[str, Any]] = {}
    for name, values in samples.items():
        if not values:
            continue
        timing[name] = {
            "full_call_wall_ms": values,
            "median_ms": float(statistics.median(values)),
            "min_ms": float(min(values)),
            "max_ms": float(max(values)),
            "repetitions_observed": len(values),
        }
    return timing, errors


def _invoke_and_sync(torch: Any, function: Any) -> Any:
    torch.xpu.synchronize()
    value = function()
    torch.xpu.synchronize()
    return value


def _run_matrix_case(
    torch: Any,
    native_op: Any,
    candidate_module: Any,
    tensors: dict[str, Any],
    record: ProjectionSpec,
    metadata: dict[str, Any],
    m: int,
    seed: int,
    config_names: tuple[str, ...],
    warmup: int,
    repetitions: int,
    reference_columns: int,
) -> dict[str, Any]:
    device = tensors["qweight_candidate"].device
    torch.manual_seed(seed)
    try:
        torch.xpu.manual_seed_all(seed)
    except Exception:
        pass
    x = torch.randn((m, record.k), device=device, dtype=torch.float16).contiguous()
    qcandidate = tensors["qweight_candidate"]
    qnative = tensors["qweight_native"]
    scales = tensors["scales"]
    qzeros = tensors["qzeros"]

    def installed() -> Any:
        return native_op(x, qnative, None, scales, qzeros, GROUP_SIZE, None)

    def make_candidate(name: str) -> Any:
        def invoke() -> Any:
            return candidate_module.candidate_int4_gemm_w4a16(
                x,
                qcandidate,
                None,
                scales,
                qzeros,
                GROUP_SIZE,
                None,
                config_name=name,
            )

        return invoke

    arms = {"installed": installed}
    arms.update({name: make_candidate(name) for name in config_names})
    _reset_peak(torch, device)
    case: dict[str, Any] = {
        "status": "running",
        "projection": record.base_name,
        "role": record.role,
        "M": m,
        "K": record.k,
        "N": record.n,
        "seed": seed,
        "weights": metadata,
        "activation": _tensor_meta(x),
        "contract": {
            "group_size": GROUP_SIZE,
            "packing": PACKING,
            "qzero": 8,
            "g_idx": None,
            "baseline": "torch.ops._xpu_C.int4_gemm_w4a16",
            "candidate": "candidate_int4_gemm_w4a16",
            "input_source": "real safetensors GPTQ projection; no synthetic weights",
        },
        "warmup_discarded": warmup,
        "repetitions_requested": repetitions,
        "timing": {},
        "numerical": {},
        "errors": {},
        "memory": {},
    }
    try:
        timing, timing_errors = _timed_matrix(torch, arms, warmup, repetitions)
        case["timing"] = timing
        case["errors"].update({f"timing:{key}": value for key, value in timing_errors.items()})
        if "installed" not in timing:
            raise RuntimeError("no installed baseline timing was observed")
        installed_output = _invoke_and_sync(torch, installed).detach()
        case["installed_output"] = _tensor_meta(installed_output)
        columns = _sample_columns(record.n, reference_columns)
        fp32_reference, columns = _fp32_reference(
            torch, x, qcandidate, scales, columns
        )
        if fp32_reference is not None:
            indices = torch.tensor(columns, dtype=torch.long, device=device)
            installed_sample = installed_output.index_select(1, indices).detach().cpu().float()
            case["numerical"]["installed_vs_fp32_reference"] = _compare(
                torch, installed_sample, fp32_reference, 1.0e-2, 1.0e-2
            )
            case["numerical"]["fp32_reference_columns"] = columns
            del installed_sample

        candidate_observed = False
        for name in config_names:
            if name not in timing:
                continue
            try:
                candidate_output = _invoke_and_sync(torch, arms[name]).detach()
                comparison = _compare(
                    torch, candidate_output, installed_output, 1.0e-2, 1.0e-2
                )
                case["numerical"][name + "_vs_installed"] = comparison
                candidate_observed = True
                if not comparison.get("finite") or not comparison.get("allclose"):
                    case["errors"]["numerical:" + name] = "candidate is not finite/allclose"
                del candidate_output
            except Exception as exc:
                case["errors"]["correctness:" + name] = _safe_text(exc)
        case["memory"] = _memory_info(torch, device)
        case["host_observed"] = candidate_observed
        case["status"] = (
            "passed"
            if candidate_observed
            and not any(key.startswith(("timing:", "correctness:", "numerical:")) for key in case["errors"])
            else "failed"
        )
        if not candidate_observed:
            case["errors"]["candidate"] = "no candidate variant produced an observed result"
    except Exception as exc:
        case["status"] = "failed"
        case["errors"]["case"] = _safe_text(exc)
        case["traceback"] = traceback.format_exc(limit=10)
        case["memory"] = _memory_info(torch, device)
    finally:
        del x
        try:
            torch.xpu.synchronize()
        except Exception:
            pass
    return case


def _initial_report(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(__file__)
    return {
        "schema_version": 1,
        "status": "running",
        "tier": "development",
        "command": list(sys.argv),
        "script_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model": str(args.model.resolve()),
        "contract": {
            "scope": "supplemental direct GEMM microbenchmark; not a serving result",
            "baseline": "installed torch.ops._xpu_C.int4_gemm_w4a16",
            "candidate": "Triton symmetric GPTQ INT4 G128, FP16 dequant and output, FP32 tl.dot accumulation",
            "intentional_difference": "candidate unpacks [N,K/8] contiguous qweight in registers instead of native oneDNN",
            "decode_policy": "future serving dispatcher preserves native op for M<=32 and unsupported layouts",
            "gpu_check": "not passed without observed host XPU results",
        },
        "research": {
            "pinned_kernel_reference": "https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h",
            "pinned_vllm_reference": "https://github.com/vllm-project/vllm/blob/ac7509e2b/vllm/model_executor/kernels/linear/mixed_precision/xpu.py",
        },
        "configuration": {
            "M_values": list(M_VALUES),
            "warmup": args.warmup,
            "repetitions": args.reps,
            "reference_columns": args.reference_columns,
            "max_projections": args.max_projections,
            "requested_configs": args.configs,
        },
        "environment": {},
        "inventory": {},
        "projections": [],
        "cases": [],
        "errors": [],
        "gpu_check": {"status": "not_run", "host_observed": False},
    }


def _parse_config_names(value: str | None, available: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return available
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--configs must name at least one candidate config")
    if len(names) > 3:
        raise ValueError("at most three static candidate configs may be exposed")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            "unknown candidate config(s): " + ", ".join(unknown) + "; available=" + ", ".join(available)
        )
    return names


def _execute(args: argparse.Namespace, report: dict[str, Any]) -> int:
    import torch

    if not bool(torch.xpu.is_available()):
        raise RuntimeError("an available host XPU is required; no GPU result was observed")
    report["environment"] = _runtime_info(torch)
    native_op = _ensure_native_operator(torch)
    candidate_module = _load_candidate_module()
    available = tuple(candidate_module.available_configs())
    config_names = _parse_config_names(args.configs, available)
    report["configuration"]["selected_configs"] = list(config_names)
    report["environment"]["candidate_configs"] = list(available)
    device = torch.device("xpu")
    from safetensors import safe_open  # noqa: F401 - records pinned runtime dependency

    quantization = _load_quantization_config(args.model)
    key_to_file, index_info = _resolve_weight_files(args.model)
    projections, discovery_errors = _discover_projections(key_to_file, args.max_projections)
    report["quantization"] = quantization
    report["inventory"] = index_info
    report["inventory"]["discovery_errors"] = discovery_errors
    report["projections"] = [record.as_dict() for record in projections]
    _write_report(args.out, report)

    seed_base = int(args.seed)
    for projection_index, record in enumerate(projections):
        try:
            tensors, metadata = _load_projection_tensors(record, key_to_file, torch, device)
            for m_index, m in enumerate(M_VALUES):
                case = _run_matrix_case(
                    torch,
                    native_op,
                    candidate_module,
                    tensors,
                    record,
                    metadata,
                    m,
                    seed_base + projection_index * 1009 + m_index * 100003,
                    config_names,
                    args.warmup,
                    args.reps,
                    args.reference_columns,
                )
                report["cases"].append(case)
                if case.get("host_observed"):
                    report["gpu_check"]["host_observed"] = True
                print(
                    f"{record.role} {record.base_name} M={m} "
                    f"status={case['status']} "
                    f"baseline_ms={case.get('timing', {}).get('installed', {}).get('median_ms', 'n/a')}",
                    flush=True,
                )
                _write_report(args.out, report)
            del tensors
            try:
                torch.xpu.synchronize()
                torch.xpu.empty_cache()
            except Exception:
                pass
        except Exception as exc:
            error = {"projection": record.base_name, "error": _safe_text(exc)}
            report["errors"].append(error)
            report["cases"].append(
                {
                    "status": "failed",
                    "projection": record.base_name,
                    "K": record.k,
                    "N": record.n,
                    "weights": record.as_dict(),
                    "errors": {"projection": _safe_text(exc)},
                }
            )
            print(f"{record.base_name} status=failed error={_safe_text(exc)}", file=sys.stderr, flush=True)
            _write_report(args.out, report)

    observed = bool(report["gpu_check"].get("host_observed"))
    failed_cases = [case for case in report["cases"] if case.get("status") != "passed"]
    if observed and not failed_cases and not report["errors"]:
        report["gpu_check"]["status"] = "passed"
        report["status"] = "passed"
    elif observed:
        report["gpu_check"]["status"] = "observed_with_failures"
        report["status"] = "failed"
    else:
        report["gpu_check"]["status"] = "not_run"
        report["status"] = "failed"
        report["errors"].append({"run": "no candidate host result was observed"})
    _write_report(args.out, report)
    return 0 if report["status"] == "passed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="real GPTQ model directory")
    parser.add_argument("--out", type=Path, default=Path("prefill-int4.json"), help="JSON result path")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="compile/warmup calls per arm (0..5)")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="interleaved timed calls (1..5)")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--max-projections", type=int, default=4, help="unique real K/N projections (1..8)")
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated candidate configs; default is all three static variants",
    )
    parser.add_argument(
        "--reference-columns",
        type=int,
        default=DEFAULT_REFERENCE_COLUMNS,
        help="small FP32 explicit-dequant sample columns; use 0 to disable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.model.is_dir():
        parser.error(f"--model must be a directory: {args.model}")
    if args.out.exists():
        parser.error(f"output already exists; choose a new path: {args.out}")
    if not 0 <= args.warmup <= 5:
        parser.error("--warmup must be in [0, 5]")
    if not 1 <= args.reps <= 5:
        parser.error("--reps must be in [1, 5]")
    if not 1 <= args.max_projections <= 8:
        parser.error("--max-projections must be in [1, 8]")
    if not 0 <= args.reference_columns <= 128:
        parser.error("--reference-columns must be in [0, 128]")
    report = _initial_report(args)
    _write_report(args.out, report)
    try:
        return _execute(args, report)
    except Exception as exc:
        report["status"] = "failed"
        report["gpu_check"] = {"status": "not_run", "host_observed": False}
        report["errors"].append({"run": _safe_text(exc)})
        report["traceback"] = traceback.format_exc(limit=14)
        _write_report(args.out, report)
        print(f"failed: {_safe_text(exc)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
