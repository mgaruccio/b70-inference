#!/usr/bin/env python3
"""Bounded, experimental draft-only LM-head probes for Muse Glimmer.

This module is intentionally not a vLLM patch or a launcher.  The target-side
integration is owned by the caller: :func:`attach_draft_head` only installs a
callable named ``glimmer_draft_head`` when ``GLIMMER_DRAFT_HEAD`` is explicitly
enabled.  The callable returns greedy token IDs and never replaces or mutates
``model.lm_head``.

The runtime modes are:

``capture``
    Run the existing ``compute_logits`` implementation, return its argmax, and
    optionally write bounded CPU captures.  Capture is fail-closed unless
    ``VLLM_XPU_ENABLE_XPU_GRAPH=0`` (or another explicit false value).
``int4``
    Build a separate symmetric G128 W4A16 representation of the shared dense
    head.  The XPU path calls the pinned ``int4_gemm_w4a16`` operator.  CPU is
    supported as a reference path for tests and offline probes.
``shortlist``
    Copy an explicit, frozen 32,768-token subset of the shared dense head and
    map local argmax IDs back to original vocabulary IDs.

The CLI is deliberately small.  ``calibrate`` creates a shortlist from saved
capture top-k frequencies using calibration labels only; ``probe`` compares
saved dense references with the two experimental heads and emits one JSON
report.  This is a naive calibration baseline, not SpecVocab replication.

Torch and vLLM are imported lazily enough that the pure validation helpers and
unit tests can run in a CPU-only Python environment without either package.
"""

import argparse
import collections
import json
import math
import numbers
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# Keep import-time behavior a strict no-op when the opt-in mode is absent.
_torch = None


VOCAB_SIZE = 202_048
HIDDEN_SIZE = 6_656
GROUP_SIZE = 128
PACK_FACTOR = 8
SHORTLIST_SIZE = 32_768
DEFAULT_TOPK = 64
DEFAULT_CAPTURE_STEPS = 64
DEFAULT_CAPTURE_STRIDE = 4
DEFAULT_CAPTURE_ROWS = 32
DEFAULT_QUANT_CHUNK_ROWS = 2_048

_MISSING = object()


class DraftHeadError(RuntimeError):
    """Base error for deliberate, fail-closed probe refusals."""


class UnsupportedDraftHeadConfiguration(DraftHeadError):
    """Raised when the existing vLLM draft/verification contract is unsafe."""


def _require_torch() -> Any:
    global _torch
    if _torch is None:
        try:
            import torch as imported_torch
        except ModuleNotFoundError as exc:  # pragma: no cover - host dependent.
            raise DraftHeadError(
                "this runtime mode requires torch; pure validation helpers do not"
            ) from exc
        _torch = imported_torch
    return _torch


def _is_tensor(value: Any) -> bool:
    return _torch is not None and isinstance(value, _torch.Tensor)


def _lookup(obj: Any, dotted_path: str) -> Any:
    """Read an attribute or mapping path without treating absent as ``None``."""

    current = obj
    for name in dotted_path.split("."):
        if isinstance(current, Mapping):
            if name not in current:
                return _MISSING
            current = current[name]
        else:
            current = getattr(current, name, _MISSING)
            if current is _MISSING:
                return _MISSING
    return current


def _first_value(*objects_and_paths: tuple[Any, str]) -> Any:
    for obj, path in objects_and_paths:
        value = _lookup(obj, path)
        if value is not _MISSING:
            return value
    return _MISSING


def _nested_options(vllm_config: Any) -> Mapping[str, Any]:
    for path in ("glimmer_draft_head", "glimmer_draft_head_config"):
        value = _lookup(vllm_config, path)
        if isinstance(value, Mapping):
            return value
    return {}


def _option(
    options: Mapping[str, Any],
    name: str,
    env_name: str,
    default: Any,
) -> Any:
    value = os.environ.get(env_name)
    if value is not None:
        return value
    return options.get(name, default)


def _parse_int(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise DraftHeadError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise DraftHeadError(f"{name} must be >= {minimum}, got {parsed}")
    return parsed


def _parse_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDraftHeadConfiguration(
            f"{name} must be a finite positive number, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise UnsupportedDraftHeadConfiguration(
            f"{name} must be finite and > 0, got {value!r}"
        )
    return parsed


def parse_mode(value: str | None = None) -> str | None:
    """Parse the opt-in environment value, returning ``None`` for no-op."""

    raw = os.environ.get("GLIMMER_DRAFT_HEAD") if value is None else value
    if raw is None or not str(raw).strip():
        return None
    mode = str(raw).strip().lower()
    if mode in {"0", "false", "off", "none", "disabled"}:
        return None
    if mode not in {"capture", "int4", "shortlist"}:
        raise DraftHeadError(
            "GLIMMER_DRAFT_HEAD must be capture, int4, shortlist, or an explicit false value"
        )
    return mode


def _false_env(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }


def require_graphs_off() -> None:
    """Capture must never accidentally run inside an XPU graph."""

    value = os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH")
    if not _false_env(value):
        raise UnsupportedDraftHeadConfiguration(
            "capture mode requires VLLM_XPU_ENABLE_XPU_GRAPH=0 (graphs are fail-closed)"
        )


def _safe_label(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._-") or "capture"


def resolve_label(label_file: str | os.PathLike[str] | None) -> str | None:
    """Return a stable label from a caller-owned label file, or ``None``.

    A missing file is intentionally a soft no-op: callers can attach the
    diagnostic mode while doing a dummy load without creating capture files.
    The first non-empty line is the label; the stem is used for an empty file.
    """

    if label_file is None:
        return None
    path = Path(label_file).expanduser()
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    first = next((line.strip() for line in lines if line.strip()), path.stem)
    return _safe_label(first)


def _resolve_path(value: Any) -> Path | None:
    if value is None or value is _MISSING or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser()


def validate_shortlist_ids(
    ids: Iterable[Any],
    *,
    vocab_size: int = VOCAB_SIZE,
    expected_size: int = SHORTLIST_SIZE,
) -> tuple[int, ...]:
    """Validate and normalize the frozen, sorted original-vocabulary IDs."""

    values: list[int] = []
    for value in ids:
        if isinstance(value, bool):
            raise ValueError("shortlist token IDs must be integers, not booleans")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"shortlist token ID is not an integer: {value!r}") from exc
        if not isinstance(value, numbers.Integral):
            raise ValueError(f"shortlist token ID is not an integer: {value!r}")
        values.append(integer)
    if len(values) != expected_size:
        raise ValueError(
            f"shortlist must contain exactly {expected_size} IDs, got {len(values)}"
        )
    if values != sorted(values):
        raise ValueError("shortlist IDs must be sorted in ascending original-ID order")
    if len(set(values)) != len(values):
        raise ValueError("shortlist IDs must be unique")
    if any(value < 0 or value >= vocab_size for value in values):
        raise ValueError(f"shortlist IDs must be in [0, {vocab_size})")
    return tuple(values)


def _coerce_id_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        for key in ("token_ids", "ids", "shortlist_ids"):
            if key in value:
                return _coerce_id_values(value[key])
        raise ValueError("shortlist JSON must contain token_ids (or ids)")
    if _is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError("shortlist artifact must contain an iterable of IDs")
    return list(value)


def _torch_load(path: Path) -> Any:
    torch = _require_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch versions before weights_only.
        return torch.load(path, map_location="cpu")


def load_shortlist_ids(
    path: str | os.PathLike[str],
    *,
    vocab_size: int = VOCAB_SIZE,
    expected_size: int = SHORTLIST_SIZE,
) -> tuple[int, ...]:
    """Load a JSON or torch artifact and enforce the immutable shortlist contract."""

    artifact = Path(path).expanduser()
    if not artifact.is_file():
        raise DraftHeadError(f"shortlist artifact does not exist: {artifact}")
    suffix = artifact.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DraftHeadError(f"could not read shortlist JSON: {artifact}") from exc
    elif suffix in {".pt", ".pth", ".bin"}:
        payload = _torch_load(artifact)
    else:
        raise DraftHeadError("shortlist artifact must be .json, .pt, .pth, or .bin")
    try:
        ids = _coerce_id_values(payload)
        return validate_shortlist_ids(
            ids, vocab_size=vocab_size, expected_size=expected_size
        )
    except ValueError as exc:
        raise DraftHeadError(f"invalid shortlist artifact {artifact}: {exc}") from exc


def pack_signed_nibbles(signed: Any) -> Any:
    """Pack ``[rows, K]`` signed values in [-8, 7] into int32 uint4 words."""

    torch = _require_torch()
    if not isinstance(signed, torch.Tensor) or signed.ndim != 2:
        raise ValueError("signed values must be a two-dimensional torch tensor")
    if signed.shape[1] % PACK_FACTOR:
        raise ValueError("the K dimension must be divisible by 8")
    signed_i32 = signed.to(dtype=torch.int32)
    if bool(torch.any(signed_i32 < -8).item()) or bool(torch.any(signed_i32 > 7).item()):
        raise ValueError("signed INT4 values must be in [-8, 7]")
    unsigned = signed_i32 + 8
    shifts = (
        torch.arange(PACK_FACTOR, device=signed.device, dtype=torch.int32)
        .mul_(4)
        .view(1, 1, PACK_FACTOR)
    )
    return (unsigned.reshape(signed.shape[0], -1, PACK_FACTOR) << shifts).sum(dim=-1)


def unpack_signed_nibbles(packed: Any, k: int | None = None, *, zero_point: int = 8) -> Any:
    """Unpack int32 words into signed values with the runtime's nibble order."""

    torch = _require_torch()
    if not isinstance(packed, torch.Tensor) or packed.ndim != 2:
        raise ValueError("packed values must be a two-dimensional torch tensor")
    if zero_point != 8:
        raise ValueError("the Glimmer prototype only supports scalar zero point 8")
    shifts = (
        torch.arange(PACK_FACTOR, device=packed.device, dtype=torch.int32)
        .mul_(4)
        .view(1, 1, PACK_FACTOR)
    )
    codes = (packed.to(dtype=torch.int32).unsqueeze(-1) >> shifts) & 0xF
    values = (codes - zero_point).reshape(packed.shape[0], -1)
    if k is not None:
        if k < 0 or k > values.shape[1]:
            raise ValueError(f"invalid unpack K={k} for {values.shape[1]} values")
        values = values[:, :k]
    return values.to(dtype=torch.int8)


@dataclass(frozen=True)
class PackedInt4Weights:
    qweight: Any
    scales: Any
    zero_point: Any
    original_shape: tuple[int, int]
    group_size: int = GROUP_SIZE


def _validate_weight_tensor(
    weight: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    torch = _require_torch()
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise UnsupportedDraftHeadConfiguration("lm_head.weight must be a rank-2 tensor")
    shape = (int(weight.shape[0]), int(weight.shape[1]))
    if expected_shape is not None and shape != expected_shape:
        raise UnsupportedDraftHeadConfiguration(
            f"lm_head.weight must have shape {expected_shape}, got {shape}"
        )
    if weight.dtype != torch.float16:
        raise UnsupportedDraftHeadConfiguration(
            f"shared draft head requires FP16 weights, got {weight.dtype}"
        )
    if not weight.is_floating_point():
        raise UnsupportedDraftHeadConfiguration("lm_head.weight must be floating point")
    return shape


def pack_int4_weight(
    weight: Any,
    *,
    group_size: int = GROUP_SIZE,
    chunk_rows: int = DEFAULT_QUANT_CHUNK_ROWS,
    expected_shape: tuple[int, int] | None = None,
) -> PackedInt4Weights:
    """Quantize a dense ``[vocab, hidden]`` matrix without mutating it.

    Scales are computed in FP32, rounded to FP16, and then used for the
    round-to-nearest signed quantization.  The quantized values are stored as
    ``signed + 8`` nibbles in the exact ``[K/8, N]`` / ``(1, K/8)`` layout used
    by the XPU operator.
    """

    torch = _require_torch()
    shape = _validate_weight_tensor(weight, expected_shape=expected_shape)
    rows, hidden = shape
    if group_size != GROUP_SIZE:
        raise ValueError("this prototype only supports G128")
    if hidden % group_size or hidden % PACK_FACTOR:
        raise ValueError("hidden size must be divisible by G128 and by 8")
    chunk_rows = _parse_int(chunk_rows, "chunk_rows", minimum=1)

    source = weight.detach()
    groups = hidden // group_size
    # A transpose view over row-major storage preserves the exact kernel
    # contract while working on the pinned CPU and XPU backends.
    qweight_storage = torch.empty(
        (rows, hidden // PACK_FACTOR), dtype=torch.int32, device=weight.device
    )
    qweight = qweight_storage.transpose(0, 1)
    scales = torch.empty(
        (groups, rows), dtype=torch.float16, device=weight.device
    )
    with torch.no_grad():
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            chunk = source[start:end].to(dtype=torch.float32).contiguous()
            grouped = chunk.reshape(end - start, groups, group_size)
            scale32 = grouped.abs().amax(dim=-1).div(7.0)
            scale16 = scale32.to(dtype=torch.float16)
            # A zero row/group is represented by signed zero with a harmless
            # unit scale; this avoids feeding a zero divisor to a kernel.
            scale16 = torch.where(scale16 > 0, scale16, torch.ones_like(scale16))
            signed = torch.round(
                grouped / scale16.to(dtype=torch.float32).unsqueeze(-1)
            ).clamp_(-8, 7).reshape(end - start, hidden)
            packed = pack_signed_nibbles(signed)
            qweight[:, start:end].copy_(packed.transpose(0, 1))
            scales[:, start:end].copy_(scale16.transpose(0, 1))
    zero_point = torch.tensor([8], dtype=torch.int8, device=weight.device)
    return PackedInt4Weights(qweight, scales, zero_point, shape, group_size)


def unpack_int4_weight(
    qweight: Any,
    scales: Any,
    *,
    zero_point: int = 8,
    group_size: int = GROUP_SIZE,
    columns: Sequence[int] | None = None,
    output_dtype: Any | None = None,
) -> Any:
    """Dequantize selected output columns as ``[columns, hidden]``."""

    torch = _require_torch()
    if group_size != GROUP_SIZE or zero_point != 8:
        raise ValueError("only symmetric G128 with scalar zero point 8 is supported")
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("qweight and scales must both be rank-2 tensors")
    hidden_packed, rows = qweight.shape
    if hidden_packed * PACK_FACTOR % group_size:
        raise ValueError("qweight K dimension is not divisible by G128")
    hidden = hidden_packed * PACK_FACTOR
    if scales.shape != (hidden // group_size, rows):
        raise ValueError(
            f"scales must have shape {(hidden // group_size, rows)}, got {tuple(scales.shape)}"
        )
    if columns is None:
        selected = torch.arange(rows, device=qweight.device, dtype=torch.long)
    else:
        selected = torch.as_tensor(columns, device=qweight.device, dtype=torch.long)
        if selected.ndim != 1 or bool(torch.any(selected < 0).item()) or bool(
            torch.any(selected >= rows).item()
        ):
            raise ValueError("selected output columns are out of range")
    packed_columns = qweight.index_select(1, selected).transpose(0, 1)
    signed = unpack_signed_nibbles(packed_columns, hidden, zero_point=zero_point)
    scale_columns = scales.index_select(1, selected).transpose(0, 1)
    expanded = scale_columns.repeat_interleave(group_size, dim=1)
    dtype = torch.float32 if output_dtype is None else output_dtype
    return signed.to(dtype=dtype) * expanded.to(dtype=dtype)


def _dequant_linear(
    hidden_states: Any,
    packed: PackedInt4Weights,
    *,
    chunk_rows: int = DEFAULT_QUANT_CHUNK_ROWS,
    logit_scale: float = 1.0,
) -> Any:
    torch = _require_torch()
    rows = packed.original_shape[0]
    hidden_fp32 = hidden_states.to(dtype=torch.float32)
    chunks = []
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        dequant = unpack_int4_weight(
            packed.qweight,
            packed.scales,
            zero_point=int(packed.zero_point.item()),
            group_size=packed.group_size,
            columns=range(start, end),
            output_dtype=torch.float32,
        )
        chunks.append(torch.nn.functional.linear(hidden_fp32, dequant))
    output = torch.cat(chunks, dim=-1)
    return output if logit_scale == 1.0 else output * logit_scale


def _load_xpu_op() -> Callable[..., Any]:
    torch = _require_torch()
    try:
        # Importing xpu_ops registers the private operator in the pinned image.
        from vllm._xpu_ops import xpu_ops as _xpu_ops  # noqa: F401
    except Exception as exc:
        raise DraftHeadError("vllm._xpu_ops is required for the XPU INT4 path") from exc
    try:
        op = torch.ops._xpu_C.int4_gemm_w4a16
    except AttributeError as exc:
        raise DraftHeadError("int4_gemm_w4a16 is not registered in torch.ops._xpu_C") from exc
    return op


def _validate_hidden(hidden_states: Any, hidden_size: int) -> Any:
    torch = _require_torch()
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("draft head expects a rank-2 [M, hidden] tensor")
    if int(hidden_states.shape[1]) != hidden_size:
        raise ValueError(
            f"draft hidden size must be {hidden_size}, got {tuple(hidden_states.shape)}"
        )
    return hidden_states


@dataclass
class Int4DraftHead:
    """Callable separate INT4 head; it never owns the dense target parameter."""

    packed: PackedInt4Weights
    logit_scale: float = 1.0
    chunk_rows: int = DEFAULT_QUANT_CHUNK_ROWS
    reference_only: bool = False

    def __post_init__(self) -> None:
        self.memory = memory_summary_for_packed(self.packed, self.logit_scale)
        self.hidden_size = self.packed.original_shape[1]
        self.vocab_size = self.packed.original_shape[0]

    def logits(self, hidden_states: Any) -> Any:
        torch = _require_torch()
        hidden_states = _validate_hidden(hidden_states, self.hidden_size)
        if hidden_states.device.type == "cpu" or self.reference_only:
            return _dequant_linear(
                hidden_states,
                self.packed,
                chunk_rows=self.chunk_rows,
                logit_scale=self.logit_scale,
            )
        if tuple(self.packed.qweight.stride()) != (
            1,
            self.hidden_size // PACK_FACTOR,
        ):
            raise DraftHeadError("qweight lost the required (1, K/8) XPU strides")
        if not self.packed.scales.is_contiguous():
            raise DraftHeadError("INT4 scales lost required contiguous FP16 layout")
        if self.packed.scales.dtype != torch.float16:
            raise DraftHeadError("INT4 scales must be FP16")
        op = _load_xpu_op()
        activation = hidden_states.contiguous()
        output = op(
            activation,
            self.packed.qweight,
            None,
            self.packed.scales,
            self.packed.zero_point,
            GROUP_SIZE,
            None,
        )
        if not isinstance(output, torch.Tensor):
            raise DraftHeadError("int4_gemm_w4a16 returned a non-tensor result")
        return output if self.logit_scale == 1.0 else output * self.logit_scale

    def __call__(self, hidden_states: Any, *, return_logits: bool = False) -> Any:
        logits = self.logits(hidden_states)
        return logits if return_logits else logits.argmax(dim=-1)


@dataclass
class ShortlistDraftHead:
    """Callable copied shortlist head with original-vocabulary ID output."""

    dense_weight: Any
    token_ids: tuple[int, ...]
    logit_scale: float = 1.0

    def __post_init__(self) -> None:
        torch = _require_torch()
        _validate_weight_tensor(self.dense_weight, expected_shape=None)
        if len(self.token_ids) == 0:
            raise ValueError("shortlist cannot be empty")
        indices = torch.tensor(
            self.token_ids, dtype=torch.long, device=self.dense_weight.device
        )
        # This is the one and only dense-to-shortlist copy.  The original
        # lm_head parameter remains untouched and is not registered here.
        self.weight = self.dense_weight.detach().index_select(0, indices).contiguous()
        self.original_ids = indices
        self.hidden_size = int(self.weight.shape[1])
        self.vocab_size = int(self.dense_weight.shape[0])
        self.memory = {
            "mode": "shortlist",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "shortlist_size": len(self.token_ids),
            "selected_weight_bytes": _tensor_bytes(self.weight),
            "token_id_bytes": _tensor_bytes(self.original_ids),
            "new_bytes": _tensor_bytes(self.weight) + _tensor_bytes(self.original_ids),
            "dense_weight_bytes": _tensor_bytes(self.dense_weight),
        }

    def logits(self, hidden_states: Any) -> Any:
        torch = _require_torch()
        hidden_states = _validate_hidden(hidden_states, self.hidden_size)
        if hidden_states.dtype != self.weight.dtype:
            raise ValueError(
                f"shortlist activation dtype {hidden_states.dtype} does not match "
                f"head dtype {self.weight.dtype}"
            )
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        return logits if self.logit_scale == 1.0 else logits * self.logit_scale

    def __call__(self, hidden_states: Any, *, return_logits: bool = False) -> Any:
        local_logits = self.logits(hidden_states)
        if return_logits:
            return local_logits
        return self.original_ids[local_logits.argmax(dim=-1)]


class CaptureRecorder:
    """Bounded per-label capture writer owned by one callable instance."""

    def __init__(
        self,
        *,
        artifact_dir: Path | None,
        label_file: Path | None,
        max_steps: int = DEFAULT_CAPTURE_STEPS,
        sample_stride: int = DEFAULT_CAPTURE_STRIDE,
        max_rows: int = DEFAULT_CAPTURE_ROWS,
        topk: int = DEFAULT_TOPK,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.label_file = label_file
        self.label = resolve_label(label_file)
        self.max_steps = _parse_int(max_steps, "max_steps", minimum=1)
        self.sample_stride = _parse_int(sample_stride, "sample_stride", minimum=1)
        self.max_rows = _parse_int(max_rows, "max_rows", minimum=1)
        self.topk = _parse_int(topk, "topk", minimum=1)
        self.step = 0
        self.saved_steps = 0
        self._label_counts: dict[str | None, tuple[int, int]] = {}
        if self.artifact_dir is not None and self.label is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.artifact_dir is not None and self.label is not None

    def record(self, hidden_states: Any, logits: Any) -> None:
        torch = _require_torch()
        label = resolve_label(self.label_file)
        if label != self.label:
            self.label = label
            self.step, self.saved_steps = self._label_counts.get(label, (0, 0))
        step = self.step
        self.step += 1
        self._label_counts[self.label] = (self.step, self.saved_steps)
        if not self.enabled or self.saved_steps >= self.max_steps:
            return
        if step % self.sample_stride:
            return
        rows = min(int(hidden_states.shape[0]), self.max_rows)
        if rows <= 0:
            return
        topk = min(self.topk, int(logits.shape[-1]))
        values, ids = torch.topk(logits[:rows], k=topk, dim=-1)
        payload = {
            "format": "glimmer-draft-head-capture-v1",
            "label": self.label,
            "label_file": str(self.label_file) if self.label_file is not None else None,
            "step": step,
            "hidden_states": hidden_states[:rows].detach().to("cpu").clone(),
            "baseline_top1": logits[:rows].argmax(dim=-1).detach().to("cpu").clone(),
            "baseline_topk_ids": ids.detach().to("cpu").clone(),
            "baseline_topk_logits": values.detach().to("cpu").clone(),
            "vocab_size": int(logits.shape[-1]),
            "hidden_size": int(hidden_states.shape[-1]),
            "topk": topk,
        }
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.label}-step{step:06d}-part{self.saved_steps:04d}.pt"
        destination = self.artifact_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        self.saved_steps += 1
        self._label_counts[self.label] = (self.step, self.saved_steps)


@dataclass
class CaptureDraftHead:
    compute_logits: Callable[[Any], Any]
    recorder: CaptureRecorder

    def __call__(self, hidden_states: Any, *, return_logits: bool = False) -> Any:
        logits = self.compute_logits(hidden_states)
        torch = _require_torch()
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise DraftHeadError("compute_logits must return a rank-2 logits tensor")
        self.recorder.record(hidden_states, logits)
        return logits if return_logits else logits.argmax(dim=-1)

    @property
    def memory(self) -> dict[str, Any]:
        return {
            "mode": "capture",
            "new_bytes": 0,
            "saved_steps": self.recorder.saved_steps,
            "max_steps": self.recorder.max_steps,
        }


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def memory_summary_for_packed(
    packed: PackedInt4Weights, logit_scale: float = 1.0
) -> dict[str, Any]:
    return {
        "mode": "int4",
        "vocab_size": packed.original_shape[0],
        "hidden_size": packed.original_shape[1],
        "group_size": packed.group_size,
        "qweight_shape": list(packed.qweight.shape),
        "qweight_strides": list(packed.qweight.stride()),
        "scales_shape": list(packed.scales.shape),
        "zero_point": int(packed.zero_point.item()),
        "qweight_bytes": _tensor_bytes(packed.qweight),
        "scale_bytes": _tensor_bytes(packed.scales),
        "zero_point_bytes": _tensor_bytes(packed.zero_point),
        "new_bytes": _tensor_bytes(packed.qweight)
        + _tensor_bytes(packed.scales)
        + _tensor_bytes(packed.zero_point),
        "dense_weight_bytes": packed.original_shape[0]
        * packed.original_shape[1]
        * 2,
        "logit_scale": logit_scale,
    }


def _write_memory_summary(summary: Mapping[str, Any], artifact_dir: Path | None) -> None:
    print("glimmer_draft_head_memory " + json.dumps(dict(summary), sort_keys=True), file=sys.stderr)
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mode = _safe_label(str(summary.get("mode", "head")))
    destination = artifact_dir / f"memory-summary-{mode}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _nonempty_mapping(value: Any) -> bool:
    if value is _MISSING or value is None:
        return False
    if _is_tensor(value):
        return value.numel() > 0
    if isinstance(value, (str, bytes, Mapping, Sequence)):
        return len(value) > 0
    return bool(value)


def _validate_existing_contract(model: Any, vllm_config: Any, *, vocab_size: int) -> float:
    """Reject mappings/samplers the standalone greedy head cannot preserve."""

    objects = ((model, "model"), (vllm_config, "vllm_config"))
    parallel = _first_value(
        (vllm_config, "parallel_config.tensor_parallel_size"),
        (vllm_config, "target_parallel_config.tensor_parallel_size"),
        (vllm_config, "tensor_parallel_size"),
        (model, "tensor_parallel_size"),
    )
    if parallel is not _MISSING and parallel is not None:
        try:
            parallel_int = int(parallel)
        except (TypeError, ValueError) as exc:
            raise UnsupportedDraftHeadConfiguration(
                f"tensor parallel size is not an integer: {parallel!r}"
            ) from exc
        if parallel_int != 1:
            raise UnsupportedDraftHeadConfiguration(
                f"draft-only head requires tensor_parallel_size=1, got {parallel_int}"
            )

    mapping_paths = (
        "draft_id_to_target_id",
        "draft_to_target_token_mapping",
        "token_id_mapping",
        "vocab_mapping",
        "vocab_map",
        "draft_vocab_map",
        "target_vocab_map",
        "speculative_config.draft_to_target_token_mapping",
        "speculative_config.token_id_mapping",
        "speculative_config.vocab_mapping",
        "speculative_config.vocab_map",
        "speculative_config.draft_vocab_map",
        "speculative_config.target_vocab_map",
    )
    for obj, owner in objects:
        for path in mapping_paths:
            value = _lookup(obj, path)
            if _nonempty_mapping(value):
                raise UnsupportedDraftHeadConfiguration(
                    f"unsupported existing {owner}.{path}; refusing to guess token-ID mapping"
                )
    heterogeneous = _first_value(
        (vllm_config, "speculative_config.use_heterogeneous_vocab"),
        (vllm_config, "use_heterogeneous_vocab"),
        (model, "use_heterogeneous_vocab"),
    )
    if heterogeneous is not _MISSING and bool(heterogeneous):
        raise UnsupportedDraftHeadConfiguration(
            "heterogeneous draft vocabularies require an explicit mapping and are unsupported"
        )
    draft_vocab = _first_value(
        (model, "config.draft_vocab_size"),
        (vllm_config, "speculative_config.draft_vocab_size"),
        (vllm_config, "draft_vocab_size"),
        (model, "draft_vocab_size"),
    )
    if draft_vocab is not _MISSING and draft_vocab is not None:
        try:
            if int(draft_vocab) != vocab_size:
                raise UnsupportedDraftHeadConfiguration(
                    f"draft vocab size {draft_vocab} does not match target {vocab_size}"
                )
        except (TypeError, ValueError) as exc:
            raise UnsupportedDraftHeadConfiguration(
                f"invalid draft vocab size {draft_vocab!r}"
            ) from exc

    sample_method = _first_value(
        (vllm_config, "speculative_config.draft_sample_method"),
        (vllm_config, "draft_sample_method"),
        (model, "draft_sample_method"),
    )
    if sample_method is not _MISSING and sample_method is not None:
        if str(sample_method).strip().lower() not in {"greedy", "argmax"}:
            raise UnsupportedDraftHeadConfiguration(
                f"draft-only head only supports greedy sampling, got {sample_method!r}"
            )
    verification = _first_value(
        (vllm_config, "speculative_config.rejection_sample_method"),
        (vllm_config, "speculative_config.verification_method"),
        (vllm_config, "speculative_config.verification_mode"),
        (vllm_config, "rejection_sample_method"),
        (model, "rejection_sample_method"),
    )
    if verification is not _MISSING and verification is not None:
        if str(verification).strip().lower() not in {"standard", "greedy"}:
            raise UnsupportedDraftHeadConfiguration(
                "draft-only head refuses nonstandard/probabilistic verification "
                f"({verification!r})"
            )

    scale = _first_value(
        (model, "logit_scale"),
        (model, "config.logit_scale"),
        (vllm_config, "model_config.logit_scale"),
        (vllm_config, "logit_scale"),
    )
    if scale is _MISSING or scale is None:
        return 1.0
    if _is_tensor(scale):
        if scale.numel() != 1:
            raise UnsupportedDraftHeadConfiguration("logit_scale must be scalar")
        scale = scale.detach().cpu().item()
    return _parse_float(scale, "logit_scale")


def _model_weight(model: Any) -> Any:
    lm_head = getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    if weight is None:
        raise UnsupportedDraftHeadConfiguration("model.lm_head.weight is required")
    _validate_weight_tensor(weight, expected_shape=(VOCAB_SIZE, HIDDEN_SIZE))
    return weight


def _artifact_dir(options: Mapping[str, Any]) -> Path | None:
    return _resolve_path(
        _option(options, "artifact_dir", "GLIMMER_DRAFT_HEAD_ARTIFACT_DIR", None)
    )


def attach_draft_head(model: Any, vllm_config: Any) -> Any | None:
    """Attach the explicitly selected standalone greedy draft-head callable.

    With no ``GLIMMER_DRAFT_HEAD`` environment value this is a strict no-op: no
    model attribute is added and torch/vLLM are not imported.  On success the
    return value is the same callable assigned to ``model.glimmer_draft_head``.
    """

    mode = parse_mode()
    if mode is None:
        return None
    _require_torch()
    options = _nested_options(vllm_config)
    weight = _model_weight(model)
    scale = _validate_existing_contract(model, vllm_config, vocab_size=int(weight.shape[0]))
    if hasattr(model, "glimmer_draft_head"):
        raise DraftHeadError("model already has glimmer_draft_head; refusing replacement")

    artifact_dir = _artifact_dir(options)
    if mode == "capture":
        require_graphs_off()
        label_file = _resolve_path(
            _option(options, "label_file", "GLIMMER_DRAFT_HEAD_LABEL_FILE", None)
        )
        recorder = CaptureRecorder(
            artifact_dir=artifact_dir,
            label_file=label_file,
            max_steps=_option(
                options,
                "max_steps",
                "GLIMMER_DRAFT_HEAD_MAX_STEPS",
                DEFAULT_CAPTURE_STEPS,
            ),
            sample_stride=_option(
                options,
                "sample_stride",
                "GLIMMER_DRAFT_HEAD_SAMPLE_STRIDE",
                DEFAULT_CAPTURE_STRIDE,
            ),
            max_rows=_option(
                options,
                "max_rows",
                "GLIMMER_DRAFT_HEAD_MAX_ROWS",
                DEFAULT_CAPTURE_ROWS,
            ),
            topk=_option(
                options, "topk", "GLIMMER_DRAFT_HEAD_TOPK", DEFAULT_TOPK
            ),
        )
        compute_logits = getattr(model, "compute_logits", None)
        if not callable(compute_logits):
            raise UnsupportedDraftHeadConfiguration("capture mode requires model.compute_logits")
        head: Any = CaptureDraftHead(compute_logits, recorder)
    elif mode == "int4":
        chunk_rows = _parse_int(
            _option(
                options,
                "chunk_rows",
                "GLIMMER_DRAFT_HEAD_CHUNK_ROWS",
                DEFAULT_QUANT_CHUNK_ROWS,
            ),
            "chunk_rows",
            minimum=1,
        )
        packed = pack_int4_weight(
            weight,
            group_size=GROUP_SIZE,
            chunk_rows=chunk_rows,
            expected_shape=(VOCAB_SIZE, HIDDEN_SIZE),
        )
        head = Int4DraftHead(packed, logit_scale=scale, chunk_rows=chunk_rows)
    else:
        shortlist_value = _option(
            options,
            "shortlist",
            "GLIMMER_DRAFT_HEAD_SHORTLIST",
            os.environ.get("GLIMMER_DRAFT_HEAD_SHORTLIST_PATH"),
        )
        if shortlist_value is None:
            raise DraftHeadError(
                "shortlist mode requires GLIMMER_DRAFT_HEAD_SHORTLIST (JSON/torch artifact)"
            )
        ids = load_shortlist_ids(shortlist_value)
        head = ShortlistDraftHead(weight, ids, logit_scale=scale)

    setattr(model, "glimmer_draft_head", head)
    _write_memory_summary(head.memory, artifact_dir)
    return head


# ---- Offline capture/calibration/probe helpers ---------------------------------


def _capture_paths(path: str | os.PathLike[str]) -> list[Path]:
    root = Path(path).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise DraftHeadError(f"capture path does not exist: {root}")
    return sorted(p for p in root.rglob("*.pt") if p.is_file())


def _load_capture(path: Path) -> Mapping[str, Any]:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping):
        raise DraftHeadError(f"capture is not a mapping: {path}")
    if "hidden_states" not in payload or "baseline_topk_ids" not in payload:
        raise DraftHeadError(f"capture is missing required fields: {path}")
    return payload


def _capture_label(payload: Mapping[str, Any], path: Path) -> str:
    value = payload.get("label")
    return _safe_label(str(value)) if value is not None else _safe_label(path.stem)


def _read_label_set(path: str | os.PathLike[str]) -> set[str]:
    label_path = Path(path).expanduser()
    if not label_path.is_file():
        raise DraftHeadError(f"label list does not exist: {label_path}")
    labels = {
        _safe_label(line)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not labels:
        raise DraftHeadError(f"label list is empty: {label_path}")
    return labels


def _payload_rows(value: Any) -> list[list[int]]:
    if _is_tensor(value):
        raw = value.detach().cpu().tolist()
    else:
        raw = value
    if raw is None:
        return []
    if isinstance(raw, list) and raw and not isinstance(raw[0], list):
        return [[int(item) for item in raw]]
    return [[int(item) for item in row] for row in (raw or [])]


def _payload_vector(value: Any) -> list[int]:
    if _is_tensor(value):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in (value or [])]


def calibrate_shortlist(
    captures: str | os.PathLike[str],
    *,
    calibration_labels: set[str],
    heldout_labels: set[str] | None = None,
    vocab_size: int = VOCAB_SIZE,
    expected_size: int = SHORTLIST_SIZE,
) -> dict[str, Any]:
    """Derive a sorted shortlist from calibration captures only.

    Top-k occurrences receive one vote and top-1 receives one extra vote.  The
    result is intentionally a naive frequency baseline and records that fact in
    the artifact metadata; it is not a SpecVocab implementation.
    """

    if not calibration_labels:
        raise DraftHeadError("calibration label set must not be empty")
    if heldout_labels and calibration_labels & heldout_labels:
        overlap = sorted(calibration_labels & heldout_labels)
        raise DraftHeadError(f"calibration/heldout labels overlap: {overlap}")
    counts: collections.Counter[int] = collections.Counter()
    used_labels: set[str] = set()
    files = _capture_paths(captures)
    for path in files:
        payload = _load_capture(path)
        label = _capture_label(payload, path)
        if label not in calibration_labels:
            continue
        used_labels.add(label)
        for row in _payload_rows(payload.get("baseline_topk_ids")):
            for token_id in row:
                if 0 <= token_id < vocab_size:
                    counts[token_id] += 1
        for token_id in _payload_vector(payload.get("baseline_top1")):
            if 0 <= token_id < vocab_size:
                counts[token_id] += 1  # top-1 tie-break bonus
    if not used_labels:
        raise DraftHeadError(
            "no saved captures matched calibration labels; refusing heldout leakage"
        )
    if len(counts) < expected_size:
        raise DraftHeadError(
            f"calibration captures contain only {len(counts)} unique IDs; "
            f"need {expected_size} without synthetic/fallback IDs"
        )
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:expected_size]
    token_ids = sorted(token_id for token_id, _count in ranked)
    validated = validate_shortlist_ids(
        token_ids, vocab_size=vocab_size, expected_size=expected_size
    )
    return {
        "format": "glimmer-draft-head-shortlist-v1",
        "token_ids": list(validated),
        "vocab_size": vocab_size,
        "shortlist_size": expected_size,
        "calibration_labels": sorted(used_labels),
        "heldout_labels": sorted(heldout_labels or set()),
        "frequency_votes": {str(token_id): counts[token_id] for token_id in validated},
        "method": "naive_top1_plus_topk_frequency",
        "not_spec_vocab_replication": True,
    }


def _load_safetensors_head(index_path: Path, key: str) -> Any:
    _require_torch()
    if index_path.suffix.lower() == ".safetensors":
        shard = index_path
    else:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DraftHeadError(f"could not read safetensors index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or key not in weight_map:
            raise DraftHeadError(f"head key {key!r} is absent from {index_path}")
        shard = index_path.parent / str(weight_map[key])
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise DraftHeadError("probe requires safetensors.torch") from exc
    if not shard.is_file():
        raise DraftHeadError(f"safetensors shard does not exist: {shard}")
    tensors = load_file(str(shard), device="cpu")
    if key not in tensors:
        matches = [name for name in tensors if name.endswith(key)]
        if len(matches) != 1:
            raise DraftHeadError(f"head key {key!r} is absent from shard {shard}")
        key = matches[0]
    # The checkpoint is BF16; the retained server explicitly loads FP16.
    weight = tensors[key].to(dtype=_require_torch().float16)
    _validate_weight_tensor(weight, expected_shape=(VOCAB_SIZE, HIDDEN_SIZE))
    return weight


def _device_from_arg(value: str) -> Any:
    torch = _require_torch()
    if value != "auto":
        return torch.device(value)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: Any) -> None:
    torch = _require_torch()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing(
    head: Any,
    *,
    hidden_size: int,
    device: Any,
    m_values: Sequence[int],
    k_values: Sequence[int],
    warmup: int,
    repeats: int,
    representative_hidden: Any,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    if warmup < 0 or repeats < 1 or representative_hidden.shape[0] < 1:
        raise DraftHeadError("timing needs nonempty captures and positive repeats")
    rows: list[dict[str, Any]] = []
    for m in m_values:
        hidden = representative_hidden.repeat(
            (math.ceil(m / representative_hidden.shape[0]), 1)
        )[:m].to(device=device, dtype=torch.float16).contiguous()
        for k in k_values:
            for _ in range(warmup):
                for _ in range(k):
                    head(hidden)
            _sync(device)
            started = time.perf_counter()
            for _ in range(repeats):
                for _ in range(k):
                    head(hidden)
            _sync(device)
            elapsed = time.perf_counter() - started
            calls = repeats * k
            rows.append(
                {
                    "m": m,
                    "k_repeats": k,
                    "warmup": warmup,
                    "repeats": repeats,
                    "us_per_call": elapsed * 1e6 / calls,
                    "device": str(device),
                    "input_source": "captured_draft_hidden_states",
                }
            )
    return rows


def _probe(args: argparse.Namespace) -> dict[str, Any]:
    torch = _require_torch()
    device = _device_from_arg(args.device)
    weight_cpu = _load_safetensors_head(Path(args.model_index).expanduser(), args.head_key)
    weight = weight_cpu.to(device=device)
    packed = pack_int4_weight(
        weight,
        group_size=GROUP_SIZE,
        chunk_rows=args.chunk_rows,
        expected_shape=(VOCAB_SIZE, HIDDEN_SIZE),
    )
    int4_head = Int4DraftHead(
        packed, logit_scale=1.0, chunk_rows=args.chunk_rows
    )
    captures = [_load_capture(path) for path in _capture_paths(args.captures)]
    if not captures:
        raise DraftHeadError("probe found no .pt captures")

    heldout_labels = (
        _read_label_set(args.heldout_label_file)
        if args.heldout_label_file
        else None
    )
    baseline_total = 0
    dense_saved_agree = 0
    int4_dense_agree = 0
    int4_dequant_agree = 0
    int4_dequant_abs_sum = 0.0
    int4_dequant_count = 0
    int4_dequant_max = 0.0
    shortlist_head: ShortlistDraftHead | None = None
    shortlist_ids: tuple[int, ...] | None = None
    if args.shortlist:
        shortlist_ids = load_shortlist_ids(args.shortlist)
        shortlist_head = ShortlistDraftHead(weight, shortlist_ids)
    heldout_total = 0
    heldout_coverage = 0
    heldout_agree = 0
    for path, payload in zip(_capture_paths(args.captures), captures):
        label = _capture_label(payload, path)
        hidden = payload["hidden_states"]
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.as_tensor(hidden)
        hidden = hidden.to(device=device)
        saved_top1 = _payload_vector(payload.get("baseline_top1"))
        if hidden.ndim != 2 or hidden.shape[0] < 1 or hidden.shape[1] != HIDDEN_SIZE:
            raise DraftHeadError(f"empty or malformed hidden capture: {path}")
        if len(saved_top1) != hidden.shape[0]:
            raise DraftHeadError(f"saved top1 length does not match captured rows: {path}")
        if not bool(torch.isfinite(hidden).all().item()):
            raise DraftHeadError(f"nonfinite captured hidden states: {path}")
        dense_logits = torch.nn.functional.linear(hidden, weight)
        int4_logits = int4_head.logits(hidden)
        dequant_logits = _dequant_linear(
            hidden,
            packed,
            chunk_rows=args.chunk_rows,
            logit_scale=1.0,
        )
        diff = (int4_logits.to(torch.float32) - dequant_logits.to(torch.float32)).abs()
        for value in (dense_logits, int4_logits, dequant_logits):
            if not bool(torch.isfinite(value).all().item()):
                raise DraftHeadError(f"nonfinite probe logits: {path}")
        if not torch.allclose(int4_logits.float(), dequant_logits.float(), atol=0.05, rtol=0.01):
            raise DraftHeadError(f"INT4/dequant numerical mismatch at {path}: max_abs={diff.max().item()}")
        int4_dequant_abs_sum += float(diff.sum().item())
        int4_dequant_count += int(diff.numel())
        int4_dequant_max = max(int4_dequant_max, float(diff.max().item()))
        dense_top1 = dense_logits.argmax(dim=-1).detach().cpu()
        int4_top1 = int4_logits.argmax(dim=-1).detach().cpu()
        dequant_top1 = dequant_logits.argmax(dim=-1).detach().cpu()
        for expected, dense_id, int4_id, dequant_id in zip(
            saved_top1,
            dense_top1.tolist(),
            int4_top1.tolist(),
            dequant_top1.tolist(),
        ):
            baseline_total += 1
            dense_saved_agree += int(expected == dense_id)
            int4_dense_agree += int(dense_id == int4_id)
            int4_dequant_agree += int(int4_id == dequant_id)
        if heldout_labels is not None and label in heldout_labels:
            shortlist_top1 = (
                shortlist_head(hidden).detach().cpu().tolist()
                if shortlist_head is not None
                else []
            )
            for dense_id, short_id in zip(dense_top1.tolist(), shortlist_top1):
                heldout_total += 1
                heldout_coverage += int(dense_id in (shortlist_ids or ()))
                heldout_agree += int(dense_id == short_id)

    if baseline_total == 0 or dense_saved_agree != baseline_total:
        raise DraftHeadError(
            f"dense reference invalid: {dense_saved_agree}/{baseline_total} saved choices reproduced"
        )
    if shortlist_head is not None and heldout_total == 0:
        raise DraftHeadError("shortlist probe needs nonempty held-out captures")
    report: dict[str, Any] = {
        "format": "glimmer-draft-head-probe-v1",
        "model_index": str(Path(args.model_index).expanduser()),
        "device": str(device),
        "captures": len(captures),
        "rows": baseline_total,
        "numerical_screen_passed": True,
        "reference_tolerance": {"atol": 0.05, "rtol": 0.01},
        "dense_memory": {
            "shape": list(weight.shape),
            "dtype": str(weight.dtype),
            "bytes": _tensor_bytes(weight),
        },
        "dense_saved_top1_agreement": (
            dense_saved_agree / baseline_total if baseline_total else None
        ),
        "int4_top1_agreement_vs_dense": (
            int4_dense_agree / baseline_total if baseline_total else None
        ),
        "int4_top1_agreement_vs_dequant": (
            int4_dequant_agree / baseline_total if baseline_total else None
        ),
        "int4_logit_abs_error_vs_dequant_reference": {
            "mean": (
                int4_dequant_abs_sum / int4_dequant_count
                if int4_dequant_count
                else None
            ),
            "max": int4_dequant_max if int4_dequant_count else None,
        },
        "int4_memory": int4_head.memory,
        "calibration_note": "naive calibration baseline; not SpecVocab replication",
    }
    if shortlist_head is not None:
        report["shortlist_memory"] = shortlist_head.memory
        report["shortlist"] = {
            "path": str(Path(args.shortlist).expanduser()),
            "heldout_labels": sorted(heldout_labels or set()),
            "heldout_top1_coverage": (
                heldout_coverage / heldout_total if heldout_total else None
            ),
            "heldout_top1_agreement": (
                heldout_agree / heldout_total if heldout_total else None
            ),
            "heldout_rows": heldout_total,
        }
    m_values = tuple(int(value) for value in args.m_values.split(",") if value)
    k_values = tuple(int(value) for value in args.k_values.split(",") if value)
    if not m_values or not k_values or any(value <= 0 for value in (*m_values, *k_values)):
        raise DraftHeadError("m-values and k-values must be positive comma-separated integers")
    heads = {
        "dense": lambda hidden: torch.nn.functional.linear(hidden, weight).argmax(dim=-1),
        "int4": int4_head,
    }
    if shortlist_head is not None:
        heads["shortlist"] = shortlist_head
    report["timing"] = {
        name: _timing(
            head, hidden_size=HIDDEN_SIZE, device=device,
            m_values=m_values, k_values=k_values,
            warmup=args.warmup, repeats=args.repeats,
            representative_hidden=captures[0]["hidden_states"],
        )
        for name, head in heads.items()
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser(
        "calibrate", help="derive a frozen shortlist from calibration captures"
    )
    calibrate.add_argument("--captures", required=True)
    calibrate.add_argument("--calibration-label-file", required=True)
    calibrate.add_argument(
        "--heldout-label-file",
        required=True,
        help="required explicit split; labels are checked but never used for votes",
    )
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    calibrate.add_argument("--size", type=int, default=SHORTLIST_SIZE)

    probe = subparsers.add_parser(
        "probe", help="compare dense, INT4, and optional shortlist heads"
    )
    probe.add_argument("--model-index", required=True, help="safetensors index JSON")
    probe.add_argument("--captures", required=True, help="capture .pt or directory")
    probe.add_argument("--shortlist", help="frozen shortlist JSON/torch artifact")
    probe.add_argument("--heldout-label-file")
    probe.add_argument("--head-key", default="lm_head.weight")
    probe.add_argument("--device", default="auto")
    probe.add_argument("--chunk-rows", type=int, default=DEFAULT_QUANT_CHUNK_ROWS)
    probe.add_argument("--m-values", default="16,24,32")
    probe.add_argument("--k-values", default="2,4")
    probe.add_argument("--warmup", type=int, default=2)
    probe.add_argument("--repeats", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "calibrate":
            calibration_labels = _read_label_set(args.calibration_label_file)
            heldout_labels = _read_label_set(args.heldout_label_file)
            artifact = calibrate_shortlist(
                args.captures,
                calibration_labels=calibration_labels,
                heldout_labels=heldout_labels,
                vocab_size=args.vocab_size,
                expected_size=args.size,
            )
            destination = Path(args.output).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, destination)
            print(json.dumps({"output": str(destination), **artifact}, sort_keys=True))
            return 0
        report = _probe(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (DraftHeadError, ValueError, OSError, RuntimeError) as exc:
        print(f"glimmer_draft_head: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised via CLI.
    raise SystemExit(main())
