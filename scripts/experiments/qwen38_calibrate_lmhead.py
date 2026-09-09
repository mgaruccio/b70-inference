#!/usr/bin/env python3
"""Standalone activation-Hessian GPTQ calibration for Qwen's dense ``lm_head``.

This tool deliberately quantizes only the dense output head.  It consumes saved
FP16 hidden states rather than starting a model server, keeps calibration and
held-out inputs separate, and writes a vLLM/XPU W4A16 artifact without touching
the source checkpoint.

The update rule is an independent adaptation of the GPTQ algorithm described in
IST-DASLab/gptq's ``gptq.py`` (canonical ``add_batch`` Hessian accumulation and
``fasterquant`` inverse-Hessian Cholesky/error propagation), Apache-2.0:
https://github.com/IST-DASLab/gptq/blob/main/LICENSE.  The implementation here
retains the algorithm's full off-diagonal Hessian, consecutive-column order,
and block-128 correction while using this repository's explicit symmetric
G128/XPU serialization contract.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

GROUP_SIZE = 128
PACK_FACTOR = 8
QZERO = 8
DEFAULT_BLOCK_SIZE = GROUP_SIZE
DEFAULT_EVAL_ROWS = 512
DEFAULT_EVAL_CHUNK_ROWS = 8
DEFAULT_HESSIAN_CHUNK_ROWS = 4096
DEFAULT_RTN_ROW_CHUNK = 4096
DEFAULT_DEQUANT_ROW_CHUNK = 4096
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 15
QWEN_HEAD_SHAPE = (248_320, 5_120)


class CalibrationError(RuntimeError):
    """Raised when the standalone calibration contract is not satisfied."""

def _require_torch() -> Any:
    """Import torch lazily; Python's module cache avoids a manual global cache."""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent.
        raise CalibrationError("this calibration tool requires PyTorch") from exc
    return torch


def _torch_load(path: Path) -> Any:
    torch = _require_torch()
    return torch.load(path, map_location="cpu", weights_only=True)


def _check_finite(tensor: Any, what: str) -> None:
    torch = _require_torch()
    if not bool(torch.isfinite(tensor).all().item()):
        raise CalibrationError(f"{what} contains NaN or infinity")


def _validate_hidden_tensor(hidden: Any, path: Path, hidden_size: int) -> Any:
    torch = _require_torch()
    if not isinstance(hidden, torch.Tensor):
        raise CalibrationError(f"{path}: hidden_states must be a torch.Tensor")
    if hidden.ndim != 2 or int(hidden.shape[1]) != hidden_size:
        raise CalibrationError(
            f"{path}: hidden_states must have shape [N, {hidden_size}], got {tuple(hidden.shape)}"
        )
    if hidden.dtype != torch.float16:
        raise CalibrationError(f"{path}: hidden_states must be float16, got {hidden.dtype}")
    if int(hidden.shape[0]) <= 0:
        raise CalibrationError(f"{path}: hidden_states must contain at least one row")
    _check_finite(hidden, f"{path}: hidden_states")
    return hidden


def _load_hidden_file(path: Path, hidden_size: int) -> Any:
    payload = _torch_load(path)
    if not isinstance(payload, Mapping) or "hidden_states" not in payload:
        raise CalibrationError(
            f"{path}: expected torch.save({{'hidden_states': tensor}}) payload"
        )
    return _validate_hidden_tensor(payload["hidden_states"], path, hidden_size)


def capture_files(directory: str | Path) -> list[Path]:
    """Return recursive, deterministic ``*.pt`` capture files in ``directory``."""

    root = Path(directory)
    if not root.is_dir():
        raise CalibrationError(f"capture directory does not exist or is not a directory: {root}")
    files = sorted(path for path in root.rglob("*.pt") if path.is_file())
    if not files:
        raise CalibrationError(f"capture directory contains no *.pt files: {root}")
    return [path.resolve() for path in files]


def validate_capture_sets(
    calibration_dir: str | Path,
    eval_dir: str | Path,
    hidden_size: int,
) -> tuple[list[Path], list[Path], int, int]:
    """Validate both input trees and reject overlapping calibration/held-out files."""

    calibration_files = capture_files(calibration_dir)
    eval_files = capture_files(eval_dir)
    overlap = sorted(set(calibration_files).intersection(eval_files))
    if overlap:
        raise CalibrationError(
            "calibration and eval inputs must be disjoint; overlapping files: "
            + ", ".join(str(path) for path in overlap[:4])
        )

    calibration_count = count_hidden_rows(calibration_files, hidden_size)
    eval_count = count_hidden_rows(eval_files, hidden_size)
    if calibration_count <= 0 or eval_count <= 0:
        raise CalibrationError("calibration and eval inputs must both contain rows")
    return calibration_files, eval_files, calibration_count, eval_count


def iter_hidden_states(paths: Iterable[str | Path], hidden_size: int) -> Iterator[Any]:
    """Yield validated CPU hidden-state tensors one capture file at a time."""

    for raw_path in paths:
        path = Path(raw_path)
        yield _load_hidden_file(path, hidden_size)


def count_hidden_rows(paths: Iterable[str | Path], hidden_size: int) -> int:
    total = 0
    for hidden in iter_hidden_states(paths, hidden_size):
        total += int(hidden.shape[0])
    return total


def compute_activation_hessian(
    paths: Iterable[str | Path],
    hidden_size: int,
    *,
    chunk_rows: int = DEFAULT_HESSIAN_CHUNK_ROWS,
    progress: bool = False,
) -> tuple[Any, int]:
    """Compute the requested full Hessian ``H = 2 X^T X / N`` on CPU.

    Capture files are buffered until ``chunk_rows`` rows are available before a
    rank-5k ``X.T @ X``.  This keeps thousands of tiny capture files from
    launching thousands of large CPU matmuls while preserving the full Hessian.
    """

    torch = _require_torch()
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    path_list = [Path(path) for path in paths]
    hessian = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device="cpu")
    sample_count = 0
    block_count = 0
    buffered: list[Any] = []
    buffered_rows = 0

    def add_block(chunks: list[Any]) -> None:
        nonlocal block_count, sample_count
        sample = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        sample32 = sample.to(dtype=torch.float32)
        hessian.add_(sample32.transpose(0, 1).matmul(sample32))
        sample_count += int(sample32.shape[0])
        block_count += 1
        if progress and (block_count == 1 or block_count % 4 == 0):
            print(
                f"[hessian] blocks={block_count} rows={sample_count} chunk_rows={chunk_rows}",
                flush=True,
            )

    file_total = len(path_list)
    for file_index, path in enumerate(path_list, 1):
        hidden = _load_hidden_file(path, hidden_size)
        if progress and (file_index == 1 or file_index % 256 == 0 or file_index == file_total):
            print(f"[hessian] files={file_index}/{file_total}", flush=True)
        offset = 0
        while offset < int(hidden.shape[0]):
            take = min(int(hidden.shape[0]) - offset, chunk_rows - buffered_rows)
            buffered.append(hidden[offset : offset + take])
            buffered_rows += take
            offset += take
            if buffered_rows == chunk_rows:
                add_block(buffered)
                buffered = []
                buffered_rows = 0
    if buffered_rows:
        add_block(buffered)
    if sample_count <= 0:
        raise CalibrationError("calibration captures contain no rows")
    if progress and block_count > 1 and block_count % 4:
        print(f"[hessian] blocks={block_count} rows={sample_count} complete", flush=True)
    hessian.mul_(2.0 / sample_count)
    _check_finite(hessian, "activation Hessian")
    return hessian, sample_count


def inverse_hessian_cholesky(hessian: Any, damp: float = 0.01) -> Any:
    """Return the upper Cholesky factor used by sequential GPTQ correction.

    ``damp`` is the canonical GPTQ fractional diagonal damping value: the
    diagonal receives ``damp * mean(diag(H))`` before inversion.
    """

    torch = _require_torch()
    if not isinstance(hessian, torch.Tensor) or hessian.ndim != 2:
        raise ValueError("hessian must be a rank-2 tensor")
    if hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square, got {tuple(hessian.shape)}")
    if not math.isfinite(float(damp)) or damp < 0:
        raise ValueError("damp must be a finite non-negative number")
    hessian_cpu = hessian.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
    _check_finite(hessian_cpu, "activation Hessian")
    diagonal = hessian_cpu.diagonal()
    diagonal_mean = float(diagonal.mean().item())
    if not math.isfinite(diagonal_mean) or diagonal_mean <= 0:
        raise CalibrationError(
            "activation Hessian has no positive average diagonal; calibration activations are unusable"
        )
    diagonal.add_(diagonal_mean * damp)
    try:
        lower = torch.linalg.cholesky(hessian_cpu)
        inverse = torch.cholesky_inverse(lower)
        try:
            upper = torch.linalg.cholesky(inverse, upper=True)
        except TypeError:  # Compatibility with torch versions lacking upper=.
            upper = torch.linalg.cholesky(inverse).transpose(0, 1).contiguous()
    except RuntimeError as exc:
        raise CalibrationError(
            "activation Hessian is not positive definite after damping; "
            "collect more diverse calibration hidden states or increase --damp"
        ) from exc
    _check_finite(upper, "inverse-Hessian Cholesky factor")
    return upper.contiguous()


def pack_signed_nibbles(signed: Any) -> Any:
    """Pack a ``[rows, K]`` tensor of signed values ``[-8, 7]`` into int32 words."""

    torch = _require_torch()
    if not isinstance(signed, torch.Tensor) or signed.ndim != 2:
        raise ValueError("signed values must be a two-dimensional torch tensor")
    if int(signed.shape[1]) % PACK_FACTOR:
        raise ValueError("the K dimension must be divisible by 8")
    values = signed.to(dtype=torch.int64)
    if values.numel() and (
        bool(torch.any(values < -8).item()) or bool(torch.any(values > 7).item())
    ):
        raise ValueError("signed INT4 values must be in [-8, 7]")
    shifts = torch.arange(PACK_FACTOR, device=signed.device, dtype=torch.int64).mul_(4)
    packed = ((values + QZERO).reshape(int(signed.shape[0]), -1, PACK_FACTOR) << shifts).sum(
        dim=-1
    )
    return packed.to(dtype=torch.int32)


def unpack_signed_nibbles(packed: Any, k: int | None = None, *, zero_point: int = QZERO) -> Any:
    """Unpack XPU-order int32 words into signed values."""

    torch = _require_torch()
    if not isinstance(packed, torch.Tensor) or packed.ndim != 2:
        raise ValueError("packed values must be a two-dimensional torch tensor")
    if zero_point != QZERO:
        raise ValueError("this artifact format only supports zero point 8")
    shifts = torch.arange(PACK_FACTOR, device=packed.device, dtype=torch.int32).mul_(4)
    codes = (packed.to(dtype=torch.int32).unsqueeze(-1) >> shifts) & 0xF
    values = (codes - zero_point).reshape(int(packed.shape[0]), -1).to(dtype=torch.int8)
    if k is not None:
        if k < 0 or k > int(values.shape[1]):
            raise ValueError(f"invalid unpack K={k} for {int(values.shape[1])} values")
        values = values[:, :k]
    return values


def unpack_qweight(qweight: Any, k: int | None = None) -> Any:
    """Unpack serialized ``[K/8, N]`` qweight into dense signed ``[N, K]`` codes."""

    torch = _require_torch()
    if not isinstance(qweight, torch.Tensor) or qweight.ndim != 2:
        raise ValueError("qweight must be a rank-2 tensor")
    return unpack_signed_nibbles(qweight.transpose(0, 1), k=k)


def _validate_quant_params(rows: int, hidden: int, group_size: int, block_size: int) -> None:
    if group_size <= 0 or group_size % PACK_FACTOR:
        raise ValueError("group_size must be positive and divisible by 8")
    if block_size <= 0 or block_size % group_size or block_size % PACK_FACTOR:
        raise ValueError("block_size must be a positive multiple of group_size and 8")
    if rows <= 0 or hidden <= 0 or hidden % group_size or hidden % PACK_FACTOR:
        raise ValueError("weight shape must have positive rows and K divisible by group_size and 8")


def _group_scales(working: Any, group_start: int, group_size: int) -> Any:
    torch = _require_torch()
    values = working[:, group_start : group_start + group_size]
    scale32 = values.abs().amax(dim=1).div(7.0)
    _check_finite(scale32, "quantization scales")
    scale16 = scale32.to(dtype=torch.float16)
    if not bool(torch.isfinite(scale16).all().item()):
        raise CalibrationError("quantization scales overflowed float16")
    # A zero group has an exact zero reconstruction; unit scale avoids division by 0.
    return torch.where(scale16 > 0, scale16, torch.ones_like(scale16))


def _validate_weight(weight: Any) -> tuple[int, int]:
    torch = _require_torch()
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head weight must be a rank-2 tensor")
    if not weight.is_floating_point():
        raise ValueError(f"lm_head weight must be floating point, got {weight.dtype}")
    shape = (int(weight.shape[0]), int(weight.shape[1]))
    _check_finite(weight, "lm_head weight")
    return shape


def _target_device(weight: Any, device: str | Any | None) -> Any:
    torch = _require_torch()
    return weight.device if device is None else torch.device(device)


def _row_chunk_size(rows: int, requested: int | None) -> int:
    if requested is None or requested == 0:
        return rows
    if requested < 0:
        raise ValueError("row_chunk_rows must be non-negative")
    return min(rows, requested)


def _new_qweight_storage(rows: int, hidden: int, *, device: Any) -> Any:
    torch = _require_torch()
    return torch.empty((rows, hidden // PACK_FACTOR), dtype=torch.int32, device=device)


def rtn_quantize(
    weight: Any,
    *,
    group_size: int = GROUP_SIZE,
    row_chunk_rows: int | None = None,
    device: str | Any | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Create the symmetric G128 round-to-nearest baseline in the same layout."""

    torch = _require_torch()
    rows, hidden = _validate_weight(weight)
    _validate_quant_params(rows, hidden, group_size, DEFAULT_BLOCK_SIZE)
    target = _target_device(weight, device)
    source = weight.detach().to(device=target)
    storage = _new_qweight_storage(rows, hidden, device=target)
    scales = torch.empty((hidden // group_size, rows), dtype=torch.float16, device=target)
    requested_rows = (
        DEFAULT_RTN_ROW_CHUNK
        if row_chunk_rows is None or row_chunk_rows == 0
        else min(row_chunk_rows, DEFAULT_RTN_ROW_CHUNK)
    )
    chunk_size = _row_chunk_size(rows, requested_rows)
    with torch.no_grad():
        for row_start in range(0, rows, chunk_size):
            row_end = min(rows, row_start + chunk_size)
            if progress:
                print(f"[rtn] rows={row_start}:{row_end}/{rows}", flush=True)
            working = source[row_start:row_end].to(dtype=torch.float32).contiguous()
            grouped = working.reshape(row_end - row_start, hidden // group_size, group_size)
            scale32 = grouped.abs().amax(dim=-1).div(7.0)
            _check_finite(scale32, "quantization scales")
            scale16 = scale32.to(dtype=torch.float16)
            if not bool(torch.isfinite(scale16).all().item()):
                raise CalibrationError("quantization scales overflowed float16")
            scale16 = torch.where(scale16 > 0, scale16, torch.ones_like(scale16))
            scales[:, row_start:row_end].copy_(scale16.transpose(0, 1))
            signed = torch.round(
                grouped / scale16.to(dtype=torch.float32).unsqueeze(-1)
            ).clamp_(-8, 7).reshape(row_end - row_start, hidden).to(dtype=torch.int8)
            storage[row_start:row_end].copy_(pack_signed_nibbles(signed))
    qweight = storage.transpose(0, 1)
    return {
        "qweight": qweight,
        "scales": scales,
        "qzeros": torch.tensor([QZERO], dtype=torch.int8, device=target),
        "group_size": group_size,
        "metadata": {
            "algorithm": "rtn",
            "original_shape": [rows, hidden],
            "group_size": group_size,
            "qzero": QZERO,
            "qweight_shape": list(qweight.shape),
            "qweight_strides": list(qweight.stride()),
            "scales_shape": list(scales.shape),
            "row_chunk_rows": chunk_size,
        },
    }


def gptq_quantize(
    weight: Any,
    hessian: Any,
    *,
    damp: float = 0.01,
    group_size: int = GROUP_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    row_chunk_rows: int | None = None,
    device: str | Any | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Quantize ``[N, K]`` with full-Hessian, sequential block-128 GPTQ.

    Rows are independent and may be chunked only to bound temporary memory.  No
    activation ordering is performed: columns are visited consecutively so every
    group remains exactly ``[g*128:(g+1)*128]``.
    """

    torch = _require_torch()
    rows, hidden = _validate_weight(weight)
    _validate_quant_params(rows, hidden, group_size, block_size)
    if not isinstance(hessian, torch.Tensor) or tuple(hessian.shape) != (hidden, hidden):
        raise ValueError(
            f"hessian must have shape {(hidden, hidden)}, got "
            f"{tuple(hessian.shape) if isinstance(hessian, torch.Tensor) else None}"
        )
    upper_cpu = inverse_hessian_cholesky(hessian, damp=damp)
    upper_diagonal = upper_cpu.diagonal()
    if bool(torch.any(upper_diagonal <= 0).item()):
        raise CalibrationError("inverse-Hessian Cholesky factor has a non-positive diagonal")
    target = _target_device(weight, device)
    source = weight.detach().to(device=target)
    upper = upper_cpu.to(device=target)
    storage = _new_qweight_storage(rows, hidden, device=target)
    scales = torch.empty((hidden // group_size, rows), dtype=torch.float16, device=target)
    chunk_size = _row_chunk_size(rows, row_chunk_rows)

    with torch.no_grad():
        for row_start in range(0, rows, chunk_size):
            row_end = min(rows, row_start + chunk_size)
            if progress:
                print(f"[gptq] rows={row_start}:{row_end}/{rows}", flush=True)
            working = source[row_start:row_end].to(dtype=torch.float32).contiguous()
            for block_start in range(0, hidden, block_size):
                block_end = min(hidden, block_start + block_size)
                width = block_end - block_start
                block_codes = torch.empty(
                    (row_end - row_start, width), dtype=torch.int8, device=target
                )
                block_errors = torch.empty_like(working[:, block_start:block_end])
                for column in range(block_start, block_end):
                    group_index = column // group_size
                    group_start = group_index * group_size
                    if column == group_start:
                        scale16 = _group_scales(working, group_start, group_size)
                        scales[group_index, row_start:row_end].copy_(scale16)
                    else:
                        scale16 = scales[group_index, row_start:row_end]
                    current = working[:, column]
                    dequant = (
                        torch.round(current / scale16.to(dtype=torch.float32))
                        .clamp_(-8, 7)
                        * scale16.to(dtype=torch.float32)
                    )
                    quantized = torch.round(
                        current / scale16.to(dtype=torch.float32)
                    ).clamp_(-8, 7).to(dtype=torch.int8)
                    block_codes[:, column - block_start].copy_(quantized)
                    diagonal_value = float(upper_diagonal[column].item())
                    error = (current - dequant) / diagonal_value
                    block_errors[:, column - block_start].copy_(error)
                    if column + 1 < block_end:
                        correction = error.unsqueeze(1) * upper[column, column + 1 : block_end]
                        working[:, column + 1 : block_end].sub_(correction)
                _check_finite(block_errors, "GPTQ error correction")
                storage[row_start:row_end, block_start // PACK_FACTOR : block_end // PACK_FACTOR].copy_(
                    pack_signed_nibbles(block_codes)
                )
                if block_end < hidden:
                    cross_correction = block_errors.matmul(upper[block_start:block_end, block_end:])
                    working[:, block_end:].sub_(cross_correction)
                if progress:
                    block_index = block_start // block_size + 1
                    block_total = (hidden + block_size - 1) // block_size
                    if block_index == 1 or block_index % 8 == 0 or block_index == block_total:
                        print(
                            f"[gptq] rows={row_start}:{row_end}/{rows} "
                            f"blocks={block_index}/{block_total}",
                            flush=True,
                        )

    qweight = storage.transpose(0, 1)
    return {
        "qweight": qweight,
        "scales": scales,
        "qzeros": torch.tensor([QZERO], dtype=torch.int8, device=target),
        "group_size": group_size,
        "metadata": {
            "algorithm": "gptq",
            "hessian": "full_off_diagonal",
            "hessian_formula": "2 X^T X / N",
            "damp": float(damp),
            "block_size": block_size,
            "actorder": False,
            "original_shape": [rows, hidden],
            "group_size": group_size,
            "qzero": QZERO,
            "qweight_shape": list(qweight.shape),
            "qweight_strides": list(qweight.stride()),
            "scales_shape": list(scales.shape),
            "scales_dtype": "float16",
        },
    }


def dequantize_weight_rows(
    qweight: Any,
    scales: Any,
    start: int,
    end: int,
    *,
    group_size: int = GROUP_SIZE,
    output_dtype: Any | None = None,
) -> Any:
    """Dequantize only output rows ``[start:end]`` for bounded CPU/reference checks."""

    torch = _require_torch()
    if group_size <= 0 or group_size % PACK_FACTOR:
        raise ValueError("group_size must be positive and divisible by 8")
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("qweight and scales must be rank-2 tensors")
    hidden_packed, rows = int(qweight.shape[0]), int(qweight.shape[1])
    hidden = hidden_packed * PACK_FACTOR
    if hidden % group_size:
        raise ValueError("qweight K dimension must be divisible by group_size")
    if scales.shape != (hidden // group_size, rows):
        raise ValueError(
            f"scales must have shape {(hidden // group_size, rows)}, got {tuple(scales.shape)}"
        )
    if not 0 <= start <= end <= rows:
        raise ValueError(f"invalid output row range [{start}, {end}) for N={rows}")
    signed = unpack_signed_nibbles(qweight[:, start:end].transpose(0, 1), k=hidden)
    row_scales = scales[:, start:end].transpose(0, 1).repeat_interleave(group_size, dim=1)
    dtype = torch.float32 if output_dtype is None else output_dtype
    return signed.to(dtype=dtype) * row_scales.to(dtype=dtype)


def dequantize_weight(artifact: Mapping[str, Any], *, output_dtype: Any | None = None) -> Any:
    """Reference helper for small fixtures; production evaluation uses row chunks."""

    rows = int(artifact["qweight"].shape[1])
    return dequantize_weight_rows(
        artifact["qweight"],
        artifact["scales"],
        0,
        rows,
        group_size=int(artifact["group_size"]),
        output_dtype=output_dtype,
    )


def _copy_tensor_to_cpu_preserve_layout(tensor: Any) -> Any:
    torch = _require_torch()
    if tensor.device.type == "cpu":
        return tensor
    result = torch.empty_strided(
        tuple(tensor.shape), tuple(tensor.stride()), dtype=tensor.dtype, device="cpu"
    )
    result.copy_(tensor)
    return result


def save_quantized_artifact(
    artifact: Mapping[str, Any], output: str | Path, *, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Write the exact qweight/scales/qzeros/group_size/metadata torch payload."""

    torch = _require_torch()
    qweight = _copy_tensor_to_cpu_preserve_layout(artifact["qweight"])
    scales = artifact["scales"].detach().to(device="cpu").contiguous()
    qzeros = artifact["qzeros"].detach().to(device="cpu").contiguous()
    if qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise CalibrationError("qweight must be int32 rank-2")
    expected_stride = (1, int(qweight.shape[0]))
    if tuple(qweight.stride()) != expected_stride:
        raise CalibrationError(
            f"qweight must have NT stride {expected_stride}, got {qweight.stride()}"
        )
    if scales.dtype != torch.float16 or not scales.is_contiguous():
        raise CalibrationError("scales must be contiguous float16")
    if qzeros.dtype != torch.int8 or tuple(qzeros.shape) != (1,) or int(qzeros.item()) != QZERO:
        raise CalibrationError("qzeros must be the one-element int8 tensor [8]")
    if int(artifact["group_size"]) != GROUP_SIZE:
        raise CalibrationError("group_size must be 128")
    final_metadata = dict(artifact.get("metadata", {}))
    if metadata:
        final_metadata.update(metadata)
    final_metadata.update(
        {
            "qweight_shape": list(qweight.shape),
            "qweight_strides": list(qweight.stride()),
            "qweight_dtype": str(qweight.dtype),
            "scales_shape": list(scales.shape),
            "scales_dtype": str(scales.dtype),
            "qzeros_shape": list(qzeros.shape),
            "qzeros_value": QZERO,
            "group_size": GROUP_SIZE,
        }
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qweight": qweight,
        "scales": scales,
        "qzeros": qzeros,
        "group_size": GROUP_SIZE,
        "metadata": final_metadata,
    }
    torch.save(payload, destination)
    return destination


def resolve_lm_head_shard(model_dir: str | Path) -> tuple[Path, Path]:
    """Resolve only ``lm_head.weight`` through a safetensors index."""

    root = Path(model_dir).resolve()
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise CalibrationError(f"missing safetensors index: {index_path}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index["weight_map"]["lm_head.weight"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"safetensors index has no lm_head.weight mapping: {index_path}") from exc
    shard = (root / str(shard_name)).resolve()
    try:
        shard.relative_to(root)
    except ValueError as exc:
        raise CalibrationError(f"lm_head shard escapes model directory: {shard_name!r}") from exc
    if not shard.is_file():
        raise CalibrationError(f"lm_head shard does not exist: {shard}")
    return index_path, shard


def load_lm_head_weight(model_dir: str | Path, *, expected_shape: tuple[int, int] | None = None) -> tuple[Any, Path]:
    """Load the dense head tensor without loading unrelated checkpoint tensors."""

    torch = _require_torch()
    _, shard = resolve_lm_head_shard(model_dir)
    try:
        from safetensors import safe_open
    except ModuleNotFoundError as exc:  # pragma: no cover - pinned image dependent.
        raise CalibrationError("loading the checkpoint requires the safetensors package") from exc
    try:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if "lm_head.weight" not in handle.keys():
                raise CalibrationError(f"lm_head.weight is absent from mapped shard: {shard}")
            weight = handle.get_tensor("lm_head.weight")
    except CalibrationError:
        raise
    except Exception as exc:
        raise CalibrationError(f"failed to read lm_head.weight from {shard}: {exc}") from exc
    if weight.dtype != torch.float16:
        raise CalibrationError(f"lm_head.weight must be float16, got {weight.dtype}")
    if expected_shape is not None and tuple(weight.shape) != expected_shape:
        raise CalibrationError(
            f"lm_head.weight must have shape {expected_shape}, got {tuple(weight.shape)}"
        )
    _check_finite(weight, "lm_head.weight")
    return weight.contiguous(), shard


def _load_xpu_op() -> Any:
    torch = _require_torch()
    try:
        return torch.ops._xpu_C.int4_gemm_w4a16
    except AttributeError:
        pass

    try:
        import vllm  # noqa: F401  # Registers the pinned vLLM XPU extensions.
    except Exception as exc:  # pragma: no cover - pinned image dependent.
        raise CalibrationError("failed to import vllm for XPU operator registration") from exc

    registration_error: Exception | None = None
    try:
        import vllm._C  # noqa: F401  # Some pinned builds register _xpu_C here.
    except Exception as exc:  # pragma: no cover - build dependent.
        registration_error = exc

    try:
        return torch.ops._xpu_C.int4_gemm_w4a16
    except AttributeError as exc:  # pragma: no cover - pinned image dependent.
        detail = f"; vllm._C import error: {registration_error}" if registration_error else ""
        raise CalibrationError(
            "int4_gemm_w4a16 remained unavailable after importing vllm and vllm._C" + detail
        ) from exc

def _first_hidden_rows(paths: Sequence[str | Path], hidden_size: int, rows: int) -> Any:
    torch = _require_torch()
    if rows <= 0:
        raise ValueError("rows must be positive")
    parts: list[Any] = []
    remaining = rows
    for hidden in iter_hidden_states(paths, hidden_size):
        take = min(int(hidden.shape[0]), remaining)
        parts.append(hidden[:take])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise CalibrationError(f"need {rows} real hidden rows, found {rows - remaining}")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def xpu_kernel_preflight(
    weight: Any,
    calibration_files: Sequence[str | Path],
    hidden_size: int,
    op: Any,
    *,
    output_rows: int = 128,
    input_columns: int = 128,
    atol: float = 0.25,
    rtol: float = 0.02,
) -> dict[str, Any]:
    """Prove the real XPU pack/kernel path against dequantized FP16 F.linear."""
    torch = _require_torch()
    if output_rows != 128 or input_columns not in (128, 256):
        raise ValueError("preflight supports N=128 and K=128 or 256")
    if tuple(weight.shape)[0] < output_rows or tuple(weight.shape)[1] < input_columns:
        raise CalibrationError("lm_head is too small for the XPU preflight fixture")
    real_hidden = _first_hidden_rows(calibration_files, hidden_size, 5)
    if int(real_hidden.shape[1]) < input_columns:
        raise CalibrationError("real hidden states are too narrow for the XPU preflight fixture")
    tiny_weight = weight[:output_rows, :input_columns].contiguous()
    tiny = rtn_quantize(
        tiny_weight,
        group_size=GROUP_SIZE,
        row_chunk_rows=output_rows,
        device="xpu",
        progress=False,
    )
    hidden = real_hidden[:5, :input_columns].to(device="xpu", dtype=torch.float16).contiguous()
    dequantized = dequantize_weight(tiny, output_dtype=torch.float16)
    report: dict[str, Any] = {
        "status": "passed",
        "input": "real calibration hidden states",
        "shape": {"M": [1, 5], "N": output_rows, "K": input_columns},
        "tolerance": {"atol": atol, "rtol": rtol},
        "comparisons": {},
    }
    with torch.no_grad():
        for rows in (1, 5):
            sample = hidden[:rows]
            kernel = op(
                sample,
                tiny["qweight"],
                None,
                tiny["scales"],
                tiny["qzeros"],
                GROUP_SIZE,
                None,
            )
            _synchronize(sample.device)
            reference = torch.nn.functional.linear(sample, dequantized)
            difference = (kernel.to(dtype=torch.float32) - reference.to(dtype=torch.float32)).abs()
            max_abs = float(difference.max().item())
            rms = float(torch.sqrt((difference * difference).mean()).item())
            max_reference = float(reference.to(dtype=torch.float32).abs().max().item())
            max_allowed = atol + rtol * max_reference
            if not bool(torch.allclose(kernel, reference, atol=atol, rtol=rtol)):
                raise CalibrationError(
                    f"XPU INT4 preflight mismatch at M={rows}: max_abs={max_abs:.6g} "
                    f"rms={rms:.6g} allowed_max={max_allowed:.6g}"
                )
            report["comparisons"][f"M{rows}"] = {
                "max_abs_error": max_abs,
                "rms_error": rms,
                "max_allowed_abs_error": max_allowed,
            }
    return report



def _synchronize(device: Any) -> None:
    torch = _require_torch()
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":  # Useful for an offline compatibility check.
        torch.cuda.synchronize(device)


def _dequantized_linear(
    hidden: Any,
    artifact: Mapping[str, Any],
    *,
    row_chunk: int = DEFAULT_DEQUANT_ROW_CHUNK,
) -> Any:
    torch = _require_torch()
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    rows = int(artifact["qweight"].shape[1])
    pieces = []
    for start in range(0, rows, row_chunk):
        dequant = dequantize_weight_rows(
            artifact["qweight"],
            artifact["scales"],
            start,
            min(rows, start + row_chunk),
            group_size=int(artifact["group_size"]),
            output_dtype=hidden.dtype,
        )
        pieces.append(torch.nn.functional.linear(hidden, dequant))
    return torch.cat(pieces, dim=-1)


def _artifact_logits(hidden: Any, artifact: Mapping[str, Any], op: Any | None, *, dequant_row_chunk: int) -> Any:
    if op is not None:
        return op(
            hidden,
            artifact["qweight"],
            None,
            artifact["scales"],
            artifact["qzeros"],
            int(artifact["group_size"]),
            None,
        )
    return _dequantized_linear(hidden, artifact, row_chunk=dequant_row_chunk)


def _new_metric_accumulator() -> dict[str, float]:
    return {"rows": 0.0, "max_abs_error": 0.0, "squared_error": 0.0, "elements": 0.0, "argmax_matches": 0.0, "top5_overlap": 0.0}


def _update_metric_accumulator(metrics: dict[str, float], reference: Any, candidate: Any) -> None:
    torch = _require_torch()
    reference32 = reference.to(dtype=torch.float32)
    candidate32 = candidate.to(dtype=torch.float32)
    if reference32.shape != candidate32.shape:
        raise CalibrationError(
            f"logit shape mismatch: dense {tuple(reference32.shape)} vs candidate {tuple(candidate32.shape)}"
        )
    difference = candidate32 - reference32
    metrics["rows"] += float(reference32.shape[0])
    metrics["elements"] += float(difference.numel())
    metrics["max_abs_error"] = max(metrics["max_abs_error"], float(difference.abs().max().item()))
    metrics["squared_error"] += float((difference * difference).sum().item())
    metrics["argmax_matches"] += float(
        (reference32.argmax(dim=-1) == candidate32.argmax(dim=-1)).sum().item()
    )
    topk = min(5, int(reference32.shape[-1]))
    reference_top = torch.topk(reference32, k=topk, dim=-1).indices
    candidate_top = torch.topk(candidate32, k=topk, dim=-1).indices
    overlap = (
        reference_top.unsqueeze(-1) == candidate_top.unsqueeze(-2)
    ).sum(dim=(-1, -2)).to(dtype=torch.float32)
    metrics["top5_overlap"] += float((overlap / topk).sum().item())


def _finalize_metric_accumulator(metrics: Mapping[str, float]) -> dict[str, float | int]:
    rows = int(metrics["rows"])
    elements = int(metrics["elements"])
    if rows <= 0 or elements <= 0:
        raise CalibrationError("held-out eval produced no rows")
    return {
        "rows": rows,
        "argmax_match_rate": metrics["argmax_matches"] / rows,
        "max_abs_error": metrics["max_abs_error"],
        "rms_error": math.sqrt(metrics["squared_error"] / elements),
        "top5_overlap": metrics["top5_overlap"] / rows,
    }


def evaluate_heldout(
    weight: Any,
    calibrated: Mapping[str, Any],
    rtn: Mapping[str, Any],
    eval_files: Sequence[str | Path],
    *,
    hidden_size: int,
    device: str | Any,
    max_rows: int = DEFAULT_EVAL_ROWS,
    chunk_rows: int = DEFAULT_EVAL_CHUNK_ROWS,
    op: Any | None = None,
    dequant_row_chunk: int = DEFAULT_DEQUANT_ROW_CHUNK,
) -> tuple[dict[str, Any], Any | None]:
    """Compare dense FP16, calibrated GPTQ, and RTN without retaining logits."""

    torch = _require_torch()
    if max_rows <= 0 or chunk_rows <= 0:
        raise ValueError("max_rows and chunk_rows must be positive")
    target = torch.device(device)
    dense_weight = weight.detach().to(device=target)
    if dense_weight.dtype != torch.float16:
        raise CalibrationError("held-out dense reference must remain FP16")
    calibrated_device = {
        **calibrated,
        "qweight": calibrated["qweight"].to(device=target),
        "scales": calibrated["scales"].to(device=target),
        "qzeros": calibrated["qzeros"].to(device=target),
    }
    rtn_device = {
        **rtn,
        "qweight": rtn["qweight"].to(device=target),
        "scales": rtn["scales"].to(device=target),
        "qzeros": rtn["qzeros"].to(device=target),
    }
    calibrated_metrics = _new_metric_accumulator()
    rtn_metrics = _new_metric_accumulator()
    seen = 0
    benchmark_hidden = None
    with torch.no_grad():
        for hidden_cpu in iter_hidden_states(eval_files, hidden_size):
            if seen >= max_rows:
                break
            take = min(int(hidden_cpu.shape[0]), max_rows - seen)
            for start in range(0, take, chunk_rows):
                hidden = hidden_cpu[start : min(take, start + chunk_rows)].to(
                    device=target, dtype=torch.float16
                ).contiguous()
                if benchmark_hidden is None:
                    benchmark_hidden = hidden[: min(5, int(hidden.shape[0]))].clone()
                elif int(benchmark_hidden.shape[0]) < 5:
                    needed = 5 - int(benchmark_hidden.shape[0])
                    benchmark_hidden = torch.cat((benchmark_hidden, hidden[:needed]), dim=0)
                dense_logits = torch.nn.functional.linear(hidden, dense_weight)
                calibrated_logits = _artifact_logits(
                    hidden,
                    calibrated_device,
                    op,
                    dequant_row_chunk=dequant_row_chunk,
                )
                rtn_logits = _artifact_logits(
                    hidden,
                    rtn_device,
                    op,
                    dequant_row_chunk=dequant_row_chunk,
                )
                _update_metric_accumulator(calibrated_metrics, dense_logits, calibrated_logits)
                _update_metric_accumulator(rtn_metrics, dense_logits, rtn_logits)
                seen += int(hidden.shape[0])
                if seen >= max_rows:
                    break
    if seen <= 0 or benchmark_hidden is None:
        raise CalibrationError("held-out eval produced no rows")
    return {
        "rows": seen,
        "dense_fp16_vs_calibrated_gptq": _finalize_metric_accumulator(calibrated_metrics),
        "dense_fp16_vs_rtn": _finalize_metric_accumulator(rtn_metrics),
    }, benchmark_hidden


def benchmark_w4a16(
    op: Any,
    hidden: Any,
    artifact: Mapping[str, Any],
    dense_weight: Any,
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Supplementary INT4 vs FP16 M=1/M=5 timing on real held-out states."""
    torch = _require_torch()
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if hidden.ndim != 2 or int(hidden.shape[0]) < 1:
        raise ValueError("benchmark hidden states must contain at least one row")
    if dense_weight.dtype != torch.float16 or dense_weight.ndim != 2:
        raise ValueError("FP16 benchmark reference must be a rank-2 float16 tensor")
    if dense_weight.device != hidden.device:
        raise ValueError("FP16 benchmark weight and hidden states must share a device")

    def measure(callable_: Any, sample: Any) -> list[float]:
        for _ in range(warmup):
            callable_(sample)
        _synchronize(sample.device)
        durations: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            callable_(sample)
            _synchronize(sample.device)
            durations.append((time.perf_counter() - started) * 1000.0)
        return durations

    def summarize(durations: list[float]) -> dict[str, float]:
        return {
            "median_ms": float(median(durations)),
            "mean_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        }

    measurements: dict[str, Any] = {
        "supplementary": True,
        "not_serving_speed": True,
        "warmup": warmup,
        "repeats": repeats,
        "operator": "torch.ops._xpu_C.int4_gemm_w4a16",
        "reference": "dense FP16 torch.nn.functional.linear",
    }
    for rows in (1, 5):
        if int(hidden.shape[0]) < rows:
            measurements[f"M{rows}"] = {"skipped": "fewer than requested held-out rows"}
            continue
        sample = hidden[:rows].contiguous()
        int4_durations = measure(
            lambda value: op(
                value,
                artifact["qweight"],
                None,
                artifact["scales"],
                artifact["qzeros"],
                int(artifact["group_size"]),
                None,
            ),
            sample,
        )
        fp16_durations = measure(
            lambda value: torch.nn.functional.linear(value, dense_weight),
            sample,
        )
        int4_summary = summarize(int4_durations)
        fp16_summary = summarize(fp16_durations)
        measurements[f"M{rows}"] = {
            "rows": rows,
            "int4": int4_summary,
            "fp16_reference": fp16_summary,
            "median_fp16_over_int4": fp16_summary["median_ms"] / int4_summary["median_ms"],
        }
    return measurements


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="existing Qwen checkpoint directory")
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new torch.save W4A16 artifact")
    parser.add_argument("--device", default="xpu", help="quantization/eval device (default: xpu)")
    parser.add_argument("--damp", type=float, default=0.01, help="fractional GPTQ diagonal damping")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--row-chunk-rows", type=int, default=0, help="0 quantizes all output rows at once")
    parser.add_argument("--hessian-chunk-rows", type=int, default=DEFAULT_HESSIAN_CHUNK_ROWS)
    parser.add_argument("--eval-rows", type=int, default=DEFAULT_EVAL_ROWS)
    parser.add_argument("--eval-chunk-rows", type=int, default=DEFAULT_EVAL_CHUNK_ROWS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch = _require_torch()
    if args.group_size != GROUP_SIZE:
        raise CalibrationError("--group-size must be 128 for the pinned XPU artifact")
    device = torch.device(args.device)
    if device.type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise CalibrationError("--device xpu requested but no XPU is available")
        op = _load_xpu_op()
    elif device.type in {"cpu", "cuda"}:
        op = None
        if device.type == "cuda" and not torch.cuda.is_available():
            raise CalibrationError("--device cuda requested but no CUDA device is available")
    else:
        raise CalibrationError(f"unsupported device for this tool: {device}")

    print("[phase] validating calibration and held-out captures", flush=True)
    calibration_files, eval_files, calibration_rows, eval_rows = validate_capture_sets(
        args.calibration_dir, args.eval_dir, QWEN_HEAD_SHAPE[1]
    )
    print(
        f"[phase] captures ready: calibration_files={len(calibration_files)} rows={calibration_rows} "
        f"eval_files={len(eval_files)} rows={eval_rows}",
        flush=True,
    )
    weight, shard = load_lm_head_weight(args.model, expected_shape=QWEN_HEAD_SHAPE)
    print(f"[phase] loaded dense lm_head from {shard}", flush=True)
    if op is not None:
        print("[phase] running real-XPU N128/K128 M1/M5 pack preflight", flush=True)
        preflight = xpu_kernel_preflight(
            weight, calibration_files, QWEN_HEAD_SHAPE[1], op
        )
        print("[phase] real-XPU pack preflight passed", flush=True)
    else:
        preflight = {
            "status": "skipped",
            "reason": "--device is not xpu",
        }
    hessian, hessian_rows = compute_activation_hessian(
        calibration_files,
        QWEN_HEAD_SHAPE[1],
        chunk_rows=args.hessian_chunk_rows,
        progress=True,
    )
    print(f"[phase] Hessian ready: rows={hessian_rows} shape={tuple(hessian.shape)}", flush=True)
    calibrated = gptq_quantize(
        weight,
        hessian,
        damp=args.damp,
        group_size=args.group_size,
        row_chunk_rows=args.row_chunk_rows,
        device=device,
        progress=True,
    )
    print("[phase] calibrated GPTQ head complete", flush=True)
    rtn = rtn_quantize(
        weight,
        group_size=args.group_size,
        row_chunk_rows=args.row_chunk_rows,
        device=device,
        progress=True,
    )
    print("[phase] RTN comparison head complete", flush=True)
    print(f"[phase] evaluating held-out logits (rows={args.eval_rows}, chunk={args.eval_chunk_rows})", flush=True)
    report, benchmark_hidden = evaluate_heldout(
        weight,
        calibrated,
        rtn,
        eval_files,
        hidden_size=QWEN_HEAD_SHAPE[1],
        device=device,
        max_rows=args.eval_rows,
        chunk_rows=args.eval_chunk_rows,
        op=op,
    )
    print("[phase] held-out comparison complete", flush=True)
    if op is not None:
        print("[phase] running supplementary INT4/FP16 M1/M5 microbench", flush=True)
        benchmark_dense_weight = weight.to(device=device)
        try:
            benchmark = benchmark_w4a16(
                op,
                benchmark_hidden,
                calibrated,
                benchmark_dense_weight,
            )
        finally:
            del benchmark_dense_weight
    else:
        benchmark = {
            "supplementary": True,
            "not_serving_speed": True,
            "skipped": "XPU int4_gemm_w4a16 benchmark requires --device xpu",
        }

    metadata = {
        "tool": "qwen38_calibrate_lmhead.py",
        "format": "xpu_w4a16_g128",
        "model": str(args.model.resolve()),
        "lm_head_shard": str(shard),
        "calibration": {
            "files": len(calibration_files),
            "rows": calibration_rows,
            "hessian_rows": hessian_rows,
            "hidden_size": QWEN_HEAD_SHAPE[1],
            "hessian_formula": "2 X^T X / N",
            "inputs_disjoint_from_eval": True,
        },
        "eval": {"files": len(eval_files), "available_rows": eval_rows, "reported_rows": report["rows"]},
        "device": str(device),
        "rtn_comparison": "same FP16 scales/zero-point contract, no Hessian correction",
        "xpu_preflight": preflight,
    }
    output = save_quantized_artifact(calibrated, args.output, metadata=metadata)
    report_payload = {
        "artifact": str(output),
        "metadata": metadata,
        "comparisons": report,
        "microbench": benchmark,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(output), "report": str(report_path), "comparisons": report, "microbench": benchmark}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI image.
    try:
        raise SystemExit(main())
    except CalibrationError as exc:
        print(f"qwen38_calibrate_lmhead: {exc}", file=sys.stderr)
        raise SystemExit(2)
