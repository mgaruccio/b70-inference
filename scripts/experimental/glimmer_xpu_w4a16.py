#!/usr/bin/env python3
"""Experimental M<=32 XPU GPTQ W4A16 GEMM and bounded microbenchmark.

This is deliberately a microbenchmark filter, not a vLLM integration.  It
implements the exact symmetric GPTQ layout used by the current Glimmer model:

* x:       [M, K] fp16 contiguous
* qweight: [K / 8, N] int32 with strides (1, K / 8); k % 8 selects a nibble
* scales:  [K / 128, N] fp16 contiguous
* zero:    one int8 value equal to 8

For split-K configurations, each split starts at a G128 boundary, writes fp32
partials, and a second kernel reduces the partials in a fixed order before one
fp16 conversion.  No fp16 atomics are used.

Run with --check, --bench, or both.  With neither flag, both modes run.  Stdout
is one JSON document; progress goes to stderr.  The defaults use six explicit
configs and bounded repetitions rather than Triton autotune.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Callable, Sequence

import torch
import triton
import triton.language as tl


PACK_FACTOR = 8
GROUP_SIZE = 128
BLOCK_M = 32
BLOCK_K = 128
FP16_UNIT_ROUNDOFF = 2.0**-11
FP16_MIN_SUBNORMAL = 2.0**-24
FP32_UNIT_ROUNDOFF = 2.0**-24


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    m: int
    k: int
    n: int
    profiled_us: float | None = None
    profiled_calls: int | None = None
    profiled_total_ms: float | None = None


@dataclass(frozen=True)
class KernelConfig:
    name: str
    block_n: int
    split_k: int
    num_warps: int
    num_stages: int
    grf_mode: str


@dataclass
class Problem:
    shape: ShapeSpec
    x: torch.Tensor
    qweight: torch.Tensor
    scales: torch.Tensor
    zero_point: torch.Tensor


@dataclass
class KernelBuffers:
    output: torch.Tensor
    partials: torch.Tensor


@dataclass
class ReferenceSlice:
    columns: list[int]
    exact: torch.Tensor
    rounded_dequant: torch.Tensor
    dequant_rounding: torch.Tensor
    rounded_sum_abs: torch.Tensor
    k: int

    def fp16_result_bound(self, split_k: int) -> torch.Tensor:
        """Conservative elementwise bound against the CPU fp32 reference.

        The terms are the measured fp16 dequantization error, Higham gamma
        bounds for both fp32 accumulation orders, fixed-order split reduction,
        and final fp16 output rounding.  The 1.02 factor is arithmetic-model
        slack, not an output-accuracy tolerance.
        """
        gamma_k = _gamma(self.k, FP32_UNIT_ROUNDOFF)
        gamma_split = _gamma(max(0, split_k - 1), FP32_UNIT_ROUNDOFF)
        accumulation = (2.0 * gamma_k + gamma_split) * self.rounded_sum_abs
        before_output = self.dequant_rounding + accumulation
        output_rounding = FP16_UNIT_ROUNDOFF * (
            self.exact.abs() + before_output
        ) + 0.5 * FP16_MIN_SUBNORMAL
        return 1.02 * (before_output + output_rounding) + FP16_MIN_SUBNORMAL


OBSERVED_SHAPES = {
    "gate": ShapeSpec(
        "gate_up", 32, 6656, 39936, profiled_us=281.0,
        profiled_calls=228, profiled_total_ms=64.0,
    ),
    "down": ShapeSpec(
        "down", 32, 19968, 6656, profiled_us=151.0,
        profiled_calls=228, profiled_total_ms=34.0,
    ),
}
TAIL_SHAPE = ShapeSpec("awkward_tail", 13, 384, 77)

# Explicitly small sweep.  It varies N tile, split count, subgroup count, and
# GRF mode, because the official XPU tutorial's 32-warp/stage-4 choice targets
# much larger M and should not be assumed optimal at M=32.
CONFIGS = (
    KernelConfig("m32n64k128-s1-w8-st2-auto", 64, 1, 8, 2, "auto"),
    KernelConfig("m32n64k128-s2-w16-st3-auto", 64, 2, 16, 3, "auto"),
    KernelConfig("m32n64k128-s4-w16-st3-grf256", 64, 4, 16, 3, "256"),
    KernelConfig("m32n128k128-s1-w16-st2-grf256", 128, 1, 16, 2, "256"),
    KernelConfig("m32n128k128-s2-w16-st3-grf256", 128, 2, 16, 3, "256"),
    KernelConfig("m32n128k128-s4-w32-st4-auto", 128, 4, 32, 4, "auto"),
)
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


@triton.jit
def _w4a16_m32_kernel(
    x_ptr,
    qweight_ptr,
    scale_ptr,
    output_ptr,
    partial_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_qk,
    stride_qn,
    stride_sg,
    stride_sn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    PACKING: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Fused symmetric INT4 dequant + fp16 tl.dot/fp32 accumulation."""
    tl.static_assert(BLOCK_SIZE_M == 32)
    tl.static_assert(BLOCK_SIZE_K == QUANT_GROUP_SIZE)
    tl.static_assert(PACKING == 8)

    pid_n = tl.program_id(axis=0)
    pid_split = tl.program_id(axis=1)

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_packed_k = tl.arange(0, BLOCK_SIZE_K // PACKING)
    offs_nibble = tl.arange(0, PACKING)

    group_count = K // QUANT_GROUP_SIZE
    groups_per_split = tl.cdiv(group_count, SPLIT_K)
    first_group = pid_split * groups_per_split
    accumulator = tl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
    )

    # A dynamic loop keeps K out of the compilation key.  Every iteration is
    # exactly one G128 block, so every split boundary is a scale-group boundary.
    for group_offset in range(0, groups_per_split):
        group_index = first_group + group_offset
        valid_group = group_index < group_count
        group_k = group_index * QUANT_GROUP_SIZE

        x_ptrs = (
            x_ptr
            + offs_m[:, None] * stride_xm
            + (group_k + offs_k[None, :]) * stride_xk
        )
        activations = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & valid_group,
            other=0.0,
        )

        packed_row = (
            group_index * (QUANT_GROUP_SIZE // PACKING)
            + offs_packed_k
        )
        q_ptrs = (
            qweight_ptr
            + packed_row[:, None] * stride_qk
            + offs_n[None, :] * stride_qn
        )
        packed = tl.load(
            q_ptrs,
            mask=valid_group & (offs_n[None, :] < N),
            other=0,
        ).to(tl.int32)

        # packed[p, n] -> codes[p, nibble, n] -> codes[k, n].  Loading each
        # int32 once preserves the 8x packing advantage before the DPAS dot.
        shifts = (offs_nibble[None, :, None] * 4).to(tl.int32)
        codes_3d = (packed[:, None, :] >> shifts) & 0xF
        codes = tl.reshape(
            codes_3d, (BLOCK_SIZE_K, BLOCK_SIZE_N), can_reorder=False
        )

        scale = tl.load(
            scale_ptr + group_index * stride_sg + offs_n * stride_sn,
            mask=valid_group & (offs_n < N),
            other=0.0,
        ).to(tl.float32)
        weights = (
            (codes.to(tl.float32) - 8.0) * scale[None, :]
        ).to(tl.float16)
        accumulator = tl.dot(activations, weights, accumulator)

    output_offsets = offs_m[:, None] * N + offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(tl.float16),
            mask=output_mask,
        )
    else:
        partial_offsets = pid_split * M * N + output_offsets
        tl.store(partial_ptr + partial_offsets, accumulator, mask=output_mask)


@triton.jit
def _reduce_split_k_kernel(
    partial_ptr,
    output_ptr,
    element_count,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Deterministic fp32 split reduction followed by one fp16 conversion."""
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    total = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_index in tl.static_range(0, SPLIT_K):
        total += tl.load(
            partial_ptr + split_index * element_count + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(output_ptr + offsets, total.to(tl.float16), mask=mask)


def _gamma(operation_count: int, unit_roundoff: float) -> float:
    product = operation_count * unit_roundoff
    if product >= 1.0:
        return math.inf
    return product / (1.0 - product)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _safe_error(exc: BaseException, limit: int = 1600) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _runtime_info() -> dict:
    device_name = None
    try:
        device_name = torch.xpu.get_device_name(0)
    except Exception:  # Device validation reports the actionable failure later.
        pass
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton_import": getattr(triton, "__version__", None),
        "triton_xpu_distribution": _version("triton-xpu"),
        "vllm": _version("vllm"),
        "vllm_xpu_kernels": _version("vllm-xpu-kernels"),
        "xpu_device": device_name,
        "timing": "torch.xpu.Event(enable_timing=True) with device synchronize",
    }


def _validate_runtime() -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("an available PyTorch XPU device is required")
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        try:
            from vllm._xpu_ops import xpu_ops as _xpu_ops_registration
        except Exception as exc:
            raise RuntimeError(
                "failed to register torch.ops._xpu_C.int4_gemm_w4a16 via "
                "vllm._xpu_ops.xpu_ops"
            ) from exc
        del _xpu_ops_registration
    if not hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):
        raise RuntimeError(
            "torch.ops._xpu_C.int4_gemm_w4a16 remained unavailable after "
            "importing vllm._xpu_ops.xpu_ops"
        )


def _validate_problem(problem: Problem) -> None:
    shape = problem.shape
    if not 1 <= shape.m <= BLOCK_M:
        raise ValueError(f"M must be in [1, {BLOCK_M}], got {shape.m}")
    if shape.k % GROUP_SIZE:
        raise ValueError(f"K must be divisible by G128, got {shape.k}")
    if problem.x.shape != (shape.m, shape.k):
        raise ValueError("x shape does not match ShapeSpec")
    if problem.x.dtype != torch.float16 or not problem.x.is_contiguous():
        raise ValueError("x must be contiguous fp16")
    if problem.qweight.shape != (shape.k // PACK_FACTOR, shape.n):
        raise ValueError("qweight shape must be [K/8, N]")
    expected_q_strides = (1, shape.k // PACK_FACTOR)
    if problem.qweight.dtype != torch.int32 or problem.qweight.stride() != expected_q_strides:
        raise ValueError(
            "qweight must be int32 with actual oneDNN strides "
            f"{expected_q_strides}, got {problem.qweight.stride()}"
        )
    if problem.scales.shape != (shape.k // GROUP_SIZE, shape.n):
        raise ValueError("scales shape must be [K/128, N]")
    if problem.scales.dtype != torch.float16 or not problem.scales.is_contiguous():
        raise ValueError("scales must be contiguous fp16")
    if problem.zero_point.dtype != torch.int8 or problem.zero_point.numel() != 1:
        raise ValueError("zero_point must be a one-element int8 tensor")
    if int(problem.zero_point.item()) != 8:
        raise ValueError("this symmetric prototype requires zero_point=[8]")


def _make_problem(shape: ShapeSpec, seed: int, device: torch.device) -> Problem:
    if shape.k % GROUP_SIZE:
        raise ValueError("synthetic problem K must be divisible by 128")
    torch.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
    x = torch.randn(
        (shape.m, shape.k), device=device, dtype=torch.float16
    ).mul_(0.125)
    # qweight_storage is [N, K/8] contiguous.  The transpose is exactly the
    # non-contiguous [K/8, N], strides=(1, K/8) view consumed by oneDNN.
    qweight_storage = torch.randint(
        -(2**31),
        2**31 - 1,
        (shape.n, shape.k // PACK_FACTOR),
        device=device,
        dtype=torch.int32,
    )
    qweight = qweight_storage.t()
    scales = torch.empty(
        (shape.k // GROUP_SIZE, shape.n),
        device=device,
        dtype=torch.float16,
    ).uniform_(0.002, 0.02)
    zero_point = torch.tensor([8], device=device, dtype=torch.int8)
    problem = Problem(shape, x, qweight, scales, zero_point)
    _validate_problem(problem)
    return problem


def _allocate_buffers(problem: Problem, config: KernelConfig) -> KernelBuffers:
    shape = problem.shape
    output = torch.empty(
        (shape.m, shape.n), device=problem.x.device, dtype=torch.float16
    )
    if config.split_k == 1:
        # The direct kernel ignores this pointer; using output avoids a dummy
        # allocation while retaining one stable launch signature.
        partials = output
    else:
        partials = torch.empty(
            (config.split_k, shape.m, shape.n),
            device=problem.x.device,
            dtype=torch.float32,
        )
    return KernelBuffers(output, partials)


def _launch_triton(
    problem: Problem, config: KernelConfig, buffers: KernelBuffers
) -> torch.Tensor:
    shape = problem.shape
    grid = (triton.cdiv(shape.n, config.block_n), config.split_k)
    _w4a16_m32_kernel[grid](
        problem.x,
        problem.qweight,
        problem.scales,
        buffers.output,
        buffers.partials,
        shape.m,
        shape.n,
        shape.k,
        problem.x.stride(0),
        problem.x.stride(1),
        problem.qweight.stride(0),
        problem.qweight.stride(1),
        problem.scales.stride(0),
        problem.scales.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=config.block_n,
        BLOCK_SIZE_K=BLOCK_K,
        QUANT_GROUP_SIZE=GROUP_SIZE,
        PACKING=PACK_FACTOR,
        SPLIT_K=config.split_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        grf_mode=config.grf_mode,
    )
    if config.split_k > 1:
        element_count = shape.m * shape.n
        reduce_grid = (triton.cdiv(element_count, 256),)
        _reduce_split_k_kernel[reduce_grid](
            buffers.partials,
            buffers.output,
            element_count,
            SPLIT_K=config.split_k,
            BLOCK_SIZE=256,
            num_warps=8,
            num_stages=1,
            grf_mode="auto",
        )
    return buffers.output


def _launch_onednn(problem: Problem) -> torch.Tensor:
    return torch.ops._xpu_C.int4_gemm_w4a16(
        problem.x,
        problem.qweight,
        None,
        problem.scales,
        problem.zero_point,
        GROUP_SIZE,
        None,
    )


def _reference_columns(n: int, count: int) -> list[int]:
    if count >= n:
        return list(range(n))
    priority = [0, n - 1, 1, n - 2, 63, 64, 127, 128, n // 2]
    columns: list[int] = []
    for value in priority:
        if 0 <= value < n and value not in columns:
            columns.append(value)
        if len(columns) == count:
            return sorted(columns)
    denominator = max(1, count - 1)
    for index in range(count):
        value = round(index * (n - 1) / denominator)
        if value not in columns:
            columns.append(value)
        if len(columns) == count:
            break
    # Very small collisions can leave a gap; fill deterministically.
    for value in range(n):
        if value not in columns:
            columns.append(value)
        if len(columns) == count:
            break
    return sorted(columns)


def _make_reference(problem: Problem, columns: list[int]) -> ReferenceSlice:
    """Build only selected dequantized columns on CPU in fp32."""
    device_columns = torch.tensor(
        columns, device=problem.qweight.device, dtype=torch.long
    )
    packed = problem.qweight.index_select(1, device_columns).cpu().to(torch.int64)
    scale = problem.scales.index_select(1, device_columns).cpu().to(torch.float32)
    x = problem.x.cpu().to(torch.float32)

    shifts = (torch.arange(PACK_FACTOR, dtype=torch.int64) * 4).view(1, -1, 1)
    codes = ((packed[:, None, :] >> shifts) & 0xF).to(torch.float32)
    codes = codes.reshape(problem.shape.k, len(columns))
    expanded_scale = scale.repeat_interleave(GROUP_SIZE, dim=0)
    exact_weight = (codes - 8.0) * expanded_scale
    rounded_weight = exact_weight.to(torch.float16).to(torch.float32)

    exact = torch.matmul(x, exact_weight)
    rounded_dequant = torch.matmul(x, rounded_weight)
    dequant_rounding = torch.matmul(
        x.abs(), (rounded_weight - exact_weight).abs()
    )
    rounded_sum_abs = torch.matmul(x.abs(), rounded_weight.abs())
    return ReferenceSlice(
        columns,
        exact,
        rounded_dequant,
        dequant_rounding,
        rounded_sum_abs,
        problem.shape.k,
    )


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _compare_to_reference(
    output: torch.Tensor, reference: ReferenceSlice, split_k: int
) -> dict:
    device_columns = torch.tensor(
        reference.columns, device=output.device, dtype=torch.long
    )
    actual = output.index_select(1, device_columns).cpu().to(torch.float32)
    expected = reference.exact
    absolute = (actual - expected).abs()
    relative = absolute / expected.abs().clamp_min(1.0e-6)
    rounded_absolute = (actual - reference.rounded_dequant).abs()
    bound = reference.fp16_result_bound(split_k)
    finite = torch.isfinite(actual)
    violations = finite & (absolute > bound)
    violation_count = int(violations.sum().item())
    finite_count = int(finite.sum().item())
    ratio = absolute / bound.clamp_min(FP16_MIN_SUBNORMAL)
    return {
        "finite": finite_count == actual.numel(),
        "within_fp16_error_bound": (
            finite_count == actual.numel() and violation_count == 0
        ),
        "elements": actual.numel(),
        "violation_count": violation_count,
        "vs_exact_fp32_dequant": {
            "max_abs": _json_float(absolute.max().item()),
            "mean_abs": _json_float(absolute.mean().item()),
            "p99_abs": _json_float(
                torch.quantile(absolute.flatten(), 0.99).item()
            ),
            "max_rel": _json_float(relative.max().item()),
        },
        "vs_fp16_rounded_dequant_fp32_accum": {
            "max_abs": _json_float(rounded_absolute.max().item()),
            "mean_abs": _json_float(rounded_absolute.mean().item()),
            "p99_abs": _json_float(
                torch.quantile(rounded_absolute.flatten(), 0.99).item()
            ),
        },
        "max_bound": _json_float(bound.max().item()),
        "max_error_to_bound_ratio": _json_float(ratio.max().item()),
    }


def _compare_outputs_sampled(
    left: torch.Tensor, right: torch.Tensor, columns: list[int]
) -> dict:
    device_columns = torch.tensor(columns, device=left.device, dtype=torch.long)
    lhs = left.index_select(1, device_columns).cpu().to(torch.float32)
    rhs = right.index_select(1, device_columns).cpu().to(torch.float32)
    absolute = (lhs - rhs).abs()
    return {
        "elements": absolute.numel(),
        "exact_equal_fraction": _json_float((lhs == rhs).float().mean().item()),
        "max_abs": _json_float(absolute.max().item()),
        "mean_abs": _json_float(absolute.mean().item()),
        "p99_abs": _json_float(torch.quantile(absolute.flatten(), 0.99).item()),
    }


def _shape_metadata(problem: Problem, reference: ReferenceSlice) -> dict:
    shape = problem.shape
    return {
        "name": shape.name,
        "M": shape.m,
        "K": shape.k,
        "N": shape.n,
        "group_size": GROUP_SIZE,
        "qweight_shape": list(problem.qweight.shape),
        "qweight_strides": list(problem.qweight.stride()),
        "scale_shape": list(problem.scales.shape),
        "reference_columns": reference.columns,
        "reference_dequantized_elements": shape.k * len(reference.columns),
        "full_target_weight_dequantized": len(reference.columns) == shape.n,
        "packed_weight_bytes": problem.qweight.numel() * problem.qweight.element_size(),
        "scale_bytes": problem.scales.numel() * problem.scales.element_size(),
        "profile_context": {
            "eager_profile_us_per_call": shape.profiled_us,
            "calls": shape.profiled_calls,
            "aggregate_ms": shape.profiled_total_ms,
        },
    }


def _try_onednn(problem: Problem, reference: ReferenceSlice) -> tuple[torch.Tensor | None, dict]:
    try:
        output = _launch_onednn(problem)
        torch.xpu.synchronize()
        comparison = _compare_to_reference(output, reference, split_k=1)
        return output, {"status": "ok", "reference": comparison}
    except Exception as exc:  # Experimental driver must preserve other results.
        return None, {"status": "error", "error": _safe_error(exc)}


def _run_check_case(
    shape: ShapeSpec,
    configs: Sequence[KernelConfig],
    seed: int,
    reference_column_count: int,
    include_onednn: bool,
    deadline: float,
) -> dict:
    _log(f"check: creating {shape.name} M={shape.m} K={shape.k} N={shape.n}")
    problem = _make_problem(shape, seed, torch.device("xpu"))
    columns = _reference_columns(shape.n, reference_column_count)
    reference = _make_reference(problem, columns)
    result = _shape_metadata(problem, reference)

    baseline: torch.Tensor | None = None
    if include_onednn:
        baseline, result["onednn"] = _try_onednn(problem, reference)
    else:
        result["onednn"] = {
            "status": "skipped",
            "reason": (
                "awkward-tail correctness is a Triton layout test; oneDNN is "
                "benchmarked on observed shapes"
            ),
        }

    config_results: list[dict] = []
    numerical_failures = 0
    valid_configs = 0
    for config in configs:
        if time.monotonic() >= deadline:
            config_results.append({
                "config": asdict(config),
                "status": "skipped_budget_exhausted",
            })
            continue
        _log(f"check: {shape.name}: {config.name}")
        entry: dict = {"config": asdict(config)}
        try:
            buffers = _allocate_buffers(problem, config)
            output = _launch_triton(problem, config, buffers)
            torch.xpu.synchronize()
            comparison = _compare_to_reference(output, reference, config.split_k)
            entry["reference"] = comparison
            if baseline is not None:
                entry["vs_onednn_sample"] = _compare_outputs_sampled(
                    output, baseline, reference.columns
                )
            if comparison["within_fp16_error_bound"]:
                entry["status"] = "pass"
                valid_configs += 1
            else:
                entry["status"] = "numerical_failure"
                numerical_failures += 1
        except Exception as exc:
            entry.update(status="unavailable", error=_safe_error(exc))
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        config_results.append(entry)
    result["configs"] = config_results
    result["passed"] = valid_configs > 0 and numerical_failures == 0
    result["valid_config_count"] = valid_configs
    result["numerical_failure_count"] = numerical_failures
    del baseline, reference, problem
    gc.collect()
    torch.xpu.empty_cache()
    return result


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _measure(
    launch: Callable[[Problem], torch.Tensor],
    problems: Sequence[Problem],
    warmup: int,
    repeats: int,
    inner: int,
    deadline: float,
) -> dict:
    warmup_count = max(warmup, len(problems))
    for index in range(warmup_count):
        launch(problems[index % len(problems)])
    torch.xpu.synchronize()

    samples_us: list[float] = []
    for repeat in range(repeats):
        if time.monotonic() >= deadline:
            raise TimeoutError("benchmark time budget exhausted")
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        base = repeat * inner
        for offset in range(inner):
            launch(problems[(base + offset) % len(problems)])
        end.record()
        torch.xpu.synchronize()
        samples_us.append(float(start.elapsed_time(end)) * 1000.0 / inner)

    ordered = sorted(samples_us)
    mean = statistics.fmean(samples_us)
    stdev = statistics.pstdev(samples_us) if len(samples_us) > 1 else 0.0
    return {
        "median_us": _json_float(statistics.median(samples_us)),
        "mean_us": _json_float(mean),
        "min_us": _json_float(ordered[0]),
        "max_us": _json_float(ordered[-1]),
        "p20_us": _json_float(_percentile(ordered, 0.20)),
        "p80_us": _json_float(_percentile(ordered, 0.80)),
        "cv": _json_float(stdev / mean if mean else 0.0),
        "sample_us": [_json_float(value) for value in samples_us],
        "repeats": repeats,
        "inner": inner,
        "weight_bank_count": len(problems),
    }


def _run_benchmark_shape(
    shape: ShapeSpec,
    configs: Sequence[KernelConfig],
    args: argparse.Namespace,
    seed: int,
    deadline: float,
) -> dict:
    _log(
        f"bench: creating {args.rotate_bank} bank(s) for {shape.name} "
        f"M={shape.m} K={shape.k} N={shape.n}"
    )
    problems = [
        _make_problem(shape, seed + bank * 1009, torch.device("xpu"))
        for bank in range(args.rotate_bank)
    ]
    reference = _make_reference(
        problems[0], _reference_columns(shape.n, args.reference_columns)
    )
    result = _shape_metadata(problems[0], reference)
    result["rotated_bank_count"] = len(problems)
    result["rotated_packed_and_scale_bytes"] = len(problems) * (
        result["packed_weight_bytes"] + result["scale_bytes"]
    )

    baseline_output, baseline_check = _try_onednn(problems[0], reference)
    result["onednn_numerical"] = baseline_check
    if baseline_output is None:
        result.update(
            status="baseline_error",
            error="oneDNN baseline is required for the microbenchmark filter",
        )
        return result

    sweep: list[dict] = []
    selectable: list[tuple[float, KernelConfig]] = []
    for config in configs:
        entry: dict = {"config": asdict(config)}
        if time.monotonic() >= deadline:
            entry["status"] = "skipped_budget_exhausted"
            sweep.append(entry)
            continue
        _log(f"bench: {shape.name}: tuning {config.name}")
        try:
            buffers = _allocate_buffers(problems[0], config)
            output = _launch_triton(problems[0], config, buffers)
            torch.xpu.synchronize()
            numerical = _compare_to_reference(output, reference, config.split_k)
            entry["reference"] = numerical
            entry["vs_onednn_sample"] = _compare_outputs_sampled(
                output, baseline_output, reference.columns
            )
            if not numerical["within_fp16_error_bound"]:
                entry["status"] = "numerical_failure"
                sweep.append(entry)
                continue

            def launch_one(problem: Problem) -> torch.Tensor:
                return _launch_triton(problem, config, buffers)

            timing = _measure(
                launch_one,
                [problems[0]],
                args.warmup,
                args.tune_repeats,
                args.inner,
                deadline,
            )
            entry.update(status="ok", warm_tuning=timing)
            selectable.append((float(timing["median_us"]), config))
        except Exception as exc:
            entry.update(status="unavailable", error=_safe_error(exc))
            try:
                torch.xpu.synchronize()
            except Exception:
                pass
        sweep.append(entry)
    result["config_sweep"] = sweep

    if not selectable:
        result.update(status="no_valid_triton_config")
        return result
    _, selected = min(selectable, key=lambda item: item[0])
    result["selected_config"] = asdict(selected)
    selected_buffers = _allocate_buffers(problems[0], selected)

    baseline_sink: list[torch.Tensor | None] = [baseline_output]

    def launch_baseline(problem: Problem) -> torch.Tensor:
        baseline_sink[0] = _launch_onednn(problem)
        return baseline_sink[0]

    def launch_selected(problem: Problem) -> torch.Tensor:
        return _launch_triton(problem, selected, selected_buffers)

    try:
        _log(f"bench: {shape.name}: final oneDNN warm")
        onednn_warm = _measure(
            launch_baseline,
            [problems[0]],
            args.warmup,
            args.repeats,
            args.inner,
            deadline,
        )
        _log(f"bench: {shape.name}: final Triton warm")
        triton_warm = _measure(
            launch_selected,
            [problems[0]],
            args.warmup,
            args.repeats,
            args.inner,
            deadline,
        )
        onednn_rotated = None
        triton_rotated = None
        if len(problems) > 1:
            _log(f"bench: {shape.name}: final oneDNN rotated")
            onednn_rotated = _measure(
                launch_baseline,
                problems,
                args.warmup,
                args.repeats,
                args.inner,
                deadline,
            )
            _log(f"bench: {shape.name}: final Triton rotated")
            triton_rotated = _measure(
                launch_selected,
                problems,
                args.warmup,
                args.repeats,
                args.inner,
                deadline,
            )
    except Exception as exc:
        result.update(status="timing_error", error=_safe_error(exc))
        return result

    warm_speedup = float(onednn_warm["median_us"]) / float(
        triton_warm["median_us"]
    )
    rotated_speedup = None
    if onednn_rotated is not None and triton_rotated is not None:
        rotated_speedup = float(onednn_rotated["median_us"]) / float(
            triton_rotated["median_us"]
        )
    compared_speedups = [warm_speedup]
    if rotated_speedup is not None:
        compared_speedups.append(rotated_speedup)
    filter_pass = min(compared_speedups) >= args.filter_speedup

    result.update(
        status="ok",
        onednn_timing={"warm": onednn_warm, "rotated": onednn_rotated},
        triton_timing={"warm": triton_warm, "rotated": triton_rotated},
        speedup_onednn_over_triton={
            "warm": _json_float(warm_speedup),
            "rotated": _json_float(rotated_speedup) if rotated_speedup is not None else None,
        },
        microbench_filter={
            "required_speedup": args.filter_speedup,
            "passes": filter_pass,
            "requires_warm_and_rotated_when_rotation_enabled": True,
            "meaning": "filter only; passing does not authorize service integration",
        },
    )
    if shape.profiled_calls is not None:
        measured_saving_us = float(onednn_warm["median_us"]) - float(
            triton_warm["median_us"]
        )
        result["simple_profile_extrapolation"] = {
            "measured_warm_saving_us_per_call": _json_float(measured_saving_us),
            "saving_ms_over_profiled_call_count": _json_float(
                measured_saving_us * shape.profiled_calls / 1000.0
            ),
            "warning": "kernel-only arithmetic; it is not a TPS prediction",
        }

    del baseline_output, baseline_sink, selected_buffers, reference, problems
    gc.collect()
    torch.xpu.empty_cache()
    return result


def _select_shapes(value: str) -> list[ShapeSpec]:
    if value == "both":
        return [OBSERVED_SHAPES["gate"], OBSERVED_SHAPES["down"]]
    if value == "none":
        return []
    return [OBSERVED_SHAPES[value]]


def _select_configs(value: str) -> list[KernelConfig]:
    if value == "all":
        return list(CONFIGS)
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIG_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown config(s): {', '.join(unknown)}; choices are "
            + ", ".join(CONFIG_BY_NAME)
        )
    if not names:
        raise ValueError("--configs selected no configs")
    return [CONFIG_BY_NAME[name] for name in names]


def _validate_args(args: argparse.Namespace) -> None:
    bounded = {
        "rotate_bank": (args.rotate_bank, 1, 4),
        "warmup": (args.warmup, 1, 50),
        "tune_repeats": (args.tune_repeats, 1, 30),
        "repeats": (args.repeats, 1, 50),
        "inner": (args.inner, 1, 100),
        "reference_columns": (args.reference_columns, 4, 64),
        "max_seconds": (args.max_seconds, 30, 300),
    }
    for name, (value, lower, upper) in bounded.items():
        if not lower <= value <= upper:
            raise ValueError(f"--{name.replace('_', '-')} must be in [{lower}, {upper}]")
    if not 1.0 <= args.filter_speedup <= 2.0:
        raise ValueError("--filter-speedup must be in [1.0, 2.0]")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and benchmark an experimental XPU M32 GPTQ W4A16 Triton GEMM."
    )
    parser.add_argument("--check", action="store_true", help="run numerical checks")
    parser.add_argument("--bench", action="store_true", help="run oneDNN comparison benchmarks")
    parser.add_argument(
        "--check-observed",
        choices=("none", "gate", "down", "both"),
        default="down",
        help="observed-size sampled-reference cases added after the awkward tail",
    )
    parser.add_argument(
        "--bench-shapes",
        choices=("gate", "down", "both"),
        default="both",
    )
    parser.add_argument(
        "--configs",
        default="all",
        help="'all' or comma-separated explicit config names",
    )
    parser.add_argument(
        "--rotate-bank",
        type=int,
        default=3,
        help="1 disables rotated-weight timing; 2-4 rotates independent weights",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tune-repeats", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--inner", type=int, default=8)
    parser.add_argument("--reference-columns", type=int, default=16)
    parser.add_argument(
        "--filter-speedup",
        type=float,
        default=1.10,
        help="minimum oneDNN/Triton median ratio for both warm and rotated timing",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=285,
        help="cooperative wall-clock budget checked between compile/timing units",
    )
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--pretty", action="store_true", help="indent JSON stdout")
    parser.add_argument(
        "--traceback", action="store_true", help="print top-level traceback to stderr"
    )
    args = parser.parse_args(argv)
    if not args.check and not args.bench:
        args.check = True
        args.bench = True
    return args


def _execute(args: argparse.Namespace) -> tuple[dict, int]:
    _validate_args(args)
    configs = _select_configs(args.configs)
    _validate_runtime()
    started = time.monotonic()
    deadline = started + args.max_seconds
    report: dict = {
        "schema_version": 1,
        "experiment": "glimmer-xpu-m32-w4a16",
        "status": "running",
        "environment": _runtime_info(),
        "expected_runtime": {
            "image": (
                "vllm/vllm-openai-xpu@sha256:"
                "f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
            ),
            "torch": "2.13.0+xpu",
            "triton_xpu": "3.7.2",
            "vllm": "0.27.2rc1.dev77+gac7509e2b.xpu",
            "vllm_xpu_kernels": "0.1.13.2",
        },
        "requested_modes": [
            mode
            for mode, enabled in (("check", args.check), ("bench", args.bench))
            if enabled
        ],
        "selected_sweep": [asdict(config) for config in configs],
        "contract": {
            "activation": "fp16 contiguous [M,K]",
            "qweight": "int32 [K/8,N], strides [1,K/8], k%8 nibble",
            "dequant": "fp16((nibble - 8) * fp16_scale), G128 symmetric",
            "accumulation": "tl.dot fp16 inputs with fp32 accumulator",
            "split_k": "G128 boundaries, fp32 workspace, deterministic fixed-order reduction",
            "output": "fp16 [M,N]",
            "unsupported": ["M>32", "K%128!=0", "bias", "g_idx", "asymmetric zero points"],
        },
        "numerical_bound": (
            "selected-column CPU fp32 reference; exact fp16 dequant rounding error + "
            "Higham fp32 accumulation/reduction gamma bounds + fp16 output rounding"
        ),
        "checks": [],
        "benchmarks": [],
        "research": {
            "official_source": "https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/python/tutorials/03-matrix-multiplication.py",
            "applied_conclusions": [
                "tl.dot accepts fp16 inputs with an fp32 accumulator",
                "XPU exposes num_warps, num_stages, and grf_mode compile options",
                "the tutorial's large-M defaults are swept rather than assumed for M32",
            ],
        },
        "limitations": [
            "Synthetic packed weights and scales only; no checkpoint weights are materialized or saved.",
            "Observed-shape fp32 references dequantize selected columns only, never the full target matrix.",
            "A small rotated bank is cache-cold-ish, not proof of a specific cache residency state.",
            "No vLLM/service integration, graph capture, serving restart, or host tuning is performed.",
            "A passing microbenchmark is only a filter; promotion requires normal graph-enabled public-API TPS and correctness verification.",
        ],
    }

    if args.check:
        report["checks"].append(
            _run_check_case(
                TAIL_SHAPE,
                configs,
                args.seed,
                TAIL_SHAPE.n,
                include_onednn=False,
                deadline=deadline,
            )
        )
        for index, shape in enumerate(_select_shapes(args.check_observed)):
            if time.monotonic() >= deadline:
                report["checks"].append({
                    "name": shape.name,
                    "passed": False,
                    "status": "skipped_budget_exhausted",
                })
                continue
            report["checks"].append(
                _run_check_case(
                    shape,
                    configs,
                    args.seed + 10_000 + index,
                    args.reference_columns,
                    include_onednn=True,
                    deadline=deadline,
                )
            )

    if args.bench:
        for index, shape in enumerate(_select_shapes(args.bench_shapes)):
            if time.monotonic() >= deadline:
                report["benchmarks"].append({
                    "name": shape.name,
                    "status": "skipped_budget_exhausted",
                })
                continue
            benchmark = _run_benchmark_shape(
                shape, configs, args, args.seed + 20_000 + index, deadline
            )
            report["benchmarks"].append(benchmark)

    elapsed = time.monotonic() - started
    report["elapsed_seconds"] = _json_float(elapsed)
    report["budget_seconds"] = args.max_seconds
    report["budget_exhausted"] = elapsed >= args.max_seconds

    checks_pass = all(case.get("passed", False) for case in report["checks"])
    benches_complete = all(
        bench.get("status") == "ok" for bench in report["benchmarks"]
    )
    numerical_bench_pass = all(
        all(
            entry.get("status") != "numerical_failure"
            for entry in bench.get("config_sweep", [])
        )
        for bench in report["benchmarks"]
    )
    successful = (
        checks_pass
        and benches_complete
        and numerical_bench_pass
        and not report["budget_exhausted"]
    )
    report["status"] = "ok" if successful else "incomplete_or_failed"
    if report["benchmarks"]:
        filters = [
            bench.get("microbench_filter", {}).get("passes")
            for bench in report["benchmarks"]
            if bench.get("status") == "ok"
        ]
        report["microbench_filter_summary"] = {
            "all_completed_shapes_pass": bool(filters) and all(filters),
            "shape_results": {
                bench.get("name", f"shape_{index}"): bench.get(
                    "microbench_filter", {}
                ).get("passes")
                for index, bench in enumerate(report["benchmarks"])
            },
            "next_step_if_true": (
                "graph-enabled vLLM public-API A/B; do not promote from this "
                "result alone"
            ),
        }
    return report, 0 if successful else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with torch.inference_mode():
            report, return_code = _execute(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "experiment": "glimmer-xpu-m32-w4a16",
            "status": "error",
            "error": _safe_error(exc),
        }
        return_code = 2
        if args.traceback:
            traceback.print_exc(file=sys.stderr)
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, sort_keys=True, allow_nan=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
