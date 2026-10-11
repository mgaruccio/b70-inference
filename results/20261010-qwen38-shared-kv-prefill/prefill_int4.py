#!/usr/bin/env python3
"""Experiment-scoped Triton FP16 symmetric GPTQ-G128 large-M GEMM.

The kernel consumes the physical packed layout used by the installed XPU path:
``qweight`` is contiguous ``[N, K / 8]`` int32 storage and ``scales`` is
contiguous ``[K / 128, N]`` FP16.  Nibbles are unpacked in registers, rounded
to FP16 before an FP32-accumulating ``tl.dot``, and written as FP16 output.

This file deliberately does not patch vLLM or keep a dequantized weight cache.
``candidate_int4_gemm_w4a16`` is the direct microbenchmark entry point.
``dispatch_int4_gemm_w4a16`` is a serving-shaped seam for a later experiment:
small M (decode) and unsupported layouts stay on the installed operator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final


PACKING: Final[int] = 8
GROUP_SIZE: Final[int] = 128
DECODE_MAX_M: Final[int] = 32


class UnsupportedConfiguration(ValueError):
    """The direct candidate refuses a layout it was not written to handle."""


@dataclass(frozen=True)
class KernelConfig:
    """One of the deliberately small, static XPU tile choices."""

    name: str
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    grf_mode: str = "auto"


# Keep this list short and explicit.  This is not a Triton autotune table.
KERNEL_CONFIGS: Final[tuple[KernelConfig, ...]] = (
    KernelConfig("m64-n128-k128", 64, 128, 128, 16, 2),
    KernelConfig("m128-n128-k128", 128, 128, 128, 16, 2),
    KernelConfig("m128-n256-k64", 128, 256, 64, 16, 2),
)


def available_configs() -> tuple[str, ...]:
    """Return the bounded config names exposed to the microbenchmark."""

    return tuple(config.name for config in KERNEL_CONFIGS) + ("tiled-fp16",)


def _find_config(name: str | None) -> KernelConfig:
    if name is None:
        raise ValueError("a concrete config is required here")
    for config in KERNEL_CONFIGS:
        if config.name == name:
            return config
    raise UnsupportedConfiguration(
        f"unknown config {name!r}; choose one of {', '.join(available_configs())}"
    )


def _choose_config(m: int, n: int) -> KernelConfig:
    """Choose a static tile without mutable tuning or shape-specific caches."""

    if m <= KERNEL_CONFIGS[0].block_m:
        return KERNEL_CONFIGS[0]
    if n > KERNEL_CONFIGS[1].block_n:
        return KERNEL_CONFIGS[2]
    return KERNEL_CONFIGS[1]


def _require_xpu_tensor(tensor: Any, name: str) -> None:
    if getattr(getattr(tensor, "device", None), "type", None) != "xpu":
        raise UnsupportedConfiguration(f"{name} must be on an XPU device")


def _validate_problem(
    x: Any,
    qweight: Any,
    g_idx: Any,
    scales: Any,
    qzeros: Any,
    group_size: int,
    bias: Any,
    config_name: str | None,
) -> tuple[Any, KernelConfig]:
    """Validate every assumption before handing pointers to Triton."""

    if getattr(x, "ndim", 0) < 1:
        raise UnsupportedConfiguration("x must have a final K dimension")
    if getattr(x, "dtype", None) != _torch_dtype("float16"):
        raise UnsupportedConfiguration("x must be FP16")
    if not bool(x.is_contiguous()):
        raise UnsupportedConfiguration("x must be contiguous")
    _require_xpu_tensor(x, "x")

    if getattr(qweight, "ndim", 0) != 2:
        raise UnsupportedConfiguration("qweight must be rank 2 [N, K/8]")
    if getattr(qweight, "dtype", None) != _torch_dtype("int32"):
        raise UnsupportedConfiguration("qweight must be int32")
    if not bool(qweight.is_contiguous()):
        raise UnsupportedConfiguration("qweight must be contiguous [N, K/8]")
    _require_xpu_tensor(qweight, "qweight")

    if getattr(scales, "ndim", 0) != 2:
        raise UnsupportedConfiguration("scales must be rank 2 [K/128, N]")
    if getattr(scales, "dtype", None) != _torch_dtype("float16"):
        raise UnsupportedConfiguration("scales must be FP16")
    if not bool(scales.is_contiguous()):
        raise UnsupportedConfiguration("scales must be contiguous")
    _require_xpu_tensor(scales, "scales")

    if g_idx is not None:
        raise UnsupportedConfiguration("act-order g_idx is unsupported; expected None")
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise UnsupportedConfiguration("group_size must be the integer 128")
    if group_size != GROUP_SIZE:
        raise UnsupportedConfiguration("only symmetric GPTQ group_size=128 is supported")

    if qzeros is None:
        raise UnsupportedConfiguration("qzeros must be the symmetric scalar [8]")
    if getattr(qzeros, "ndim", 0) != 1 or int(qzeros.numel()) != 1:
        raise UnsupportedConfiguration("qzeros must contain exactly one scalar")
    if getattr(qzeros, "dtype", None) != _torch_dtype("int8"):
        raise UnsupportedConfiguration("qzeros must be int8")
    _require_xpu_tensor(qzeros, "qzeros")
    if int(qzeros.reshape(-1)[0].item()) != 8:
        raise UnsupportedConfiguration("only symmetric qzeros=[8] is supported")

    if bias is not None:
        if getattr(bias, "ndim", 0) != 1:
            raise UnsupportedConfiguration("bias must be rank 1 [N]")
        if getattr(bias, "dtype", None) != _torch_dtype("float16"):
            raise UnsupportedConfiguration("bias must be FP16")
        if not bool(bias.is_contiguous()):
            raise UnsupportedConfiguration("bias must be contiguous")
        _require_xpu_tensor(bias, "bias")

    devices = (x.device, qweight.device, scales.device, qzeros.device)
    if any(device != devices[0] for device in devices[1:]):
        raise UnsupportedConfiguration("x, qweight, scales, and qzeros need one XPU device")
    if bias is not None and bias.device != devices[0]:
        raise UnsupportedConfiguration("bias must share the activation device")

    k = int(x.shape[-1])
    n = int(qweight.shape[0])
    packed_k = int(qweight.shape[1])
    if k <= 0 or n <= 0 or packed_k * PACKING != k:
        raise UnsupportedConfiguration(
            f"shape mismatch: x K={k}, qweight={tuple(qweight.shape)}"
        )
    if k % GROUP_SIZE:
        raise UnsupportedConfiguration("K must be divisible by GPTQ group_size=128")
    if tuple(scales.shape) != (k // GROUP_SIZE, n):
        raise UnsupportedConfiguration(
            f"scales must be [{k // GROUP_SIZE}, {n}], got {tuple(scales.shape)}"
        )
    if bias is not None and int(bias.shape[0]) != n:
        raise UnsupportedConfiguration(f"bias must have N={n} elements")

    m = 1
    for dimension in x.shape[:-1]:
        m *= int(dimension)
    if m <= 0:
        raise UnsupportedConfiguration("x must contain at least one matrix row")

    config = _find_config(config_name) if config_name is not None else _choose_config(m, n)
    if config.block_k <= 0 or GROUP_SIZE % config.block_k:
        raise UnsupportedConfiguration(f"invalid static config {config.name}")
    return x.reshape(m, k), config


def _torch_dtype(name: str) -> Any:
    """Import torch only when a candidate call is actually made."""

    import torch

    return getattr(torch, name)


# This GPU-only module has runtime imports; benchmark --help does not import it.
# A stable JITFunction lets discarded warmup actually populate its launch cache.
import triton
import triton.language as tl


@triton.jit
def _int4_gemm_kernel(
    x_ptr, qweight_ptr, scale_ptr, bias_ptr, output_ptr,
    M, N, K, stride_xm, stride_xk, stride_qn, stride_qk,
    stride_sg, stride_sn, stride_b, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr, PACKING: tl.constexpr, GROUP_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_packed_k = tl.arange(0, BLOCK_K // PACKING)
    offs_nibble = tl.arange(0, PACKING)
    row_mask = offs_m < M
    col_mask = offs_n < N
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    group_count = K // GROUP_SIZE
    packed_per_group = GROUP_SIZE // PACKING
    for group_index in range(0, group_count):
        group_k = group_index * GROUP_SIZE
        scale = tl.load(
            scale_ptr + group_index * stride_sg + offs_n * stride_sn,
            mask=col_mask, other=0.0,
        ).to(tl.float32)
        for k_offset in range(0, GROUP_SIZE, BLOCK_K):
            x_ptrs = (x_ptr + offs_m[:, None] * stride_xm
                      + (group_k + k_offset + offs_k[None, :]) * stride_xk)
            activations = tl.load(
                x_ptrs,
                mask=row_mask[:, None] & ((group_k + k_offset + offs_k)[None, :] < K),
                other=0.0,
            )
            packed_rows = group_index * packed_per_group + k_offset // PACKING + offs_packed_k
            q_ptrs = (qweight_ptr + offs_n[None, :] * stride_qn
                      + packed_rows[:, None] * stride_qk)
            packed = tl.load(q_ptrs, mask=col_mask[None, :], other=0).to(tl.int32)
            shifts = (offs_nibble[None, :, None] * 4).to(tl.int32)
            codes = (packed[:, None, :] >> shifts) & 0xF
            codes = tl.reshape(codes, (BLOCK_K, BLOCK_N), can_reorder=False)
            weights = ((codes.to(tl.float32) - 8.0) * scale[None, :]).to(tl.float16)
            accumulator = tl.dot(activations, weights, accumulator)
    if HAS_BIAS:
        bias_values = tl.load(bias_ptr + offs_n * stride_b, mask=col_mask, other=0.0).to(tl.float32)
        accumulator += bias_values[None, :]
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, accumulator.to(tl.float16), mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _dequant_fp16_tile(q_ptr, s_ptr, w_ptr, K, N, START_N,
                       TILE_N: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
    ks = tl.program_id(0) * BK + tl.arange(0, BK)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    actual_n = START_N + ns
    mask = (ks[:, None] < K) & (ns[None, :] < TILE_N)
    packed = tl.load(q_ptr + actual_n[None, :] * (K // 8) + ks[:, None] // 8,
                     mask=mask, other=0).to(tl.int32)
    values = ((packed >> ((ks[:, None] % 8) * 4)) & 15).to(tl.float32) - 8.0
    scales = tl.load(s_ptr + (ks[:, None] // 128) * N + actual_n[None, :],
                     mask=mask, other=0).to(tl.float32)
    tl.store(w_ptr + ns[None, :] * K + ks[:, None],
             (values * scales).to(tl.float16), mask=mask)


def _launch_tiled_fp16(x: Any, qweight: Any, scales: Any, output: Any) -> None:
    """Dequantize <=64MiB at a time; dense FP16 GEMM writes into output slices.

    This is an opt-in supplemental experiment, not the default dispatcher.
    No tensor or dense weight survives the call. Native matmul may also allocate
    a bounded output tile/scratchpad; report actual peak memory rather than
    claiming the 64MiB weight limit is a total-memory limit.
    """
    import torch

    k = int(x.shape[1])
    n = int(qweight.shape[0])
    cap = 64 * 1024 * 1024
    width = min(n, (cap // (2 * k) // 32) * 32)
    if width <= 0:
        raise UnsupportedConfiguration("K exceeds the 64MiB dequant tile cap")
    for start in range(0, n, width):
        count = min(width, n - start)
        dense = torch.empty((count, k), device=x.device, dtype=torch.float16)
        assert dense.numel() * dense.element_size() <= cap
        _dequant_fp16_tile[(triton.cdiv(k, 128), triton.cdiv(count, 32))](
            qweight, scales, dense, k, n, start,
            TILE_N=count, BK=128, BN=32, num_warps=4,
        )
        # Match the native packed weight's N-major physical orientation.
        torch.mm(x, dense.transpose(0, 1), out=output[:, start:start + count])
        del dense

def _launch_kernel(
    x: Any,
    qweight: Any,
    scales: Any,
    bias: Any,
    output: Any,
    config: KernelConfig,
) -> None:
    """Launch the stable kernel without a weight-dependent cache or workspace."""

    import torch

    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(qweight.shape[0])
    grid = (triton.cdiv(m, config.block_m), triton.cdiv(n, config.block_n))
    # When bias is absent the pointer is intentionally unused because HAS_BIAS
    # specializes the branch away; reusing x avoids a dummy allocation.
    bias_ptr = bias if bias is not None else x
    _int4_gemm_kernel[grid](
        x,
        qweight,
        scales,
        bias_ptr,
        output,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        bias.stride(0) if bias is not None else 1,
        output.stride(0),
        output.stride(1),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        HAS_BIAS=bias is not None,
        PACKING=PACKING,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        grf_mode=config.grf_mode,
    )


def candidate_int4_gemm_w4a16(
    x: Any,
    qweight: Any,
    bias: Any,
    scales: Any,
    qzeros: Any,
    group_size: int,
    g_idx: Any = None,
    *,
    config_name: str | None = None,
) -> Any:
    """Run the direct fused candidate on a validated FP16 XPU problem.

    The positional arguments intentionally mirror
    ``torch.ops._xpu_C.int4_gemm_w4a16``.  Unlike the serving dispatcher, this
    entry point raises ``UnsupportedConfiguration`` instead of silently falling
    back when its exact layout contract is not met.
    """

    import torch

    flat_x, config = _validate_problem(
        x, qweight, g_idx, scales, qzeros, group_size, bias,
        None if config_name == "tiled-fp16" else config_name
    )
    output = torch.empty(
        (int(flat_x.shape[0]), int(qweight.shape[0])),
        device=x.device,
        dtype=torch.float16,
    )
    if config_name == "tiled-fp16":
        if bias is not None:
            raise UnsupportedConfiguration("tiled-fp16 screen requires bias=None")
        _launch_tiled_fp16(flat_x, qweight, scales, output)
    else:
        _launch_kernel(flat_x, qweight, scales, bias, output, config)
    return output


def _installed_operator(torch: Any) -> Any:
    try:
        op = getattr(torch.ops._xpu_C, "int4_gemm_w4a16")
    except AttributeError as exc:
        raise RuntimeError(
            "torch.ops._xpu_C.int4_gemm_w4a16 is unavailable; import the "
            "pinned vLLM XPU registration first"
        ) from exc
    return op



def _native_candidate_weight_view(x: Any, qweight: Any, scales: Any) -> Any:
    """Turn the installed transposed view into the candidate's zero-copy view."""

    try:
        packed_k = int(qweight.shape[0])
        expected_n = int(scales.shape[1])
        if (
            getattr(qweight, "ndim", 0) == 2
            and tuple(qweight.shape) == (packed_k, expected_n)
            and packed_k * PACKING == int(x.shape[-1])
            and tuple(qweight.stride()) == (1, packed_k)
        ):
            return qweight.transpose(0, 1)
    except Exception:
        pass
    return qweight
def dispatch_int4_gemm_w4a16(
    x: Any,
    qweight: Any,
    bias: Any,
    scales: Any,
    qzeros: Any,
    group_size: int,
    g_idx: Any = None,
    *,
    config_name: str | None = None,
) -> Any:
    """Serving-shaped dispatch preserving the installed decode path.

    M<=32 always calls the installed operator.  Larger M calls the candidate
    only when its complete layout contract is valid; every unsupported
    candidate configuration falls back to the same installed operator.
    """

    import torch

    native = _installed_operator(torch)
    m = 0
    try:
        if getattr(x, "ndim", 0) >= 1:
            m = 1
            for dimension in x.shape[:-1]:
                m *= int(dimension)
    except Exception:
        m = 0
    if m and m <= DECODE_MAX_M:
        return native(x, qweight, bias, scales, qzeros, group_size, g_idx)
    try:
        return candidate_int4_gemm_w4a16(
            x,
            _native_candidate_weight_view(x, qweight, scales),
            bias,
            scales,
            qzeros,
            group_size,
            g_idx,
            config_name=config_name,
        )
    except UnsupportedConfiguration:
        return native(x, qweight, bias, scales, qzeros, group_size, g_idx)


# Explicit aliases make the two intended public seams easy to discover without
# changing the native operator's positional calling convention.
prefill_int4_gemm_w4a16 = candidate_int4_gemm_w4a16
serving_int4_gemm_w4a16 = dispatch_int4_gemm_w4a16


__all__ = [
    "DECODE_MAX_M",
    "GROUP_SIZE",
    "KERNEL_CONFIGS",
    "KernelConfig",
    "UnsupportedConfiguration",
    "available_configs",
    "candidate_int4_gemm_w4a16",
    "dispatch_int4_gemm_w4a16",
    "prefill_int4_gemm_w4a16",
    "serving_int4_gemm_w4a16",
]
