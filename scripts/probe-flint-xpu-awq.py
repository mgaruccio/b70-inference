#!/usr/bin/env python3
"""Reproduce the pinned Flint AWQ/XPU MoE packing blocker; never run a MoE kernel.

Uses tiny CPU tensors, not model weights. Exit 2 means an observed incompatibility;
exit 3 means unverified (including a changed runtime or no detected mismatch).
This diagnostic has no successful-inference or compatibility-pass outcome.
"""

import argparse
import inspect
import sys
import traceback


IMAGE = (
    "vllm/vllm-openai-xpu@sha256:"
    "f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
)
VLLM_VERSION = "0.27.2rc1.dev77+gac7509e2b"
TORCH_VERSION = "2.13.0+xpu"
SOURCE = (
    "https://github.com/vllm-project/vllm/blob/ac7509e2b/"
    "vllm/model_executor/layers/fused_moe/oracle/int_wna16.py"
)
CHECKPOINT = "mattbucci/Flint-AWQ@2cae47a65e71d151a002172d18bcf2589fe09177"
EXPERTS, INPUTS, OUTPUTS, GROUP_SIZE = 2, 128, 64, 64
AWQ_SHAPES = ((EXPERTS, INPUTS, 2 * OUTPUTS // 8), (EXPERTS, OUTPUTS, INPUTS // 8))
NATIVE_SHAPES = ((EXPERTS, 2 * OUTPUTS, INPUTS // 2), (EXPERTS, INPUTS, OUTPUTS // 2))


def report_conversion(actual_shapes, zero_points_present):
    """Diagnose metadata only; matching metadata cannot prove numerical correctness."""
    print(f"AWQ input int32 shapes (w13, w2): {AWQ_SHAPES}")
    print("AWQ zero points: w13=5, w2=7 (noncentral, packed int32); scales=BF16")
    print(f"Expected native uint8 shapes (w13, w2): {NATIVE_SHAPES}")
    print(f"Actual converted shapes (w13, w2): {actual_shapes}")
    print(f"Returned zero points present (w13, w2): {zero_points_present}")
    issues = []
    if actual_shapes != NATIVE_SHAPES:
        issues.append("AWQ output-axis packing was not converted to native input-axis packing")
    if not all(zero_points_present):
        issues.append("noncentral AWQ zero points were omitted from the converted result")
    if issues:
        print("INCOMPATIBLE with the pinned native XPU MoE contract:")
        for issue in issues:
            print(f"- {issue}")
        return 2
    print("UNVERIFIED: metadata checks found no mismatch; values/kernel/serving were not tested.")
    return 3


def probe_runtime():
    # Keep third-party imports here: --help and helper tests require only Python.
    import torch
    import vllm

    print(f"Python: {sys.version.split()[0]}")
    print(f"Installed torch: {torch.__version__}; vLLM: {vllm.__version__}")
    if (str(torch.__version__), vllm.__version__) != (TORCH_VERSION, VLLM_VERSION):
        print("UNVERIFIED: runtime versions differ from the pinned reproduction; not probing.")
        return 3

    from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
        WNA16MoEBackend,
        convert_to_wna16_moe_kernel_format,
    )
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    print(f"Converter file: {inspect.getsourcefile(convert_to_wna16_moe_kernel_format)}")
    print(f"Converter signature: {inspect.signature(convert_to_wna16_moe_kernel_format)}")
    config = AutoAWQConfig.from_config(
        {"quant_method": "awq", "bits": 4, "group_size": GROUP_SIZE, "zero_point": True}
    )
    # Constant nibbles are valid in AWQ's interleaved output-axis order too.
    # This deliberately does NOT implement a converter, repack, or dequantizer.
    packed = convert_to_wna16_moe_kernel_format(
        WNA16MoEBackend.XPU,
        torch.nn.Module(),
        config,
        None,
        torch.full(AWQ_SHAPES[0], 0x44444444, dtype=torch.int32, device="cpu"),
        torch.full(AWQ_SHAPES[1], 0x33333333, dtype=torch.int32, device="cpu"),
        torch.ones((EXPERTS, INPUTS // GROUP_SIZE, 2 * OUTPUTS), dtype=torch.bfloat16, device="cpu"),
        torch.ones((EXPERTS, OUTPUTS // GROUP_SIZE, INPUTS), dtype=torch.bfloat16, device="cpu"),
        w13_qzeros=torch.full(
            (EXPERTS, INPUTS // GROUP_SIZE, 2 * OUTPUTS // 8),
            0x55555555, dtype=torch.int32, device="cpu",
        ),
        w2_qzeros=torch.full(
            (EXPERTS, OUTPUTS // GROUP_SIZE, INPUTS // 8),
            0x77777777, dtype=torch.int32, device="cpu",
        ),
    )
    if not isinstance(packed, tuple) or len(packed) != 14:
        print("UNVERIFIED: unexpected converter return contract.")
        return 3
    return report_conversion(
        (tuple(packed[0].shape), tuple(packed[1].shape)),
        (packed[8] is not None, packed[9] is not None),
    )


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(f"Pinned image (caller must select it): {IMAGE}")
    print(f"Expected torch: {TORCH_VERSION}; vLLM: {VLLM_VERSION}")
    print(f"Checkpoint format reference only, not downloaded: {CHECKPOINT}")
    print(f"Converter source: {SOURCE}")
    print("CPU tensors only; MoE kernel dispatched: NO; full-model HTTP E2E: NOT RUN.")
    try:
        return probe_runtime()
    except Exception:
        traceback.print_exc()
        print("UNVERIFIED: probe could not complete; see the exception above.")
        return 3


if __name__ == "__main__":
    sys.exit(main())
