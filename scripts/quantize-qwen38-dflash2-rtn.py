#!/usr/bin/env python3
"""CPU-only RTN INT4 conversion of the native, single-file Qwen3.8 DFlash2 draft.

This is calibration-free round-to-nearest, NOT calibrated GPTQ. GPTQ v1 is
only the storage/dispatch format. QKV, conditioning, selector, convolutions,
norms and native vocabulary stay unchanged; no target weights are loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

try:
    import torch
except ImportError:  # Allows dependency-free config/path checks.
    torch = None

BITS = 4
GROUP_SIZE = 128
PACK_VALUES = 8
ZERO_POINT = 8
ROW_CHUNK_SIZE = 256
EXPECTED_TENSOR_COUNT = 81
EXPECTED_PARAMETER_COUNT = 1_924_404_480
QUANTIZATION_CONFIG = {
    "quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False,
    "sym": True, "lm_head": False, "checkpoint_format": "gptq",
    "modules_in_block_to_quantize": ["self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
}
QUANTIZED_TENSOR_NAMES = frozenset(
    f"layers.{layer}.{suffix}.weight"
    for layer in range(5)
    for suffix in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
)
EXPECTED_SHAPES = {
    "candidate_selector.hidden_projection.weight": (256, 5120),
    "candidate_selector.predecessor_codebook": (248320, 256),
    "candidate_selector.successor_codebook": (248320, 256),
    "fc.weight": (5120, 25600), "hidden_norm.weight": (5120,), "norm.weight": (5120,),
    **{
        f"layers.{layer}.{suffix}": shape
        for layer in range(5)
        for suffix, shape in {
            "attention_conv.base_kernel": (2, 2, 5120),
            "attention_conv.kernel_projection.weight": (1280, 5120),
            "input_layernorm.weight": (5120,),
            "mlp.down_proj.weight": (5120, 17408),
            "mlp.gate_proj.weight": (17408, 5120),
            "mlp.up_proj.weight": (17408, 5120),
            "mlp_conv.base_kernel": (2, 2, 5120),
            "mlp_conv.kernel_projection.weight": (1280, 5120),
            "post_attention_layernorm.weight": (5120,),
            "self_attn.k_norm.weight": (128,),
            "self_attn.k_proj.weight": (1024, 5120),
            "self_attn.o_proj.weight": (5120, 4096),
            "self_attn.q_norm.weight": (128,),
            "self_attn.q_proj.weight": (4096, 5120),
            "self_attn.v_proj.weight": (1024, 5120),
        }.items()
    },
}


def _is_quantized_name(name: str) -> bool:
    return name in QUANTIZED_TENSOR_NAMES


def _finite(tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"nonfinite {label}")


def _pack_nibbles(values):
    shifts = torch.arange(8, dtype=torch.int64) * 4
    return (values.to(torch.int64) << shifts).sum(dim=-1).to(torch.int32)


def quantize_weight(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """RTN [N,K] -> GPTQ-v1 qweight[K/8,N], BF16 scales[K/128,N].

    Each output-channel/input-group scale is 2*absmax/15, rounded to BF16
    BEFORE computing codes. Zero groups use scale 1. Codes are ties-to-even
    round(weight/scale)+8, clipped to [0,15]. Eight consecutive input codes
    occupy an int32's low-to-high nibbles. Stored zero points use v1's zp-1.
    """
    if torch is None:
        raise RuntimeError("conversion requires PyTorch")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("weight must be a two-dimensional [N,K] tensor")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("weight must have a floating-point dtype")
    n, k = weight.shape
    if n == 0 or k == 0 or n % 8 or k % 128:
        raise ValueError("nonzero N must be divisible by eight and K by group_size 128")
    source = weight.detach().to(device="cpu")
    _finite(source, "weight")
    groups = k // 128
    qweight = torch.empty((k // 8, n), dtype=torch.int32)
    scales = torch.empty((groups, n), dtype=torch.bfloat16)
    for start in range(0, n, ROW_CHUNK_SIZE):
        end = min(start + ROW_CHUNK_SIZE, n)
        grouped = source[start:end].float().reshape(end - start, groups, 128)
        absmax = grouped.abs().amax(dim=-1)
        _finite(absmax, "weight absmax")
        scale = (2 * absmax / 15).bfloat16()
        scale = torch.where(absmax == 0, torch.ones_like(scale), scale)
        _finite(scale, "RTN scales")
        if bool((scale <= 0).any().item()):
            raise ValueError("RTN scales must be positive")
        scales[:, start:end] = scale.T
        codes = (torch.round(grouped / scale.float().unsqueeze(-1)) + 8).clamp(0, 15).int()
        qweight[:, start:end] = _pack_nibbles(codes.reshape(end - start, k // 8, 8)).T
    return {
        "qweight": qweight, "scales": scales,
        "qzeros": torch.full((groups, n // 8), 0x77777777, dtype=torch.int32),
        "g_idx": torch.arange(k, dtype=torch.int32) // 128,
    }


def _validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"source checkpoint directory not found: {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("source and output must be unrelated paths (no ancestor/descendant conversion)")
    return source, output


def _validate_source_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("source config must be a JSON object")
    for key, value in config.items():
        if key == "tie_word_embeddings":
            if value is not False:
                raise ValueError("source must not contain tied head/embedding parameters")
            continue
        if (key in {"head", "head_dtype", "embedding", "embeddings", "embed"}
                or any(token in key.lower() for token in ("quant", "remap", "lm_head", "embed_tokens"))):
            raise ValueError("source config contains quantization/remap/head/embed metadata")
    required = {"model_type": "qwen3", "architectures": ["DFlash2DraftModel"],
                "hidden_size": 5120, "num_hidden_layers": 5, "vocab_size": 248320}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("source must be the native Qwen3.8 DFlash2 config")
    dtypes = [config[k] for k in ("dtype", "torch_dtype") if k in config]
    if not dtypes or any(v != "bfloat16" for v in dtypes):
        raise ValueError("source config must declare BF16 dtype")
    dflash = config.get("dflash_config")
    required = {"block_size": 8, "selector_rank": 256, "selector_top_k": 16,
                "conv_kernel_size": 2, "conv_group_size": 16,
                "target_layer_ids": [5, 19, 33, 47, 61]}
    if not isinstance(dflash, dict) or any(dflash.get(k) != v for k, v in required.items()):
        raise ValueError("source dflash_config must match the native checkpoint")


def _forbidden_tensor_name(name: str) -> bool:
    return any(token in name.lower() for token in ("qweight", "qzeros", "g_idx", ".scales",
                                                 "quant", "remap", "lm_head", "embed"))


def _output_config(source_config: dict) -> dict:
    config = copy.deepcopy(source_config)
    config["quantization_config"] = copy.deepcopy(QUANTIZATION_CONFIG)
    return config


def quantize_checkpoint(source: Path, output: Path) -> dict:
    """Convert only the known 81-tensor single-file checkpoint; never overwrite.

    Standard safetensors handles validation/serialization. Failed writes may
    leave this newly owned output directory for inspection; nothing is deleted.
    """
    source, output = _validate_paths(Path(source), Path(output))
    config = json.loads((source / "config.json").read_text())
    _validate_source_config(config)
    for filename in ("quantize_config.json", "quantization_config.json", "vocab_remap.json",
                     "model.safetensors.index.json"):
        if (source / filename).exists():
            raise ValueError(f"not the native single-file source: {filename}")
    if torch is None:
        raise RuntimeError("conversion requires PyTorch and safetensors")
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as sf:
        if any(_forbidden_tensor_name(name) for name in sf.keys()):
            raise ValueError("source contains quantization/remap/head/embed tensors")
        if set(sf.keys()) != set(EXPECTED_SHAPES) or len(sf.keys()) != EXPECTED_TENSOR_COUNT:
            raise ValueError("source tensor inventory must match all 81 native BF16 tensors")
        count = 0
        for name, shape in EXPECTED_SHAPES.items():
            header = sf.get_slice(name)
            if header.get_dtype() != "BF16" or tuple(header.get_shape()) != shape:
                raise ValueError(f"source tensor shape/dtype mismatch: {name}")
            count += math.prod(shape)
        if count != EXPECTED_PARAMETER_COUNT:
            raise ValueError("source parameter count mismatch")
        for name in sf.keys():
            weight = sf.get_tensor(name)
            _finite(weight, name)
            if _is_quantized_name(name):
                for component, tensor in quantize_weight(weight).items():
                    tensors[name.removesuffix(".weight") + "." + component] = tensor
                print("RTN_INT4", name, flush=True)
            else:
                tensors[name] = weight  # No cast: retained native BF16 bytes.
        output.mkdir(parents=True, exist_ok=False)
        save_file(tensors, str(output / "model.safetensors"), metadata=sf.metadata())
        (output / "config.json").write_text(json.dumps(_output_config(config), indent=2) + "\n")
    return {"method": "RTN", "bits": BITS, "group_size": GROUP_SIZE,
            "quantized_matrices": len(QUANTIZED_TENSOR_NAMES),
            "retained_native_tensors": EXPECTED_TENSOR_COUNT - len(QUANTIZED_TENSOR_NAMES),
            "output_tensors": len(tensors),
            "tensor_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(quantize_checkpoint(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
