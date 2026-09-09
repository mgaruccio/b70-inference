#!/usr/bin/env python3
"""CPU-only, calibration-free RTN conversion for the Qwen3.8 DFlash2 drafter.

This is round-to-nearest (RTN), not calibrated GPTQ.  The resulting tensors use
standard GPTQ v1 W4A16 packing so the disposable drafter can be loaded by a
GPTQ-aware runtime.  The native BF16 source is never modified and no target
vocabulary, embedding, or head is copied into the output.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ImportError:  # pragma: no cover - exercised by environments without torch
    torch = None  # type: ignore[assignment]



BITS = 4
GROUP_SIZE = 128
PACK_VALUES = 8
ZERO_POINT = 8
ROW_CHUNK_SIZE = 256
EXPECTED_TENSOR_COUNT = 81
EXPECTED_PARAMETER_COUNT = 1_924_404_480

QUANTIZATION_CONFIG = {
    "quant_method": "gptq",
    "bits": 4,
    "group_size": 128,
    "desc_act": False,
    "sym": True,
    "lm_head": False,
    "checkpoint_format": "gptq",
    "modules_in_block_to_quantize": [
        "self_attn.o_proj",
        "mlp.gate_up_proj",
        "mlp.down_proj",
    ],
}



# This set is deliberately explicit: Q/K/V, conditioning FC, selector/codebooks,
# convolutions, and all normalization tensors are not quantization candidates.
QUANTIZED_TENSOR_NAMES = frozenset(
    f"layers.{layer}.{suffix}"
    for layer in range(5)
    for suffix in (
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
)

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "candidate_selector.hidden_projection.weight": (256, 5120),
    "candidate_selector.predecessor_codebook": (248320, 256),
    "candidate_selector.successor_codebook": (248320, 256),
    "fc.weight": (5120, 25600),
    "hidden_norm.weight": (5120,),
    "norm.weight": (5120,),
}
for _layer in range(5):
    for _suffix, _shape in {
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
    }.items():
        EXPECTED_SHAPES[f"layers.{_layer}.{_suffix}"] = _shape


class _TensorHeader:
    __slots__ = ("dtype", "shape", "data_offsets")

    def __init__(self, dtype: str, shape: tuple[int, ...], data_offsets: tuple[int, int]):
        self.dtype = dtype
        self.shape = shape
        self.data_offsets = data_offsets


class _ShardHeader:
    __slots__ = ("path", "tensors", "metadata")

    def __init__(self, path: Path, tensors: dict[str, _TensorHeader], metadata: dict[str, str] | None):
        self.path = path
        self.tensors = tensors
        self.metadata = metadata


class _CheckpointLayout:
    __slots__ = ("source", "shards", "index_path", "index_data", "weight_map")

    def __init__(
        self,
        source: Path,
        shards: tuple[Path, ...],
        index_path: Path | None,
        index_data: dict[str, Any] | None,
        weight_map: dict[str, str],
    ):
        self.source = source
        self.shards = shards
        self.index_path = index_path
        self.index_data = index_data
        self.weight_map = weight_map


def _require_torch():
    if torch is None:
        raise RuntimeError("quantize-qwen38-dflash2-rtn.py requires PyTorch")
    return torch


def _is_quantized_name(name: str) -> bool:
    """Return whether *name* is one of the exactly twenty RTN candidates."""

    return name in QUANTIZED_TENSOR_NAMES


def _pack_nibbles(values):
    """Pack the last dimension's eight low-to-high 4-bit values into int32."""

    t = _require_torch()
    if values.ndim == 0 or values.shape[-1] != PACK_VALUES:
        raise ValueError("nibble packing requires a final dimension of eight values")
    if values.dtype not in (t.int32, t.int64, t.uint8):
        raise ValueError("nibble packing requires integer values")
    shifts = t.arange(PACK_VALUES, dtype=t.int64, device=values.device) * BITS
    packed = (values.to(t.int64) << shifts).sum(dim=-1)
    return packed.to(dtype=t.int32)


def _finite(tensor, what: str) -> None:
    t = _require_torch()
    if not bool(t.isfinite(tensor).all().item()):
        raise ValueError(f"nonfinite {what}")

def _positive_finite_scales(scales, absmax):
    _require_torch()
    _finite(scales, "RTN scales")
    if bool(((absmax > 0) & (scales <= 0)).any().item()):
        raise ValueError("nonpositive RTN scale for nonzero group")
    if bool((scales <= 0).any().item()):
        raise ValueError("RTN scales must be positive")
    return scales


def quantize_weight(weight: "torch.Tensor") -> dict[str, "torch.Tensor"]:
    """Quantize one ``[N, K]`` BF16/floating matrix with deterministic RTN.

    The result is standard GPTQ v1 orientation and packing:

    * ``qweight`` is ``[K/8, N]`` int32, with input values in low-to-high
      nibbles;
    * ``scales`` is ``[K/128, N]`` BF16;
    * ``qzeros`` is ``[K/128, N/8]`` int32, storing actual zero-point 8 minus
      one (all packed nibbles are therefore 7);
    * ``g_idx`` is a sequential int32 group id for each input column.

    It is calibration-free RTN: no activations or target model are consulted.
    Rows are processed in bounded chunks so the temporary code tensor does not
    scale with the complete matrix.
    """

    t = _require_torch()
    if not isinstance(weight, t.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    if weight.ndim != 2:
        raise ValueError("weight must be a two-dimensional [N, K] tensor")
    if weight.dtype not in (t.float16, t.bfloat16, t.float32, t.float64):
        raise ValueError("weight must have a floating-point dtype")

    n, k = (int(weight.shape[0]), int(weight.shape[1]))
    if n == 0 or k == 0 or n % PACK_VALUES:
        raise ValueError("weight output dimension N must be nonzero and divisible by eight")
    if k % GROUP_SIZE:
        raise ValueError("weight input dimension K must be divisible by group_size 128")

    # Check the source dtype before conversion, then do all arithmetic in FP32.
    source = weight.detach().to(device="cpu")
    _finite(source, "weight")
    groups = k // GROUP_SIZE
    qweight = t.empty((k // PACK_VALUES, n), dtype=t.int32, device="cpu")
    scales = t.empty((groups, n), dtype=t.bfloat16, device="cpu")

    for row_start in range(0, n, ROW_CHUNK_SIZE):
        row_end = min(row_start + ROW_CHUNK_SIZE, n)
        rows = source[row_start:row_end].to(dtype=t.float32)
        grouped = rows.reshape(row_end - row_start, groups, GROUP_SIZE)
        absmax = grouped.abs().amax(dim=-1)
        _finite(absmax, "weight absmax")

        # Round the exact symmetric range scale to BF16 before deriving codes.
        rounded_scales = (2.0 * absmax / 15.0).to(dtype=t.bfloat16)
        _finite(rounded_scales, "RTN scales")
        zero_groups = absmax == 0
        rounded_scales = t.where(
            zero_groups,
            t.ones_like(rounded_scales, dtype=t.bfloat16),
            rounded_scales,
        )
        rounded_scales = _positive_finite_scales(rounded_scales, absmax)
        scale_fp32 = rounded_scales.to(dtype=t.float32)
        scales[:, row_start:row_end] = rounded_scales.transpose(0, 1)

        codes = t.round(grouped / scale_fp32.unsqueeze(-1)) + ZERO_POINT
        codes = codes.clamp(min=0, max=(1 << BITS) - 1).to(dtype=t.int32)
        packed = _pack_nibbles(codes.reshape(row_end - row_start, k // PACK_VALUES, PACK_VALUES))
        qweight[:, row_start:row_end] = packed.transpose(0, 1).contiguous()

    qzeros = _pack_nibbles(
        t.full((groups, n), ZERO_POINT - 1, dtype=t.int32, device="cpu").reshape(
            groups, n // PACK_VALUES, PACK_VALUES
        )
    )
    g_idx = t.arange(k, dtype=t.int32, device="cpu") // GROUP_SIZE
    _finite(scales, "RTN scales")
    if bool((scales <= 0).any().item()):
        raise ValueError("RTN scales must be positive")
    return {
        "qweight": qweight,
        "scales": scales,
        "qzeros": qzeros,
        "g_idx": g_idx,
    }


def _output_name(weight_name: str, component: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError(f"quantized candidate is not a weight tensor: {weight_name}")
    return f"{weight_name[:-len('.weight')]}.{component}"


def _transform_tensor_map(tensors: Mapping[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
    """Replace only candidate weights; retained tensors keep their native object."""

    output: dict[str, "torch.Tensor"] = {}
    for name, tensor in tensors.items():
        if not _is_quantized_name(name):
            if name in output:
                raise ValueError(f"duplicate output tensor name: {name}")
            output[name] = tensor
            continue
        quantized = quantize_weight(tensor)
        for component, packed in quantized.items():
            out_name = _output_name(name, component)
            if out_name in output:
                raise ValueError(f"duplicate output tensor name: {out_name}")
            output[out_name] = packed
    return output


def _load_safetensors_api():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - depends on conversion image
        raise RuntimeError("checkpoint conversion requires the safetensors package") from exc
    return safe_open, save_file


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_paths(source: Path, output: Path) -> tuple[Path, Path]:
    source = Path(source)
    output = Path(output)
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"source checkpoint directory not found: {source}")
    source_abs = source.resolve()
    output_exists = output.exists() or output.is_symlink()
    output_abs = output.resolve(strict=False)
    if output_exists:
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    if source_abs == output_abs or _is_within(output_abs, source_abs) or _is_within(source_abs, output_abs):
        raise ValueError("source and output must be unrelated paths (no ancestor/descendant conversion)")
    return source_abs, output_abs


def _forbidden_feature_text(value: object) -> bool:
    text = str(value).lower()
    return any(token in text for token in ("quant", "gptq", "remap", "lm_head", "embed_tokens", "embedding"))


def _validate_source_config(config: Mapping[str, Any]) -> None:
    # Reject an already transformed source before looking at its tensor files.
    for key, value in config.items():
        key_text = str(key).lower()
        if (key_text in {"head", "head_dtype", "embedding", "embed_tokens"}
                or any(token in key_text for token in ("quant", "remap", "lm_head", "embed_tokens", "embedding"))):
            raise ValueError("source config contains quantization/remap/head/embed metadata")
        if key_text == "tie_word_embeddings" and value is True:
            raise ValueError("source must not contain tied head/embedding parameters")

    required = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "vocab_size": 248320,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"source config {key!r} must be {expected!r}")
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("source config architectures must be ['DFlash2DraftModel']")

    dtype_values = [config[key] for key in ("torch_dtype", "dtype") if key in config]
    if not dtype_values or any(value != "bfloat16" for value in dtype_values):
        raise ValueError("source config must declare BF16 dtype")

    dflash = config.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("source config dflash_config is required")
    expected_dflash = {
        "block_size": 8,
        "selector_rank": 256,
        "selector_top_k": 16,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "target_layer_ids": [5, 19, 33, 47, 61],
    }
    for key, expected in expected_dflash.items():
        if dflash.get(key) != expected:
            raise ValueError(f"source dflash_config {key!r} must be {expected!r}")


def _forbidden_tensor_name(name: str) -> bool:
    lower = name.lower()
    return any(
        token in lower
        for token in (
            "qweight",
            "qzeros",
            "g_idx",
            ".scales",
            "quant",
            "remap",
            "lm_head",
            "head.",
            ".head",
            "embed_tokens",
            "embed.",
            ".embed",
            "embedding",
        )
    )


def _read_safetensor_header(path: Path) -> _ShardHeader:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError("missing safetensors header length")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length > file_size - 8 or header_length > 128 * 1024 * 1024:
                raise ValueError("invalid or oversized safetensors header")
            raw_header = handle.read(header_length)
    except OSError as exc:
        raise ValueError(f"cannot read safetensors shard: {path}") from exc
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")

    metadata = header.get("__metadata__")
    if metadata is not None:
        if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
        ):
            raise ValueError(f"invalid safetensors metadata: {path}")
        if any(_forbidden_feature_text(key) or _forbidden_feature_text(value) for key, value in metadata.items()):
            raise ValueError("source safetensors metadata contains quantization/remap/head/embed metadata")

    payload_size = file_size - 8 - header_length
    tensors: dict[str, _TensorHeader] = {}
    ranges: list[tuple[int, int]] = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError(f"invalid tensor descriptor in safetensors shard: {path}")
        dtype = descriptor.get("dtype")
        shape_value = descriptor.get("shape")
        offsets_value = descriptor.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape_value, list) or not isinstance(offsets_value, list):
            raise ValueError(f"malformed tensor descriptor {name!r} in {path}")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape_value):
            raise ValueError(f"malformed tensor shape for {name!r}")
        if len(offsets_value) != 2 or any(
            isinstance(offset, bool) or not isinstance(offset, int) for offset in offsets_value
        ):
            raise ValueError(f"malformed tensor offsets for {name!r}")
        start, end = offsets_value
        if start < 0 or end < start or end > payload_size:
            raise ValueError(f"tensor offsets outside shard payload for {name!r}")
        if dtype != "BF16":
            # The source contract is BF16-only.  Reject here before loading data.
            raise ValueError(f"source tensor {name!r} must have BF16 dtype, got {dtype!r}")
        element_count = math.prod(shape_value)
        if end - start != element_count * 2:
            raise ValueError(f"tensor byte length does not match shape for {name!r}")
        tensors[name] = _TensorHeader(dtype, tuple(shape_value), (start, end))
        ranges.append((start, end))

    previous_end = 0
    for start, end in sorted(ranges):
        if start < previous_end:
            raise ValueError(f"overlapping tensor ranges in safetensors shard: {path}")
        previous_end = end
    return _ShardHeader(path, tensors, metadata)


def _safe_source_file(source: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"index shard path escapes source directory: {relative_name}")
    candidate = (source / relative).resolve(strict=False)
    if not _is_within(candidate, source) or not candidate.is_file():
        raise FileNotFoundError(f"missing or external safetensors shard: {relative_name}")
    return candidate


def _discover_layout(source: Path) -> _CheckpointLayout:
    index_paths = sorted(source.glob("*.safetensors.index.json"))
    if len(index_paths) > 1:
        raise ValueError("source contains multiple safetensors index files")
    if index_paths:
        index_path = index_paths[0]
        index_data = _read_json(index_path)
        if any(_forbidden_feature_text(key) for key in index_data if key != "weight_map"):
            raise ValueError("source index contains forbidden quantization/remap/head/embed metadata")
        raw_weight_map = index_data.get("weight_map")
        if not isinstance(raw_weight_map, dict) or not raw_weight_map:
            raise ValueError("safetensors index must contain a nonempty weight_map")
        if any(not isinstance(name, str) or not isinstance(shard, str) for name, shard in raw_weight_map.items()):
            raise ValueError("safetensors index weight_map must map strings to strings")
        shard_names = sorted(set(raw_weight_map.values()))
        shards = tuple(_safe_source_file(source, name) for name in shard_names)
        referenced = {path.resolve() for path in shards}
        all_safetensors = {path.resolve() for path in source.rglob("*.safetensors")}
        if all_safetensors != referenced:
            raise ValueError("safetensors index does not account for every source shard")
        return _CheckpointLayout(source, shards, index_path, index_data, dict(raw_weight_map))
    shards = tuple(sorted(source.glob("*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards found in source: {source}")
    if len(shards) > 1:
        raise ValueError("multiple safetensors shards require a safetensors index")
    return _CheckpointLayout(source, shards, None, None, {})


def _validate_source_headers(layout: _CheckpointLayout) -> dict[str, _TensorHeader]:
    headers = [_read_safetensor_header(path) for path in layout.shards]
    by_name: dict[str, _TensorHeader] = {}
    for header in headers:
        for name, descriptor in header.tensors.items():
            if name in by_name:
                raise ValueError(f"duplicate source tensor name: {name}")
            by_name[name] = descriptor

    names = set(by_name)
    expected_names = set(EXPECTED_SHAPES)
    unexpected = names - expected_names
    forbidden = sorted(name for name in unexpected if _forbidden_tensor_name(name))
    if forbidden:
        raise ValueError(f"source contains forbidden quantization/remap/head/embed tensors: {forbidden}")
    if len(names) != EXPECTED_TENSOR_COUNT or names != expected_names:
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise ValueError(
            f"source tensor contract requires {EXPECTED_TENSOR_COUNT} exact tensors; missing={missing}, extra={extra}"
        )
    parameter_count = sum(math.prod(by_name[name].shape) for name in names)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ValueError(
            f"source tensor contract requires {EXPECTED_PARAMETER_COUNT} parameters, got {parameter_count}"
        )
    for name, expected_shape in EXPECTED_SHAPES.items():
        descriptor = by_name[name]
        if descriptor.dtype != "BF16" or descriptor.shape != tuple(expected_shape):
            raise ValueError(
                f"source tensor {name!r} must be BF16 with shape {tuple(expected_shape)}, "
                f"got dtype={descriptor.dtype!r} shape={descriptor.shape!r}"
            )

    if layout.index_data is not None:
        index_names = set(layout.weight_map)
        if index_names != names:
            raise ValueError("safetensors index weight_map does not exactly match source tensors")
        for header in headers:
            relative_shard = header.path.relative_to(layout.source).as_posix()
            for name in header.tensors:
                if layout.weight_map[name] != relative_shard:
                    raise ValueError(f"safetensors index points {name!r} at the wrong shard")
    return by_name


def _expanded_weight_map(layout: _CheckpointLayout) -> dict[str, str]:
    if layout.index_data is None:
        source_map = {name: layout.shards[0].name for name in EXPECTED_SHAPES}
    else:
        source_map = layout.weight_map
    expanded: dict[str, str] = {}
    for name, shard in source_map.items():
        if _is_quantized_name(name):
            for component in ("qweight", "scales", "qzeros", "g_idx"):
                expanded[_output_name(name, component)] = shard
        else:
            expanded[name] = shard
    return expanded


def _validate_output_tensors(tensors: Mapping[str, "torch.Tensor"]) -> None:
    t = _require_torch()
    for name, tensor in tensors.items():
        if name.endswith(".scales"):
            if tensor.dtype != t.bfloat16:
                raise ValueError(f"output scale tensor {name} is not BF16")
            _finite(tensor, "output scales")
            if bool((tensor <= 0).any().item()):
                raise ValueError(f"output scale tensor {name} is not positive")
        elif any(name.endswith(f".{component}") for component in ("qweight", "qzeros", "g_idx")):
            if tensor.dtype != t.int32:
                raise ValueError(f"output GPTQ tensor {name} is not int32")


def _process_shard(source_path: Path, output_path: Path) -> tuple[int, int]:
    safe_open, save_file = _load_safetensors_api()
    t = _require_torch()
    source_tensors: dict[str, "torch.Tensor"] = {}
    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if tensor.dtype != t.bfloat16:
                raise ValueError(f"source tensor {name!r} is not BF16")
            _finite(tensor, f"source tensor {name}")
            source_tensors[name] = tensor
        metadata = handle.metadata()

    output_tensors = _transform_tensor_map(source_tensors)
    del source_tensors
    _validate_output_tensors(output_tensors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if metadata is None else {"metadata": metadata}
    save_file(output_tensors, str(output_path), **kwargs)
    tensor_bytes = sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in output_tensors.values())
    return len(output_tensors), tensor_bytes


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_config(source_config: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(source_config))
    config["quantization_config"] = copy.deepcopy(QUANTIZATION_CONFIG)
    return config


def quantize_checkpoint(source: Path, output: Path) -> dict[str, Any]:
    """Convert a strict native Qwen3.8 DFlash2 BF16 checkpoint on the CPU.

    Only the twenty drafter projection matrices are replaced.  The output is
    staged in a newly-created sibling directory and atomically published only
    after every shard and config has succeeded.
    """

    source_abs, output_abs = _validate_paths(Path(source), Path(output))
    config_path = source_abs / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"source config.json not found: {config_path}")
    source_config = _read_json(config_path)
    _validate_source_config(source_config)
    for forbidden_file in ("quantize_config.json", "quantization_config.json", "vocab_remap.json"):
        if (source_abs / forbidden_file).is_file():
            raise ValueError(f"source contains forbidden quantization/remap artifact: {forbidden_file}")
    layout = _discover_layout(source_abs)
    _validate_source_headers(layout)

    output_abs.parent.mkdir(parents=True, exist_ok=True)
    stage_prefix = f".{output_abs.name}.rtn-"
    stage = Path(tempfile.mkdtemp(prefix=stage_prefix, dir=str(output_abs.parent)))
    try:
        total_output_tensors = 0
        total_tensor_bytes = 0
        for source_shard in layout.shards:
            relative = source_shard.relative_to(source_abs)
            staged_shard = stage / relative
            count, tensor_bytes = _process_shard(source_shard, staged_shard)
            total_output_tensors += count
            total_tensor_bytes += tensor_bytes

        _write_json(stage / "config.json", _output_config(source_config))
        if layout.index_path is not None and layout.index_data is not None:
            output_index = copy.deepcopy(layout.index_data)
            output_index["weight_map"] = _expanded_weight_map(layout)
            metadata = output_index.get("metadata")
            if isinstance(metadata, dict) and "total_size" in metadata:
                metadata["total_size"] = total_tensor_bytes
            _write_json(stage / layout.index_path.relative_to(source_abs), output_index)

        if output_abs.exists() or output_abs.is_symlink():
            raise FileExistsError(f"refusing to overwrite output path created during conversion: {output_abs}")
        stage.rename(output_abs)
        stage = None  # type: ignore[assignment]
    except BaseException:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        raise

    retained = EXPECTED_TENSOR_COUNT - len(QUANTIZED_TENSOR_NAMES)
    return {
        "method": "RTN",
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "quantized_matrices": len(QUANTIZED_TENSOR_NAMES),
        "retained_native_tensors": retained,
        "output_tensors": total_output_tensors,
        "tensor_bytes": total_tensor_bytes,
        "output": str(output_abs),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU calibration-free RTN GPTQ conversion for Qwen3.8 DFlash2")
    parser.add_argument("--source", type=Path, required=True, help="immutable native BF16 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="new GPTQ-v1 drafter checkpoint directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = quantize_checkpoint(args.source, args.output)
    print("method=RTN (calibration-free round-to-nearest)")
    print(f"quantized_matrices={summary['quantized_matrices']}")
    print(f"retained_native_tensors={summary['retained_native_tensors']}")
    print(f"output_tensors={summary['output_tensors']}")
    print(f"tensor_bytes={summary['tensor_bytes']}")
    print(f"output={summary['output']}")


if __name__ == "__main__":
    main()
