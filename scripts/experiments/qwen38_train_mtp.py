#!/usr/bin/env python3
"""Bounded offline tuning of the stock, single-layer Qwen3.5/3.8 native MTP.

Run in the pinned inference-host environment, never in the interactive Pi runtime.
Only mtp.*, the embedding, and lm_head are read from the local safetensors index;
no verifier, model download, new architecture, or serving configuration is created.

Records contain prompt_id, input_ids int64[T], loss_mask bool[T], and observed
positions int64[H]/target_last_hidden_states floating[H,hidden_size], T-2 <= H <= T.
Short native captures must be a contiguous observed prefix; no hidden is fabricated.
Legacy full-length captures retain their actual increasing positions. Overlength or
corrupt records fail; train/dev prompt IDs must be disjoint. --eval-dir is DEVELOPMENT
data only, never a fresh test set. Default training remains single-depth.

Pinned native alignment and recursion:
https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py
https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/v1/spec_decode/step3p5.py
https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/v1/spec_decode/llm_base_proposer.py
https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/cache_utils.py
Base row j consumes x[j+1], observed h[j], p[j], predicting x[j+2]. Opt-in depth 4
teacher-forces x[t+d] into its OWN previous draft hidden at p[t]+d-1, predicting
x[t+d+1]. Each root attends base KV <= t plus only its own appended branch nodes.
HF's gated decoder, norms, RoPE and differentiable DynamicCache stay unmodified.
Depth 1 uses every valid label; deeper roots require all four valid label masks.
The objective is sum(depth_weight * depth_mean_CE), with per-update denominators.

Precision regimes: FP32 trainable masters/AdamW and BF16 export. Native
unquantized BF16 checkpoints use frozen original BF16 embedding/LM-head weights and
CUDA BF16 autocast; GPTQ checkpoints retain FP16 XPU autocast/loss scaling and the
deployed RTN INT4-g128 effective LM head. CPU is intended only for tiny synthetic
tests. Native BF16 dev metrics use the BF16 export stage; GPTQ retains the existing
BF16 export and RTN effective-core stages. Dense dequantized GEMM is not bitwise
INT4-kernel equivalence. Core RTN is not used during training (no QAT). Step 0,
intermediate and final BF16 overlays are reloaded for dev metrics. Stock remains
eligible; the reported dev choice is NOT a serving promotion or an acceptance
guarantee.

Examples (new output paths, verifier unloaded before training):
  python qwen38_train_mtp.py --model MODEL --export-stock --output stock.safetensors
  python qwen38_train_mtp.py --model MODEL --train-dir TRAIN --eval-dir HELDOUT \
      --output tuned.safetensors --steps 10 --lr 1e-5 --device xpu
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace


class TrainingError(ValueError):
    """Unsupported model, invalid capture, or unsafe experiment configuration."""


@lru_cache(maxsize=1)
def runtime():
    try:
        import torch
        import transformers
        from safetensors import safe_open
        from safetensors.torch import save_file
        from transformers.cache_utils import DynamicCache
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
        )
    except ImportError as exc:
        raise TrainingError(
            "Requires the pinned inference-host torch/transformers/safetensors runtime "
            "with native Qwen3_5DecoderLayer; no dependencies will be installed."
        ) from exc
    return SimpleNamespace(
        torch=torch, transformers=transformers, safe_open=safe_open, save_file=save_file,
        Config=Qwen3_5TextConfig, Decoder=Qwen3_5DecoderLayer,
        Norm=Qwen3_5RMSNorm, Rotary=Qwen3_5TextRotaryEmbedding, Cache=DynamicCache,
    )


def precision_regime(raw):
    quant = raw.get("quantization_config")
    if quant in (None, {}):
        return "native_bf16"
    if not isinstance(quant, dict) or (
            quant.get("quant_method") != "gptq"
            or quant.get("bits") != 4
            or quant.get("group_size") != 128
            or quant.get("sym") is not True):
        raise TrainingError("Only dense, gated-Q, one-layer native Qwen3.5 GPTQ INT4-g128 or unquantized BF16 is supported")
    return "gptq_int4_g128"


def native_config(raw):
    text = raw.get("text_config", {})
    precision_regime(raw)
    if (
        raw.get("model_type") != "qwen3_5"
        or text.get("model_type") != "qwen3_5_text"
        or text.get("mtp_num_hidden_layers") != 1
        or text.get("mtp_use_dedicated_embeddings", False)
        or raw.get("tie_word_embeddings", False)
        or text.get("tie_word_embeddings", False)
        or text.get("attention_bias", False)
        or text.get("qkv_bias", False)
        or not text.get("attn_output_gate", True)
        or text.get("hidden_act") != "silu"
        or text.get("layer_scale", False)
        or not text.get("is_causal", True)
        or text.get("attention_dropout", 0) != 0
    ):
        raise TrainingError("Only dense, gated-Q, one-layer native Qwen3.5 GPTQ INT4-g128 or unquantized BF16 is supported")
    for key in ("hidden_size", "intermediate_size", "head_dim", "num_attention_heads",
                "num_key_value_heads", "vocab_size", "max_position_embeddings"):
        if type(text.get(key)) is not int or text[key] <= 0:
            raise TrainingError(f"Invalid model text_config.{key}")
    rope = text.get("rope_parameters", {})
    rotary_dim = int(text["head_dim"] * rope.get("partial_rotary_factor", 1.0))
    if (text["hidden_size"] % 128 or text["num_attention_heads"] % text["num_key_value_heads"]
            or rope.get("rope_type") != "default" or rope.get("rope_theta", 0) <= 0
            or rotary_dim <= 0 or rotary_dim > text["head_dim"] or rotary_dim % 2):
        raise TrainingError("Unsupported head dimensions or RoPE configuration")
    # Construct only the MTP decoder. Never construct the target's 64-layer model.
    text = dict(text, num_hidden_layers=1, layer_types=["full_attention"], use_cache=False)
    config = runtime().Config.from_dict(text)
    config._attn_implementation = "sdpa"
    return config

def expected_mtp_shapes(config):
    h, d = config.hidden_size, config.head_dim
    q, k = config.num_attention_heads * d, config.num_key_value_heads * d
    i = config.intermediate_size
    shapes = {
        "fc.weight": (h, 2 * h), "norm.weight": (h,),
        "pre_fc_norm_embedding.weight": (h,), "pre_fc_norm_hidden.weight": (h,),
        "layers.0.input_layernorm.weight": (h,),
        "layers.0.post_attention_layernorm.weight": (h,),
        "layers.0.self_attn.q_proj.weight": (2 * q, h),
        "layers.0.self_attn.k_proj.weight": (k, h),
        "layers.0.self_attn.v_proj.weight": (k, h),
        "layers.0.self_attn.o_proj.weight": (h, q),
        "layers.0.self_attn.q_norm.weight": (d,),
        "layers.0.self_attn.k_norm.weight": (d,),
        "layers.0.mlp.gate_proj.weight": (i, h),
        "layers.0.mlp.up_proj.weight": (i, h),
        "layers.0.mlp.down_proj.weight": (h, i),
    }
    return {"mtp." + key: shape for key, shape in shapes.items()}


def validate_mtp_state(state, shapes, *, stock=False):
    torch = runtime().torch
    if set(state) != set(shapes):
        raise TrainingError(f"MTP keys mismatch: missing={sorted(set(shapes) - set(state))}, "
                            f"extra={sorted(set(state) - set(shapes))}")
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shapes[key]:
            raise TrainingError(f"MTP shape mismatch: {key}, expected {shapes[key]}")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise TrainingError(f"Nonfinite or nonfloating MTP tensor: {key}")
        if stock and tensor.dtype != torch.bfloat16:
            raise TrainingError(f"Stock identity export requires original BF16 MTP tensors: {key}")


class Checkpoint:
    EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
    LM_HEAD_KEY = "lm_head.weight"

    def __init__(self, model):
        self.path = Path(model).resolve()
        try:
            self.raw = json.loads((self.path / "config.json").read_text())
            self.weight_map = json.loads(
                (self.path / "model.safetensors.index.json").read_text()
            )["weight_map"]
        except (OSError, ValueError, KeyError) as exc:
            raise TrainingError(f"Local checkpoint config/shard index unavailable: {self.path}") from exc
        self.regime = precision_regime(self.raw)
        self.precision_regime = self.regime
        self.config = native_config(self.raw)
        self.shapes = expected_mtp_shapes(self.config)
        if {k for k in self.weight_map if k.startswith("mtp.")} != set(self.shapes):
            raise TrainingError("Checkpoint must contain exactly the native 15 mtp.* keys")
        if not {self.EMBEDDING_KEY, self.LM_HEAD_KEY} <= self.weight_map.keys():
            raise TrainingError("Checkpoint is missing the shared embedding or separate lm_head")

    def tensor(self, key):
        if key not in self.shapes and key not in (self.EMBEDDING_KEY, self.LM_HEAD_KEY):
            raise TrainingError(f"Refusing to load verifier tensor: {key}")
        try:
            shard = (self.path / self.weight_map[key]).resolve()
            if not shard.is_relative_to(self.path):
                raise TrainingError(f"Shard outside model directory: {key}")
            with runtime().safe_open(shard, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        except (OSError, KeyError) as exc:
            raise TrainingError(f"Cannot read checkpoint tensor: {key}") from exc

    def mtp_state(self):
        state = {key: self.tensor(key) for key in self.shapes}
        validate_mtp_state(state, self.shapes, stock=True)
        return state


def build_native_mtp(config, state, device="cpu"):
    rt = runtime()
    torch, nn = rt.torch, rt.torch.nn
    validate_mtp_state(state, expected_mtp_shapes(config))

    class NativeMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = rt.Norm(config.hidden_size, eps=config.rms_norm_eps)
            self.pre_fc_norm_hidden = rt.Norm(config.hidden_size, eps=config.rms_norm_eps)
            self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
            self.layers = nn.ModuleList([rt.Decoder(config, layer_idx=0)])
            self.norm = rt.Norm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(self, input_ids, target_hidden, positions, embedding, *, past_key_values=None):
            embedded = self.pre_fc_norm_embedding(embedding(input_ids))
            hidden = self.pre_fc_norm_hidden(target_hidden)
            hidden = self.fc(torch.cat([embedded, hidden], dim=-1)).unsqueeze(0)
            position_ids = positions.unsqueeze(0)
            rotary = self.rotary_emb(hidden, position_ids)
            length = input_ids.numel()
            prefix = past_key_values.get_seq_length() if past_key_values is not None else 0
            # Prefill is causal; a single decode row attends every supplied KV.
            # The caller must structurally exclude future base rows and siblings.
            mask = None
            if length > 1:
                mask = torch.full((length, prefix + length), float("-inf"), dtype=hidden.dtype,
                                  device=hidden.device).triu(prefix + 1)[None, None]
            hidden = self.layers[0](hidden, position_embeddings=rotary,
                                    attention_mask=mask, position_ids=position_ids,
                                    past_key_values=past_key_values)
            return self.norm(hidden).squeeze(0)

    with torch.device("meta"):
        model = NativeMTP()
    model.load_state_dict({key.removeprefix("mtp."): value.to(device=device, dtype=torch.float32)
                           for key, value in state.items()}, strict=True, assign=True)
    # RoPE has only nonpersistent buffers; initialize outside the meta context.
    model.rotary_emb = rt.Rotary(config).to(device)
    return model


def rtn_effective_lm_head(weight, *, device="cpu", row_chunk=4096):
    """Dequantize patch_draft_lmhead_int4.py's exact RTN codes/FP16 scales.

    The serving loader first casts checkpoint weights to FP16, then calculates
    maxabs/7 and codes in FP32, and ONLY THEN stores scales as FP16. All-zero
    groups dequantize to zero regardless of the runtime's undefined 0/0 code.
    """
    torch = runtime().torch
    if weight.ndim != 2 or weight.shape[1] % 128 or row_chunk <= 0:
        raise TrainingError("LM head must be [vocab, hidden] with hidden divisible by 128")
    effective = torch.empty(weight.shape, dtype=torch.float16, device=device)
    with torch.no_grad():
        for start in range(0, weight.shape[0], row_chunk):
            rows = weight[start:start + row_chunk].to(device=device, dtype=torch.float16).float()
            if not torch.isfinite(rows).all().item():
                raise TrainingError("Nonfinite FP16 LM-head weight")
            grouped = rows.reshape(rows.shape[0], -1, 128)
            scales = grouped.abs().amax(dim=-1, keepdim=True) / 7.0
            codes = (grouped / torch.where(scales == 0, 1.0, scales)).round().clamp(-8, 7)
            effective[start:start + row_chunk] = (codes * scales.half().float()).reshape_as(rows).half()
    return effective


def frozen_heads(checkpoint, device):
    torch = runtime().torch
    shape = (checkpoint.config.vocab_size, checkpoint.config.hidden_size)
    native_bf16 = getattr(checkpoint, "regime",
                           getattr(checkpoint, "precision_regime", "gptq_int4_g128")) == "native_bf16"
    weight = checkpoint.tensor(checkpoint.EMBEDDING_KEY)
    if (tuple(weight.shape) != shape or not torch.is_floating_point(weight)
            or not torch.isfinite(weight).all().item()):
        raise TrainingError("Invalid shared embedding")
    if native_bf16 and weight.dtype != torch.bfloat16:
        raise TrainingError("Native BF16 checkpoint requires a BF16 shared embedding")
    device_type = torch.device(device).type
    if native_bf16:
        dtype = torch.bfloat16
        embedding_weight = weight.to(device=device, dtype=dtype)
    else:
        dtype = torch.float16 if device_type in ("xpu", "cuda") else torch.float32
        # CPU tests reproduce the serving FP16 rounding before FP32 computation.
        embedding_weight = weight.half().to(device=device, dtype=dtype)
    embedding = torch.nn.Embedding.from_pretrained(embedding_weight, freeze=True)
    weight = checkpoint.tensor(checkpoint.LM_HEAD_KEY)
    if (tuple(weight.shape) != shape or not torch.is_floating_point(weight)
            or not torch.isfinite(weight).all().item()):
        raise TrainingError("Invalid separate LM-head shape")
    if native_bf16:
        if weight.dtype != torch.bfloat16:
            raise TrainingError("Native BF16 checkpoint requires an original BF16 LM head")
        effective = weight.to(device=device, dtype=torch.bfloat16)
    else:
        effective = rtn_effective_lm_head(weight, device=device).to(dtype=dtype)
    with torch.device("meta"):
        head = torch.nn.Linear(shape[1], shape[0], bias=False)
    head.weight = torch.nn.Parameter(effective, requires_grad=False)
    return embedding, head


def validate_record(record, config, max_length):
    torch = runtime().torch
    if not isinstance(record, dict) or not isinstance(record.get("prompt_id"), str) or not record["prompt_id"].strip():
        raise TrainingError("Each complete-sequence record requires a nonempty prompt_id string")
    fields = ("input_ids", "positions", "target_last_hidden_states", "loss_mask")
    if any(not isinstance(record.get(key), torch.Tensor) for key in fields):
        raise TrainingError("Missing complete-sequence tensor fields")
    ids, positions, hidden, mask = (record[key] for key in fields)
    if ids.ndim != 1 or ids.dtype != torch.int64 or not 3 <= ids.numel() <= max_length:
        raise TrainingError(f"input_ids must be int64[T], 3 <= T <= {max_length}; no truncation")
    length = ids.numel()
    if (hidden.ndim != 2 or hidden.shape[1] != config.hidden_size
            or not length - 2 <= hidden.shape[0] <= length
            or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise TrainingError("target_last_hidden_states must be runtime float[H, hidden_size], T-2 <= H <= T")
    observed = hidden.shape[0]
    if positions.shape != (observed,) or positions.dtype != torch.int64:
        raise TrainingError("positions must be int64[H], matching observed hidden rows")
    if positions[0].item() < 0 or not (positions[1:] > positions[:-1]).all().item():
        raise TrainingError("positions must be nonnegative and strictly increasing")
    if observed < length and not torch.equal(positions, torch.arange(observed, device=positions.device)):
        raise TrainingError("Short native captures require a contiguous observed prefix starting at zero")
    if positions[-1].item() >= config.max_position_embeddings:
        raise TrainingError("positions exceed model context")
    if ids.min().item() < 0 or ids.max().item() >= config.vocab_size:
        raise TrainingError("input_ids outside model vocabulary")
    if not torch.isfinite(hidden).all().item():
        raise TrainingError("Nonfinite target_last_hidden_states")
    if mask.shape != (length,) or mask.dtype != torch.bool or not mask[2:].any().item():
        raise TrainingError("loss_mask must be bool[T] with at least one supervised x[t+2] label")
    return record


def load_record(path, config, max_length):
    try:
        record = runtime().torch.load(path, map_location="cpu", weights_only=True)
        return validate_record(record, config, max_length)
    except Exception as exc:
        raise TrainingError(f"Invalid capture {path}: {exc}") from exc


def aligned_inputs(record, device="cpu"):
    """Match pinned vLLM set_inputs_first_pass, NOT a rebased/shifted RoPE grid."""
    return (
        record["input_ids"][1:-1].to(device),
        record["target_last_hidden_states"][:record["input_ids"].numel() - 2].to(device),
        record["positions"][:record["input_ids"].numel() - 2].to(device),
        record["input_ids"][2:].to(device),
        record["loss_mask"][2:].to(device),
    )


def capture_split(directory, config, max_length):
    directory = Path(directory)
    paths = sorted(directory.glob("*.pt"))
    if not directory.is_dir() or not paths:
        raise TrainingError(f"No complete-sequence .pt captures in {directory}")
    ids = set()
    for path in paths:
        record = load_record(path, config, max_length)
        ids.add(record["prompt_id"])
    return paths, ids


def capture_sets(train_dir, eval_dir, config, max_length):
    train, train_ids = capture_split(train_dir, config, max_length)
    evaluation, eval_ids = capture_split(eval_dir, config, max_length) if eval_dir else ([], set())
    if train_ids & eval_ids:
        raise TrainingError(f"Train/heldout prompt_id overlap: {sorted(train_ids & eval_ids)}")
    return train, evaluation


def autocast(device, *, dtype=None):
    torch = runtime().torch
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device_type == "xpu":
        compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else torch.float16
        return torch.autocast("xpu", dtype=compute_dtype)
    if device_type == "cpu" and dtype == torch.bfloat16:
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def kl_divergence(student_logits, teacher_logits, temperature=1.0):
    """Return summed KL(teacher || student) in FP32 with T^2 scaling."""
    torch = runtime().torch
    if not math.isfinite(temperature) or temperature <= 0:
        raise TrainingError("KL temperature must be positive and finite")
    if (student_logits.ndim != 2 or teacher_logits.shape != student_logits.shape):
        raise TrainingError("KL logits must be matching rank-2 tensors")
    student_logits = student_logits.float()
    with torch.no_grad():
        teacher_logits = teacher_logits.float()
        if not torch.isfinite(teacher_logits).all().item():
            raise TrainingError("Nonfinite KL teacher logits")
        teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    if not torch.isfinite(student_logits).all().item():
        raise TrainingError("Nonfinite KL student logits")
    student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    result = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="sum")
    result = result * (temperature * temperature)
    if not torch.isfinite(result).item():
        raise TrainingError("Nonfinite KL divergence")
    return result


def chunked_ce(hidden, head, labels, mask, *, chunk_tokens=128, backward=False,
                normalizer=None, scaler=None, weight=1.0, gradients=None, stats=None,
                kl_teacher=None, kl_mask=None, kl_temperature=1.0, kl_weight=0.0,
                kl_normalizer=None, kl_hidden=None):
    """Token-chunked CE plus an optional token-chunked teacher KL.

    Each chunk frees its logits graph before the next chunk. A detached leaf
    accumulates output gradients, then backpropagates through the full-context
    native core once. Accumulating graph-connected chunk losses would retain the
    entire T*vocab logits tensor and defeat the memory bound.
    """
    torch = runtime().torch
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise TrainingError("KL weight must be finite and nonnegative")
    kl_enabled = kl_weight > 0
    indices = mask.nonzero(as_tuple=True)[0]
    count = indices.numel()
    denominator = count if normalizer is None else normalizer
    if count == 0 or chunk_tokens <= 0 or denominator <= 0:
        raise TrainingError("CE requires supervised labels and positive chunk/normalizer")
    kl_indices = None
    if kl_enabled:
        if kl_teacher is None or kl_mask is None:
            raise TrainingError("KL requires teacher states and a pair mask")
        if (kl_teacher.shape != hidden.shape or kl_mask.shape != mask.shape
                or kl_mask.dtype != torch.bool):
            raise TrainingError("KL teacher states and pair mask must match student rows")
        if not math.isfinite(kl_temperature) or kl_temperature <= 0:
            raise TrainingError("KL temperature must be positive and finite")
        kl_indices = kl_mask.nonzero(as_tuple=True)[0]
        if kl_indices.numel():
            kl_normalizer = (kl_indices.numel() if kl_normalizer is None else kl_normalizer)
            if kl_normalizer <= 0:
                raise TrainingError("KL requires a positive valid-pair normalizer")
        if stats is not None:
            stats["kl_loss_sum"] = stats.get("kl_loss_sum", 0.0)
            stats["kl_pairs"] = stats.get("kl_pairs", 0) + int(kl_indices.numel())
    features = hidden.detach().requires_grad_(True) if backward else hidden
    if kl_hidden is not None and (not kl_enabled or kl_hidden.shape != hidden.shape):
        raise TrainingError("Separate KL student states require KL and matching CE row shapes")
    kl_features = features if kl_hidden is None else (
        kl_hidden.detach().requires_grad_(True) if backward else kl_hidden)
    total = 0.0
    for offset in range(0, count, chunk_tokens):
        rows = indices[offset:offset + chunk_tokens]
        with autocast(hidden.device, dtype=head.weight.dtype):
            logits = head(features.index_select(0, rows))
            loss = torch.nn.functional.cross_entropy(logits.float(), labels[rows], reduction="sum")
        if not torch.isfinite(loss).item():
            raise TrainingError("Nonfinite cross entropy")
        total += loss.detach().item()
        if stats is not None:
            stats["correct"] += int((logits.argmax(-1) == labels[rows]).sum().item())
        if backward:
            scaled = loss * (weight / denominator)
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
        del logits, loss
    if kl_enabled and kl_indices.numel():
        for offset in range(0, kl_indices.numel(), chunk_tokens):
            rows = kl_indices[offset:offset + chunk_tokens]
            with autocast(hidden.device, dtype=head.weight.dtype):
                student_logits = head(kl_features.index_select(0, rows))
                with torch.no_grad():
                    teacher_logits = head(kl_teacher.index_select(0, rows).to(dtype=head.weight.dtype))
            loss = kl_divergence(student_logits, teacher_logits, temperature=kl_temperature)
            if stats is not None:
                stats["kl_loss_sum"] += loss.detach().item()
            if backward:
                scaled = loss * (weight * kl_weight / kl_normalizer)
                (scaler.scale(scaled) if scaler is not None else scaled).backward()
            del student_logits, teacher_logits, loss
    if backward:
        pending = [(hidden, features.grad)]
        if kl_hidden is not None and kl_features.grad is not None:
            pending.append((kl_hidden, kl_features.grad))
        if gradients is None:
            torch.autograd.backward(*zip(*pending))
        else:
            # Joint backward after ALL depths: both graphs share student masters.
            gradients.extend(pending)
    return total, count

def sequence_hidden(model, embedding, record, device):
    ids, hidden, positions, labels, mask = aligned_inputs(record, device)
    hidden = hidden.to(dtype=embedding.weight.dtype)
    with autocast(device, dtype=embedding.weight.dtype):
        result = model(ids, hidden, positions, embedding)
    return result, labels, mask


def sample_roots(record, depth, count, rng):
    """One seeded sample without replacement, valid at EVERY supervised depth."""
    if depth == 1:
        return []
    mask = record["loss_mask"]
    eligible = [t for t in range(record["input_ids"].numel() - depth - 1)
                if mask[t + 2:t + depth + 2].all().item()]
    return sorted(rng.sample(eligible, min(count, len(eligible))))


def prefix_cache(base_cache, length):
    """A fresh cache per root, with differentiable, already-RoPE'd prefix slices."""
    if not 1 <= length <= base_cache.get_seq_length():
        raise TrainingError("Branch prefix outside observed base KV")
    layer = base_cache.layers[0]
    cache = runtime().Cache()
    cache.update(layer.keys[..., :length, :], layer.values[..., :length, :], 0)
    return cache


def sequence_depths(model, embedding, record, device, depth=1, roots=()):
    if depth == 1:
        return [sequence_hidden(model, embedding, record, device)]
    if depth != 4:
        raise TrainingError("Only single-depth or native recursive depth 4 is supported")
    torch = runtime().torch
    ids, hidden, positions, labels, mask = aligned_inputs(record, device)
    if (len(set(roots)) != len(roots) or any(
            type(t) is not int or not 0 <= t < ids.numel() - depth + 1
            or not mask[t:t + depth].all().item() for t in roots)):
        raise TrainingError("Recursive roots require distinct in-bounds, fully supervised label chains")
    cache = runtime().Cache()
    with autocast(device, dtype=embedding.weight.dtype):
        base = model(ids, hidden.to(embedding.weight.dtype), positions, embedding,
                     past_key_values=cache)
        outputs = [[] for _ in range(depth - 1)]
        tokens = record["input_ids"].to(device)
        for root in roots:
            branch_cache = prefix_cache(cache, root + 1)
            previous = base[root:root + 1]
            for d in range(2, depth + 1):
                previous = model(tokens[root + d:root + d + 1], previous,
                                 positions[root:root + 1] + d - 1, embedding,
                                 past_key_values=branch_cache)
                outputs[d - 2].append(previous)
    result = [(base, labels, mask)]
    for d, rows in enumerate(outputs, start=2):
        indices = torch.tensor([root + d + 1 for root in roots], device=device, dtype=torch.long)
        result.append((torch.cat(rows, dim=0) if rows else base[:0], tokens[indices],
                       torch.ones(len(roots), device=device, dtype=torch.bool)))
    return result

PINNED_BF16_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def validate_greedy_checkpoint(checkpoint, device):
    if checkpoint.regime != "native_bf16":
        raise TrainingError("Greedy KL requires an unquantized native BF16 checkpoint")
    if checkpoint.raw.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise TrainingError("Greedy KL requires the original Qwen3_5ForConditionalGeneration architecture")
    # Only tiny CPU fixtures may bypass the real-checkpoint revision guard.
    tiny_cpu = (runtime().torch.device(device).type == "cpu"
                and sum(math.prod(s) for s in checkpoint.shapes.values()) <= 5_000_000
                and checkpoint.raw["text_config"].get("num_hidden_layers", 0) <= 4)
    if not tiny_cpu:
        revision = checkpoint.raw.get("_commit_hash")
        snapshot_revision = (checkpoint.path.name if checkpoint.path.parent.name == "snapshots" else None)
        if ((revision or snapshot_revision) != PINNED_BF16_REVISION
                or (revision and snapshot_revision and revision != snapshot_revision)):
            raise TrainingError(f"Greedy KL requires exact BF16 model revision {PINNED_BF16_REVISION}")


def validate_greedy_positions(record):
    torch = runtime().torch
    positions = record["positions"]
    if not torch.equal(positions, torch.arange(positions.numel(), device=positions.device)):
        raise TrainingError("Greedy KL requires contiguous native positions starting at zero")


class FrozenTarget:
    """Original multimodal checkpoint; fresh causal text replay, never a hybrid-cache fork."""

    def __init__(self, checkpoint, device):
        validate_greedy_checkpoint(checkpoint, device)
        torch = runtime().torch
        # Import/load ONLY on the opted-in path. No CausalLM key remapping or downloads.
        from transformers import Qwen3_5ForConditionalGeneration
        for shard_name in set(checkpoint.weight_map.values()):
            shard = (checkpoint.path / shard_name).resolve()
            if not shard.is_relative_to(checkpoint.path):
                raise TrainingError("Teacher shard outside model directory")
            with runtime().safe_open(shard, framework="pt", device="cpu") as handle:
                if any(handle.get_slice(key).get_dtype() != "BF16" for key in handle.keys()):
                    raise TrainingError("Teacher requires original BF16 checkpoint tensors")
        self.model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint.path, local_files_only=True, use_safetensors=True,
            dtype=torch.bfloat16, device_map=str(device), attn_implementation="sdpa",
            output_loading_info=True,
        )
        # HF intentionally ignores mtp.*. Everything else must load exactly, including vision.
        expected = set(checkpoint.weight_map) - set(checkpoint.shapes)
        actual = set(self.model.state_dict())
        unexpected = set(info.get("unexpected_keys", [])) - set(checkpoint.shapes)
        if (expected != actual or unexpected or info.get("missing_keys")
                or info.get("mismatched_keys") or info.get("error_msgs")):
            raise TrainingError(f"Strict teacher loading failed: missing={sorted(actual - expected)}, "
                                f"extra={sorted(expected - actual)}, diagnostics={info}")
        self.model.requires_grad_(False).eval()
        self.text = self.model.model.language_model
        self.embedding, self.head = self.text.embed_tokens, self.model.lm_head
        self.device = device
        for module in (self.embedding, self.head):
            if (module.weight.dtype != torch.bfloat16
                    or not torch.isfinite(module.weight).all().item()):
                raise TrainingError("Teacher shared embedding/head must be finite frozen BF16")

    def replay(self, record, roots, proposals):
        torch = runtime().torch
        tokens = record["input_ids"].to(self.device)
        if proposals.shape != (len(roots), 4):
            raise TrainingError("Teacher replay requires one depth-four proposal per root")
        rows = []
        with torch.no_grad():
            for root, draft in zip(roots, proposals):
                # Row r+1 predicts y1; the last four rows predict y1..y4.
                prefix = torch.cat([tokens[:root + 2], draft[:3].detach()])
                result = self.text(
                    input_ids=prefix.unsqueeze(0),
                    position_ids=torch.arange(prefix.numel(), device=self.device).unsqueeze(0),
                    past_key_values=None, use_cache=False,
                ).last_hidden_state[0, -4:]
                if result.shape != (4, self.head.in_features) or not torch.isfinite(result).all().item():
                    raise TrainingError("Nonfinite or invalid full-prefix teacher replay rows")
                # Copy only the scored rows, not the entire full-prefix backing storage.
                rows.append(result.detach().clone())
        return (torch.stack(rows) if rows else
                self.head.weight.new_empty((0, 4, self.head.in_features)))


def native_greedy_teacher_rows(record, roots, proposals):
    """Same-request native hidden states. No second model forward.

    Row d is the captured state at position root+d+1. It is valid only when that
    state was produced by the drafted prefix: row 0 does not depend on the draft,
    and later rows require draft[:d] to equal the captured tokens.
    """
    torch = runtime().torch
    tokens = record["input_ids"].detach().cpu()
    target = record["target_last_hidden_states"].detach()
    if proposals.shape != (len(roots), 4):
        raise TrainingError("Native teacher rows require one depth-four proposal per root")
    if not target.ndim == 2 or target.shape[1] == 0:
        raise TrainingError("Native teacher rows require captured hidden states")
    rows, valid = [], []
    drafts = proposals.detach().cpu()
    for root, draft in zip(roots, drafts):
        positions = [root + depth + 1 for depth in range(4)]
        if any(position < 0 or position >= tokens.numel() for position in positions):
            raise TrainingError("Native teacher row is outside the captured token request")
        present = [0 <= position < target.shape[0] and bool(torch.isfinite(target[position]).all())
                   for position in positions]
        row = target.new_zeros((4, target.shape[1]))
        for depth, position in enumerate(positions):
            if present[depth]:
                row[depth] = target[position]
        matched = [present[0]]
        for depth in range(1, 4):
            expected = tokens[root + 2:root + 2 + depth]
            matched.append(present[depth] and bool(torch.equal(draft[:depth], expected)))
        rows.append(row)
        valid.append(matched)
    stacked = torch.stack(rows) if rows else target.new_empty((0, 4, target.shape[1]))
    mask = torch.tensor(valid, dtype=torch.bool).reshape(len(rows), 4)
    return stacked, mask


def greedy_depths(model, embedding, head, record, device, roots):
    """Own-token fixed-depth proposals with a differentiable serving-BF16 parameter view.

    vLLM0.27.1 Qwen MTP uses the shared logits head and argmax, with no EOS
    truncation inside its fixed-depth proposer. Discrete choices alone are detached.
    """
    torch = runtime().torch
    validate_greedy_positions(record)
    ids, hidden, positions, _, mask = aligned_inputs(record, device)
    if (len(set(roots)) != len(roots) or any(
            type(r) is not int or not 0 <= r < ids.numel() - 3
            or not mask[r:r + 4].all().item() for r in roots)):
        raise TrainingError("Greedy roots require distinct in-bounds, fully supervised label chains")
    if not roots:
        return [hidden[:0].detach()] * 4, ids.new_empty((0, 4))
    parameters = {name: value.to(torch.bfloat16) for name, value in model.named_parameters()}
    def forward(tokens, state, position, cache):
        return torch.func.functional_call(
            model, parameters, (tokens, state, position, embedding),
            {"past_key_values": cache},
        )
    def choose(state):
        with torch.no_grad():
            logits = head(state.detach())
            if not torch.isfinite(logits).all().item():
                raise TrainingError("Nonfinite greedy proposal logits")
            return logits.argmax(-1)
    cache = runtime().Cache()
    states, proposals = [[] for _ in range(4)], []
    with autocast(device, dtype=torch.bfloat16):
        base = forward(ids, hidden.detach().to(torch.bfloat16), positions, cache)
        for root in roots:
            branch = prefix_cache(cache, root + 1)
            previous = base[root:root + 1]
            draft = []
            for d in range(4):
                if d:
                    previous = forward(draft[-1], previous, positions[root:root + 1] + d, branch)
                states[d].append(previous)
                draft.append(choose(previous))
            proposals.append(torch.cat(draft))
    return [torch.cat(rows) for rows in states], torch.stack(proposals)


def synchronized_time(device):
    torch = runtime().torch
    device = torch.device(device)
    if device.type in ("cuda", "xpu"):
        getattr(torch, device.type).synchronize(device)
    return time.perf_counter()


def recurrence_counts():
    return {"roots": 0, "teacher_replays": 0, "native_teacher_rows": 0,
            "native_teacher_masked": 0, "teacher_rows_by_depth": [0] * 4,
            "proposal_divergences_by_depth": [0] * 4, "divergent_histories_by_depth": [0] * 4,
            "first_divergence_by_depth": [0] * 4, "student_seconds": 0.0, "teacher_seconds": 0.0}


def objective_inputs(model, embedding, head, record, args, roots, teacher=None, diagnostics=None):
    """Keep the CE graph and captured depth-one KL exactly as on the legacy path."""
    outputs = sequence_depths(model, embedding, record, args.device, args.recursive_depth, roots)
    if not getattr(args, "greedy_kl", False):
        pairs = (kl_teacher_pairs(record, outputs, args.recursive_depth, roots, args.device)
                 if getattr(args, "kl_weight", 0.0) > 0 else None)
        return outputs, pairs, None
    if teacher is not None and not hasattr(teacher, "replay"):
        raise TrainingError("Greedy KL teacher handle is not a frozen target")
    start = synchronized_time(args.device)
    states, proposals = greedy_depths(model, embedding, head, record, args.device, roots)
    drafted = synchronized_time(args.device)
    replay, row_valid = native_greedy_teacher_rows(record, roots, proposals)
    replay = replay.to(args.device)
    row_valid = row_valid.to(args.device)
    replayed = synchronized_time(args.device)
    pairs = kl_teacher_pairs(record, outputs[:1], 1, (), args.device)
    pairs.extend((replay[:, depth], outputs[depth][2] & row_valid[:, depth]) for depth in range(1, 4))
    if diagnostics is not None:
        diagnostics["roots"] += len(roots)
        diagnostics["native_teacher_rows"] += len(roots)
        diagnostics["native_teacher_masked"] += int((~row_valid).sum().item())
        diagnostics["student_seconds"] += drafted - start
        diagnostics["teacher_seconds"] += replayed - drafted
        for root, draft in zip(roots, proposals.tolist()):
            diverged = False
            for d, token in enumerate(draft):
                mismatch = token != record["input_ids"][root + d + 2].item()
                diagnostics["teacher_rows_by_depth"][d] += 1
                diagnostics["divergent_histories_by_depth"][d] += int(diverged)
                diagnostics["proposal_divergences_by_depth"][d] += int(mismatch)
                diagnostics["first_divergence_by_depth"][d] += int(mismatch and not diverged)
                diverged |= mismatch
        if "records" in diagnostics:
            diagnostics["records"].append({"prompt_id": record["prompt_id"], "roots": list(roots),
                                           "proposals": proposals.tolist()})
    return outputs, pairs, [None] + states[1:]


def objective_pair_counts(record, args, roots):
    counts = kl_pair_counts(record, args.recursive_depth, roots)
    if getattr(args, "greedy_kl", False):
        # Depths 2-4 use same-request native rows, not a second model replay.
        counts[1:] = [len(roots)] * 3
    return counts


def _kl_teacher_entries(record, depth, roots, device):
    torch = runtime().torch
    if depth not in (1, 4):
        raise TrainingError("Only single-depth or native recursive depth 4 is supported")
    roots = tuple(roots)
    length = record["input_ids"].numel()
    supervised = record["loss_mask"].to(device=device, dtype=torch.bool)
    entries = [(torch.arange(length - 2, device=device, dtype=torch.long) + 1, supervised[2:])]
    for d in range(2, depth + 1):
        indices = torch.tensor([root + d for root in roots], device=device, dtype=torch.long)
        labels = torch.tensor([root + d + 1 for root in roots], device=device, dtype=torch.long)
        label_mask = torch.zeros(len(roots), device=device, dtype=torch.bool)
        in_bounds = (labels >= 0) & (labels < length)
        if in_bounds.any().item():
            label_mask[in_bounds] = supervised.index_select(0, labels[in_bounds])
        entries.append((indices, label_mask))
    return entries


def kl_teacher_pairs(record, outputs, depth, roots, device):
    """Align detached future target states and masks with CE output rows."""
    torch = runtime().torch
    target = record["target_last_hidden_states"].to(device=device).detach()
    entries = _kl_teacher_entries(record, depth, roots, device)
    if len(outputs) != len(entries):
        raise TrainingError("KL teacher/output depth mismatch")
    pairs = []
    for (student, _, output_mask), (indices, supervised) in zip(outputs, entries):
        if student.shape[0] != indices.numel():
            raise TrainingError("KL teacher/output row mismatch")
        if target.shape[0]:
            in_bounds = (indices >= 0) & (indices < target.shape[0])
            safe = indices.clamp(min=0, max=target.shape[0] - 1)
            teacher = target.index_select(0, safe)
            finite = torch.isfinite(teacher).all(dim=-1)
        else:
            in_bounds = torch.zeros(indices.shape, device=device, dtype=torch.bool)
            teacher = target.new_zeros((indices.numel(), target.shape[1]))
            finite = torch.zeros(indices.shape, device=device, dtype=torch.bool)
        valid = output_mask.to(device=device, dtype=torch.bool) & supervised & in_bounds & finite
        pairs.append((teacher.detach(), valid))
    return pairs


def kl_pair_counts(record, depth, roots, device="cpu"):
    """Count finite, supervised future teacher rows without running the model."""
    torch = runtime().torch
    target = record["target_last_hidden_states"].to(device=device)
    counts = []
    for indices, supervised in _kl_teacher_entries(record, depth, roots, device):
        if target.shape[0]:
            in_bounds = (indices >= 0) & (indices < target.shape[0])
            safe = indices.clamp(min=0, max=target.shape[0] - 1)
            finite = torch.isfinite(target.index_select(0, safe)).all(dim=-1)
            counts.append(int((supervised & in_bounds & finite).sum().item()))
        else:
            counts.append(0)
    return counts

def depth_losses(outputs, head, weights, *, chunk_tokens, backward=False, normalizers=None, scaler=None,
                 teachers=None, kl_weight=0.0, kl_temperature=1.0, kl_normalizers=None, kl_students=None):
    """Separate CE/KL chunks, with one connected backward through recursion and base KV."""
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise TrainingError("KL weight must be finite and nonnegative")
    kl_enabled = kl_weight > 0
    if kl_enabled:
        if teachers is None or len(teachers) != len(outputs):
            raise TrainingError("KL requires one teacher pair set per depth")
        if not math.isfinite(kl_temperature) or kl_temperature <= 0:
            raise TrainingError("KL temperature must be positive and finite")
        if kl_normalizers is not None and len(kl_normalizers) != len(outputs):
            raise TrainingError("KL normalizers must match recursive depth")
    gradients, losses = [], []
    for index, (hidden, labels, mask) in enumerate(outputs):
        stats = {"loss_sum": 0.0, "tokens": int(mask.sum().item()), "correct": 0}
        if kl_enabled:
            stats.update(kl_loss_sum=0.0, kl_pairs=0)
        if stats["tokens"]:
            kwargs = {}
            if kl_enabled:
                kwargs["kl_teacher"], kwargs["kl_mask"] = teachers[index]
                if kl_students is not None:
                    kwargs["kl_hidden"] = kl_students[index]
                kwargs["kl_temperature"] = kl_temperature
                kwargs["kl_weight"] = kl_weight
                kwargs["kl_normalizer"] = (
                    kl_normalizers[index] if kl_normalizers is not None else None)
            stats["loss_sum"], _ = chunked_ce(
                hidden, head, labels, mask, chunk_tokens=chunk_tokens, backward=backward,
                normalizer=normalizers[index] if normalizers is not None else None,
                scaler=scaler, weight=weights[index], gradients=gradients, stats=stats,
                **kwargs)
        losses.append(stats)
    if backward:
        runtime().torch.autograd.backward(*zip(*gradients))
    return losses


def loss_metrics(totals, weights):
    report_kl = any("kl_pairs" in total for total in totals)
    depths = [{**total, "depth": d + 1, "weight": weights[d],
               "ce": total["loss_sum"] / total["tokens"] if total["tokens"] else None,
               "argmax_agreement": total["correct"] / total["tokens"] if total["tokens"] else None}
              for d, total in enumerate(totals)]
    if report_kl:
        for total, row in zip(totals, depths):
            pairs = total["kl_pairs"]
            row["kl"] = total["kl_loss_sum"] / pairs if pairs else None
            row["kl_pairs"] = pairs
    result = {"ce": depths[0]["ce"], "tokens": depths[0]["tokens"],
              "argmax_agreement": depths[0]["argmax_agreement"], "depths": depths,
              "objective": sum(row["weight"] * row["ce"] for row in depths if row["tokens"])}
    if report_kl:
        result["kl"] = sum(row["weight"] * row["kl"] for row in depths if row["kl"] is not None)
        result["kl_pairs"] = sum(row["kl_pairs"] for row in depths)
    return result


def evaluate(model, embedding, head, files, config, args, teacher=None):
    if not files:
        return None
    torch = runtime().torch
    was_training = model.training
    model.eval()
    kl_weight = getattr(args, "kl_weight", 0.0)
    kl_temperature = getattr(args, "kl_temperature", 1.0)
    kl_enabled = kl_weight > 0
    totals = [{"loss_sum": 0.0, "tokens": 0, "correct": 0}
              for _ in range(args.recursive_depth)]
    if kl_enabled:
        for total in totals:
            total.update(kl_loss_sum=0.0, kl_pairs=0)
    # Reset for EVERY candidate/stage, independently of training order or RNG.
    rng = random.Random(args.seed)
    with torch.no_grad():
        for path in files:
            record = load_record(path, config, args.max_length)
            roots = sample_roots(record, args.recursive_depth, args.roots, rng)
            outputs, teachers, kl_students = objective_inputs(
                model, embedding, head, record, args, roots, teacher)
            losses = depth_losses(
                outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk,
                teachers=teachers, kl_weight=kl_weight, kl_temperature=kl_temperature,
                kl_students=kl_students)
            for total, loss in zip(totals, losses):
                for key in total:
                    total[key] += loss[key]
    model.train(was_training)
    return {**loss_metrics(totals, args.depth_weights), "sequences": len(files)}


def output_paths(output, model_path):
    output = Path(output).resolve()
    sidecar = output.with_suffix(".json")
    if output.suffix != ".safetensors" or output.is_relative_to(Path(model_path).resolve()):
        raise TrainingError("Output must be a new .safetensors path outside the stock model directory")
    if output.exists() or sidecar.exists():
        raise TrainingError(f"Refusing to overwrite an existing export or sidecar: {output}")
    return output, sidecar


def export_mtp(state, checkpoint, output, report):
    rt = runtime()
    validate_mtp_state(state, checkpoint.shapes)
    output, sidecar = output_paths(output, checkpoint.path)
    tensors = {key: value.detach().to(device="cpu", dtype=rt.torch.bfloat16).contiguous()
               for key, value in state.items()}
    validate_mtp_state(tensors, checkpoint.shapes, stock=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    rt.save_file(tensors, output)
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def rtn_effective_core(state):
    """Reuse diagnosis/fourway-eval.sh: per-row g128 RTN, norms unchanged.

    q/k/v and gate/up fuse along OUTPUT rows in serving, so quantizing their
    separate rows gives the identical codes/scales. No train-time QAT or STE.
    """
    return {key: (rtn_effective_lm_head(value, row_chunk=512).float()
                  if value.ndim == 2 else value.float()) for key, value in state.items()}


def evaluate_export(path, checkpoint, embedding, head, files, args, teacher=None):
    metrics = {}
    native_bf16 = getattr(checkpoint, "regime",
                           getattr(checkpoint, "precision_regime", "gptq_int4_g128")) == "native_bf16"
    stages = ("BF16_export",) if native_bf16 else ("BF16_export", "RTN_effective_dense")
    for stage in stages:
        if not files:
            metrics[stage] = None
            continue
        with runtime().safe_open(path, framework="pt", device="cpu") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        validate_mtp_state(state, checkpoint.shapes, stock=True)
        if stage == "RTN_effective_dense":
            state = rtn_effective_core(state)
        model = build_native_mtp(checkpoint.config, state, args.device)
        del state
        metrics[stage] = evaluate(model, embedding, head, files, checkpoint.config, args, teacher)
        del model
    if native_bf16:
        # RTN is not an actual regime for an unquantized BF16 checkpoint.
        metrics["RTN_effective_dense"] = None
    return metrics


def select_dev_checkpoint(candidates):
    """Strict improvement only: stock wins ties; no test-set inputs or promotion."""
    if any(candidate["metrics"].get("RTN_effective_dense") is not None for candidate in candidates):
        stage = "RTN_effective_dense"
    elif any(candidate["metrics"].get("BF16_export") is not None for candidate in candidates):
        stage = "BF16_export"
    else:
        return None
    eligible = [candidate for candidate in candidates if candidate["metrics"].get(stage) is not None]
    winner = min(eligible, key=lambda candidate: candidate["metrics"][stage]["objective"])
    return {"step": winner["step"], "path": winner["path"],
            "metric": f"dev.{stage}.objective",
            "value": winner["metrics"][stage]["objective"],
            "promotion": False}


def checkpoint_path(output, step):
    output = Path(output)
    return output.with_name(f"{output.stem}.step{step:04d}.safetensors")


def memory_peak(device):
    import resource
    result = {"process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    torch = runtime().torch
    if device.type == "xpu":
        result.update(xpu_max_allocated_bytes=torch.xpu.max_memory_allocated(device),
                      xpu_max_reserved_bytes=torch.xpu.max_memory_reserved(device))
    elif device.type == "cuda":
        result.update(cuda_max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_max_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def mtp_digest(state):
    """Content hash for the existing stock/export boundary, independent of file metadata."""
    digest = hashlib.sha256()
    torch = runtime().torch
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_only(model, embedding, head, teacher, checkpoint, train, dev, args, report, output):
    """Exercise the actual loss/backward/export CLI path without constructing an optimizer."""
    torch = runtime().torch
    started = synchronized_time(args.device)
    stock_hash = mtp_digest(checkpoint.mtp_state())
    report.update(status="validation-only", validation={}, recurrence=recurrence_counts())
    report["recurrence"]["records"] = []
    report["counts"] = {"train_sequences": len(train), "dev_sequences": len(dev),
                        "sequences_seen": 0, "loss_tokens_by_depth": [0] * args.recursive_depth,
                        "kl_pairs_by_depth": [0] * args.recursive_depth, "sampled_roots": 0}
    frozen = list(teacher.model.parameters()) if teacher is not None else list(embedding.parameters()) + list(head.parameters())
    gradients = {"student_connected": True, "student_finite": True, "student_nonzero": False,
                 "frozen_no_grad": True, "group_grad_norms": []}
    ce_identical, checks = True, []
    for split, files in (("train", train), ("dev", dev)):
        rng = random.Random(args.seed)
        totals = [{"loss_sum": 0.0, "tokens": 0, "correct": 0, "kl_loss_sum": 0.0, "kl_pairs": 0}
                  for _ in range(args.recursive_depth)]
        for start in range(0, len(files), args.grad_accum):
            records = [load_record(path, checkpoint.config, args.max_length)
                       for path in files[start:start + args.grad_accum]]
            roots = [sample_roots(record, args.recursive_depth, args.roots, rng) for record in records]
            denominators = [sum(int(r["loss_mask"][2:].sum()) for r in records)] + [sum(map(len, roots))] * (args.recursive_depth - 1)
            kl_denominators = [sum(counts[d] for counts in (objective_pair_counts(r, args, s)
                               for r, s in zip(records, roots))) for d in range(args.recursive_depth)]
            model.zero_grad(set_to_none=True)
            for record, selected in zip(records, roots):
                outputs, teachers, students = objective_inputs(
                    model, embedding, head, record, args, selected, teacher, report["recurrence"])
                with torch.no_grad():
                    anchor = depth_losses(outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk)
                losses = depth_losses(
                    outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk, backward=True,
                    normalizers=denominators, teachers=teachers, kl_students=students,
                    kl_weight=args.kl_weight, kl_temperature=args.kl_temperature, kl_normalizers=kl_denominators)
                ce_identical &= all(all(a[key] == b[key] for key in ("loss_sum", "tokens", "correct"))
                                    for a, b in zip(anchor, losses))
                for d, (total, loss) in enumerate(zip(totals, losses)):
                    for key in total:
                        total[key] += loss.get(key, 0)
                    report["counts"]["loss_tokens_by_depth"][d] += loss["tokens"]
                    report["counts"]["kl_pairs_by_depth"][d] += loss.get("kl_pairs", 0)
                report["counts"]["sequences_seen"] += 1
                report["counts"]["sampled_roots"] += len(selected)
                checks.append((split, record["prompt_id"], selected))
                del outputs, teachers, students
            parameters = list(model.parameters())
            gradients["student_connected"] &= all(p.grad is not None for p in parameters)
            gradients["student_finite"] &= all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in parameters)
            norm = math.sqrt(sum(p.grad.float().norm().item() ** 2 for p in parameters if p.grad is not None))
            gradients["group_grad_norms"].append(norm)
            gradients["student_nonzero"] |= norm > 0
            gradients["frozen_no_grad"] &= all(not p.requires_grad and p.grad is None for p in frozen)
            model.zero_grad(set_to_none=True)
        report["validation"][split] = {**loss_metrics(totals, args.depth_weights), "sequences": len(files)}
    report["validation"]["ce_anchor_identical"] = ce_identical
    report["validation"]["gradient_checks"] = gradients
    if not ce_identical or not all(gradients[key] for key in (
            "student_connected", "student_finite", "student_nonzero", "frozen_no_grad")):
        raise TrainingError("Validation CE-anchor/gradient checks failed")
    forward_backward_done = synchronized_time(args.device)
    state = {"mtp." + key: value for key, value in model.state_dict().items()}
    if mtp_digest(state) != stock_hash:
        raise TrainingError("Validation changed the stock MTP weights")
    export_mtp(state, checkpoint, output, report)
    with runtime().safe_open(output, framework="pt", device="cpu") as handle:
        reloaded_state = {key: handle.get_tensor(key) for key in handle.keys()}
    validate_mtp_state(reloaded_state, checkpoint.shapes, stock=True)
    export_hash = mtp_digest(reloaded_state)
    if export_hash != stock_hash:
        raise TrainingError("Validation export differs from the stock MTP")
    reloaded = build_native_mtp(checkpoint.config, reloaded_state, args.device)
    del reloaded_state, state
    proposal_checks = []
    with torch.no_grad():
        for (split, prompt_id, selected), path in zip(checks, list(train) + list(dev)):
            record = load_record(path, checkpoint.config, args.max_length)
            _, before = greedy_depths(model, embedding, head, record, args.device, selected)
            _, after = greedy_depths(reloaded, embedding, head, record, args.device, selected)
            if not torch.equal(before, after):
                raise TrainingError("BF16 export/reload greedy proposals differ")
            proposal_checks.append({"split": split, "prompt_id": prompt_id, "roots": selected,
                                    "proposals": before.tolist()})
    report["validation"]["export_reload"] = {
        "greedy_proposals_equal": True, "proposal_tokens": sum(len(c["roots"]) * 4 for c in proposal_checks),
        "stock_mtp_sha256": stock_hash, "export_mtp_sha256": export_hash, "records": proposal_checks,
    }
    report["validation"]["seconds"] = {"forward_backward": forward_backward_done - started,
                                        "export_reload": synchronized_time(args.device) - forward_backward_done}
    report["memory_peak"] = memory_peak(torch.device(args.device))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def run(args):
    rt = runtime()
    torch = rt.torch
    args.kl_weight = getattr(args, "kl_weight", 0.0)
    args.kl_temperature = getattr(args, "kl_temperature", 1.0)
    if (not 3 <= args.max_length <= 2048 or not 1 <= args.steps <= 10000
            or not math.isfinite(args.lr) or args.lr <= 0
            or not 1 <= args.grad_accum <= 64 or not 1 <= args.logits_chunk <= 256
            or args.recursive_depth not in (1, 4) or not 1 <= args.roots <= 32
            or not 1 <= args.checkpoint_every <= 1000
            or (args.epochs is not None and not 1 <= args.epochs <= 100)
            or not math.isfinite(args.kl_weight) or args.kl_weight < 0
            or not math.isfinite(args.kl_temperature) or args.kl_temperature <= 0):
        raise TrainingError("Invalid bounds: length 3..2048, steps 1..10000, positive lr, accumulation 1..64, "
                            "logits chunk 1..256, depth 1/4, roots 1..32, checkpoint interval 1..1000, "
                            "epochs 1..100, nonnegative finite KL weight, positive finite KL temperature")
    args.depth_weights = args.depth_weights if args.depth_weights is not None else [1.0] * args.recursive_depth
    if (len(args.depth_weights) != args.recursive_depth
            or any(not math.isfinite(w) or w < 0 for w in args.depth_weights)
            or sum(args.depth_weights) <= 0):
        raise TrainingError("Supply one finite nonnegative weight per depth, with positive total weight")
    greedy_kl = getattr(args, "greedy_kl", False)
    validation_only = getattr(args, "validate_only", False)
    if greedy_kl and (args.recursive_depth != 4 or args.kl_weight <= 0 or args.export_stock):
        raise TrainingError("--greedy-kl requires depth 4, positive KL weight and no --export-stock")
    if validation_only and (args.recursive_depth != 4 or args.export_stock):
        raise TrainingError("--validate-only requires depth 4 and no --export-stock")
    kl_enabled = args.kl_weight > 0
    checkpoint = Checkpoint(args.model)
    native_bf16 = checkpoint.regime == "native_bf16"
    if greedy_kl:
        validate_greedy_checkpoint(checkpoint, args.device)
    if validation_only and not native_bf16:
        raise TrainingError("--validate-only requires native BF16 for the greedy export boundary")
    precision = {
        "masters": "float32",
        "export": "bfloat16",
        "cpu_compute": "bfloat16_autocast (tiny tests only)" if native_bf16 else "float32 (tiny tests only)",
    }
    if native_bf16:
        precision.update(
            regime=checkpoint.regime,
            cuda_compute="bfloat16_autocast",
            xpu_compute="bfloat16_autocast",
            lm_head="frozen original BF16 checkpoint weight",
            limitations="Native BF16 metrics are actual BF16 exports; RTN-effective metrics are not applicable",
        )
    else:
        precision.update(
            xpu_compute="float16_autocast",
            lm_head="frozen dequantized RTN INT4-g128, FP16 input/scales/effective weights",
            limitations="No core QAT; exported/RTN dev CE != serving acceptance; dense GEMM != bitwise INT4 kernel",
        )
    output, sidecar = output_paths(args.output, checkpoint.path)
    report = {
        "mode": "export_stock" if args.export_stock else (
            "recursive_teacher_forcing" if args.recursive_depth == 4 else "first_step_teacher_forcing"),
        "config": {key: value for key, value in vars(args).items()
                   if key not in ("greedy_kl", "validate_only") or value}, "torch": torch.__version__,
        "transformers": rt.transformers.__version__, "python": platform.python_version(),
        "optimizer_steps": 0, "train_steps": [], "eval_before": None, "eval_after": None,
        "checkpoints": [], "dev_selection": None,
        "alignment": "base: x[j+1], observed h[j], p[j] -> x[j+2]; branch d: x[t+d], y[d-1], p[t]+d-1 -> x[t+d+1]",
        "precision": precision,
    }
    if args.export_stock:
        if args.train_dir or args.eval_dir:
            raise TrainingError("--export-stock does not consume train/eval data")
        export_mtp(checkpoint.mtp_state(), checkpoint, output, report)
        return report
    if not args.train_dir:
        raise TrainingError("Training requires --train-dir")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("Requested CUDA device is not available")
        if not torch.cuda.is_bf16_supported():
            raise TrainingError("BF16 CUDA support required")
        if not native_bf16:
            raise TrainingError("CUDA training requires an unquantized BF16 checkpoint")
    elif device.type == "xpu":
        if not torch.xpu.is_available():
            raise TrainingError("Requested inference-host XPU is not available")
    elif device.type != "cpu":
        raise TrainingError("Use CPU for tiny synthetic tests, or an available inference-host XPU/CUDA device")
    if device.type == "cpu" and sum(math.prod(s) for s in checkpoint.shapes.values()) > 5_000_000:
        raise TrainingError("Stock-size CPU training is forbidden; use inference-host XPU after unloading verifier")
    train, dev = capture_sets(args.train_dir, args.eval_dir, checkpoint.config, args.max_length)
    if greedy_kl or validation_only:
        for path in train + dev:
            validate_greedy_positions(load_record(path, checkpoint.config, args.max_length))
    sequence_budget = (len(train) if validation_only else (
        args.epochs * len(train) if args.epochs is not None else args.steps * args.grad_accum))
    steps = math.ceil(sequence_budget / args.grad_accum)
    if steps > 10000:
        raise TrainingError("Epoch budget exceeds 10000 optimizer updates")
    # Fail before training if ANY scheduled output would clobber an existing run.
    if not validation_only:
        for step in range(0, steps, args.checkpoint_every):
            output_paths(checkpoint_path(output, step), checkpoint.path)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    rng, root_rng = random.Random(args.seed), random.Random(args.seed)
    model = build_native_mtp(checkpoint.config, checkpoint.mtp_state(), args.device)
    teacher = FrozenTarget(checkpoint, args.device) if greedy_kl else None
    embedding, head = ((teacher.embedding, teacher.head) if teacher is not None
                       else frozen_heads(checkpoint, args.device))
    if greedy_kl:
        report["mode"] = "greedy_deeper_kl"
        report["recurrence"] = recurrence_counts()
    if validation_only:
        return validate_only(model, embedding, head, teacher, checkpoint, train, dev, args, report, output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler("xpu", init_scale=128.0) if device.type == "xpu" and not native_bf16 else None

    def save_candidate(step, path):
        state = {"mtp." + key: value for key, value in model.state_dict().items()}
        export_mtp(state, checkpoint, path, {"step": step})
        candidate = {"step": step, "path": str(path),
                     "metrics": evaluate_export(path, checkpoint, embedding, head, dev, args,
                                                **({"teacher": teacher} if teacher is not None else {}))}
        Path(path).with_suffix(".json").write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
        report["checkpoints"].append(candidate)
        return candidate["metrics"]["BF16_export"]

    report["eval_before"] = save_candidate(0, checkpoint_path(output, 0))
    report["counts"] = {"train_sequences": len(train), "dev_sequences": len(dev), "sequences_seen": 0,
                        "input_tokens_seen": 0, "observed_hidden_rows_seen": 0, "useful_positions_seen": 0,
                        "loss_tokens_by_depth": [0] * args.recursive_depth}
    if kl_enabled:
        report["counts"]["kl_pairs_by_depth"] = [0] * args.recursive_depth
    order, cursor = [], 0
    model.train()
    for step in range(steps):
        records = []
        for _ in range(min(args.grad_accum, sequence_budget - report["counts"]["sequences_seen"])):
            if cursor == len(order):
                order = list(train)
                rng.shuffle(order)
                cursor = 0
            records.append(load_record(order[cursor], checkpoint.config, args.max_length))
            cursor += 1
        roots = [sample_roots(record, args.recursive_depth, args.roots, root_rng) for record in records]
        tokens = sum(int(record["loss_mask"][2:].sum().item()) for record in records)
        denominators = [tokens] + [sum(map(len, roots))] * (args.recursive_depth - 1)
        kl_denominators = None
        if kl_enabled:
            kl_denominators = [0] * args.recursive_depth
            for record, selected in zip(records, roots):
                for index, count in enumerate(objective_pair_counts(record, args, selected)):
                    kl_denominators[index] += count
        totals = [{"loss_sum": 0.0, "tokens": 0, "correct": 0} for _ in denominators]
        if kl_enabled:
            for total in totals:
                total.update(kl_loss_sum=0.0, kl_pairs=0)
        optimizer.zero_grad(set_to_none=True)
        for record, selected in zip(records, roots):
            outputs, teachers, kl_students = objective_inputs(
                model, embedding, head, record, args, selected, teacher, report.get("recurrence"))
            losses = depth_losses(
                outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk,
                backward=True, normalizers=denominators, scaler=scaler,
                teachers=teachers, kl_weight=args.kl_weight, kl_students=kl_students,
                kl_temperature=args.kl_temperature, kl_normalizers=kl_denominators)
            for total, loss in zip(totals, losses):
                for key in total:
                    total[key] += loss[key]
            del outputs, teachers, kl_students
        if scaler is not None:
            scaler.unscale_(optimizer)
        # Abort rather than report a skipped/overflowed update as a step.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        # Export validation builds another core; do not retain obsolete gradients.
        optimizer.zero_grad(set_to_none=True)
        result = {"step": step + 1, **loss_metrics(totals, args.depth_weights),
                  "sequences": len(records), "grad_norm": grad_norm.item()}
        report["train_steps"].append(result)
        report["optimizer_steps"] += 1
        counts = report["counts"]
        counts["sequences_seen"] += len(records)
        counts["input_tokens_seen"] += sum(r["input_ids"].numel() for r in records)
        counts["observed_hidden_rows_seen"] += sum(r["positions"].numel() for r in records)
        counts["useful_positions_seen"] += tokens
        counts["loss_tokens_by_depth"] = [a + b for a, b in zip(counts["loss_tokens_by_depth"], denominators)]
        if kl_enabled:
            counts["kl_pairs_by_depth"] = [a + b for a, b in zip(
                counts["kl_pairs_by_depth"], [total["kl_pairs"] for total in totals])]
        print(json.dumps(result, allow_nan=False), flush=True)
        if (step + 1) % args.checkpoint_every == 0 and step + 1 < steps:
            save_candidate(step + 1, checkpoint_path(output, step + 1))
    report["eval_after"] = save_candidate(steps, output)
    report["dev_selection"] = select_dev_checkpoint(report["checkpoints"])
    report["memory_peak"] = memory_peak(device)
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", required=True, help="Local GPTQ or unquantized BF16 checkpoint directory; no downloads")
    result.add_argument("--train-dir")
    result.add_argument("--eval-dir", help="Development captures for checkpoint selection; NEVER the fresh test set")
    result.add_argument("--output", required=True, help="Final mtp-only BF16 file; step 0/intermediates saved beside it, no promotion")
    result.add_argument("--export-stock", action="store_true", help="Identity export only; no optimizer or target load")
    budget = result.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, default=10, help="Optimizer updates (1..10000)")
    budget.add_argument("--epochs", type=int, help="Complete shuffled passes (1..100), instead of --steps; at most 10000 updates")
    result.add_argument("--recursive-depth", type=int, choices=(1, 4), default=1)
    result.add_argument("--roots", type=int, default=8, help="Max sampled recursive roots per sequence (1..32)")
    result.add_argument("--depth-weights", type=float, nargs="+", help="One nonnegative weight per depth; default all 1, weighted SUM of means")
    result.add_argument("--checkpoint-every", type=int, default=10, help="Export/evaluate every N updates; step 0 and final always saved")
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--device", default="xpu")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-length", type=int, default=2048, help="Reject, never truncate, longer sequences")
    result.add_argument("--grad-accum", type=int, default=4, help="Complete sequences per optimizer update")
    result.add_argument("--logits-chunk", type=int, default=128, help="Supervised tokens per CE/KL logits allocation")
    result.add_argument("--kl-weight", type=float, default=0.0, help="Auxiliary teacher-to-student KL weight; default disables KL")
    result.add_argument("--kl-temperature", type=float, default=1.0, help="Positive finite KL temperature")
    result.add_argument("--greedy-kl", action="store_true",
                        help="Opt-in native BF16 own-token deeper KL; frozen full target, depth 4 only")
    result.add_argument("--validate-only", action="store_true",
                        help="Native BF16 depth-4 diagnostics/backward/export/reload; no optimizer or updates")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except (TrainingError, RuntimeError) as exc:
        raise SystemExit(f"MTP trainer: {exc}") from exc


if __name__ == "__main__":
    main()
