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

Precision limits: FP32 trainable masters/AdamW, FP16 XPU autocast + loss scaling,
BF16 export. CPU is FP32 and only intended for tiny synthetic tests. The frozen
LM head is the deployed RTN INT4-g128 effective weight (FP16 input and stored
scales), NOT the checkpoint BF16 head. Dense dequantized GEMM is not bitwise XPU
INT4-kernel equivalence. Core RTN is not used during training (no QAT). Step 0,
intermediate and final BF16 overlays are reloaded for dev metrics both before and
after the existing group-128 RTN effective-core transform. Stock remains eligible;
the reported dev choice is NOT a serving promotion or an acceptance guarantee.

Examples (new output paths, verifier unloaded before training):
  python qwen38_train_mtp.py --model MODEL --export-stock --output stock.safetensors
  python qwen38_train_mtp.py --model MODEL --train-dir TRAIN --eval-dir HELDOUT \
      --output tuned.safetensors --steps 10 --lr 1e-5 --device xpu
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import random
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


def native_config(raw):
    text = raw.get("text_config", {})
    quant = raw.get("quantization_config", {})
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
        or quant.get("quant_method") != "gptq"
        or quant.get("bits") != 4
        or quant.get("group_size") != 128
        or quant.get("sym") is not True
    ):
        raise TrainingError("Only dense, gated-Q, one-layer native Qwen3.5 GPTQ INT4-g128 is supported")
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
    weight = checkpoint.tensor(checkpoint.EMBEDDING_KEY)
    if tuple(weight.shape) != shape or not torch.isfinite(weight).all().item():
        raise TrainingError("Invalid shared embedding")
    dtype = torch.float16 if torch.device(device).type == "xpu" else torch.float32
    # CPU tests reproduce the serving FP16 rounding before FP32 computation.
    embedding = torch.nn.Embedding.from_pretrained(weight.half().to(device=device, dtype=dtype),
                                                   freeze=True)
    weight = checkpoint.tensor(checkpoint.LM_HEAD_KEY)
    if tuple(weight.shape) != shape:
        raise TrainingError("Invalid separate LM-head shape")
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


def autocast(device):
    torch = runtime().torch
    return torch.autocast("xpu", dtype=torch.float16) if torch.device(device).type == "xpu" else nullcontext()


def chunked_ce(hidden, head, labels, mask, *, chunk_tokens=128, backward=False,
               normalizer=None, scaler=None, weight=1.0, gradients=None, stats=None):
    """Token-chunked CE; backward frees EACH logits graph before the next chunk.

    A detached leaf accumulates output gradients, then backpropagates through the
    full-context native core once. Accumulating graph-connected chunk losses would
    retain the entire T*vocab logits tensor and defeat the memory bound.
    """
    torch = runtime().torch
    indices = mask.nonzero(as_tuple=True)[0]
    count = indices.numel()
    denominator = count if normalizer is None else normalizer
    if count == 0 or chunk_tokens <= 0 or denominator <= 0:
        raise TrainingError("CE requires supervised labels and positive chunk/normalizer")
    features = hidden.detach().requires_grad_(True) if backward else hidden
    total = 0.0
    for offset in range(0, count, chunk_tokens):
        rows = indices[offset:offset + chunk_tokens]
        with autocast(hidden.device):
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
    if backward:
        if gradients is None:
            hidden.backward(features.grad)
        else:
            # Joint backward after ALL depths: later outputs depend on earlier
            # outputs and base KV, so freeing an earlier graph here is incorrect.
            gradients.append((hidden, features.grad))
    return total, count


def sequence_hidden(model, embedding, record, device):
    ids, hidden, positions, labels, mask = aligned_inputs(record, device)
    hidden = hidden.to(dtype=embedding.weight.dtype)
    with autocast(device):
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
    with autocast(device):
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


def depth_losses(outputs, head, weights, *, chunk_tokens, backward=False, normalizers=None, scaler=None):
    """Separate logits chunks, one connected backward through recursion AND base KV."""
    gradients, losses = [], []
    for index, (hidden, labels, mask) in enumerate(outputs):
        stats = {"loss_sum": 0.0, "tokens": int(mask.sum().item()), "correct": 0}
        if stats["tokens"]:
            stats["loss_sum"], _ = chunked_ce(
                hidden, head, labels, mask, chunk_tokens=chunk_tokens, backward=backward,
                normalizer=normalizers[index] if normalizers is not None else None,
                scaler=scaler, weight=weights[index], gradients=gradients, stats=stats)
        losses.append(stats)
    if backward:
        runtime().torch.autograd.backward(*zip(*gradients))
    return losses


def loss_metrics(totals, weights):
    depths = [{**total, "depth": d + 1, "weight": weights[d],
               "ce": total["loss_sum"] / total["tokens"] if total["tokens"] else None,
               "argmax_agreement": total["correct"] / total["tokens"] if total["tokens"] else None}
              for d, total in enumerate(totals)]
    return {"ce": depths[0]["ce"], "tokens": depths[0]["tokens"],
            "argmax_agreement": depths[0]["argmax_agreement"], "depths": depths,
            "objective": sum(row["weight"] * row["ce"] for row in depths if row["tokens"])}


def evaluate(model, embedding, head, files, config, args):
    if not files:
        return None
    torch = runtime().torch
    was_training = model.training
    model.eval()
    totals = [{"loss_sum": 0.0, "tokens": 0, "correct": 0} for _ in range(args.recursive_depth)]
    # Reset for EVERY candidate/stage, independently of training order or RNG.
    rng = random.Random(args.seed)
    with torch.no_grad():
        for path in files:
            record = load_record(path, config, args.max_length)
            roots = sample_roots(record, args.recursive_depth, args.roots, rng)
            outputs = sequence_depths(model, embedding, record, args.device, args.recursive_depth, roots)
            losses = depth_losses(outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk)
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


def evaluate_export(path, checkpoint, embedding, head, files, args):
    metrics = {}
    for stage in ("BF16_export", "RTN_effective_dense"):
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
        metrics[stage] = evaluate(model, embedding, head, files, checkpoint.config, args)
        del model
    return metrics


def select_dev_checkpoint(candidates):
    """Strict improvement only: stock wins ties; no test-set inputs or promotion."""
    eligible = [candidate for candidate in candidates if candidate["metrics"]["RTN_effective_dense"] is not None]
    if not eligible:
        return None
    winner = min(eligible, key=lambda candidate: candidate["metrics"]["RTN_effective_dense"]["objective"])
    return {"step": winner["step"], "path": winner["path"],
            "metric": "dev.RTN_effective_dense.objective",
            "value": winner["metrics"]["RTN_effective_dense"]["objective"],
            "promotion": False}


def checkpoint_path(output, step):
    output = Path(output)
    return output.with_name(f"{output.stem}.step{step:04d}.safetensors")


def memory_peak(device):
    import resource
    result = {"process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    if device.type == "xpu":
        result.update(xpu_max_allocated_bytes=runtime().torch.xpu.max_memory_allocated(device),
                      xpu_max_reserved_bytes=runtime().torch.xpu.max_memory_reserved(device))
    return result


def run(args):
    rt = runtime()
    torch = rt.torch
    if (not 3 <= args.max_length <= 2048 or not 1 <= args.steps <= 1000
            or not math.isfinite(args.lr) or args.lr <= 0
            or not 1 <= args.grad_accum <= 64 or not 1 <= args.logits_chunk <= 256
            or args.recursive_depth not in (1, 4) or not 1 <= args.roots <= 32
            or not 1 <= args.checkpoint_every <= 1000
            or (args.epochs is not None and not 1 <= args.epochs <= 100)):
        raise TrainingError("Invalid bounds: length 3..2048, steps 1..1000, positive lr, accumulation 1..64, "
                            "logits chunk 1..256, depth 1/4, roots 1..32, checkpoint interval 1..1000, epochs 1..100")
    args.depth_weights = args.depth_weights if args.depth_weights is not None else [1.0] * args.recursive_depth
    if (len(args.depth_weights) != args.recursive_depth
            or any(not math.isfinite(w) or w < 0 for w in args.depth_weights)
            or sum(args.depth_weights) <= 0):
        raise TrainingError("Supply one finite nonnegative weight per depth, with positive total weight")
    checkpoint = Checkpoint(args.model)
    output, sidecar = output_paths(args.output, checkpoint.path)
    report = {
        "mode": "export_stock" if args.export_stock else (
            "recursive_teacher_forcing" if args.recursive_depth == 4 else "first_step_teacher_forcing"),
        "config": vars(args).copy(), "torch": torch.__version__,
        "transformers": rt.transformers.__version__, "python": platform.python_version(),
        "optimizer_steps": 0, "train_steps": [], "eval_before": None, "eval_after": None,
        "checkpoints": [], "dev_selection": None,
        "alignment": "base: x[j+1], observed h[j], p[j] -> x[j+2]; branch d: x[t+d], y[d-1], p[t]+d-1 -> x[t+d+1]",
        "precision": {
            "masters": "float32", "export": "bfloat16", "xpu_compute": "float16_autocast",
            "cpu_compute": "float32 (tiny tests only)",
            "lm_head": "frozen dequantized RTN INT4-g128, FP16 input/scales/effective weights",
            "limitations": "No core QAT; exported/RTN dev CE != serving acceptance; dense GEMM != bitwise INT4 kernel",
        },
    }
    if args.export_stock:
        if args.train_dir or args.eval_dir:
            raise TrainingError("--export-stock does not consume train/eval data")
        export_mtp(checkpoint.mtp_state(), checkpoint, output, report)
        return report
    if not args.train_dir:
        raise TrainingError("Training requires --train-dir")
    device = torch.device(args.device)
    if device.type not in ("cpu", "xpu") or (device.type == "xpu" and not torch.xpu.is_available()):
        raise TrainingError("Use CPU for tiny synthetic tests or an available inference-host XPU")
    if device.type == "cpu" and sum(math.prod(s) for s in checkpoint.shapes.values()) > 5_000_000:
        raise TrainingError("Stock-size CPU training is forbidden; use inference-host XPU after unloading verifier")
    train, dev = capture_sets(args.train_dir, args.eval_dir, checkpoint.config, args.max_length)
    sequence_budget = args.epochs * len(train) if args.epochs is not None else args.steps * args.grad_accum
    steps = math.ceil(sequence_budget / args.grad_accum)
    if steps > 1000:
        raise TrainingError("Epoch budget exceeds 1000 optimizer updates")
    # Fail before training if ANY scheduled output would clobber an existing run.
    for step in range(0, steps, args.checkpoint_every):
        output_paths(checkpoint_path(output, step), checkpoint.path)
    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    rng, root_rng = random.Random(args.seed), random.Random(args.seed)
    model = build_native_mtp(checkpoint.config, checkpoint.mtp_state(), args.device)
    embedding, head = frozen_heads(checkpoint, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler("xpu", init_scale=128.0) if device.type == "xpu" else None

    def save_candidate(step, path):
        state = {"mtp." + key: value for key, value in model.state_dict().items()}
        export_mtp(state, checkpoint, path, {"step": step})
        candidate = {"step": step, "path": str(path),
                     "metrics": evaluate_export(path, checkpoint, embedding, head, dev, args)}
        Path(path).with_suffix(".json").write_text(json.dumps(candidate, indent=2, allow_nan=False) + "\n")
        report["checkpoints"].append(candidate)
        return candidate["metrics"]["BF16_export"]

    report["eval_before"] = save_candidate(0, checkpoint_path(output, 0))
    report["counts"] = {"train_sequences": len(train), "dev_sequences": len(dev), "sequences_seen": 0,
                        "input_tokens_seen": 0, "observed_hidden_rows_seen": 0, "useful_positions_seen": 0,
                        "loss_tokens_by_depth": [0] * args.recursive_depth}
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
        totals = [{"loss_sum": 0.0, "tokens": 0, "correct": 0} for _ in denominators]
        optimizer.zero_grad(set_to_none=True)
        for record, selected in zip(records, roots):
            outputs = sequence_depths(model, embedding, record, args.device, args.recursive_depth, selected)
            losses = depth_losses(outputs, head, args.depth_weights, chunk_tokens=args.logits_chunk,
                                  backward=True, normalizers=denominators, scaler=scaler)
            for total, loss in zip(totals, losses):
                for key in total:
                    total[key] += loss[key]
            del outputs
        if scaler is not None:
            scaler.unscale_(optimizer)
        # Abort rather than report a skipped/overflowed update as a step.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
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
    result.add_argument("--model", required=True, help="Local GPTQ checkpoint directory; no downloads")
    result.add_argument("--train-dir")
    result.add_argument("--eval-dir", help="Development captures for checkpoint selection; NEVER the fresh test set")
    result.add_argument("--output", required=True, help="Final mtp-only BF16 file; step 0/intermediates saved beside it, no promotion")
    result.add_argument("--export-stock", action="store_true", help="Identity export only; no optimizer or target load")
    budget = result.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int, default=10, help="Optimizer updates (1..1000)")
    budget.add_argument("--epochs", type=int, help="Complete shuffled passes (1..100), instead of --steps; at most 1000 updates")
    result.add_argument("--recursive-depth", type=int, choices=(1, 4), default=1)
    result.add_argument("--roots", type=int, default=8, help="Max sampled recursive roots per sequence (1..32)")
    result.add_argument("--depth-weights", type=float, nargs="+", help="One nonnegative weight per depth; default all 1, weighted SUM of means")
    result.add_argument("--checkpoint-every", type=int, default=10, help="Export/evaluate every N updates; step 0 and final always saved")
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--device", default="xpu")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-length", type=int, default=2048, help="Reject, never truncate, longer sequences")
    result.add_argument("--grad-accum", type=int, default=4, help="Complete sequences per optimizer update")
    result.add_argument("--logits-chunk", type=int, default=128, help="Supervised tokens per CE logits allocation")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except (TrainingError, RuntimeError) as exc:
        raise SystemExit(f"MTP trainer: {exc}") from exc


if __name__ == "__main__":
    main()
