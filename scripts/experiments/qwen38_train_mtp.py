#!/usr/bin/env python3
"""Bounded offline tuning of the stock, single-layer Qwen3.5/3.8 native MTP.

Run in the pinned inference-host environment, never in the interactive Pi runtime.
Only mtp.*, the embedding, and lm_head are read from the local safetensors index;
no verifier, model download, new architecture, or serving configuration is created.

Records are complete text sequences: prompt_id (nonempty string), input_ids int64[T],
positions int64[T], target_last_hidden_states floating[T,H], loss_mask bool[T].
The assistant-label mask is shifted by TWO. All prefix tokens condition attention;
overlength/corrupt records fail, never crop or skip. Train/eval prompt IDs must be
disjoint. This is first-step teacher forcing, NOT recursive four-depth training.

Pinned source inspection established native +1 RMS norms, cat [embedding, hidden],
fc, one full-attention gated-Q decoder, final norm, and direct checkpoint key mapping:
https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/model_executor/models/qwen3_5_mtp.py
https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/vllm/v1/spec_decode/llm_base_proposer.py
Step3p5MTPProposer inherits set_inputs_first_pass: IDs shift, positions DO NOT.
Thus use x[1:-1], hidden[:-2], positions[:-2], labels/mask[2:].

Precision limits: FP32 trainable masters/AdamW, FP16 XPU autocast + loss scaling,
BF16 export. CPU is FP32 and only intended for tiny synthetic tests. The frozen
LM head is the deployed RTN INT4-g128 effective weight (FP16 input and stored
scales), NOT the checkpoint BF16 head. Dense dequantized GEMM is not bitwise XPU
INT4-kernel equivalence. MTP core RTN is applied only by the existing serving
loader, not during training (no QAT); real GPTQ-verifier features supply the
quantization-aware input distribution. CE alone does not establish acceptance.

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
        Norm=Qwen3_5RMSNorm, Rotary=Qwen3_5TextRotaryEmbedding,
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

        def forward(self, input_ids, target_hidden, positions, embedding):
            embedded = self.pre_fc_norm_embedding(embedding(input_ids))
            hidden = self.pre_fc_norm_hidden(target_hidden)
            hidden = self.fc(torch.cat([embedded, hidden], dim=-1)).unsqueeze(0)
            position_ids = positions.unsqueeze(0)
            rotary = self.rotary_emb(hidden, position_ids)
            length = input_ids.numel()
            # Explicit mask: unmasked prompt rows still participate in every layer.
            mask = torch.full((length, length), float("-inf"), dtype=hidden.dtype,
                              device=hidden.device).triu(1)[None, None]
            hidden = self.layers[0](hidden, position_embeddings=rotary,
                                    attention_mask=mask, position_ids=position_ids)
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
    if positions.shape != (length,) or positions.dtype != torch.int64:
        raise TrainingError("positions must be int64[T]")
    if positions[0].item() < 0 or not (positions[1:] > positions[:-1]).all().item():
        raise TrainingError("positions must be nonnegative and strictly increasing")
    if positions[-1].item() >= config.max_position_embeddings:
        raise TrainingError("positions exceed model context")
    if ids.min().item() < 0 or ids.max().item() >= config.vocab_size:
        raise TrainingError("input_ids outside model vocabulary")
    if hidden.shape != (length, config.hidden_size) or hidden.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TrainingError("target_last_hidden_states must be runtime float[T, hidden_size]")
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
        record["target_last_hidden_states"][:-2].to(device),
        record["positions"][:-2].to(device),
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
               normalizer=None, scaler=None):
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
        if backward:
            scaled = loss / denominator
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
        del logits, loss
    if backward:
        hidden.backward(features.grad)
    return total, count


def sequence_hidden(model, embedding, record, device):
    ids, hidden, positions, labels, mask = aligned_inputs(record, device)
    hidden = hidden.to(dtype=embedding.weight.dtype)
    with autocast(device):
        result = model(ids, hidden, positions, embedding)
    return result, labels, mask


def evaluate(model, embedding, head, files, config, args):
    if not files:
        return None
    torch = runtime().torch
    was_training = model.training
    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for path in files:
            record = load_record(path, config, args.max_length)
            hidden, labels, mask = sequence_hidden(model, embedding, record, args.device)
            loss, count = chunked_ce(hidden, head, labels, mask, chunk_tokens=args.logits_chunk)
            total += loss
            tokens += count
    model.train(was_training)
    return {"ce": total / tokens, "tokens": tokens, "sequences": len(files)}


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


def run(args):
    rt = runtime()
    torch = rt.torch
    if (not 3 <= args.max_length <= 2048 or not 1 <= args.steps <= 1000
            or not math.isfinite(args.lr) or args.lr <= 0
            or not 1 <= args.grad_accum <= 64 or not 1 <= args.logits_chunk <= 256):
        raise TrainingError("Invalid bounds: length 3..2048, steps 1..1000, positive lr, accumulation 1..64, logits chunk 1..256")
    checkpoint = Checkpoint(args.model)
    output_paths(args.output, checkpoint.path)
    report = {
        "mode": "export_stock" if args.export_stock else "first_step_teacher_forcing",
        "config": vars(args).copy(), "torch": torch.__version__,
        "transformers": rt.transformers.__version__, "python": platform.python_version(),
        "optimizer_steps": 0, "train_steps": [], "eval_before": None, "eval_after": None,
        "alignment": "ids[1:-1], hidden[:-2], positions[:-2], labels/mask[2:]",
        "precision": {
            "masters": "float32", "export": "bfloat16", "xpu_compute": "float16_autocast",
            "cpu_compute": "float32 (tiny tests only)",
            "lm_head": "frozen dequantized RTN INT4-g128, FP16 input/scales/effective weights",
            "limitations": "No MTP-core QAT; dense GEMM != bitwise INT4 kernel; no recursive-depth or acceptance claim",
        },
    }
    if args.export_stock:
        if args.train_dir or args.eval_dir:
            raise TrainingError("--export-stock does not consume train/eval data")
        export_mtp(checkpoint.mtp_state(), checkpoint, args.output, report)
        return report
    if not args.train_dir:
        raise TrainingError("Training requires --train-dir")
    device = torch.device(args.device)
    if device.type not in ("cpu", "xpu") or (device.type == "xpu" and not torch.xpu.is_available()):
        raise TrainingError("Use CPU for tiny synthetic tests or an available inference-host XPU")
    if device.type == "cpu" and sum(math.prod(s) for s in checkpoint.shapes.values()) > 5_000_000:
        raise TrainingError("Stock-size CPU training is forbidden; use inference-host XPU after unloading verifier")
    train, heldout = capture_sets(args.train_dir, args.eval_dir, checkpoint.config, args.max_length)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = build_native_mtp(checkpoint.config, checkpoint.mtp_state(), args.device)
    embedding, head = frozen_heads(checkpoint, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    scaler = torch.amp.GradScaler("xpu", init_scale=128.0) if device.type == "xpu" else None
    report["eval_before"] = evaluate(model, embedding, head, heldout, checkpoint.config, args)
    order, cursor = [], 0
    model.train()
    for step in range(args.steps):
        records = []
        for _ in range(args.grad_accum):
            if cursor == len(order):
                order = list(train)
                rng.shuffle(order)
                cursor = 0
            records.append(load_record(order[cursor], checkpoint.config, args.max_length))
            cursor += 1
        tokens = sum(int(record["loss_mask"][2:].sum().item()) for record in records)
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for record in records:
            hidden, labels, mask = sequence_hidden(model, embedding, record, args.device)
            loss, _ = chunked_ce(hidden, head, labels, mask, chunk_tokens=args.logits_chunk,
                                 backward=True, normalizer=tokens, scaler=scaler)
            total += loss
        if scaler is not None:
            scaler.unscale_(optimizer)
        # Abort rather than silently report a skipped/overflowed update as a step.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        result = {"step": step + 1, "ce": total / tokens, "tokens": tokens,
                  "sequences": len(records), "grad_norm": grad_norm.item()}
        report["train_steps"].append(result)
        report["optimizer_steps"] += 1
        print(json.dumps(result, allow_nan=False), flush=True)
    report["eval_after"] = evaluate(model, embedding, head, heldout, checkpoint.config, args)
    state = {"mtp." + key: value for key, value in model.state_dict().items()}
    export_mtp(state, checkpoint, args.output, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--model", required=True, help="Local GPTQ checkpoint directory; no downloads")
    result.add_argument("--train-dir")
    result.add_argument("--eval-dir")
    result.add_argument("--output", required=True, help="New mtp-only .safetensors file outside stock model")
    result.add_argument("--export-stock", action="store_true", help="Identity export only; no optimizer or target load")
    result.add_argument("--steps", type=int, default=10, help="Optimizer updates (1..1000)")
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
