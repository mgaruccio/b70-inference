"""Disposable, pinned Qwen3.8-27B MTP training runtime (not a launcher).

B70_MTP_WEIGHTS: complete stock-keyed BF16 mtp-only safetensors overlay. The
original loader still remaps/ packs weights, then the existing RTN hooks run.

B70_MTP_CAPTURE_DIR: existing directory with capture-request.json containing
{"name": "example-0001", "input_ids": [...], "loss_mask": [false, ..., true]}.
Replay those EXACT tokens as a single /v1/completions token-ID prompt with
max_tokens=1, speculation OFF, prefix caching OFF, async scheduling OFF,
TP=PP=DP=1 and max_num_seqs=1. Generate on-policy sequences separately using
native MTP first; do not re-tokenize the generated token IDs. Text only; no
LoRA, embedding inputs, KV transfer, or preemption. Chunked prefill is allowed.
The mask denotes assistant tokens at token positions (the trainer shifts by 2).

After the response, <name>.pt contains ONLY input_ids/positions [T] int64,
loss_mask [T] bool, target_last_hidden_states [T,5120] in the runtime dtype.
No file is published for incomplete sequences. Change control between requests;
never change it during a replay. Limits per process: 32768 tokens/sequence,
128 sequences, B70_MTP_CAPTURE_MAX_TOKENS total (default 131072, max 1048576).

The runner seam is AFTER _model_forward AND its set_forward_context, before
logit-row selection. Qwen3_5Model inherits Qwen3NextModel.forward's final norm;
Qwen has no get_mtp_target_hidden_states override. These are the same tensor
rows the native proposer's non-speculative/prefill branch passes as
hidden_states[:num_scheduled_tokens], not compute_logits' sampled subset.
Replay excludes rejected drafts by construction and slices away graph padding.
Copies run in Python outside compilation/graph execution, never in forward.
Replay is causal teacher forcing with the same weights/backend, NOT a claim of
bitwise parity between prefill kernels and on-policy speculative decode kernels.
That numerical parity and the stock-overlay identity gate require the live E2E.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch


# Checkpoint revision 9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e: 424699392 BF16
# parameters. Deliberately pinned, not a permissive generic adapter.
MTP_SHAPES = {
    "mtp.fc.weight": (5120, 10240),
    "mtp.layers.0.input_layernorm.weight": (5120,),
    "mtp.layers.0.mlp.down_proj.weight": (5120, 17408),
    "mtp.layers.0.mlp.gate_proj.weight": (17408, 5120),
    "mtp.layers.0.mlp.up_proj.weight": (17408, 5120),
    "mtp.layers.0.post_attention_layernorm.weight": (5120,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj.weight": (5120, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (12288, 5120),
    "mtp.layers.0.self_attn.v_proj.weight": (1024, 5120),
    "mtp.norm.weight": (5120,),
    "mtp.pre_fc_norm_embedding.weight": (5120,),
    "mtp.pre_fc_norm_hidden.weight": (5120,),
}
HIDDEN_SIZE = 5120
MAX_SEQUENCE_TOKENS = 32768
MAX_SEQUENCES = 128


def _finite(tensor):
    # Avoid an extra full-sized boolean tensor for the largest MTP matrix.
    flat = tensor.reshape(-1)
    return all(torch.isfinite(chunk).all().item() for chunk in flat.split(1048576))


def overlay_weights(weights):
    """Substitute HF tensors before remapping/packing; unset returns the iterable itself."""
    path = os.environ.get("B70_MTP_WEIGHTS")
    if path is None:
        return weights
    if not path or Path(path).suffix != ".safetensors":
        raise ValueError("B70_MTP_WEIGHTS must name a complete .safetensors overlay")

    from safetensors import safe_open

    replacement = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != MTP_SHAPES.keys():
            raise ValueError(
                "MTP overlay keys mismatch: "
                f"missing={sorted(MTP_SHAPES.keys() - keys)}, "
                f"unexpected={sorted(keys - MTP_SHAPES.keys())}"
            )
        # Validate the entire header before materializing any weights.
        for name, shape in MTP_SHAPES.items():
            view = handle.get_slice(name)
            if tuple(view.get_shape()) != shape or view.get_dtype() != "BF16":
                raise ValueError(f"MTP overlay shape/dtype mismatch: {name}; expected {shape} BF16")
        for name in MTP_SHAPES:
            tensor = handle.get_tensor(name)
            if not _finite(tensor):
                raise ValueError(f"MTP overlay contains nonfinite values: {name}")
            replacement[name] = tensor

    def substitute():
        seen = set()
        for name, weight in weights:
            if name.startswith("mtp."):
                if name not in MTP_SHAPES or name in seen:
                    raise ValueError(f"unexpected/duplicate checkpoint MTP key: {name}")
                if tuple(weight.shape) != MTP_SHAPES[name] or weight.dtype != torch.bfloat16:
                    raise ValueError(f"checkpoint MTP shape/dtype mismatch: {name}")
                seen.add(name)
                yield name, replacement.pop(name)
            else:
                # Shared embeddings, LM head, and target are untouched, including identity.
                yield name, weight
        if seen != MTP_SHAPES.keys():
            raise ValueError(f"checkpoint missing MTP keys: {sorted(MTP_SHAPES.keys() - seen)}")

    return substitute()


def _validate_runner(runner):
    def reject(condition, message):
        if condition:
            raise ValueError("B70 MTP capture requires " + message)

    reject(runner.speculative_config is not None, "target-only replay (disable speculation)")
    reject(runner.cache_config.enable_prefix_caching, "prefix caching disabled")
    reject(runner.cache_config.kv_sharing_fast_prefill, "KV-sharing fast prefill disabled")
    reject(runner.scheduler_config.max_num_seqs != 1, "max_num_seqs=1")
    reject(runner.scheduler_config.async_scheduling, "async scheduling disabled")
    parallel = runner.parallel_config
    reject(any(getattr(parallel, key) != 1 for key in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"
    )), "TP=PP=DP=1")
    reject(parallel.use_ubatching, "microbatching disabled")
    reject(runner.vllm_config.kv_transfer_config is not None, "KV transfer disabled")
    reject(runner.vllm_config.ec_transfer_config is not None, "encoder transfer disabled")
    reject(runner.use_aux_hidden_state_outputs or runner.is_pooling_model,
           "ordinary Qwen target final hidden states")
    reject(runner.model_config.hf_text_config.hidden_size != HIDDEN_SIZE,
           "the pinned 5120-wide Qwen target")
    model = runner.get_model()
    reject(type(model).__name__ not in (
        "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"
    ), "the pinned dense Qwen3_5 target class")
    reject(hasattr(model, "get_mtp_target_hidden_states"),
           "Qwen's unchanged post-final-norm hidden-state source")


class ReplayCapture:
    """One complete sequential teacher replay at a time, with a bounded CPU cache."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError("B70_MTP_CAPTURE_DIR must be an existing directory")
        self.max_tokens = int(os.environ.get("B70_MTP_CAPTURE_MAX_TOKENS", "131072"))
        if not 1 <= self.max_tokens <= 1048576:
            raise ValueError("B70_MTP_CAPTURE_MAX_TOKENS must be in [1, 1048576]")
        self.total_tokens = 0
        self.sequences = 0
        self.active = None
        self.completed_request = None

    def _start(self, req_id, request, computed):
        control = self.root / "capture-request.json"
        if not control.is_file():
            return False
        data = json.loads(control.read_text())
        if set(data) != {"name", "input_ids", "loss_mask"}:
            raise ValueError("capture control requires exactly name, input_ids, loss_mask")
        name, ids, mask = data["name"], data["input_ids"], data["loss_mask"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("invalid capture name")
        if not isinstance(ids, list) or not 3 <= len(ids) <= MAX_SEQUENCE_TOKENS:
            raise ValueError("capture requires 3..32768 complete sequence tokens")
        if any(type(token) is not int or not 0 <= token < 248320 for token in ids):
            raise ValueError("capture input_ids must be pinned-vocabulary integer tokens")
        if (not isinstance(mask, list) or len(mask) != len(ids)
                or any(type(value) is not bool for value in mask) or not any(mask[2:])):
            raise ValueError("capture loss_mask must be a token-position bool mask with shifted labels")
        if request.prompt_token_ids != ids or computed != 0:
            raise ValueError("capture requires the complete exact token-ID prompt starting at position 0")
        if self.sequences >= MAX_SEQUENCES or self.total_tokens + len(ids) > self.max_tokens:
            raise ValueError("B70 MTP capture budget exceeded; no truncated sequences are saved")
        if (self.root / f"{name}.pt").exists() or (self.root / f"{name}.pt.partial").exists():
            raise FileExistsError(f"capture already exists: {name}")
        self.active = {
            "request": req_id, "control": data, "rows": 0, "chunks": [], "dtype": None,
        }
        return True

    def step(self, runner, scheduler_output, hidden_states):
        _validate_runner(runner)
        req_ids = runner.input_batch.req_ids
        if len(req_ids) != 1 or runner.input_batch.num_reqs != 1:
            raise ValueError("B70 MTP capture requires exactly one active request")
        req_id = req_ids[0]
        if scheduler_output.scheduled_spec_decode_tokens:
            raise ValueError("B70 MTP capture cannot include speculative/rejected positions")
        if scheduler_output.scheduled_cached_reqs.resumed_req_ids:
            raise ValueError("B70 MTP capture does not support resumed/preempted replay")
        request = runner.requests[req_id]
        if (request.mm_features or request.prompt_embeds is not None
                or request.lora_request is not None
                or (request.prompt_is_token_ids is not None and not all(request.prompt_is_token_ids))):
            raise ValueError("B70 MTP capture supports text token IDs only, no multimodal/LoRA/embeds")
        if request.sampling_params is None or request.sampling_params.max_tokens != 1:
            raise ValueError("B70 MTP replay requires max_tokens=1; capture prompt rows only")
        if req_id == self.completed_request:
            return
        computed = int(runner.input_batch.num_computed_tokens_cpu[0])
        if self.active is None and not self._start(req_id, request, computed):
            return
        active = self.active
        if active["request"] != req_id or active["rows"] != computed:
            raise ValueError("B70 MTP replay interrupted, reordered, or missing prefix rows")
        if json.loads((self.root / "capture-request.json").read_text()) != active["control"]:
            raise ValueError("capture control changed during replay")
        data = active["control"]
        n = scheduler_output.num_scheduled_tokens[req_id]
        if (n <= 0 or n != scheduler_output.total_num_scheduled_tokens
                or computed + n > len(data["input_ids"])):
            raise ValueError("invalid replay chunk length; only complete prompt tokens are allowed")
        if (not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2
                or hidden_states.shape[0] < n or hidden_states.shape[1] != HIDDEN_SIZE
                or hidden_states.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
            raise ValueError("invalid target final hidden-state shape/dtype")

        # Same source/slicing as propose_draft_token_ids' non-speculative branch.
        # Blocking independent CPU copies; never mutate model or persistent input buffers.
        ids = runner.input_ids.gpu[:n].detach().to(device="cpu", copy=True)
        positions = runner._get_positions(n).detach().to(device="cpu", copy=True)
        if ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 1:
            raise ValueError("invalid runtime token IDs")
        if positions.dtype != torch.int64:
            raise ValueError("invalid runtime position dtype")
        if positions.ndim == 2 and positions.shape[0] == 3:
            if not torch.equal(positions, positions[0:1].expand_as(positions)):
                raise ValueError("non-text M-RoPE positions are unsupported")
            positions = positions[0].clone()
        if positions.ndim != 1 or not torch.equal(positions, torch.arange(computed, computed + n)):
            raise ValueError("non-contiguous or non-text replay positions")
        ids = ids.to(torch.int64)
        if ids.tolist() != data["input_ids"][computed:computed + n]:
            raise ValueError("runtime token IDs do not match the complete replay sequence")
        hidden = hidden_states[:n].detach().to(device="cpu", copy=True)
        if not _finite(hidden):
            raise ValueError("nonfinite target hidden states")
        if active["dtype"] is not None and active["dtype"] != hidden.dtype:
            raise ValueError("target hidden-state dtype changed within replay")
        active["dtype"] = hidden.dtype
        active["chunks"].append((ids, positions, hidden))
        active["rows"] += n
        if active["rows"] != len(data["input_ids"]):
            return

        chunks = active["chunks"]
        payload = {
            "input_ids": torch.cat([chunk[0] for chunk in chunks]),
            "positions": torch.cat([chunk[1] for chunk in chunks]),
            "target_last_hidden_states": torch.cat([chunk[2] for chunk in chunks]),
            "loss_mask": torch.tensor(data["loss_mask"], dtype=torch.bool),
        }
        output = self.root / f"{data['name']}.pt"
        partial = output.with_suffix(".pt.partial")
        with partial.open("xb") as handle:
            torch.save(payload, handle)
        # C1, local disposable directory: publish only a complete torch.save payload.
        # link is no-clobber, unlike rename/replace; an existing example is an error.
        os.link(partial, output)
        partial.unlink()
        self.total_tokens += active["rows"]
        self.sequences += 1
        self.completed_request = req_id
        self.active = None


def capture_replay_step(runner, scheduler_output, model_output):
    """Called ONLY from GPUModelRunner.execute_model after leaving forward context."""
    root = os.environ.get("B70_MTP_CAPTURE_DIR")
    if root is None:
        return
    from vllm.forward_context import is_forward_context_available

    if torch.compiler.is_compiling() or is_forward_context_available():
        raise RuntimeError("B70 MTP capture must run outside compilation/XPU graph forward context")
    collector = getattr(runner, "_b70_mtp_training_capture", None)
    if collector is None:
        collector = ReplayCapture(root)
        runner._b70_mtp_training_capture = collector
    if collector.root != Path(root):
        raise ValueError("B70_MTP_CAPTURE_DIR cannot change during a run")
    collector.step(runner, scheduler_output, model_output)
