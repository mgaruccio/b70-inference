"""Released Qwen3.8 EAGLE-4 weights on vLLM's EAGLE3 *chain* proposer.

This is an independent interoperability adapter, not the Eagle adaptive-tree
harness. It imports vLLM's Apache-2.0 EAGLE3 layer implementation; no Eagle source
or kernels are vendored. The checkpoint remains subject to its own license.

Math/config references (inspected 2026-10-21):
https://github.com/eagle-hq/EAGLE-4/blob/v0.1.0/docs/architecture.md
https://github.com/eagle-hq/EAGLE-4/blob/v0.1.0/eagle_harness/vllm/models.py
https://huggingface.co/EAGLE-Corp/Qwen3.8-27B-EAGLE4/tree/f58a5985223040552ad404fba98f0dd27f5f676a

Only TP1/C1, eager, FP16 compute, dense draft weights and unquantized KV are
supported here. BF16 checkpoint tensors are converted by vLLM's weight loader.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader


ARCHITECTURE = "LlamaForCausalLMEagle4"
CHECKPOINT_REVISION = "f58a5985223040552ad404fba98f0dd27f5f676a"


def validate_config(config) -> None:
    """Fail closed on structures other than the released dense Qwen3.8 draft."""
    required = {
        "hidden_size": 5120,
        "target_hidden_size": 5120,
        "intermediate_size": 16384,
        "vocab_size": 248320,
        "draft_vocab_size": 248320,
        "num_hidden_layers": 1,
        "num_attention_heads": 40,
        "num_key_value_heads": 40,
        "head_dim": 128,
        "num_aux_hidden_states": 3,
        "hidden_act": "silu",
        "fc_norm": True,
        "norm_output": True,
        "layer_types": ["sliding_attention"],
        "sliding_window": 4096,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
    }
    for name, value in required.items():
        if getattr(config, name, None) != value:
            raise ValueError(f"EAGLE4 requires {name}={value!r}")
    if getattr(config, "architectures", None) != [ARCHITECTURE]:
        raise ValueError(f"EAGLE4 requires architecture {ARCHITECTURE}")
    if getattr(config, "efficient_lm_head_config", None) != {
        "type": "low_rank", "rank": 128, "num_candidates": 128
    }:
        raise ValueError("EAGLE4 requires the released rank-128, 128-candidate head")
    eagle = getattr(config, "eagle_config", {})
    if (eagle.get("eagle_aux_hidden_state_layer_ids") != [3, 31, 59]
            or eagle.get("use_aux_hidden_state") is not True
            or eagle.get("norm_before_fc", False)):
        raise ValueError("EAGLE4 requires auxiliary layers [3, 31, 59] and per-chunk norms")
    for name in ("norm_before_fc", "norm_before_residual", "attn_norm", "enable_moe"):
        if getattr(config, name, False):
            raise ValueError(f"EAGLE4 does not support {name}")
    if getattr(config, "use_rope", True) is not True:
        raise ValueError("EAGLE4 requires rotary positions")
    rope = getattr(config, "rope_parameters", {})
    if rope.get("rope_type") != "default" or rope.get("rope_theta") != 10000000.0:
        raise ValueError("EAGLE4 requires the released default rotary configuration")
    if getattr(config, "logit_scale", 1.0) != 1.0:
        raise ValueError("EAGLE4 requires unscaled candidate logits")


def validate_runtime(vllm_config) -> None:
    spec = vllm_config.speculative_config
    if spec is None or spec.method != "eagle3" or spec.parallel_drafting:
        raise ValueError("EAGLE4 requires method=eagle3 with sequential chain drafting")
    if (getattr(spec, "draft_sample_method", "greedy") != "greedy"
            or getattr(spec, "rejection_sample_method", "standard") != "standard"
            or getattr(spec, "use_local_argmax_reduction", False)):
        raise ValueError("EAGLE4 supports greedy proposals and standard verification only")
    parallel = vllm_config.parallel_config
    if any(getattr(parallel, name, 1) != 1 for name in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "decode_context_parallel_size", "prefill_context_parallel_size",
    )):
        raise ValueError("EAGLE4 supports TP=PP=DP=CP=1 only")
    if vllm_config.scheduler_config.max_num_seqs != 1:
        raise ValueError("EAGLE4 supports concurrency 1 only")
    target, draft = spec.target_model_config, spec.draft_model_config
    if any(config.dtype != torch.float16 or not config.enforce_eager
           for config in (target, draft)):
        raise ValueError("EAGLE4 requires eager FP16 target and draft compute")
    if spec.quantization is not None or draft.quantization is not None:
        raise ValueError("EAGLE4 draft weights must be unquantized")
    if (vllm_config.cache_config.cache_dtype not in ("auto", "float16")
            or spec.kv_cache_dtype not in (None, "auto", "float16")):
        raise ValueError("EAGLE4 requires unquantized FP16 KV")


def checkpoint_shapes(config) -> dict[str, tuple[int, ...]]:
    """Tensor names/shapes from the pinned safetensors header, not name heuristics."""
    h, v, i = config.hidden_size, config.vocab_size, config.intermediate_size
    rank = config.efficient_lm_head_config["rank"]
    shapes = {
        "embed_tokens.weight": (v, h),
        "lm_head.weight": (v, h),
        "efficient_lm_head.weight": (v, rank),
        "efficient_lm_head_down_proj.weight": (rank, h),
        "fc.weight": (h, config.target_hidden_size * config.num_aux_hidden_states),
        "norm.weight": (h,),
        "layers.0.hidden_norm.weight": (h,),
        "layers.0.input_layernorm.weight": (h,),
        "layers.0.post_attention_layernorm.weight": (h,),
        "layers.0.self_attn.o_proj.weight": (h, h),
        "layers.0.mlp.down_proj.weight": (h, i),
        "layers.0.mlp.gate_proj.weight": (i, h),
        "layers.0.mlp.up_proj.weight": (i, h),
    }
    for chunk in range(config.num_aux_hidden_states):
        shapes[f"fc_norm.{chunk}.weight"] = (config.target_hidden_size,)
    for projection in ("q", "k", "v"):
        heads = (config.num_attention_heads if projection == "q"
                 else config.num_key_value_heads)
        shapes[f"layers.0.self_attn.{projection}_proj.weight"] = (
            heads * config.head_dim, 2 * h
        )
    return shapes


def candidate_ids(scores: torch.Tensor, count: int) -> torch.Tensor:
    """Top candidates, cutoff ties going to lower token IDs (also on XPU).

    torch.topk alone has backend-dependent ties. Use it only to find the cutoff;
    integer priorities then put strictly larger scores first, in vocabulary
    order, followed by cutoff ties in vocabulary order. This matches the
    documented NVIDIA unsorted-topk membership/order without CUDA kernels.
    """
    if scores.ndim != 2 or not 0 < count <= scores.shape[1]:
        raise ValueError("candidate count must be in [1, vocab_size] for 2D scores")
    cutoff = scores.topk(count, dim=-1, sorted=False).values.amin(-1, keepdim=True)
    vocab = scores.shape[1]
    token_ids = torch.arange(vocab, device=scores.device)
    priority = torch.where(
        scores > cutoff, token_ids,
        torch.where(scores == cutoff, token_ids + vocab, 2 * vocab),
    )
    return priority.topk(count, dim=-1, largest=False, sorted=True).indices


def candidate_logits(
    hidden: torch.Tensor,
    down_proj: torch.Tensor,
    efficient_head: torch.Tensor,
    full_head: torch.Tensor,
    count: int,
    vocab_size: int,
) -> torch.Tensor:
    """Learned low-rank selection, then FP32 exact dot products only for candidates."""
    low = hidden @ down_proj.T
    scores = low.to(efficient_head.dtype) @ efficient_head[:vocab_size].T
    ids = candidate_ids(scores, count)
    selected_weights = full_head[ids].float()
    exact = (selected_weights * hidden.float().unsqueeze(1)).sum(-1)
    logits = torch.full(
        (hidden.shape[0], vocab_size), -torch.inf,
        dtype=torch.float32, device=hidden.device,
    )
    return logits.scatter_(1, ids, exact)


class LlamaForCausalLMEagle4(Eagle3LlamaForCausalLM):
    # The actual release includes both. Never silently replace them with target weights.
    has_own_lm_head = True
    has_own_embed_tokens = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.speculative_config.draft_model_config.hf_config
        validate_config(config)
        validate_runtime(vllm_config)
        # Checkpoints name layer outputs; pinned vLLM captures layer inputs.
        # Publish the next-layer indices before the runner configures aux capture.
        config.eagle_aux_hidden_state_layer_ids = tuple(
            layer + 1 for layer in config.eagle_config["eagle_aux_hidden_state_layer_ids"]
        )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.num_candidates = config.efficient_lm_head_config["num_candidates"]
        rank = config.efficient_lm_head_config["rank"]
        # TP1 only: ordinary dense parameters use vLLM's default weight loader.
        self.efficient_lm_head = nn.Linear(
            rank, config.vocab_size, bias=False, dtype=torch.float16
        )
        self.efficient_lm_head_down_proj = nn.Linear(
            config.hidden_size, rank, bias=False, dtype=torch.float16
        )
        self.draft_id_to_target_id = None  # released head uses the full target vocabulary

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return candidate_logits(
            hidden_states, self.efficient_lm_head_down_proj.weight,
            self.efficient_lm_head.weight, self.lm_head.weight,
            self.num_candidates, self.config.vocab_size,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expected = checkpoint_shapes(self.config)
        normalized = {}
        seen = set()
        # Validate the whole checkpoint before loading anything. In particular,
        # neither candidate-head tensor belongs under model.; trunk MLP down_proj does.
        for name, tensor in weights:
            key = name.removeprefix("model.")
            if key not in expected:
                raise ValueError(f"Unexpected EAGLE4 checkpoint tensor: {name}")
            if key in seen:
                raise ValueError(f"Duplicate EAGLE4 checkpoint tensor: {name}")
            if tuple(tensor.shape) != expected[key] or not tensor.is_floating_point():
                raise ValueError(
                    f"Invalid EAGLE4 tensor {name}: expected floating {expected[key]}, "
                    f"got {tuple(tensor.shape)} {tensor.dtype}"
                )
            seen.add(key)
            root_head = key in {
                "lm_head.weight", "efficient_lm_head.weight",
                "efficient_lm_head_down_proj.weight",
            }
            normalized[key if root_head else "model." + key] = tensor
        missing = expected.keys() - seen
        if missing:
            raise ValueError(f"Missing EAGLE4 checkpoint tensors: {sorted(missing)}")
        # Delegates trunk Q/K/V and gate/up stacking to the pinned EAGLE3 LlamaModel.
        return AutoWeightsLoader(self).load_weights(normalized.items())
