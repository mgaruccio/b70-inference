#!/usr/bin/env python3
"""Opt-in mixed FP16-target/BF16-DSpark overlay for the pinned B70 XPU image.

B70_DSPARK_BF16=1 python -P /overlay.py [--root /exported/vllm]
Keep B70_DSPARK_BF16=1 and VLLM_USE_V2_MODEL_RUNNER=1 in the server environment.
Only K=7, greedy (default) or native probabilistic proposals, standard rejection,
explicit draft kv_cache_dtype=bfloat16, TP=PP=DP=CP=1 are supported. Target GPTQ Int4/sym/G128
FP16 compute and FP8 KV stay untouched. No adaptive verification or top-k draft
truncation. This does not install/promote a launcher or promise a speedup.

Pinned image: vllm/vllm-openai-xpu@sha256:
7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
Sources inspected from its installed package (not /workspace/vllm), vLLM 73029d424.
Official research (lead, 2026-09-10):
https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/tree/b9a5dbdf03bc999c6c73c426b19c2d9041cea393
https://github.com/vllm-project/vllm/tree/73029d424/vllm/v1/worker/gpu/spec_decode/dspark
https://github.com/vllm-project/vllm/blob/73029d424/vllm/model_executor/models/qwen3_dspark.py
Probability/precision helpers inspected from the same installed image (2026-09-11):
https://github.com/vllm-project/vllm/blob/73029d424/vllm/model_executor/layers/logits_processor.py
https://github.com/vllm-project/vllm/blob/73029d424/vllm/v1/worker/gpu/sample/gumbel.py
https://github.com/vllm-project/vllm/blob/73029d424/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py

The loader's default-dtype context already creates implicit parameters in BF16.
The confidence projection remains explicitly FP32, as upstream (unused with
fixed verification); only activation boundaries to shared FP16 vocab modules
are cast. The draft-local logits processor must not inherit the target head dtype:
BF16 Markov logits plus FP16 base logits promote to FP32 in either sampling mode.
Probabilistic drafts cache that sum losslessly in FP32 before temperature, using
native Gumbel draws and standard probability-ratio rejection. Temperature is
applied once on sampling and once on reading the unscaled cache for verification.
The draft KV cache stays explicitly BF16, while its FlashAttention implementation
selector is normalized to 'auto' because the pinned XPU kernel rejects the explicit
'bfloat16' selector.
All touched sources AND their dtype/sharing/routing helpers are whole-file
pinned. All transforms compile before any writes; only exact full replay is
accepted. Offline tests are not the lead's real XPU/API acceptance gate.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path


PINNED_SHA256 = {
    "_custom_ops.py": "eb439f4656903c11c8789cf8fd0eb86300f69540cf07e29a0564a9476aedecb3",
    "config/speculative.py": "4e3d0a9b93f54fc0f25897e8c7bba5412e3851dd819950964c201ed6376e08b7",
    "v1/worker/gpu/spec_decode/speculator.py": "9587daa4a84743e051b4bca586767b1c4d6ee9c53850db98d5727591316bea7f",
    "v1/worker/gpu/spec_decode/dflash/speculator.py": "4a6b86232fa98eeb62143c469f2f5789c94d3af0f80851f623ddaa8c78e23f7b",
    "model_executor/models/qwen3_dflash.py": "51d2b55f883d34393e9b732952fc866ae0308b2a39cfcc58243e0990802afcfd",
    "model_executor/models/qwen3_dspark.py": "77d9707385fe0ac15651ccd442864989694c57262076e9a2dbc69c8557dc31dd",
    "v1/worker/gpu/spec_decode/dspark/utils.py": "56f6b81f5712817689e24088b2c5302d5832fd4e6d5b2630d7e28be73c84b298",
    # Unmodified dependencies: loader dtype context, sharing, cache and V2 routing.
    "model_executor/model_loader/__init__.py": "984790cf22eb999b796b3ecc1c2ed3d930b1f0eef1f126140dcc786a042d3ec2",
    "model_executor/model_loader/base_loader.py": "a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08",
    "model_executor/model_loader/utils.py": "6cd928c158703de94223b056d25fb58b38db49befb2f3c7aee1b5e97daca1e86",
    "model_executor/models/utils.py": "745e79363f54b16666ea40e8ac84bf9ddcba630bd04710e305a1dc1d74333954",
    "model_executor/models/registry.py": "fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7",
    "v1/worker/gpu/spec_decode/eagle/utils.py": "65d882b8fb476eb0dd6161247346cd7b0392b22832abf36fbb11e795979f9e28",
    "v1/worker/gpu/spec_decode/eagle/eagle3_utils.py": "43e5ba8d3997712878ead63e9d5d5bc7c3e656650b8aa6a41b3e19f84d2f6487",
    "v1/worker/gpu/spec_decode/dspark/speculator.py": "ed020e35bccf6281382f24132717e2f48575acc35d9f26acba2e930cb8f2e8a7",
    "v1/worker/gpu/spec_decode/dflash/cudagraph.py": "2d4afe57efb13586decbd42943873467a3fe618e2baf3441e344e8a864721ac6",
    "v1/worker/gpu/spec_decode/__init__.py": "1f1f84bfe6f4f5af3a7e21a7a8716ecf7b9013da1d03327668a3ee36afc24c32",
    "v1/worker/gpu/model_runner.py": "5c72998ac7a3aa97ff31ed5e5576d9bcd27c79656e3989e3c00e5adf6e3bdfa6",
    "config/cache.py": "1c33754731efad7000b55034a49aae05ee9e2f54763d2dd6c4672224f501e1d5",
    "v1/kv_cache_interface.py": "4e9238f1b2a5a1fe9a10322d8046d11023dc82800b536761b7579aedab4a1008",
    "model_executor/layers/attention/attention.py": "64a1b218f04a7178d9d41c12fc6ae08f6cff648031bf118216019a48d15572c7",
    "v1/attention/backends/flash_attn.py": "5d9c676d8d03ec4e583183f9f42fac33c223768fdc3d6088dff97b5ac0cd4c3a",
    "platforms/xpu.py": "5951dc8c6e25f57c80fa4f4c5a55b8d5b39799a7fc578e36f9c98e89b9897af0",
    # Unmodified probability path: native head projection, cache write/read and rejection dispatch.
    "model_executor/layers/logits_processor.py": "6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f",
    "v1/worker/gpu/sample/gumbel.py": "3ec1df510bdad13e8b0a457b5b67c98affdf6e57369e56cb471246cc8f40dd34",
    "v1/worker/gpu/spec_decode/rejection_sampler.py": "e20adc9b6c8a62a5be232bae15a1e361e898142966a68ea3ef244b39ffc5ce44",
    "v1/worker/gpu/spec_decode/rejection_sampler_utils.py": "20ca2e5ac34e9ef93dca388bed72a00d4ff67d569ec6ec2393b21a92d74a1fae",
}

CONFIG_HELPERS = '''
# B70_DSPARK_BF16: private controls for this pinned experiment only.
import os as _b70_os

_B70_DSPARK_BF16 = _b70_os.environ.get("B70_DSPARK_BF16") == "1"


def _b70_dspark_enabled(spec):
    return _B70_DSPARK_BF16 and spec is not None and spec.method == "dspark"


def _b70_dspark_requested(spec):
    if not _B70_DSPARK_BF16:
        return False
    import torch
    from vllm.platforms import current_platform

    if (not current_platform.is_xpu() or spec.method != "dspark"
            or _b70_os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1"
            or spec.model is None):
        raise ValueError("B70 DSpark requires XPU, explicit method=dspark, standalone draft and V2=1")
    target = spec.target_model_config
    if (target is None or target.dtype != torch.float16
            or target.quantization not in ("gptq", "auto_gptq")
            or target.hf_config.architectures != ["Qwen3_5ForConditionalGeneration"]
            or target.hf_config.model_type != "qwen3_5"
            or target.get_hidden_size() != 5120 or target.get_vocab_size() != 248320
            or target.hf_text_config.num_hidden_layers != 64
            or getattr(target, "head_dtype", None) not in (None, torch.float16)):
        raise ValueError("B70 DSpark requires the Qwen3.8-27B FP16-compute GPTQ target and FP16 head")
    qc = getattr(target.hf_config, "quantization_config", {})
    if (not isinstance(qc, dict) or qc.get("quant_method") != "gptq"
            or qc.get("bits") != 4 or qc.get("group_size") != 128
            or qc.get("sym") is not True or qc.get("desc_act") is not False
            or qc.get("lm_head") is not False):
        raise ValueError("B70 DSpark requires target GPTQ Int4 symmetric G128, no act-order/head quantization")
    if (spec.quantization is not None or spec.num_speculative_tokens != 7
            or spec.draft_sample_method not in ("greedy", "probabilistic")
            or spec.rejection_sample_method != "standard"
            or spec.enable_adaptive_verification or spec.dspark_draft_topk is not None
            or spec.use_heterogeneous_vocab or spec.use_local_argmax_reduction
            or spec.kv_cache_dtype != "bfloat16"):
        raise ValueError("B70 DSpark requires dense draft, K=7, greedy/probabilistic, standard rejection, "
                         "fixed verification, full vocab and explicit draft kv_cache_dtype=bfloat16")
    if (spec.revision not in (None, "b9a5dbdf03bc999c6c73c426b19c2d9041cea393")
            or (spec.revision is None and not _b70_os.path.isdir(spec.model))):
        raise ValueError("B70 DSpark requires the pinned draft revision or its local snapshot")
    return True


def _b70_dspark_validate(spec):
    if not _b70_dspark_requested(spec):
        return
    import torch

    draft = spec.draft_model_config
    hf = draft.hf_config
    required = {
        "model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 5,
        "vocab_size": 248320, "draft_vocab_size": 248320, "intermediate_size": 17408,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "block_size": 7, "markov_rank": 256, "markov_head_type": "vanilla",
        "projector_type": "dspark", "attention_mode": "gqa", "mask_token_id": 248070,
        "target_layer_ids": [5, 19, 33, 47, 61], "layer_types": ["full_attention"] * 5,
        "enable_confidence_head": True, "confidence_head_with_markov": True,
    }
    dc = getattr(hf, "dflash_config", None)
    if (draft.dtype != torch.bfloat16 or draft.quantization is not None
            or getattr(hf, "quantization_config", None) is not None
            or str(getattr(hf, "dtype", None)) not in ("bfloat16", "torch.bfloat16")
            or hf.architectures not in (["DSparkDraftModel"], ["Qwen3DSparkModel"])
            or any(getattr(hf, key, None) != value for key, value in required.items())
            or not isinstance(dc, dict)
            or any(value != required[key] for key, value in dc.items() if key in required)
            or dc.get("target_layer_ids") != required["target_layer_ids"]
            or dc.get("mask_token_id") != required["mask_token_id"]
            or getattr(hf, "eagle_config", {})
            or getattr(hf, "eagle_aux_hidden_state_layer_ids", None)
            or getattr(hf, "dspark_target_layer_ids", required["target_layer_ids"]) != required["target_layer_ids"]
            or getattr(hf, "target_hidden_size", None) not in (None, 5120)
            or not dc.get("use_aux_hidden_state", getattr(hf, "use_aux_hidden_state", True))
            or dc.get("use_swa", False) or dc.get("causal", False)
            or dc.get("sample_from_anchor", False)
            or getattr(hf, "is_causal", False) or getattr(hf, "sliding_window", None) is not None
            or getattr(hf, "use_sliding_window", False) or getattr(hf, "tie_word_embeddings", False)
            or getattr(hf, "sample_from_anchor", True) is not True
            or getattr(hf, "dspark_draft_topk", None) is not None):
        raise ValueError("B70 DSpark requires the native BF16 RadixArk Qwen3.8-27B DSpark config")


def _b70_dspark_runtime(vllm_config):
    spec = vllm_config.speculative_config
    if not _B70_DSPARK_BF16:
        return
    if spec is None:
        raise ValueError("B70 DSpark requires a speculative config")
    _b70_dspark_validate(spec)
    pc = vllm_config.parallel_config
    if (not vllm_config.use_v2_model_runner
            or any(getattr(pc, key, 1) != 1 for key in (
                "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size"))
            or spec.draft_parallel_config.tensor_parallel_size != 1
            or vllm_config.cache_config.cache_dtype not in ("fp8", "fp8_e4m3")
            or vllm_config.cache_config.kv_cache_dtype_skip_layers
            or getattr(vllm_config, "lora_config", None) is not None
            or getattr(spec.attention_backend, "name", None) not in (None, "FLASH_ATTN")):
        raise ValueError("B70 DSpark requires actual V2, TP=PP=DP=CP=1, FP8 target KV, "
                         "no LoRA and FlashAttention draft")


def _b70_dspark_dtype(vllm_config):
    if _b70_dspark_enabled(vllm_config.speculative_config):
        return vllm_config.speculative_config.draft_model_config.dtype
    return vllm_config.model_config.dtype

'''

MODEL_HELPERS = '''
# B70_DSPARK_BF16: startup-only validation; no tensor hooks or capture-time sync.
from vllm.config.speculative import _b70_dspark_enabled, _b70_dspark_dtype


def _b70_dspark_loaded(model, target_embed, target_head):
    import torch

    if (model.model.embed_tokens is not target_embed or model.lm_head is not target_head
            or target_embed is None or target_head is None
            or target_embed.weight.dtype != torch.float16 or target_head.weight.dtype != torch.float16
            or getattr(model, "has_own_embed_tokens", False)
            or getattr(model, "has_own_lm_head", False)
            or model.draft_id_to_target_id is not None):
        raise ValueError("B70 DSpark must alias the target's dense FP16 embedding and head")
    shared = {id(p) for m in (target_embed, target_head) for p in m.parameters()}
    for name, param in model.named_parameters():
        # Confidence computation is explicitly FP32 upstream; keep it that way.
        dtype = (torch.float16 if id(param) in shared else
                 torch.float32 if name.startswith("model.confidence_head.") else torch.bfloat16)
        if param.dtype != dtype:
            raise ValueError(f"B70 DSpark unexpected parameter dtype: {name} {param.dtype}, expected {dtype}")
    for name in ("_fused_kv_weight", "_hidden_norm_weight", "_k_norm_weights"):
        if getattr(model.model, name).dtype != torch.bfloat16:
            raise ValueError(f"B70 DSpark requires BF16 fused context parameter {name}")
    for layer in model.model.layers:
        attn = layer.self_attn.attn
        if (attn.dtype != torch.bfloat16 or attn.kv_cache_dtype != "bfloat16"
                or attn.kv_cache_torch_dtype != torch.bfloat16
                or attn.backend.name != "FLASH_ATTN"):
            raise ValueError("B70 DSpark requires BF16 query/context/cache and FlashAttention")
        impl = getattr(attn, "impl", None)
        if getattr(impl, "kv_cache_dtype", None) not in ("bfloat16", "auto"):
            raise ValueError("B70 DSpark requires unquantized native FlashAttention KV dispatch")
        # XPU 0.1.14.1 accepts only 'auto' for an unquantized cache. Keep
        # the resolved BF16 allocation/spec; normalize the implementation
        # selector shared by context updates and query forward.
        impl.kv_cache_dtype = "auto"

'''

CACHE_CHECK = '''
        if _b70_dspark_enabled(self.speculative_config):
            # V2 set_attn precedes allocation: validate specs, not empty tensors.
            for name in self.model.get_draft_kv_cache_layer_names():
                group = next(g for g in kv_cache_config.kv_cache_groups if name in g.layer_names)
                spec = group.kv_cache_spec
                if hasattr(spec, "kv_cache_specs"):
                    spec = spec.kv_cache_specs[name]
                if spec.dtype != torch.bfloat16:
                    raise ValueError("B70 DSpark cache allocation spec must be BF16")
            if self.hidden_states.dtype != torch.bfloat16:
                raise ValueError("B70 DSpark context buffer must be BF16")
'''


def transformations():
    return {
        "config/speculative.py": [
            ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n' + CONFIG_HELPERS),
            ('    def __post_init__(self):\n',
             '    def __post_init__(self):\n        _b70_dspark_requested(self)\n'),
            ('                    dtype=self.target_model_config.dtype,\n',
             '                    dtype=("bfloat16" if _b70_dspark_requested(self)\n'
             '                           else self.target_model_config.dtype),\n'),
            ('                    config_format=self.target_model_config.config_format,\n                )\n',
             '                    config_format=self.target_model_config.config_format,\n                )\n'
             '                _b70_dspark_validate(self)\n'),
        ],
        "v1/worker/gpu/spec_decode/speculator.py": [
            ('from vllm.config.compilation import CUDAGraphMode\n',
             'from vllm.config.compilation import CUDAGraphMode\n'
             'from vllm.config.speculative import _b70_dspark_dtype, _b70_dspark_runtime, _b70_dspark_enabled\n'),
            ('        self.dtype = vllm_config.model_config.dtype\n',
             '        _b70_dspark_runtime(vllm_config)\n'
             '        self.dtype = _b70_dspark_dtype(vllm_config)\n'),
            ('        return vllm_config.model_config.head_dtype, 0.0\n',
             '        if (_b70_dspark_enabled(vllm_config.speculative_config)\n'
             '                and vllm_config.speculative_config.draft_sample_method == "probabilistic"):\n'
             '            # BF16 Markov + FP16 base logits promote to FP32. Do not round q.\n'
             '            return torch.float32, 0.0\n'
             '        return vllm_config.model_config.head_dtype, 0.0\n'),
        ],
        "v1/worker/gpu/spec_decode/dflash/speculator.py": [
            ('from vllm.config.compilation import CUDAGraphMode\n',
             'from vllm.config.compilation import CUDAGraphMode\n'
             'from vllm.config.speculative import _b70_dspark_enabled\n'),
            ('        config = copy.copy(super().attn_vllm_config)\n',
             '        config = copy.copy(super().attn_vllm_config)\n'
             '        if _b70_dspark_enabled(self.speculative_config):\n'
             '            config.model_config = self.draft_model_config\n'
             '            config.cache_config = replace(\n'
             '                self.vllm_config.cache_config, cache_dtype="bfloat16"\n'
             '            )\n'),
            ('        self.draft_kv_cache_group_ids = [\n',
             CACHE_CHECK + '\n        self.draft_kv_cache_group_ids = [\n'),
            ('        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)\n',
             '        if _b70_dspark_enabled(self.speculative_config):\n'
             '            for layer in self.model.model.layers:\n'
             '                cache = layer.self_attn.attn.kv_cache\n'
             '                if cache.numel() and cache.dtype != torch.bfloat16:\n'
             '                    raise ValueError("B70 DSpark allocated query cache must be BF16")\n'
             '        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)\n'),
        ],
        "model_executor/models/qwen3_dflash.py": [
            ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n' + MODEL_HELPERS),
            ('        self.config = vllm_config.speculative_config.draft_model_config.hf_config\n',
             '        self.config = vllm_config.speculative_config.draft_model_config.hf_config\n'
             '        self._b70_dspark_bf16 = _b70_dspark_enabled(vllm_config.speculative_config)\n'),
            ('            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),\n',
             '            torch.zeros(self.config.hidden_size, dtype=_b70_dspark_dtype(vllm_config)),\n'),
            ('                params_dtype=vllm_config.model_config.dtype,\n',
             '                params_dtype=_b70_dspark_dtype(vllm_config),\n'),
            ('        embeds = self.embed_tokens(input_ids)\n',
             '        embeds = self.embed_tokens(input_ids)\n'
             '        if self._b70_dspark_bf16:\n'
             '            embeds = embeds.to(torch.bfloat16)\n'),
            ('        result = self.model.fc(hidden_states)\n',
             '        if self.model._b70_dspark_bf16:\n'
             '            hidden_states = hidden_states.to(torch.bfloat16)\n'
             '        result = self.model.fc(hidden_states)\n'),
            ('        ops.rms_norm(\n'
             '            all_k_normed,\n'
             '            all_k,\n'
             '            self._k_norm_weights,\n'
             '            self._rms_norm_eps,\n'
             '        )\n',
             '        if self._b70_dspark_bf16:\n'
             '            # Pinned XPU grouped RMSNorm reuses layer 0; select each layer weight.\n'
             '            for i in range(all_k.shape[0]):\n'
             '                ops.rms_norm(\n'
             '                    all_k_normed[i],\n'
             '                    all_k[i],\n'
             '                    self._k_norm_weights[i],\n'
             '                    self._rms_norm_eps,\n'
             '                )\n'
             '        else:\n'
             '            ops.rms_norm(\n'
             '                all_k_normed,\n'
             '                all_k,\n'
             '                self._k_norm_weights,\n'
             '                self._rms_norm_eps,\n'
             '            )\n'),
            ('            kv_cache = attn.kv_cache\n',
             '            kv_cache = attn.kv_cache\n'
             '            if self._b70_dspark_bf16 and kv_cache.dtype != torch.bfloat16:\n'
             '                raise ValueError("B70 DSpark allocated context cache must be BF16")\n'),
        ],
        "model_executor/models/qwen3_dspark.py": [
            ('        self.target_vocab_size = vllm_config.model_config.get_vocab_size()\n',
             '        if self.model._b70_dspark_bf16:\n'
             '            # Keep each projection in its own dtype, not the target head dtype.\n'
             '            self.logits_processor.head_dtype = None\n'
             '        self.target_vocab_size = vllm_config.model_config.get_vocab_size()\n'),
            ('        return self.logits_processor(self.lm_head, hidden_states)\n',
             '        if self.model._b70_dspark_bf16:\n'
             '            hidden_states = hidden_states.to(self.lm_head.weight.dtype)\n'
             '        return self.logits_processor(self.lm_head, hidden_states)\n'),
            ('        for name, loaded_weight in weights:\n',
             '        for name, loaded_weight in weights:\n'
             '            if self.model._b70_dspark_bf16 and (\n'
             '                loaded_weight.dtype != torch.bfloat16\n'
             '                or any(part in name for part in ("embed_tokens", "lm_head", "t2d", "d2t"))\n'
             '            ):\n'
             '                raise ValueError(f"B70 DSpark requires BF16-only draft weights without vocab/mappings: {name}")\n'),
            ('            model_weights[name] = loaded_weight\n',
             '            if self.model._b70_dspark_bf16 and name in model_weights:\n'
             '                raise ValueError(f"B70 DSpark duplicate checkpoint tensor: {name}")\n'
             '            model_weights[name] = loaded_weight\n'),
            ('        orig_to_new_substr = {"mask_embedding": None}\n',
             '        if self.model._b70_dspark_bf16 and (len(model_weights) != 62 or not includes_confidence_head):\n'
             '            raise ValueError("B70 DSpark requires the 62-tensor RadixArk checkpoint including confidence head")\n'
             '        orig_to_new_substr = {"mask_embedding": None}\n'),
        ],
        "v1/worker/gpu/spec_decode/dspark/utils.py": [
            ('    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n',
             '    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)\n'
             '    from vllm.config.speculative import _b70_dspark_enabled, _b70_dspark_runtime\n'
             '    _b70_dspark_runtime(vllm_config)\n'
             '    if _b70_dspark_enabled(speculative_config) and draft_vllm_config.quant_config is not None:\n'
             '        raise ValueError("B70 DSpark draft must remain unquantized")\n'),
            ('    return draft_model\n',
             '    if _b70_dspark_enabled(speculative_config):\n'
             '        from vllm.model_executor.models.qwen3_dflash import _b70_dspark_loaded\n'
             '        _b70_dspark_loaded(draft_model, target_embed, target_lm_head)\n'
             '    return draft_model\n'),
        ],
    }


def prepare(sources):
    """Validate full pristine/full exact replay, including unchanged dependencies."""
    if set(sources) != set(PINNED_SHA256):
        raise RuntimeError("source set differs from pinned manifest")
    edits = transformations()
    originals, states = {}, set()
    for name, digest in PINNED_SHA256.items():
        source = sources[name]
        if hashlib.sha256(source.encode()).hexdigest() == digest:
            originals[name] = source
            if name in edits:
                states.add("pristine")
            continue
        restored = source
        for old, new in reversed(edits.get(name, [])):
            if restored.count(new) != 1:
                raise RuntimeError(f"{name}: incompatible source or damaged overlay")
            restored = restored.replace(new, old, 1)
        if hashlib.sha256(restored.encode()).hexdigest() != digest:
            raise RuntimeError(f"{name}: source fingerprint differs from pinned image")
        originals[name] = restored
        states.add("patched")
    if len(states) != 1:
        raise RuntimeError("refusing partially applied/mixed overlay; restore pristine sources")
    result = {}
    for name, source in originals.items():
        for old, new in edits.get(name, []):
            if source.count(old) != 1:
                raise RuntimeError(f"{name}: source anchor missing or ambiguous")
            source = source.replace(old, new, 1)
        compile(source, name, "exec")  # Not merely AST parsing: before ANY writes.
        result[name] = source
    if states == {"patched"} and result != sources:
        raise RuntimeError("overlay replay differs from exact pinned transformation")
    return result


def apply(root):
    if os.environ.get("B70_DSPARK_BF16") != "1":
        raise RuntimeError("explicit B70_DSPARK_BF16=1 is required; no files changed")
    root = Path(root).resolve(strict=True)
    paths = {name: root / name for name in PINNED_SHA256}
    for path in paths.values():
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"refusing source outside package root: {path}")
    # read_bytes avoids accepting newline normalization as an exact source match.
    sources = {name: path.read_bytes().decode("utf-8") for name, path in paths.items()}
    patched = prepare(sources)
    changed = [name for name in sources if sources[name] != patched[name]]
    for name in changed:
        paths[name].write_bytes(patched[name].encode("utf-8"))
    print(f"B70 DSpark BF16 overlay: {'applied' if changed else 'already applied'} ({root})")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="exported vllm package root for offline tests")
    args = parser.parse_args()
    if os.environ.get("B70_DSPARK_BF16") != "1":
        parser.error("explicit B70_DSPARK_BF16=1 is required; no files changed")
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("cannot locate installed vllm package")
        root = Path(next(iter(spec.submodule_search_locations)))
    apply(root)


if __name__ == "__main__":
    main()
