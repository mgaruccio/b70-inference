#!/usr/bin/env python3
"""Research-only mixed-dtype DFlash2 overlay for the pinned B70 XPU image.

Run B70_DFLASH2_BF16=1 python /overlay.py before starting vLLM. --root accepts
an exported vllm package directory for offline tests. All seven source files must
match the actual 73029d424 image, or this exact overlay, before any are written.
No target parameter is cast, copied, or quantized. vLLM's temporary draft vocab
allocations before sharing are unchanged (upstream #53612); no vocab remapping.

Only the legacy DFlashProposer, TP=PP=DP=1, K=7, unquantized native BF16 draft,
greedy draft proposals and STANDARD rejection are supported. Target sampling is
untouched. The legacy proposer otherwise ignores DFlash2's learned selector;
the bounded greedy walk here follows the pinned V2 DFlash2Speculator, without
porting its probabilistic sampler. Audit is opt-in, synchronous, eager-only.

Primary sources consulted by the lead / this implementation:
https://github.com/vllm-project/vllm/issues/55250 (FP16 draft collapse)
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/dedf8df68adfb1afeaf7b7480c0a0243108177b4/config.json
https://github.com/vllm-project/vllm-xpu-kernels/pull/579 (layerwise K RMSNorm)
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
The loader sets torch's default dtype from draft ModelConfig, but leaves the
VllmConfig model_config pointing at the target: explicit params_dtype and the
Attention 'auto' KV dtype therefore need separate draft-only overrides.
"""

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path


PINNED_SHA256 = {
    "config/speculative.py": "4e3d0a9b93f54fc0f25897e8c7bba5412e3851dd819950964c201ed6376e08b7",
    "config/vllm.py": "06b918459d5ab7694bd32fbf4e2b16a576757502d737db7197e8b0d2a1076c6e",
    "v1/spec_decode/llm_base_proposer.py": "7b404e8de2a0068510cbda29d3b36c16fb7515a5e5260ab26be9f381692d3c91",
    "v1/spec_decode/dflash.py": "38c10f7b922905562f1af9b64eb98be71c398a7c79e94b40124f4496cab905af",
    "model_executor/models/qwen3_dflash.py": "51d2b55f883d34393e9b732952fc866ae0308b2a39cfcc58243e0990802afcfd",
    "model_executor/models/qwen3_dflash2.py": "c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f",
    "model_executor/models/registry.py": "fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7",
}

CONFIG_HELPERS = '''
# B70_DFLASH2_BF16: deliberately private, pinned research-only controls.
import os as _b70_os

_B70_DFLASH2_BF16 = _b70_os.environ.get("B70_DFLASH2_BF16") == "1"
_B70_DFLASH2_AUDIT = _b70_os.environ.get("B70_DFLASH2_AUDIT") == "1"


def _b70_dflash2_requested(spec):
    if not _B70_DFLASH2_BF16:
        if _B70_DFLASH2_AUDIT:
            raise ValueError("B70_DFLASH2_AUDIT requires B70_DFLASH2_BF16=1")
        return False
    import torch
    from vllm.platforms import current_platform

    if not current_platform.is_xpu() or spec.method != "dflash":
        raise ValueError("B70 DFlash2 BF16 requires XPU and explicit method=dflash")
    if _b70_os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "0":
        raise ValueError("B70 DFlash2 BF16 supports only the legacy model runner")
    target = spec.target_model_config
    if target.dtype != torch.float16 or target.quantization not in ("gptq", "auto_gptq"):
        raise ValueError("B70 DFlash2 requires the FP16-compute GPTQ target")
    if getattr(target, "head_dtype", None) not in (None, torch.float16):
        raise ValueError("B70 DFlash2 requires a dense FP16 shared target head")
    if (spec.quantization is not None or spec.num_speculative_tokens != 7
            or spec.draft_sample_method != "greedy"
            or spec.rejection_sample_method != "standard"
            or spec.use_heterogeneous_vocab or spec.use_local_argmax_reduction
            or spec.kv_cache_dtype not in ("auto", "bfloat16")):
        raise ValueError("B70 DFlash2 supports K=7, greedy draft, standard rejection, "
                         "same vocab and explicit draft kv_cache_dtype=auto/bfloat16 only")
    if _B70_DFLASH2_AUDIT and (not target.enforce_eager or spec.enforce_eager is False):
        raise ValueError("B70_DFLASH2_AUDIT requires eager target AND draft execution")
    return True


def _b70_dflash2_enabled(spec):
    return (_B70_DFLASH2_BF16 and spec is not None
            and spec.method == "dflash"
            and spec.draft_model_config.hf_config.architectures == ["DFlash2DraftModel"])


def _b70_dflash2_validate(spec):
    if not _b70_dflash2_requested(spec):
        return
    import torch

    draft = spec.draft_model_config
    hf = draft.hf_config
    dc = getattr(hf, "dflash_config", {})
    if (not _b70_dflash2_enabled(spec) or draft.dtype != torch.bfloat16
            or draft.quantization is not None
            or str(getattr(hf, "dtype", None)) not in ("bfloat16", "torch.bfloat16")
            or hf.num_hidden_layers != 5 or hf.hidden_size != 5120
            or hf.vocab_size != spec.target_model_config.get_vocab_size()
            or spec.target_model_config.get_hidden_size() != 5120
            or dc.get("block_size") != 8 or dc.get("selector_rank") != 256
            or dc.get("selector_top_k") != 16 or dc.get("conv_kernel_size") != 2
            or dc.get("conv_group_size") != 16
            or dc.get("target_layer_ids") != [5, 19, 33, 47, 61]
            or getattr(hf, "draft_vocab_size", hf.vocab_size) not in (None, hf.vocab_size)
            or not dc.get("use_aux_hidden_state", getattr(hf, "use_aux_hidden_state", True))
            or getattr(hf, "tie_word_embeddings", False)):
        raise ValueError("B70 overlay requires the native BF16 Qwen3.8-27B DFlash2 config")


def _b70_dflash2_dtype(vllm_config):
    spec = vllm_config.speculative_config
    if _b70_dflash2_enabled(spec):
        return spec.draft_model_config.dtype
    return vllm_config.model_config.dtype

'''

MODEL_HELPERS = '''
# B70_DFLASH2_BF16: no hooks, tensor retention, or synchronization when audit is off.
from vllm.config.speculative import (
    _B70_DFLASH2_AUDIT, _b70_dflash2_dtype, _b70_dflash2_enabled,
)


def _b70_audit_tensor(name, tensor, dtype=None):
    if not _B70_DFLASH2_AUDIT:
        return
    if torch.compiler.is_compiling():
        raise RuntimeError("B70 DFlash2 audit cannot run under torch.compile")
    if dtype is not None and tensor.dtype != dtype:
        raise RuntimeError(f"B70 DFlash2 {name}: expected {dtype}, got {tensor.dtype}")
    finite = bool(torch.isfinite(tensor).all().item())
    print(f"B70_DFLASH2_AUDIT {name}: dtype={tensor.dtype} "
          f"shape={tuple(tensor.shape)} finite={finite}", flush=True)
    if not finite:
        raise RuntimeError(f"B70 DFlash2 nonfinite tensor: {name}")

'''

DRAFT_METHODS = '''
    # B70_DFLASH2_BF16: cast activation boundaries, never shared modules/weights.
    def combine_hidden_states(self, hidden_states):
        if self.model._b70_dflash2_bf16:
            _b70_audit_tensor("target_aux", hidden_states, torch.float16)
            hidden_states = hidden_states.to(torch.bfloat16)
            _b70_audit_tensor("aux_projection_input", hidden_states, torch.bfloat16)
        result = super().combine_hidden_states(hidden_states)
        _b70_audit_tensor("context_hidden", result, torch.bfloat16)
        return result

    def _b70_head_input(self, hidden_states):
        if not self.model._b70_dflash2_bf16:
            return hidden_states
        if not getattr(self, "_b70_shared_checked", False):
            raise RuntimeError("B70 DFlash2 shared weights were not validated after load")
        _b70_audit_tensor("draft_head_hidden", hidden_states, torch.bfloat16)
        head_input = hidden_states.to(self.lm_head.weight.dtype)
        _b70_audit_tensor("shared_head_input", head_input, torch.float16)
        return head_input

    def compute_logits(self, hidden_states):
        # Inherited DFlash LogitsProcessor includes scale/vocab mapping. Do not
        # substitute an F.linear path or confuse it with candidate softcapping.
        logits = super().compute_logits(self._b70_head_input(hidden_states))
        if logits is not None:
            _b70_audit_tensor("logits", logits)
        return logits

    def load_weights(self, weights):
        def checked():
            for name, weight in weights:
                if self.model._b70_dflash2_bf16:
                    if any(part in name for part in ("embed_tokens", "lm_head", "d2t", "t2d")):
                        raise ValueError("B70 DFlash2 expects shared target vocab weights, no remapping")
                    if weight.dtype != torch.bfloat16:
                        raise ValueError(f"B70 DFlash2 checkpoint tensor {name} is not native BF16")
                yield name, weight
        return super().load_weights(checked())

    def _b70_validate_shared(self, target):
        if not self.model._b70_dflash2_bf16:
            return
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod
        from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

        embedding = getattr(target.model, "embed_tokens", None)
        if (self.lm_head is not target.lm_head or self.model.embed_tokens is not embedding
                or self.draft_id_to_target_id is not None):
            raise RuntimeError("B70 DFlash2 must share the exact target head/embedding objects")
        shared_ids = set()
        for name, module in (("shared_lm_head", self.lm_head), ("shared_embedding", embedding)):
            if (not isinstance(module.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod))
                    or module.weight.dtype != torch.float16 or module.weight.ndim != 2):
                raise RuntimeError(f"B70 DFlash2 {name} must be dense FP16")
            shared_ids.update(id(p) for p in module.parameters())
            if _B70_DFLASH2_AUDIT:
                print(f"B70_DFLASH2_AUDIT {name}: shared=True dtype={module.weight.dtype} "
                      f"shape={tuple(module.weight.shape)}", flush=True)
        for name, param in self.named_parameters():
            if id(param) in shared_ids:
                continue
            if param.dtype != torch.bfloat16:
                raise RuntimeError(f"B70 DFlash2 loaded parameter {name} is {param.dtype}, not BF16")
            _b70_audit_tensor("parameter." + name, param, torch.bfloat16)
        for layer in self.model.layers:
            attn = layer.self_attn.attn
            if attn.dtype != torch.bfloat16 or attn.kv_cache_torch_dtype != torch.bfloat16:
                raise RuntimeError("B70 DFlash2 attention compute AND KV cache must be BF16")
            # XPU 0.1.14.1 accepts only 'auto' for an unquantized cache. Keep
            # the already-resolved BF16 allocation/spec; normalize dispatch only.
            if attn.kv_cache_dtype not in ("bfloat16", "auto"):
                raise RuntimeError("B70 DFlash2 expected unquantized draft KV dispatch")
            if attn.impl.kv_cache_dtype not in ("bfloat16", "auto"):
                raise RuntimeError("B70 DFlash2 expected unquantized backend KV dispatch")
            attn.kv_cache_dtype = attn.impl.kv_cache_dtype = "auto"
            if _B70_DFLASH2_AUDIT:
                print("B70_DFLASH2_AUDIT draft_kv: storage=torch.bfloat16 dispatch=auto", flush=True)
        for name in ("_fused_kv_weight", "_k_norm_weights", "_hidden_norm_weight"):
            tensor = getattr(self.model, name)
            if tensor.dtype != torch.bfloat16:
                raise RuntimeError(f"B70 DFlash2 context buffer {name} is not BF16")
            _b70_audit_tensor(name, tensor, torch.bfloat16)
        # LogitsProcessor.head_dtype may otherwise cast the entire shared head.
        for processor in (self.logits_processor, self.candidate_logits_processor):
            if processor.head_dtype not in (None, torch.float16):
                raise RuntimeError("B70 DFlash2 LogitsProcessor would copy/cast the target head")
        self._b70_shared_checked = True

'''

SELECTOR_METHOD = '''
    # B70_DFLASH2_BF16: legacy proposer lacks the V2 DFlash2 selector path.
    def _sample_draft_tokens(self, hidden_states, sampling_metadata):
        if not _b70_dflash2_enabled(self.speculative_config):
            return super()._sample_draft_tokens(hidden_states, sampling_metadata)
        if (self.speculative_config.draft_sample_method != "greedy"
                or self.speculative_config.rejection_sample_method != "standard"
                or self.num_speculative_tokens != 7):
            raise ValueError("B70 DFlash2 supports fixed K=7 greedy draft + standard rejection only")
        # Same predecessor-conditioned greedy walk as pinned V2's
        # _selector_walk_kernel with SAMPLE_PROBABILISTIC=False. No change to
        # the target sampler or standard deterministic-proposal rejection.
        steps = self.num_speculative_tokens
        hidden = hidden_states.reshape(-1, steps, hidden_states.shape[-1])
        batch = hidden.shape[0]
        selector = self.model.model.candidate_selector
        ids, unary = self.model.compute_candidates(hidden_states)
        ids = ids.reshape(batch, steps, selector.top_k)
        unary = unary.reshape_as(ids)
        anchors = self.input_ids[:batch * (steps + 1):steps + 1]
        scores = selector(ids, unary, hidden, anchors)
        rows = torch.arange(batch, device=hidden.device)
        previous = torch.zeros(batch, dtype=torch.long, device=hidden.device)
        path = []
        for step in range(steps):
            previous = scores[rows, step, previous].argmax(dim=-1)
            path.append(ids[rows, step, previous])
        return torch.stack(path, dim=1).flatten(), None

'''

NORM_OLD = '''        all_k_normed = torch.empty_like(all_k)
        ops.rms_norm(
            all_k_normed,
            all_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return all_k_normed
'''
NORM_NEW = '''        all_k_normed = torch.empty_like(all_k)
        # B70_DFLASH2_BF16: XPU kernels 0.1.14.1 read only stacked weight row zero.
        if (getattr(self, "_b70_dflash2_bf16", False)
                and all_k.device.type == "xpu"):
            if self._k_norm_weights.shape != (all_k.shape[0], all_k.shape[-1]):
                raise ValueError("DFlash2 context K norm weights must match layers/head_dim")
            for layer_idx in range(all_k.shape[0]):
                ops.rms_norm(
                    all_k_normed[layer_idx], all_k[layer_idx],
                    self._k_norm_weights[layer_idx], self._rms_norm_eps,
                )
        else:
            ops.rms_norm(
                all_k_normed, all_k, self._k_norm_weights, self._rms_norm_eps,
            )
        return all_k_normed
'''


def transformations():
    """Literal unique source anchors; every edited file is also fingerprinted."""
    return {
        "config/speculative.py": [
            ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n' + CONFIG_HELPERS),
            ('                    dtype=self.target_model_config.dtype,\n',
             '                    dtype=("bfloat16" if _b70_dflash2_requested(self)\n'
             '                           else self.target_model_config.dtype),\n'),
            ('                    config_format=self.target_model_config.config_format,\n                )\n',
             '                    config_format=self.target_model_config.config_format,\n                )\n'
             '                _b70_dflash2_validate(self)\n'),
        ],
        "config/vllm.py": [
            ('        if self._is_dflash2_draft():\n'
             '            unsupported.append("dflash2 drafts")\n',
             '        if self._is_dflash2_draft():\n'
             '            from vllm.config.speculative import _b70_dflash2_enabled\n'
             '            # Only this guarded overlay supplies the legacy selector walk.\n'
             '            if not _b70_dflash2_enabled(self.speculative_config):\n'
             '                unsupported.append("dflash2 drafts")\n'),
        ],
        "v1/spec_decode/llm_base_proposer.py": [
            ('class SpecDecodeBaseProposer:\n',
             'from vllm.config.speculative import _b70_dflash2_dtype, _b70_dflash2_enabled\n\n\n'
             'class SpecDecodeBaseProposer:\n'),
            ('        self.dtype = vllm_config.model_config.dtype\n',
             '        self.dtype = _b70_dflash2_dtype(vllm_config)\n'
             '        if _b70_dflash2_enabled(self.speculative_config):\n'
             '            pc = vllm_config.parallel_config\n'
             '            if (device.type != "xpu" or pc.tensor_parallel_size != 1\n'
             '                    or pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1\n'
             '                    or vllm_config.lora_config is not None):\n'
             '                raise ValueError("B70 DFlash2 requires single-XPU TP=PP=DP=1, no LoRA")\n'),
            ('        self._maybe_share_lm_head(target_language_model)\n',
             '        self._maybe_share_lm_head(target_language_model)\n'
             '        if _b70_dflash2_enabled(self.speculative_config):\n'
             '            self.model._b70_validate_shared(target_language_model)\n'),
        ],
        "v1/spec_decode/dflash.py": [
            ('from vllm.config import VllmConfig\n',
             'from vllm.config import VllmConfig\n'
             'from vllm.config.speculative import _b70_dflash2_enabled\n'),
            ('        base = super()._create_draft_vllm_config()\n',
             '        base = super()._create_draft_vllm_config()\n'
             '        if _b70_dflash2_enabled(self.speculative_config):\n'
             '            # auto resolves against TARGET ModelConfig (FP16) in Attention.\n'
             '            # Make only the draft cache explicit; never mutate target config.\n'
             '            base = replace(base, cache_config=replace(\n'
             '                base.cache_config, cache_dtype="bfloat16"))\n'),
            ('    @override\n    def _warn_if_multimodal(self):\n',
             SELECTOR_METHOD + '    @override\n    def _warn_if_multimodal(self):\n'),
        ],
        "model_executor/models/qwen3_dflash.py": [
            ('logger = init_logger(__name__)\n', 'logger = init_logger(__name__)\n' + MODEL_HELPERS),
            ('        self.config = vllm_config.speculative_config.draft_model_config.hf_config\n',
             '        self.config = vllm_config.speculative_config.draft_model_config.hf_config\n'
             '        self._b70_dflash2_bf16 = _b70_dflash2_enabled(vllm_config.speculative_config)\n'),
            ('torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype)',
             'torch.zeros(self.config.hidden_size, dtype=_b70_dflash2_dtype(vllm_config))'),
            ('                params_dtype=vllm_config.model_config.dtype,\n',
             '                params_dtype=_b70_dflash2_dtype(vllm_config),\n'),
            ('        embeds = self.embed_tokens(input_ids)\n',
             '        embeds = self.embed_tokens(input_ids)\n'
             '        if self._b70_dflash2_bf16:\n'
             '            embeds = embeds.to(torch.bfloat16)\n'),
            (NORM_OLD, NORM_NEW),
            ('        num_ctx = context_states.shape[0]\n',
             '        if self._b70_dflash2_bf16:\n'
             '            _b70_audit_tensor("context_kv_input", context_states, torch.bfloat16)\n'
             '        num_ctx = context_states.shape[0]\n'),
            ('        all_k_normed = self._normalize_context_k(all_k)\n',
             '        all_k_normed = self._normalize_context_k(all_k)\n'
             '        if self._b70_dflash2_bf16:\n'
             '            _b70_audit_tensor("context_k", all_k_normed, torch.bfloat16)\n'
             '            _b70_audit_tensor("context_v", all_v, torch.bfloat16)\n'),
        ],
        "model_executor/models/qwen3_dflash2.py": [
            ('    DFlashQwen3Model,\n',
             '    DFlashQwen3Model,\n    _b70_audit_tensor,\n'
             '    _b70_dflash2_dtype,\n    _B70_DFLASH2_AUDIT,\n'),
            ('\n            params_dtype=vllm_config.model_config.dtype,\n',
             '\n            params_dtype=_b70_dflash2_dtype(vllm_config),\n'),
            ('                params_dtype=vllm_config.model_config.dtype,\n',
             '                params_dtype=_b70_dflash2_dtype(vllm_config),\n'),
            ('    blocks = hidden_states.unflatten(-1, (num_groups, group_size))\n',
             '    _b70_audit_tensor("conv.hidden", hidden_states, torch.bfloat16)\n'
             '    _b70_audit_tensor("conv.delta", delta, torch.bfloat16)\n'
             '    _b70_audit_tensor("conv.base", base, torch.bfloat16)\n'
             '    blocks = hidden_states.unflatten(-1, (num_groups, group_size))\n'),
            ('    output = coefficients[:, 0] * blocks\n',
             '    _b70_audit_tensor("conv.coefficients", coefficients, torch.bfloat16)\n'
             '    output = coefficients[:, 0] * blocks\n'),
            ('    return output.flatten(-2)\n',
             '    _b70_audit_tensor("conv.output", output, torch.bfloat16)\n'
             '    return output.flatten(-2)\n'),
            ('        hidden = self.hidden_projection(hidden_states)\n        return _score_edges(\n',
             '        _b70_audit_tensor("selector.hidden", hidden_states, torch.bfloat16)\n'
             '        hidden = self.hidden_projection(hidden_states)\n'
             '        _b70_audit_tensor("selector.projection", hidden, torch.bfloat16)\n'
             '        scores = _score_edges(\n'),
            ('            self.top_k,\n        )\n',
             '            self.top_k,\n        )\n'
             '        _b70_audit_tensor("selector.scores", scores)\n'
             '        return scores\n'),
            ('        return super().embed_input_ids(input_ids) * self.input_embedding_scale\n',
             '        embeds = super().embed_input_ids(input_ids)\n'
             '        if self._b70_dflash2_bf16:\n'
             '            embeds = embeds.to(torch.bfloat16)\n'
             '        result = embeds * self.input_embedding_scale\n'
             '        _b70_audit_tensor("input_embedding", result, torch.bfloat16)\n'
             '        return result\n'),
            ('    def compute_candidates(\n', DRAFT_METHODS + '    def compute_candidates(\n'),
            ('        return self.candidate_logits_processor.get_top_k_tokens(\n'
             '            self.lm_head, hidden_states, self.model.candidate_selector.top_k\n        )\n',
             '        ids, values = self.candidate_logits_processor.get_top_k_tokens(\n'
             '            self.lm_head, self._b70_head_input(hidden_states),\n'
             '            self.model.candidate_selector.top_k\n        )\n'
             '        _b70_audit_tensor("candidate.logits", values)\n'
             '        return ids, values\n'),
        ],
        "model_executor/models/registry.py": [],
    }


def prepare(sources):
    """Validate complete pristine or complete patched input; never accept a mix."""
    edits = transformations()
    originals = {}
    states = set()
    for relative, digest in PINNED_SHA256.items():
        source = sources[relative]
        if hashlib.sha256(source.encode()).hexdigest() == digest:
            originals[relative] = source
            if edits[relative]:
                states.add("pristine")
            continue
        restored = source
        for old, new in reversed(edits[relative]):
            if restored.count(new) != 1:
                raise RuntimeError(f"{relative}: incompatible source or damaged overlay anchor")
            restored = restored.replace(new, old, 1)
        if hashlib.sha256(restored.encode()).hexdigest() != digest:
            raise RuntimeError(f"{relative}: source fingerprint differs from pinned image")
        originals[relative] = restored
        states.add("patched")
    if len(states) != 1:
        raise RuntimeError("refusing partially applied/mixed overlay; restore pristine source files")
    result = {}
    for relative, source in originals.items():
        for old, new in edits[relative]:
            if source.count(old) != 1:
                raise RuntimeError(f"{relative}: source anchor missing or ambiguous")
            source = source.replace(old, new, 1)
        ast.parse(source, filename=relative)
        result[relative] = source
    if states == {"patched"} and result != sources:
        raise RuntimeError("overlay replay differs; refusing modified patched sources")
    return result


def apply(root):
    root = Path(root).resolve(strict=True)
    paths = {name: root / name for name in PINNED_SHA256}
    for path in paths.values():
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise RuntimeError(f"refusing source outside package root: {path}")
    sources = {name: path.read_text() for name, path in paths.items()}
    patched = prepare(sources)  # All fingerprints, anchors and syntax BEFORE writes.
    changed = [name for name in sources if sources[name] != patched[name]]
    for name in changed:
        paths[name].write_text(patched[name])
    print(f"B70 DFlash2 BF16 overlay: {'applied' if changed else 'already applied'} ({root})")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="exported vllm package root (offline fixtures)")
    args = parser.parse_args()
    if os.environ.get("B70_DFLASH2_BF16") != "1":
        parser.error("explicit B70_DFLASH2_BF16=1 is required; no files changed")
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("cannot locate installed vllm package")
        root = Path(next(iter(spec.submodule_search_locations)))
    apply(root)


if __name__ == "__main__":
    main()
