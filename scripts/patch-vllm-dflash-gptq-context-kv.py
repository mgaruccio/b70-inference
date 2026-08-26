#!/usr/bin/env python3
"""Guarded overlay for vLLM DFlash GPTQ QKV context precompute.

DFlash normally fuses the BF16 K/V weights from every draft QKV projection and
uses F.linear once per verify. GPTQ QKV modules intentionally expose qweight,
not .weight, so the existing fast path cannot build that fusion. This patch
keeps the fast path unchanged and adds a packed-QKV fallback: run each
already-configured QKV module on normalized context states, discard Q, stack
K/V, then follow the original normalization/RoPE/cache-store path.

Usage inside the vLLM container:
  python /patch-dflash-gptq-context-kv.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py
"""

from pathlib import Path
import sys

if len(sys.argv) != 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} VLLM_QWEN3_DFLASH_PY")

path = Path(sys.argv[1])
source = path.read_text()
marker = "# DFLASH_GPTQ_CONTEXT_KV_FALLBACK"
if marker in source:
    print(f"already patched: {path}")
    raise SystemExit(0)

old_build = '''        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
'''
new_build = '''        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._context_kv_attn_layers = layers_attn

        # DFLASH_GPTQ_CONTEXT_KV_FALLBACK
        # AutoGPTQ QKV modules expose packed qweight/scales rather than a
        # materialized .weight. Preserve the one-GEMM BF16 path when all
        # weights exist; packed QKV falls back to its own quantized projection
        # in _project_context_kv below.
        self._packed_context_qkv = any(
            not hasattr(attn.qkv_proj, "weight") for attn in layers_attn
        )
        if not self._packed_context_qkv:
            # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
            kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
            self._fused_kv_weight = torch.cat(kv_weights, dim=0)
            if has_bias:
                kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
                self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
            else:
                self._fused_kv_bias = None

        # K-norm weights stacked into one contiguous [num_layers, head_dim]
'''
if source.count(old_build) != 1:
    raise RuntimeError("unexpected qwen3_dflash.py: build-context anchor missing or ambiguous")
source = source.replace(old_build, new_build)

old_project = '''        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v
'''
new_project = '''        if self._packed_context_qkv:
            # A quantized QKV module has no materialized weight for the fast
            # fused F.linear path. Use its existing quantized kernel per layer
            # and retain only K/V. This runs only for packed-QKV drafts.
            all_k = []
            all_v = []
            for attn in self._context_kv_attn_layers:
                qkv, _ = attn.qkv_proj(normed_context_states)
                _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                all_k.append(k.view(num_ctx, num_kv_heads, head_dim))
                all_v.append(v.view(num_ctx, num_kv_heads, head_dim))
            return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)

        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous
        return all_k, all_v
'''
if source.count(old_project) != 1:
    raise RuntimeError("unexpected qwen3_dflash.py: project-context anchor missing or ambiguous")
path.write_text(source.replace(old_project, new_project))
print(f"patched: {path}")
