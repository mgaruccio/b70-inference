# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 ac7509e2b bridge from fused XPU projections to the official GDN hook.

This is dispatch/layout code, not a tree kernel. The registered extension owns
both tree convolution and recurrence. Native prefill/ordinary decode rows still
use the frozen SYCL op. With no extension the caller's original path is intact.
"""
from __future__ import annotations

import torch

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first


def forward_xpu_spec_rows(layer, metadata, projected_qkvz, projected_ba, out, z_out):
    """Dispatch speculative rows only; preserve the non-speculative SYCL path.

    All indices and copies remain on device (including graph capture). Padded
    speculative requests retain the metadata builder's NULL state indices and
    zero query lengths, which the extension must mask exactly as on CUDA.
    """
    count = metadata.num_actual_tokens
    qkvz, ba = projected_qkvz[:count], projected_ba[:count]
    if layer.gqa_interleaved_layout:
        q, k, v, z, b, a = layer.fix_query_key_value_ordering(qkvz, ba)
        mixed_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1)
    else:
        qkv_dim = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
        mixed_qkv, z = qkvz.split((qkv_dim, layer.value_dim // layer.tp_size), dim=-1)
        z = z.reshape(count, layer.num_v_heads // layer.tp_size, layer.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
    z_out[:count].copy_(z)

    conv_state = layer.kv_cache[0]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    conv_weight = layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
    pure_spec = metadata.num_prefills == 0 and metadata.num_decodes == 0
    if pure_spec:
        spec_out = out[:count]
    else:
        spec_indices = metadata.spec_token_indx.long()
        mixed_qkv = mixed_qkv.index_select(0, spec_indices)
        a = a.index_select(0, spec_indices)
        b = b.index_select(0, spec_indices)
        spec_out = out.new_empty((spec_indices.numel(), layer.num_v_heads // layer.tp_size, layer.head_v_dim))
    layer._forward_spec_rows(
        metadata.verify_extension, metadata, mixed_qkv, a, b,
        conv_state, conv_weight, layer.kv_cache[1], spec_out,
    )
    if pure_spec:
        return
    out.index_copy_(0, spec_indices, spec_out)

    # The SYCL kernel must never see the speculative rows a second time. Gather
    # only the remaining rows, whose state/query metadata is already compacted
    # by GDNAttentionMetadataBuilder; then scatter the outputs to original order.
    indices = metadata.non_spec_token_indx.long()
    native_out = out.new_empty((indices.numel(), layer.num_v_heads // layer.tp_size, layer.head_v_dim))
    native_z = torch.empty_like(native_out)
    torch.ops._xpu_C.gdn_attention(
        native_out, native_z, qkvz.index_select(0, indices), ba.index_select(0, indices),
        layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
        conv_state=layer.kv_cache[0], ssm_state=layer.kv_cache[1],
        conv_weights=conv_weight, conv_bias=layer.conv1d.bias, activation=layer.activation,
        A_log=layer.A_log, dt_bias=layer.dt_bias,
        num_prefills=metadata.num_prefills, num_decodes=metadata.num_decodes,
        num_spec_decodes=0, has_initial_state=metadata.has_initial_state,
        non_spec_query_start_loc=metadata.non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=metadata.non_spec_state_indices_tensor.contiguous(),
        spec_query_start_loc=None, spec_token_indx=None, spec_state_indices_tensor=None,
        num_accepted_tokens=None, num_actual_tokens=indices.numel(),
        tp_size=layer.tp_size, reorder_input=not layer.gqa_interleaved_layout,
    )
    out.index_copy_(0, indices, native_out)
    z_out.index_copy_(0, indices, native_z)
