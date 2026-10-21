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



def forward_xpu_native_rows(
    layer, metadata, projected_qkvz, projected_ba, out, z_out, *, token_indices=None
):
    """Run native GDN rows with compact state IDs for extension dispatch.

    The frozen XPU kernel computes the state pointer offset from an int32 state
    ID. Keep the native op unchanged, but give it a compact per-call state view
    and local IDs so large global state pools cannot overflow that arithmetic.
    """
    if token_indices is None:
        native_qkvz = projected_qkvz
        native_ba = projected_ba
        native_out = out
        native_z = z_out
        native_num_tokens = metadata.num_actual_tokens
    else:
        token_indices = token_indices.long()
        native_qkvz = projected_qkvz.index_select(0, token_indices)
        native_ba = projected_ba.index_select(0, token_indices)
        native_out = out.new_empty(
            (token_indices.numel(), layer.num_v_heads // layer.tp_size, layer.head_v_dim)
        )
        native_z = torch.empty_like(native_out)
        native_num_tokens = token_indices.numel()

    full_conv_state = layer.kv_cache[0]
    full_ssm_state = layer.kv_cache[1]
    state_indices = metadata.non_spec_state_indices_tensor
    if state_indices is None:
        conv_state = full_conv_state
        ssm_state = full_ssm_state
        native_state_indices = None
    else:
        # Block 0 is the engine's NULL state row; -1 is accepted for older
        # padded metadata. Only these sentinels are remapped. Positive IDs are
        # passed through to index_select so invalid IDs still fail loudly.
        padding = (state_indices == 0) | (state_indices == -1)
        safe_state_indices = torch.where(
            padding, torch.zeros_like(state_indices), state_indices
        )
        gather_indices = safe_state_indices.to(dtype=torch.long)
        conv_state = full_conv_state.index_select(0, gather_indices)
        ssm_state = full_ssm_state.index_select(0, gather_indices)
        pad_indices = torch.zeros_like(gather_indices)
        pad_conv_state = full_conv_state.index_select(0, pad_indices)
        pad_ssm_state = full_ssm_state.index_select(0, pad_indices)
        pad_conv_mask = padding.view((-1,) + (1,) * (conv_state.ndim - 1))
        pad_ssm_mask = padding.view((-1,) + (1,) * (ssm_state.ndim - 1))
        conv_state.masked_fill_(pad_conv_mask, 0)
        ssm_state.masked_fill_(pad_ssm_mask, 0)
        native_state_indices = torch.arange(
            state_indices.numel(), dtype=torch.int32, device=state_indices.device
        )

    torch.ops._xpu_C.gdn_attention(
        native_out,
        native_z,
        native_qkvz,
        native_ba,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
        conv_state=conv_state,
        ssm_state=ssm_state,
        conv_weights=layer.conv1d.weight.view(
            layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
        ),
        conv_bias=layer.conv1d.bias,
        activation=layer.activation,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_prefills=metadata.num_prefills,
        num_decodes=metadata.num_decodes,
        num_spec_decodes=0,
        has_initial_state=metadata.has_initial_state,
        non_spec_query_start_loc=metadata.non_spec_query_start_loc,
        non_spec_token_indx=None,
        non_spec_state_indices_tensor=native_state_indices,
        spec_query_start_loc=None,
        spec_token_indx=None,
        spec_state_indices_tensor=None,
        num_accepted_tokens=None,
        num_actual_tokens=native_num_tokens,
        tp_size=layer.tp_size,
        reorder_input=not layer.gqa_interleaved_layout,
    )

    if state_indices is not None:
        # Scatter only the selected rows. NULL rows target local row zero, whose
        # original contents are restored before the copy, so padding cannot
        # overwrite the reserved cache row or any other untouched state.
        conv_state = torch.where(pad_conv_mask, pad_conv_state, conv_state)
        ssm_state = torch.where(pad_ssm_mask, pad_ssm_state, ssm_state)
        full_conv_state.index_copy_(0, gather_indices, conv_state)
        full_ssm_state.index_copy_(0, gather_indices, ssm_state)

    if token_indices is not None:
        out.index_copy_(0, token_indices, native_out)
        z_out.index_copy_(0, token_indices, native_z)


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
    forward_xpu_native_rows(
        layer, metadata, qkvz, ba, out, z_out, token_indices=indices
    )
