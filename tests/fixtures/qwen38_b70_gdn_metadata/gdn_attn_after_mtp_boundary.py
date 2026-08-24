"""Narrow syntax fixture extracted from pinned gdn_attn.py after B70 boundary.

Source pin: vllm/vllm-openai-xpu sha256:f01e24f6...
Source commit: ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9
The fixture keeps the exact overlay anchors and surrounding Python shape; it
intentionally omits unrelated attention implementation.
"""
from __future__ import annotations

import torch


class GDNAttentionMetadataBuilder:
    def __init__(self, decode_cudagraph_max_bs, num_spec, device):
        self.decode_cudagraph_max_bs = decode_cudagraph_max_bs
        self.num_spec = num_spec
        self.spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )
        self.non_spec_token_indx: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
            dtype=torch.int32,
            device=device,
        )

    def build(self, block_table_tensor, query_start_loc, query_start_loc_cpu):
        if spec_sequence_masks is None:
            pass
        else:
            if num_prefills == 0 and num_decodes == 0:
                expected_spec_token_size = num_spec_decodes * (self.num_spec + 1)
                actual_spec_token_size = query_start_loc_cpu[-1].item()
                if actual_spec_token_size < expected_spec_token_size:
                    # B70_MTP_PARTIAL_FINAL_GROUP: The max-sequence boundary can
                    # truncate the final speculative group. The XPU GDN kernel
                    # requires complete groups, so process this final partial
                    # group through the existing stateful non-spec prefill path.
                    spec_sequence_masks = None
                    spec_sequence_masks_cpu = None
                    num_prefills = num_spec_decodes
                    num_prefill_tokens = actual_spec_token_size
                    num_spec_decodes = 0
                    num_spec_decode_tokens = 0
                    spec_token_indx = None
                    non_spec_token_indx = None
                    spec_state_indices_tensor = None
                    non_spec_state_indices_tensor = block_table_tensor[:, 0]
                    spec_query_start_loc = None
                    non_spec_query_start_loc = query_start_loc
                    non_spec_query_start_loc_cpu = query_start_loc_cpu
                    num_accepted_tokens = None
                else:
                    spec_token_indx = torch.arange(
                        expected_spec_token_size,
                        dtype=torch.int32,
                        device=query_start_loc.device,
                    )
                    non_spec_token_indx = torch.empty(
                        0, dtype=torch.int32, device=query_start_loc.device
                    )
                    # Filter by spec_sequence_masks to exclude padded sequences
                    spec_state_indices_tensor = block_table_tensor[
                        spec_sequence_masks_cpu, : self.num_spec + 1
                    ]
                    non_spec_state_indices_tensor = None
                    # Padded sequences are always at the back, so the first
                    # num_spec_decodes + 1 entries of query_start_loc already
                    # contain the correct cumulative token counts.
                    spec_query_start_loc = query_start_loc[: num_spec_decodes + 1]
                    non_spec_query_start_loc = None
                    non_spec_query_start_loc_cpu = None
            else:
                pass

            if spec_sequence_masks_cpu is not None:
                assert num_accepted_tokens is not None
                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]

        if use_full_cuda_graph:
            assert non_spec_token_indx is not None and spec_token_indx is not None
            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
                non_spec_token_indx, non_blocking=True
            )
            non_spec_token_indx = self.non_spec_token_indx[
                : non_spec_token_indx.size(0)
            ]

            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
                spec_token_indx, non_blocking=True
            )
            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]
