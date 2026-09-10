#!/usr/bin/env python3
"""Pinned 73029d424 / XPU kernels 0.1.14.1 C1 partial speculative GDN fix.

For disposable Qwen3.8 DFlash K7 research cells only; no scheduler, prompt,
context-limit, rejection, launcher, or native-kernel changes. The fixed-width
native kernel needs K+1 rows even when the scheduler supplies a shorter final
block. Pad only internal GDN scratch buffers and publish only the real prefix.
Keep every rollback slot and the PREVIOUS step's accepted count (possibly K+1).
Future dummy checkpoints must never be selected by acceptance of this prefix.

Pinned metadata and wrapper:
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/attention/backends/gdn_attn.py
https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/_xpu_ops.py
Native state-continuation verification must pass before API boundary testing;
CPU routing tests alone do not establish native correctness.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path

MARKER = "B70_XPU_GDN_PARTIAL_BLOCK"
START = "def _gdn_attention_core_xpu_impl(\n"
END = "\n\ndef _gdn_attention_core_xpu_fake(\n"
# Entire original function, including its final newline, from the stopped image.
ORIGINAL_SHA256 = "8c40a40b75e50e7ec9b3aa85683809c7d8cd478802fa86875bff190dcccb2b44"
REGISTRATION = '''                op_name="gdn_attention_core_xpu",
                op_func=eager_break_during_capture(_gdn_attention_core_xpu_impl),
                mutates_args=["core_attn_out", "z"],
                fake_impl=_gdn_attention_core_xpu_fake,
'''
CALL = "    torch.ops._xpu_C.gdn_attention(\n"
BEFORE_CALL = '''    # B70_XPU_GDN_PARTIAL_BLOCK: scratch padding, never scheduled tokens.
    boundary_outputs = None
    real_spec_tokens = attn_metadata.num_spec_decode_tokens
    if (
        num_spec_decodes == 1
        and num_prefills == 0
        and num_decodes == 0
        and spec_state_indices_tensor is not None
        and spec_state_indices_tensor.ndim == 2
        and 0 < real_spec_tokens < spec_state_indices_tensor.size(1)
    ):
        # This overlay is deliberately C1-only. Do not reinterpret mixed rows,
        # slice rollback slots, or infer real rows from graph-padded buffers.
        if (
            attn_metadata.num_prefill_tokens != 0
            or attn_metadata.num_decode_tokens != 0
            or spec_state_indices_tensor.size(0) != 1
            or spec_sequence_masks is None
            or spec_query_start_loc is None
            or spec_query_start_loc.numel() != 2
            or spec_token_indx is None
            or spec_token_indx.numel() != real_spec_tokens
            or num_accepted_tokens is None
            or num_accepted_tokens.numel() != 1
            or non_spec_query_start_loc is not None
            or non_spec_state_indices_tensor is not None
            or (non_spec_token_indx is not None and non_spec_token_indx.numel() != 0)
            or num_actual_tokens < real_spec_tokens
            or any(t.size(0) < real_spec_tokens for t in (
                core_attn_out, z, projected_states_qkvz, projected_states_ba
            ))
        ):
            raise RuntimeError("unsupported C1 partial XPU GDN metadata/buffers")
        block_tokens = spec_state_indices_tensor.size(1)
        padded_qkvz = projected_states_qkvz.new_zeros(
            (block_tokens, *projected_states_qkvz.shape[1:])
        )
        padded_ba = projected_states_ba.new_zeros(
            (block_tokens, *projected_states_ba.shape[1:])
        )
        padded_qkvz[:real_spec_tokens].copy_(projected_states_qkvz[:real_spec_tokens])
        padded_ba[:real_spec_tokens].copy_(projected_states_ba[:real_spec_tokens])
        projected_states_qkvz = padded_qkvz
        projected_states_ba = padded_ba
        boundary_outputs = (core_attn_out, z)
        core_attn_out = core_attn_out.new_zeros((block_tokens, *core_attn_out.shape[1:]))
        z = z.new_empty((block_tokens, *z.shape[1:]))
        spec_query_start_loc = spec_query_start_loc.new_tensor([0, block_tokens])
        spec_token_indx = torch.arange(
            block_tokens, dtype=torch.int32, device=spec_token_indx.device
        )
        # Leave metadata, state indices, caches and accepted counts untouched.
        # The registered op already uses eager_break_during_capture.
        num_actual_tokens = block_tokens

'''
AFTER_CALL = '''    if boundary_outputs is not None:
        boundary_outputs[0][:real_spec_tokens].copy_(core_attn_out[:real_spec_tokens])
        boundary_outputs[1][:real_spec_tokens].copy_(z[:real_spec_tokens])
'''


def patch_text(source):
    """Apply once, or validate an exact replay; refuse drift before any write."""
    for anchor in (START, END, REGISTRATION):
        if source.count(anchor) != 1:
            raise RuntimeError("pinned XPU GDN function/capture anchors changed")
    start = source.index(START)
    end = source.index(END, start)
    function = source[start:end]
    original = function
    already_patched = MARKER in source
    if already_patched:
        if (source.count(MARKER) != 1 or function.count(BEFORE_CALL) != 1
                or not function.endswith(AFTER_CALL)):
            raise RuntimeError("partial/changed XPU GDN boundary patch")
        original = function.replace(BEFORE_CALL, "", 1)[:-len(AFTER_CALL)]
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("pinned XPU GDN implementation changed")
    if original.count(CALL) != 1:
        raise RuntimeError("pinned native GDN call anchor changed")
    expected = original.replace(CALL, BEFORE_CALL + CALL, 1) + AFTER_CALL
    if already_patched and function != expected:
        raise RuntimeError("moved/changed XPU GDN boundary patch")
    result = source[:start] + expected + source[end:]
    compile(result, "_xpu_ops.py", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="vllm package root; omit inside serving image")
    args = parser.parse_args()
    root = args.root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vllm package not found")
        root = Path(spec.origin).parent
    path = root / "_xpu_ops.py"
    before = path.read_text()
    after = patch_text(before)
    if after != before:
        path.write_text(after)
    print(f"{MARKER}: {path}", flush=True)


if __name__ == "__main__":
    main()
