#!/usr/bin/env python3
"""Focused native-XPU state gate, run by the lead in the disposable serving image.

Preconditions: vLLM 73029d424, XPU kernels 0.1.14.1, the boundary overlay already
applied, and the target model's local config.json (no weights/downloads needed).
Example, INSIDE that isolated image, before the real API boundary journey:
  python scripts/check-qwen38-xpu-boundary-state.py --config /models/target/config.json --tp-size 1
Retain the exact command, stdout/stderr and exit status. No files, launchers,
servers or persistent settings are written. This is correctness, not timing.

Calls the installed torch.ops.vllm.gdn_attention_core_xpu entry point and the
real _xpu_C.gdn_attention full-width reference, not a numerical Python model.
For L=1..7 and previous acceptance=1..8, compare real output/z and conv/SSM
checkpoints against full blocks with zero/random dummy suffixes. Then execute
a full next step from EVERY valid accepted prefix, retaining divergent dummy
states to expose leakage. Include poisoned physical padding and padded token
counts. This does not exercise graph capture/replay; the subsequent API gate
with the production-equivalent graph settings remains mandatory.

State shapes come from the installed MambaStateShapeCalculator, as in pinned
QwenGatedDeltaNetAttention.get_state_shape; projections match forward_xpu.
"""
import argparse
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="local target config.json")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gqa-interleaved", action="store_true", help="match the target's projection layout")
    args = parser.parse_args()

    # Never import the serving runtime into the ordinary CPU test process.
    import torch
    import vllm
    from vllm import _xpu_ops
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    version = importlib.metadata.version("vllm-xpu-kernels")
    if version != "0.1.14.1":
        raise RuntimeError(f"expected XPU kernels 0.1.14.1, got {version}")
    if not torch.xpu.is_available():
        raise RuntimeError("native XPU required; there is no CPU fallback")
    spec = importlib.util.spec_from_file_location(
        "boundary_overlay", Path(__file__).with_name("patch-vllm-qwen38-xpu-boundary.py")
    )
    overlay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlay)
    installed_source = inspect.getsource(_xpu_ops)
    if overlay.MARKER not in installed_source or overlay.patch_text(installed_source) != installed_source:
        raise RuntimeError("installed pinned boundary overlay must be applied before this test")
    _xpu_ops.xpu_ops.register_ops_once()

    config = json.loads(args.config.read_text())
    config = config.get("text_config", config)
    nk, nv = config["linear_num_key_heads"], config["linear_num_value_heads"]
    dk, dv = config["linear_key_head_dim"], config["linear_value_head_dim"]
    kernel = config["linear_conv_kernel_dim"]
    tp = args.tp_size
    if tp < 1 or nk % tp or nv % tp:
        raise ValueError("tp-size must divide both head counts")
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp, nk, nv, dk, dv, kernel, 7
    )
    device, dtype = torch.device("xpu:0"), torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(73029)

    def random(shape, dtype=dtype):
        return (torch.randn(shape, generator=generator) * 0.2).to(device=device, dtype=dtype)

    def integer(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    # Non-contiguous slot IDs catch accidental assumptions about cache order.
    slots = integer([6, 2, 9, 4, 11, 1, 8, 3]).reshape(1, 8)
    valid_slots = slots[0].to(torch.long)
    other_slots = torch.tensor([0, 5, 7, 10], dtype=torch.long, device=device)
    conv_dim = (2 * nk * dk + nv * dv) // tp
    layer = SimpleNamespace(
        prefix="boundary.state", num_k_heads=nk, num_v_heads=nv,
        head_k_dim=dk, head_v_dim=dv, tp_size=tp,
        gqa_interleaved_layout=args.gqa_interleaved, activation=config.get("hidden_act", "silu"),
        conv1d=SimpleNamespace(weight=random((conv_dim, 1, kernel)), bias=None),
        A_log=random((nv // tp,), torch.float32), dt_bias=random((nv // tp,)),
    )
    base_states = (random((12, *conv_shape)), random((12, *ssm_shape), torch.float32))
    qkvz_shape, ba_shape = (8, (2 * nk * dk + 2 * nv * dv) // tp), (8, 2 * nv // tp)
    print(json.dumps({
        "test": "C1 K7 partial GDN state continuation", "torch": torch.__version__,
        "vllm": vllm.__version__, "xpu_kernels": version,
        "device": torch.xpu.get_device_name(0), "config": str(args.config),
        "tp_size": tp, "gqa_interleaved": args.gqa_interleaved,
        "conv_shape": list(base_states[0].shape), "ssm_shape": list(base_states[1].shape),
        "state_dtypes": [str(t.dtype) for t in base_states],
        "qkvz_shape": qkvz_shape, "ba_shape": ba_shape, "seed": 73029,
        "rtol": 0, "atol": 0,
    }), flush=True)

    def metadata(length, accepted, actual):
        return GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=1, num_spec_decode_tokens=length, num_actual_tokens=actual,
            spec_query_start_loc=integer([0, length]), spec_state_indices_tensor=slots,
            spec_sequence_masks=torch.tensor([True], device=device),
            spec_token_indx=integer(list(range(length))), non_spec_token_indx=integer([]),
            num_accepted_tokens=integer([accepted]),
        )

    def run(inputs, states, length, accepted, *, patched, actual=None):
        layer.kv_cache = tuple(t.clone() for t in states)
        m = metadata(length, accepted, length if actual is None else actual)
        shape = (inputs[0].shape[0], nv // tp, dv)
        output = torch.zeros(shape, dtype=dtype, device=device)
        output[length:].fill_(-17)
        z = torch.full(shape, -23, dtype=dtype, device=device)
        if patched:
            context = ForwardContext(
                no_compile_layers={layer.prefix: layer},
                attn_metadata={layer.prefix: m}, slot_mapping={},
            )
            with override_forward_context(context):
                torch.ops.vllm.gdn_attention_core_xpu(output, z, *inputs, layer.prefix)
        else:
            assert length == 8 and inputs[0].shape[0] == 8
            # Exact native ABI used by the pinned wrapper; fixed-width oracle.
            torch.ops._xpu_C.gdn_attention(
                output, z, *inputs, nk, nv, dk, dv,
                conv_state=layer.kv_cache[0], ssm_state=layer.kv_cache[1],
                conv_weights=layer.conv1d.weight.view(conv_dim, kernel), conv_bias=None,
                activation=layer.activation, A_log=layer.A_log, dt_bias=layer.dt_bias,
                num_prefills=0, num_decodes=0, num_spec_decodes=1, has_initial_state=None,
                non_spec_query_start_loc=None, non_spec_token_indx=m.non_spec_token_indx,
                non_spec_state_indices_tensor=None, spec_query_start_loc=m.spec_query_start_loc,
                spec_token_indx=m.spec_token_indx, spec_state_indices_tensor=slots,
                num_accepted_tokens=m.num_accepted_tokens, num_actual_tokens=8,
                tp_size=tp, reorder_input=not args.gqa_interleaved,
            )
        torch.xpu.synchronize()
        assert m.num_actual_tokens == (length if actual is None else actual)
        assert m.num_spec_decode_tokens == length
        assert m.num_accepted_tokens.item() == accepted
        assert m.spec_query_start_loc.tolist() == [0, length]
        assert m.spec_token_indx.tolist() == list(range(length))
        return SimpleNamespace(outputs=(output, z), states=layer.kv_cache)

    def same(a, b, label):
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise AssertionError(f"non-finite real values: {label}")
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=label)

    def compare(a, b, length, label):
        for name, left, right in zip(("output", "z"), a.outputs, b.outputs):
            same(left[:length], right[:length], f"{label}: {name}")
        for name, left, right in zip(("conv", "ssm"), a.states, b.states):
            same(left[valid_slots[:length]], right[valid_slots[:length]], f"{label}: {name}")

    cases = continuations = 0
    with torch.inference_mode():
        for length in range(1, 8):
            for previous in range(1, 9):
                label = f"L={length} previous={previous}"
                print(f"BEGIN {label}", flush=True)
                full_random = (random(qkvz_shape), random(ba_shape))
                full_zero = tuple(t.clone() for t in full_random)
                for t in full_zero:
                    t[length:].zero_()
                zero = run(full_zero, base_states, 8, previous, patched=False)
                different = run(full_random, base_states, 8, previous, patched=False)
                compare(zero, different, length, label + " suffix independence")
                if torch.equal(zero.states[1][valid_slots[length:]], different.states[1][valid_slots[length:]]):
                    raise AssertionError(f"dummy SSM suffix did not diverge: {label}")
                next_inputs = (random(qkvz_shape), random(ba_shape))
                expected_next = {}
                for accepted in range(1, length + 1):
                    a = run(next_inputs, zero.states, 8, accepted, patched=False)
                    b = run(next_inputs, different.states, 8, accepted, patched=False)
                    compare(a, b, 8, f"{label} accept={accepted} reference continuation")
                    expected_next[accepted] = a
                for physical, actual in ((length, length), (8, length), (8, 8), (12, 12)):
                    case_label = f"{label} physical={physical} actual={actual}"
                    inputs = tuple(torch.full((physical, t.shape[1]), float("nan"),
                                              dtype=dtype, device=device) for t in full_random)
                    for t, original in zip(inputs, full_random):
                        t[:length].copy_(original[:length])
                    result = run(inputs, base_states, length, previous, patched=True, actual=actual)
                    compare(result, zero, length, case_label)
                    # Scratch suffix must be zero, not read from physical poison.
                    for left, right in zip(result.states, zero.states):
                        same(left[valid_slots], right[valid_slots], case_label + " zero scratch suffix")
                    for name, state, initial in zip(("conv", "ssm"), result.states, base_states):
                        same(state[other_slots], initial[other_slots], case_label + f" untouched {name} slots")
                    for t, original in zip(inputs, full_random):
                        same(t[:length], original[:length], case_label + " input prefix unchanged")
                        assert torch.isnan(t[length:]).all().item(), case_label
                    for t, canary in zip(result.outputs, (-17, -23)):
                        assert torch.all(t[length:] == canary).item(), case_label + " output tail overwritten"
                    for accepted, expected in expected_next.items():
                        continuation = run(next_inputs, result.states, 8, accepted, patched=True)
                        compare(continuation, expected, 8, f"{case_label} accept={accepted} continuation")
                        continuations += 1
                    cases += 1
                print(f"PASS {label}", flush=True)
    print(json.dumps({"result": "PASS", "partial_cases": cases, "continuations": continuations,
                      "api_and_graph_verification": "still required"}), flush=True)


if __name__ == "__main__":
    main()
