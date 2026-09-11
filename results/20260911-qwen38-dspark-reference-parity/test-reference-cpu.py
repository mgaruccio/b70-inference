#!/usr/bin/env python3
"""Supplemental official-model CPU smoke in the existing image, NOT API parity.

No pretrained weights. Reduced dimensions exercise the actual unchanged official
classes, hooks, two independent cache-free forwards, Markov head and report code.
"""
import argparse
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-source", type=Path, required=True)
    args = parser.parse_args()
    ns = runpy.run_path(str(Path(__file__).with_name("replay-reference.py")))
    dflash, dspark = ns["official"](args.official_source)
    import torch
    from parity_capture import Capture
    from parity_common import tensor_identity
    torch.set_num_threads(2)
    torch.manual_seed(123)
    config = dspark.DSparkConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=5, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=128, markov_rank=8,
        num_target_layers=64, block_size=7, max_position_embeddings=262144, layer_types=["full_attention"] * 5,
        dflash_config={"target_layer_ids": [5, 19, 33, 47, 61], "projector_type": "dspark", "mask_token_id": 127},
        rope_parameters={"rope_type": "yarn", "rope_theta": 10000000.0, "factor": 32.0,
                         "original_max_position_embeddings": 8192, "beta_fast": 32.0, "beta_slow": 1.0})
    config._attn_implementation = "eager"
    model = dspark.DSparkDraftModel(config).eval().to(torch.bfloat16)
    inputs = {"target": torch.randn(1, 3, 160, dtype=torch.bfloat16),
              "query": torch.randn(1, 7, 32, dtype=torch.bfloat16),
              "positions": torch.arange(10)[None]}
    data = {"query_ids": torch.tensor([11] + [127] * 6)}
    data.update({f"markov_prev.{i}": torch.tensor([11]) for i in range(7)})
    head = torch.randn(128, 32, dtype=torch.float16)
    with torch.inference_mode():
        plain = model(position_ids=inputs["positions"], noise_embedding=inputs["query"],
                      target_hidden=inputs["target"], past_key_values=None, use_cache=False, is_causal=False)
        one = ns["replay"](model, dflash, inputs, data, head, 3)
        two = ns["replay"](model, dflash, inputs, data, head, 3)
    assert isinstance(plain, torch.Tensor) and tuple(plain.shape) == (1, 7, 32)
    assert torch.equal(plain[0], one["final_hidden"]), "reference instrumentation altered output"
    report = ns["compare"](one, two)
    assert report["first_nonexact_stage"] is None
    assert report["proposed_ids"]["equal"]
    # Metrics must expose real errors and argmax rank/margins, not just finite.
    stats = ns["metrics"](torch.tensor([[1., 3., 2.]]), torch.tensor([[1., 2., 3.]]), True)
    assert stats["max_abs"] == 1 and stats["rmse"] > 0 and stats["top1_agreement"] == 0
    assert stats["left_top1_rank_in_right"] == [2] and stats["right_top1_margin"] == [1.0]
    assert ns["metrics"](torch.tensor([float("nan")]), torch.tensor([0.]))["finite"] is False
    # Native observation snapshots must not alias mutable source buffers.
    observer = object.__new__(Capture)
    observer.data, observer.errors, observer.handles, observer.restores, observer.bytes = {}, [], [], [], 0
    value = torch.tensor([1., 2.], dtype=torch.bfloat16)
    observer.put("snapshot", value)
    value.zero_()
    assert observer.data["snapshot"].tolist() == [1., 2.]
    linear = torch.nn.Linear(4, 4).to(torch.bfloat16)
    observer.post(linear, "module")
    query = torch.ones(1, 4, dtype=torch.bfloat16)
    output = linear(query)
    assert torch.equal(output, observer.data["module"])
    observer.detach()
    assert torch.equal(output, linear(query))
    assert len(tensor_identity(head)["sha256"]) == 64
    assert len(model.state_dict()) == 62
    print("PASS: unchanged official HF model returns tensor; instrumentation leaves output identical")
    print("PASS: two independent official eager BF16 zero-cache forwards + official Markov/head replay")
    print("PASS: all 62 official checkpoint keys; stage metrics/ranks/margins and immutable snapshots")
    print("SUPPLEMENTAL ONLY: tiny CPU model; real API capture/XPU numerical result remains unrun")


if __name__ == "__main__":
    main()
