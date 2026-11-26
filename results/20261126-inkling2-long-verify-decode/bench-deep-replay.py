# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interleave stock and prefix-peeled C1 k=7 target attention on an idle XPU.

Run in the serving Python environment. This measures one attention launch plus
its split merge, not serving throughput. Capture at 128 keys and then replay
at four lengths. Deep bit mismatches are reported, not accepted as quality parity.
"""

import argparse
import json
from statistics import median
from time import perf_counter_ns


def _run(repeats, replays):
    import torch

    if not torch.xpu.is_available():
        raise SystemExit("This benchmark requires an Intel XPU.")

    from vllm_inkling2_xpu.ops.attention import (
        plan_decode_attention,
        rel_decode_attention,
    )

    lengths = (128, 30_080, 131_072, 375_000)
    q_len, heads, kv_heads, dim, page, extent = 8, 24, 8, 128, 64, 1024
    pages = (max(lengths) + page - 1) // page
    rng = torch.Generator().manual_seed(71)
    stream = torch.xpu.Stream()
    with torch.inference_mode(), torch.xpu.stream(stream):
        kv = torch.randn(pages, page, kv_heads, 2 * dim, generator=rng).mul_(0.25)
        kv_u8 = kv.to(torch.float8_e4m3fn).view(torch.uint8).xpu()
        k, v = kv_u8.view(torch.float8_e4m3fn).split(dim, dim=-1)
        q = torch.randn(q_len, heads, dim, generator=rng).bfloat16().xpu()
        # Fill the sheared layout directly with finite synthetic relative logits.
        rel = (
            torch.randn(q_len + 128, heads, extent + 256, generator=rng)
            .bfloat16()
            .xpu()
        )
        gate = torch.randn(q_len, heads * dim, generator=rng).bfloat16().xpu()
        sinks = torch.randn(q_len, heads, generator=rng).bfloat16().xpu()
        bt = torch.randperm(pages, generator=rng).int().unsqueeze(0).xpu()
        lens = torch.tensor([lengths[0]], dtype=torch.int32, device="xpu")
        cu = torch.tensor([0, q_len], dtype=torch.int32, device="xpu")
        splits = plan_decode_attention(False, 1, q_len, heads, kv_heads, extent)

        def forward(out, enabled):
            rel_decode_attention(
                q,
                k,
                v,
                rel,
                gate,
                sinks,
                out,
                block_table=bt,
                cache_seqlens=lens,
                cu_seqlens_q=cu,
                max_query_len=q_len,
                num_actual_tokens=q_len,
                softmax_scale=dim**-0.5,
                is_local=False,
                num_splits=splits,
                verify_bias_free_prefix=enabled,
            )

        graphs, outputs = {}, {}
        for name, enabled in (("control", False), ("candidate", True)):
            out = torch.empty_like(gate)
            forward(out, enabled)  # Compile outside capture.
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                forward(out, enabled)
            graphs[name], outputs[name] = graph, out

        rows = []
        for length in lengths:
            lens.fill_(length)  # Same graph and pointers as capture.
            for graph in graphs.values():
                for _ in range(3):
                    graph.replay()
            control = outputs["control"]
            candidate = outputs["candidate"]
            if not (control.isfinite().all() & candidate.isfinite().all()).item():
                raise RuntimeError(f"non-finite graph output at {length=}")
            mismatches = (control.view(torch.int16) != candidate.view(torch.int16)).sum().item()
            if length == 128 and mismatches:
                raise RuntimeError('candidate changed output on inactive path')
            samples = {"control": [], "candidate": []}
            for i in range(repeats):
                order = ("control", "candidate")
                for name in order if i % 2 == 0 else order[::-1]:
                    stream.synchronize()
                    start = perf_counter_ns()
                    for _ in range(replays):
                        graphs[name].replay()
                    stream.synchronize()
                    elapsed = perf_counter_ns() - start
                    samples[name].append(elapsed / (1_000 * replays))
            rows.append(
                {
                    "kv_len": length,
                    "bf16_differing_elements": mismatches,
                    "median_us": {name: median(us) for name, us in samples.items()},
                    "samples_us": samples,
                }
            )

    return {
        "device": torch.xpu.get_device_name(),
        "query_len": q_len,
        "splits": splits,
        "repeats": repeats,
        "replays_per_sample": replays,
        "method": "synchronized_xpu_graph_replay_wall",
        "fixtures": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--replays", type=int, default=32)
    args = parser.parse_args()
    if min(args.repeats, args.replays) < 1:
        parser.error("--repeats and --replays must be positive")
    print(json.dumps(_run(args.repeats, args.replays), indent=2))


if __name__ == "__main__":
    main()
