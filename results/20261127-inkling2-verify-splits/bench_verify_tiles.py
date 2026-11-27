# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare split counts or KV tile widths for full C1 k=7 attention on B70.

Run with the stock plugin in the pinned serving Python environment on an idle XPU:

    PYTHONPATH=src python scripts/bench_verify_attention_splits.py --compare-block-n -o tiles.json

The same two graphs (attention + merge) replay with identical buffers for each
KV length. --compare-block-n changes only BLOCK_N from 32 to 64 at 32 splits.
Without it, the original split comparison runs. Scratch sizes are logical FP32
allocations, not allocator-pool sizes or DRAM traffic. Error is diagnostic, not
serving parity.
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

# Inkling-2-Mini ed40e7da full target attention, 409K FP8 KV serving configuration.
Q_LEN, HEADS, KV_HEADS, HEAD_DIM, PAGE, REL_EXTENT = 8, 24, 8, 128, 64, 1024
SCALE = HEAD_DIM**-0.5
STOCK_SPLITS = 32
STOCK_BLOCK_N, WIDE_BLOCK_N = 32, 64
LENGTHS = (30_080, 65_536, 131_072, 375_000)
SEED = 71
BANDWIDTH_GB_S = 560  # Measured streaming reference, not a DRAM counter.
REF_ROWS = tuple((t, h) for t in (0, 4, 7) for h in (0, 2, 3, 12, 23))


def _make_cache(pages, rng):
    import torch

    # LBNHC backing storage for one layer; interleave K and V within each slot.
    # Fill in slabs so the 375K fixture doesn't need a full FP32 KV allocation.
    cache = torch.empty((pages, PAGE, KV_HEADS, 2 * HEAD_DIM), dtype=torch.uint8)
    for slab in cache.split(64):
        vals = torch.randn(slab.shape, generator=rng, dtype=torch.float32).mul_(0.25)
        slab.copy_(vals.to(torch.float8_e4m3fn).view(torch.uint8))
    return cache


def _shear_bias(bias, kv_len):
    import torch

    # All eight queries belong to the same 128-token shear tile. Each bias row
    # is stored by key position (not distance) with the tile end rounded up.
    end = (kv_len + 127) // 128 * 128
    keys = torch.arange(end - REL_EXTENT - 256, end)
    pos = torch.arange(kv_len - Q_LEN, kv_len)
    dist = pos[:, None, None] - keys
    idx = dist.clamp(0, REL_EXTENT - 1).expand(-1, bias.shape[1], -1)
    sheared = bias.gather(2, idx)
    sheared.masked_fill_(dist >= REL_EXTENT, 0)
    return sheared.masked_fill_(dist < 0, -float("inf"))


def _fp64_reference(q, cache, block_table, bias, gate, sinks, kv_len):
    import torch

    # Exact dequantization of the same FP8 bytes and BF16 inputs as the kernels.
    # Use the unsheared bias and dense softmax over every visible key, gathering
    # only one KV head at a time rather than expanding to all query heads.
    group = HEADS // KV_HEADS
    pages = (kv_len + PAGE - 1) // PAGE
    page_ids = block_table[0, :pages].long()
    out = torch.empty((len(REF_ROWS), HEAD_DIM), dtype=torch.float64)
    gate = gate.view(Q_LEN, HEADS, HEAD_DIM)
    for kv_head in sorted({h // group for _, h in REF_ROWS}):
        rows = [i for i, (_, h) in enumerate(REF_ROWS) if h // group == kv_head]
        tokens, heads = zip(*(REF_ROWS[i] for i in rows))
        packed = cache[page_ids, :, kv_head].flatten(0, 1)[:kv_len]
        k, v = packed.view(torch.float8_e4m3fn).split(HEAD_DIM, dim=-1)
        scores = (q[tokens, heads].double() @ k.double().T) * SCALE
        for i, (token, head) in enumerate(zip(tokens, heads)):
            end = kv_len - Q_LEN + token + 1
            start = max(0, end - REL_EXTENT)
            scores[i, start:end] += bias[token, head, : end - start].flip(0).double()
            scores[i, end:] = -float("inf")
        lse = scores.logsumexp(-1)
        attn = scores.softmax(-1) @ v.double()
        gate_weight = gate[tokens, heads].double().sigmoid()
        sink_weight = (lse - sinks[tokens, heads].double()).sigmoid()[:, None]
        out[rows] = attn * gate_weight * sink_weight
    return out


def _scratch_bytes(splits):
    lse = splits * Q_LEN * HEADS * 4
    partial = lse * HEAD_DIM
    return {"o_part": partial, "lse_part": lse, "total": partial + lse}


def _bits(tensor):
    import torch

    return tensor.view(torch.int16)  # BF16, including the sign of zero.


def _abs_error(a, b):
    err = (a.double() - b.double()).abs()
    return {"max_abs": err.max().item(), "mean_abs": err.mean().item()}


def _measure(graphs, forward, out, stream, kv_len, args):
    import torch

    expected = {}

    def check_output(arm):
        if not torch.equal(_bits(out.cpu()), _bits(expected[arm])):
            raise RuntimeError(f"{kv_len=} {arm=}: replay bits differ from eager")

    for arm, graph in graphs.items():
        out.fill_(float("nan"))
        forward(arm)
        expected[arm] = out.to("cpu", copy=True)
        if not expected[arm].isfinite().all():
            raise RuntimeError(f"{kv_len=} {arm=}: non-finite eager output")
        out.fill_(float("nan"))  # Catch missed writes, even with a shared output.
        graph.replay()
        check_output(arm)
    for _ in range(args.warmup):
        for graph in graphs.values():
            graph.replay()
    stream.synchronize()

    # A0, B0, A1, B1, A2, ...: each candidate has two neighbouring controls.
    stock, candidate = graphs
    samples = {arm: [] for arm in graphs}
    order = [stock] + [candidate, stock] * args.repeats
    for arm in order:
        graph = graphs[arm]
        start = perf_counter_ns()
        for _ in range(args.replays):
            graph.replay()
        stream.synchronize()
        samples[arm].append((perf_counter_ns() - start) / (1_000 * args.replays))
        check_output(arm)  # Validate the last replay of each batch, untimed.
    return samples, expected


def _summarize(kv_len, samples, outputs, ref, parameter):
    kv_bytes = kv_len * KV_HEADS * HEAD_DIM * 2  # FP8 K and V, read once each.
    tokens, heads = zip(*REF_ROWS)
    arms = {}
    for arm, times in samples.items():
        actual = outputs[arm].view(Q_LEN, HEADS, HEAD_DIM)[tokens, heads]
        latency = median(times)
        arms[arm] = {
            "samples_us": times,
            "median_us": latency,
            "effective_kv_GB_s": kv_bytes / (latency * 1_000),
            "sampled_error_vs_fp64": _abs_error(actual, ref),
        }
    control, candidate = samples
    stock_times, candidate_times = samples[control], samples[candidate]
    stock, other = outputs[control], outputs[candidate]
    differences = _bits(stock) != _bits(other)
    latency_ratio = arms[candidate]["median_us"] / arms[control]["median_us"]
    diff_prefix = "cross_tile" if parameter == "block_n" else "cross_split"
    return {
        "kv_len": kv_len,
        "compulsory_kv_bytes": kv_bytes,
        "kv_only_floor_us": kv_bytes / (BANDWIDTH_GB_S * 1_000),
        f"arms_by_{parameter}": arms,
        "median_latency_reduction_pct": 100 * (1 - latency_ratio),
        "candidate_batches_faster_than_both_neighbors": sum(
            b < min(a0, a1)
            for a0, b, a1 in zip(stock_times, candidate_times, stock_times[1:])
        ),
        f"{diff_prefix}_bf16_differing_elements": differences.sum().item(),
        f"{diff_prefix}_total_elements": stock.numel(),
        f"{diff_prefix}_error": _abs_error(stock, other),
    }


def _launch_wide_kv_tile(attention, launch_decode):
    from unittest.mock import patch

    # Use the original rel_decode_attention dispatcher, partial allocation,
    # Triton launcher and merge. The override is active only for this launch.
    launch = attention.rel_attention
    called = False

    def override(*args, block_n, **kwargs):
        nonlocal called
        if (
            called
            or block_n != STOCK_BLOCK_N
            or kwargs["num_splits"] != STOCK_SPLITS
            or kwargs["window"] != 0
            or kwargs["max_query_len"] != Q_LEN
            or kwargs["mode"] != attention.MODE_PARTIAL
            or kwargs["num_warps"] != 4
        ):
            raise RuntimeError("Expected the stock full C1/q8 verifier launch")
        called = True
        return launch(*args, block_n=WIDE_BLOCK_N, **kwargs)

    with patch.object(attention, "rel_attention", override):
        launch_decode(STOCK_SPLITS)
    if not called:
        raise RuntimeError("Stock decoder did not dispatch rel_attention")


def _run(args):
    import torch

    if not torch.xpu.is_available():
        raise SystemExit(
            "This benchmark requires an Intel XPU with PyTorch XPU support."
        )

    from vllm_inkling2_xpu.ops import attention

    planned = attention.plan_decode_attention(
        False, 1, Q_LEN, HEADS, KV_HEADS, REL_EXTENT
    )
    if planned != STOCK_SPLITS:
        raise SystemExit(
            f"Expected stock B70 split count {STOCK_SPLITS}, got {planned}."
        )

    if args.compare_block_n:
        parameter = "block_n"
        arms = (STOCK_BLOCK_N, WIDE_BLOCK_N)
        comparison = {
            "num_splits": planned,
            "stock_block_n": STOCK_BLOCK_N,
            "candidate_block_n": WIDE_BLOCK_N,
            "scratch_bytes_per_arm": _scratch_bytes(planned),
        }
    else:
        parameter = "splits"
        arms = (planned, args.candidate_splits)
        comparison = {
            "stock_splits": planned,
            "candidate_splits": args.candidate_splits,
            "scratch_bytes_by_splits": {s: _scratch_bytes(s) for s in arms},
        }

    rng = torch.Generator(device="cpu").manual_seed(SEED)
    pages = (max(LENGTHS) + PAGE - 1) // PAGE
    block_table = torch.randperm(pages, dtype=torch.int32, generator=rng)[None, :]
    q = torch.randn(Q_LEN, HEADS, HEAD_DIM, generator=rng).bfloat16()
    bias = torch.randn(Q_LEN, HEADS, REL_EXTENT, generator=rng).bfloat16()
    gate = torch.randn(Q_LEN, HEADS * HEAD_DIM, generator=rng).bfloat16()
    sinks = torch.randn(Q_LEN, HEADS, generator=rng).bfloat16()
    cache = _make_cache(pages, rng)
    device = torch.device("xpu", torch.xpu.current_device())
    stream = torch.xpu.Stream(device=device)
    measurements = []
    with torch.inference_mode(), torch.xpu.stream(stream):
        k, v = cache.to(device).view(torch.float8_e4m3fn).split(HEAD_DIM, dim=-1)
        dq, dg, ds, dt = (t.to(device) for t in (q, gate, sinks, block_table))
        rel = dq.new_full((Q_LEN + 128, HEADS, REL_EXTENT + 256), float("nan"))
        out = torch.empty_like(dg)
        lens = torch.tensor([LENGTHS[0]], dtype=torch.int32, device=device)
        cu_q = torch.tensor([0, Q_LEN], dtype=torch.int32, device=device)

        def launch_decode(splits):
            attention.rel_decode_attention(
                dq,
                k,
                v,
                rel,
                dg,
                ds,
                out,
                block_table=dt,
                cache_seqlens=lens,
                cu_seqlens_q=cu_q,
                max_query_len=Q_LEN,
                num_actual_tokens=Q_LEN,
                softmax_scale=SCALE,
                is_local=False,
                num_splits=splits,
            )

        def forward(arm):
            if not args.compare_block_n:
                launch_decode(arm)
            elif arm == STOCK_BLOCK_N:
                launch_decode(planned)
            else:
                _launch_wide_kv_tile(attention, launch_decode)

        graphs = {}
        rel[:Q_LEN].copy_(_shear_bias(bias, LENGTHS[0]))
        for arm in arms:
            forward(arm)  # Compile both attention and combine outside capture.
            stream.synchronize()
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph, stream=stream):
                forward(arm)
            graphs[arm] = graph  # Separate scratch pools, same inputs and output.

        for kv_len in LENGTHS:
            lens.fill_(kv_len)
            rel[:Q_LEN].copy_(_shear_bias(bias, kv_len))
            samples, outputs = _measure(graphs, forward, out, stream, kv_len, args)
            measurements.append((kv_len, samples, outputs))

    # Calculate the oracle after all timing so CPU work cannot interrupt the sweep.
    fixtures = []
    for kv_len, samples, outputs in measurements:
        ref = _fp64_reference(q, cache, block_table, bias, gate, sinks, kv_len)
        fixtures.append(_summarize(kv_len, samples, outputs, ref, parameter))
    return {
        "implementation": str(Path(attention.__file__).resolve()),
        "device": torch.xpu.get_device_name(device),
        "torch_version": str(torch.__version__),
        "seed": SEED,
        "shape": {
            "requests": 1,
            "query_len": Q_LEN,
            "query_heads": HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "page_size": PAGE,
            "rel_extent": REL_EXTENT,
            "kv_dtype": "float8_e4m3fn",
            "kv_layout": "LBNHC",
        },
        **comparison,
        "kv_allocation_bytes": cache.nbytes,
        "reference_bandwidth_GB_s": BANDWIDTH_GB_S,
        "fp64_sample_rows_token_head": REF_ROWS,
        "fp64_sample_elements": len(REF_ROWS) * HEAD_DIM,
        "determinism": "bitwise eager matches for validation replay and batch endpoints",
        "timing": {
            "method": "synchronized_xpu_graph_replay_wall",
            "order": "stock, (candidate, stock) repeated",
            "warmup_replays_per_arm": args.warmup,
            "candidate_batches": args.repeats,
            "replays_per_batch": args.replays,
        },
        "fixtures": fixtures,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="JSON result path"
    )
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument(
        "--candidate-splits", type=int, choices=(64, 128), default=64
    )
    experiment.add_argument(
        "--compare-block-n",
        action="store_true",
        help="compare 32 and 64 keys per tile with stock 32 splits in both arms",
    )
    parser.add_argument("--warmup", type=int, default=3, help="warmup replays per arm")
    parser.add_argument(
        "--repeats", type=int, default=10, help="timed candidate batches"
    )
    parser.add_argument(
        "--replays", type=int, default=32, help="replays per timed batch"
    )
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.replays) < 1:
        parser.error("--warmup, --repeats, and --replays must be positive")
    text = json.dumps(_run(args), indent=2, allow_nan=False)
    args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
