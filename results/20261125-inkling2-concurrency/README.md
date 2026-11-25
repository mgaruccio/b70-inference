# Inkling 2 Mini: 320K context and concurrency on the B70

**Development tier, 2026-11-25 UTC.** One Arc Pro B70 at 275 W, with the [same Inkling 2 Mini checkpoint and plugin](../20261124-inkling2-hybrid-prefix-sdpa/), FP8 E4M3 KV, CUTLASS Xe2, hybrid XPU flash prefill, DSpark k=7 probabilistic drafting and decode graphs. The 327,680-token per-request cap, 2,048-token scheduling budget, 64-token pages and 0.98 memory utilization were held fixed. Tests below vary client concurrency on the same **32-sequence server**; C1 is the comparison baseline, not a separate one-sequence engine.

## Short-request sweep

`vllm bench serve` used seeded random 512-token prompts, greedy generation, 128 output tokens with EOS ignored, loopback completions, and an infinite offered request rate capped at C concurrent clients. Each point has three measured repeats of `max(8, 4×C)` requests. The server kept production prefix caching enabled; the random workload requested no fixed shared prefix. All measured runs completed their full input and output budgets, observed C running simultaneously, and recorded zero preemptions. Rates include prefill and decode over the measured wall time. Latencies are medians of three per-run statistics; TPOT is time per generated token after the first.

| Concurrency | Aggregate output tok/s | Input + output tok/s | Median TTFT (ms) | P95 TTFT (ms) | Median TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.7 | 343.6 | 144 | 150 | 9.1 |
| 2 | 110.4 | 551.8 | 177 | 267 | 11.8 |
| 4 | 145.6 | 727.8 | 206 | 491 | 16.7 |
| 8 | 197.2 | 986.0 | 240 | 1,059 | 30.7 |
| 16 | 284.5 | 1,422.3 | 387 | 2,069 | 40.4 |
| **32** | **353.5** | **1,767.4** | 603 | 3,790 | 62.6 |

C32 was the highest tested concurrency that worked at 320K and the fastest tested point: **5.14× C1** aggregate output throughput, or **+24.3% over C16**. Its three individual rates were 351.9, 353.5 and 355.5 output tok/s; a separate fourth verification run reached 363.3 output tok/s (not included in the median). Median DSpark acceptance length at C32 was 2.34 including warmup. See [per-run summaries and commands](raw/sweep/) for draft and acceptance counts.

**C64 would not boot at 320K** with the same settings: vLLM reported 7.4 GiB KV memory available versus 8.04 GiB required for one maximum-length request, estimating a 294,528-token context ceiling. C33–63 were not tested. The C32 engine reported **335,515 KV tokens** (8.23 GiB); that pool is shared, not replicated per sequence. It does not hold 32 maximum-length requests at once. Eligible single-request prefill steps can use hybrid flash; multi-prefill steps use the regular Triton path.

## Long context and live rollout

A separate request on the same C32/320K configuration completed **327,424 prompt + 128 generated tokens** in 175.4 s with zero preemptions. This checks capacity, not answer quality at 320K. In the earlier C2/320K mixed workload, a 311,296-token prompt and 8,192-token prompt completed with zero preemptions, but the short request took 38.3 s to first token and queued for most of the long prefill. The short-request sweep does not measure that mixed-workload latency at C32.

The [serving launcher](../../scripts/inkling-vllm-xpu/serve/serve-inkling.sh) now runs `--max-model-len 327680 --max-num-seqs 32`. After promotion, `/health` was 200, authenticated `/v1/models` showed the correct limit, and authenticated chat and tool-call smoke tests passed. Unauthenticated `/v1/models` returned 401. Live single-stream smoke decode remained about 107.1 tok/s short and 90.9 tok/s at +30K on the existing chat fixtures. The previous live launcher is saved as `serve-inkling.sh.pre-concurrency-20261125-063047` on the guest.

## Reproduce and inspect

- With an isolated matching C32/320K loopback engine on port 18080 (see [launch command](raw/boot-32/launch.txt) and [environment](raw/environment.txt)), run `python -B sweep.py /path/to/new/results /path/to/inkling-weights 32`; rerun into a separate directory with `--verify` for the fourth C32 run and near-limit test. The helper records every exact [vLLM CLI invocation](raw/sweep/) and samples scheduler metrics during each run. See the [vLLM benchmark CLI documentation](https://docs.vllm.ai/en/stable/cli/bench/serve/) and [scheduler tuning documentation](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill).
- On `b70-dev`, full benchmark JSON and server logs are under `~/inkling-vllm-xpu/concurrency-sweep/run-20261125-061916/` and `run-20261125-063047/` in that same directory. The first maintenance run hit its harness timeout during the second C32 trial; the second reused only completed trials and restarted the same C32 engine to finish. No partial trial is in the medians. [Boot error](raw/c64-boot-error.txt), [near-limit output](raw/verify/near-limit.json), and [live args](raw/live-args.txt) are retained here.
- Production smoke command on the guest: `bash ~/cmp/live/live_check.sh after-final-concurrency`; evidence at `~/cmp/live/20261125-063649-after-final-concurrency/`.
