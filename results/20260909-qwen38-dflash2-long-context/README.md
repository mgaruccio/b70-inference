# Qwen3.8 DFlash2: measured 48K and 64K context

**All four cells passed. Both BF16 and partial RTN INT4 DFlash now successfully generate at a 65536-token total context.** The largest tested prompt is **65408 input + 128 output**, not 65536 input plus output. The former 32768 setting was a configured test ceiling, not a demonstrated model/hardware maximum.

These are **development-tier, sequential long-context measurements** of the fixed runtime. They do not replace the full [earlier benchmark package](../20260909-qwen38-dflash2-rtn-standard/), establish quality equivalence, or promote a production configuration. No context above 65536 total was tested here.

## Longer-context numbers

Each row is the median of **six valid measured trials**, after one excluded warmup. Parentheses contain **inclusive IQR**. Every successful request generated exactly 128 tokens, with greedy sampling, seed 42, ignored EOS and prefix caching disabled.

| Configured total limit | Actual prompt | Drafter | Decode proxy, tok/s (IQR) | TTFT, seconds (IQR) | E2E output tok/s (IQR) |
|---:|---:|---|---:|---:|---:|
| 49152 | 49024 | BF16 | 41.08 (2.41) | 34.595 (0.001) | 3.396 (0.016) |
| 49152 | 49024 | RTN INT4 | **44.17 (3.13)** | 34.572 (0.003) | 3.418 (0.019) |
| 65536 | 65408 | BF16 | 42.29 (1.93) | 51.398 (0.003) | 2.353 (0.006) |
| 65536 | 65408 | RTN INT4 | **42.95 (3.61)** | 51.375 (0.010) | 2.356 (0.010) |

Median end-to-end request latency is **37.687 / 37.450 seconds** for BF16 / INT4 at 48K and **54.400 / 54.330 seconds** at 64K. Full latency distributions are retained in the machine-readable analysis.

The INT4/BF16 near-limit decode-median differences are **+7.52% at 48K** and **+1.54% at 64K**. These are observed sequential-run differences, not statistically significant gains. The 64K difference is small relative to measured variation; decode sample CV is 6.88% for BF16 and 9.19% for INT4. TTFT dominates these cold, short-output requests, so the end-to-end improvement is much smaller than the decode-only difference.

“Decode proxy” is `(128 - 1) / (stream end - first nonempty chunk)`. Speculation streams bursts; this is not a measurement of individual-token latency. TTFT includes more than pure prefill kernel time. Full-precision medians, IQR, sample CV and measured-only speculative counters are in [analysis.json](analysis.json).

### Common shorter-context controls

All entries below are decode-proxy median tok/s (inclusive IQR), again six measurements per point. Keeping these controls distinguishes a raised configuration ceiling from changing the input workload itself.

| Prompt tokens | BF16, 48K limit | INT4, 48K limit | BF16, 64K limit | INT4, 64K limit |
|---:|---:|---:|---:|---:|
| 512 | 67.77 (4.62) | 70.21 (5.71) | 67.02 (4.22) | 70.15 (5.74) |
| 8192 | 53.11 (6.19) | 54.75 (6.41) | 52.51 (1.75) | 60.86 (6.12) |
| 16384 | 52.31 (2.85) | 53.00 (9.42) | 53.58 (3.88) | 55.32 (3.41) |
| 32768 | 47.12 (4.46) | 48.94 (1.19) | 45.61 (3.60) | 49.42 (0.91) |

## Capacity and memory

| Drafter / total context limit | Combined model loading | Available KV pool | Runtime-reported KV tokens, **not a tested maximum** |
|---|---:|---:|---:|
| BF16 / 49152 | 20.34 GiB | 4.69 GiB | 64731 |
| INT4 / 49152 | 18.34 GiB | 6.65 GiB | 91721 |
| BF16 / 65536 | 20.34 GiB | 4.69 GiB | 73231 |
| INT4 / 65536 | 18.34 GiB | 6.65 GiB | 103765 |

**The KV token estimate is configuration-dependent.** At the former 32768 limit, the same stack reported 52535 tokens for BF16 and 74440 for INT4; increasing the configured ceiling changed these estimates despite unchanged reported KV GiB. Neither the old 74K estimate nor the new 104K estimate is an experimentally established maximum. The precise allocation mechanism is not diagnosed in this sweep. Actual successful inference now establishes **64K total for both drafters**. INT4 still saves 2.00 GiB of combined model-loading memory; workload peak device memory was **not sampled**.

## Correctness and boundary checks

- Four cells × five supported points × (one warmup + six measurements): **140 successful forced-length streams**, including **120 measured requests**. No measured rows were invalid or excluded.
- Both 48K cells pass **49024 + 128**; both 64K cells pass **65408 + 128**. Each effective limit was verified through `/v1/models`.
- Each cell also makes four real over-limit probes: the exact **one-token-over** request (49025 or 65409 input + 128 output), plus 65536/120000/160000 input + 128 output. **All 16 return the expected context-limit HTTP 400**, with health checks passing. A 65536-input rejection on a 65536-total configuration is expected; it is not evidence of a 32K ceiling.
- Every supported stream has exact prompt/output usage, `[DONE]`, `finish_reason=length`, no SSE parse errors and healthy post-request checks. All corresponding BF16/INT4 request payloads match exactly within each context configuration.
- Per cell: **3/3 answer canaries, 131 finite boundaries and 8/8 executable functional checks** pass. Both graph capture and the fixed GDN boundary overlay are present in all server logs. No log contains a traceback or the former `Expected spec_token` assertion.
- **39 focused CPU tests pass**, no skips; [test output and commands](qwen38-long-context-cpu-tests.txt). Syntax checks pass for the changed runner/tests and analysis script. The frozen executed Python sources match the delivered source byte-for-byte.
- All four [cell exits](cell-exits.tsv) are zero. Overall pipeline: **58m05s, exit 0**. [Final cleanup](final-cleanup.txt) confirms the launcher hash, 275 W cap, stopped Glimmer and no running containers.

### Near-limit speculation, measured trials only

| Cell | Draft steps | Proposed / accepted | Acceptance | Mean accepted + bonus length |
|---|---:|---:|---:|---:|
| BF16 48K | 295 | 1825 / 455 | 24.93% | 2.542 |
| INT4 48K | 289 | 1812 / 462 | 25.50% | 2.599 |
| BF16 64K | 282 | 1724 / 468 | 27.15% | 2.660 |
| INT4 64K | 279 | 1710 / 471 | 27.54% | 2.688 |

The length is `1 + accepted / draft_steps`, not a reconstruction of every returned token. All exposed zero-based position counts and the other context points are retained in [analysis.json](analysis.json) and raw per-request Prometheus snapshots. No unavailable histogram is invented.

## Configuration and artifact index

Only the configured total context ceiling changes relative to the fixed 32K DFlash stack. BF16 versus INT4 within a limit changes only the draft checkpoint/quantization. The pinned XPU image, target, strict acceptance, K7, legacy V1, C1, 8192 batched-token budget, graphs `[1,2,4,8]`, prefix-cache-off and thinking-off settings stay fixed. Original weights are mounted read-only; no memory-budget tuning or kernel changes were added.

- Image: `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`; vLLM `73029d424`, XPU kernels 0.1.14.1, Torch `2.13.0+xpu`.
- Target: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`; FP16 compute, FP8 target KV. Draft: `incoai/Qwen3.8-27B-DFlash2`, revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`, original BF16 or existing partial RTN INT4/G128; draft KV remains BF16. [Recorded model revisions](../20260909-qwen38-dflash2-rtn-standard/model-download-revisions.json).
- [Predeclared process and primary-source research](code/docs/qwen38-dflash2-long-context-20260909.md); [process with observed results](../../docs/qwen38-dflash2-long-context-20260909.md).
- Exact executed shell pipeline: [run-command.sh](run-command.sh); source base [commit](qwen38-long-context-base-commit.txt) and [source diff](qwen38-long-context-source.diff). `code/` contains the frozen sources; Python caches are omitted.
- **INT4 48K:** [exact argv](int4-48k.command.txt), [summary](int4-48k/summary.json), [cold results](int4-48k/long-context/summary.json), [environment](int4-48k/collect_env.txt), [server log](int4-48k/server.log).
- **INT4 64K:** [exact argv](int4-64k.command.txt), [summary](int4-64k/summary.json), [cold results](int4-64k/long-context/summary.json), [environment](int4-64k/collect_env.txt), [server log](int4-64k/server.log).
- **BF16 48K:** [exact argv](bf16-48k.command.txt), [summary](bf16-48k/summary.json), [cold results](bf16-48k/long-context/summary.json), [environment](bf16-48k/collect_env.txt), [server log](bf16-48k/server.log).
- **BF16 64K:** [exact argv](bf16-64k.command.txt), [summary](bf16-64k/summary.json), [cold results](bf16-64k/long-context/summary.json), [environment](bf16-64k/collect_env.txt), [server log](bf16-64k/server.log).
- Each cell also retains `launcher.sh`, `launch-argv.json`, full client `.command.txt`, hardware/package/model configuration, raw tokenized prompts, requests, SSE and metrics under `long-context/points/`.

Recompute the report with `python3 analyze.py > analysis.json` in this directory; it uses the committed previous campaign's standard-library reducers. Reproduce inference with the archived runner and exact per-cell command on the same isolated host/model layout, **changing to a new output directory**. Do not overwrite this campaign or silently substitute another runtime. Canonical host directory: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-long-context/`.

## Limits and production status

The current-best MTP4 reference was already measured through 190000 input tokens in the separate benchmark package, but it was not rerun here and is not a matched-stack control for these deltas. This sweep does not establish that DFlash matches MTP4's larger capacity.

These results have no interleaved confidence interval, sampled workload memory peak, full new BetterBench/serving run or broad quality suite. Strict rejection is unchanged, but no bitwise-output or quality-equivalence claim is made. The production launcher retains SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; no experimental configuration was promoted. **64K is now tested successfully, not proven to be the maximum.**
