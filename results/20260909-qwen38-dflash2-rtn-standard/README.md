# Qwen3.8 DFlash2 RTN INT4 benchmark artifacts

**Execution complete; development-tier results, not publication-ready.** This is the **pre-boundary-fix** campaign. Both DFlash drafters crash at 32640 input + 128 output tokens. Both pass the explicit 32000-token follow-up; the MTP4 reference passes through 190000 input tokens. Quality sensitivity is unresolved and peak device memory was not sampled. Do not present this as a quality-preserving, standard-compliant result or a 200K replacement.

## Findings

- Drafter-only comparison: INT4 raises BetterBench's weighted category-median decode score from **100.79 to 109.51 tok/s (+8.65%)**, and reduces combined target/draft loading memory by **2.00 GiB**.
- Current-best MTP4 reference: **94.05 tok/s** on that BetterBench score. INT4's +16.44% difference is a **whole-stack/configuration comparison**, not an isolated quantization gain.
- There is no universal winner: MTP4 is faster on the fixed-length vLLM serving workload and at 8K/16K cold decode, and supports much longer context. Even against BF16 DFlash, INT4's 32000-token decode median is lower, with broad dispersion.
- Runs were sequential, not interleaved. No drift-cancelled confidence interval or statistically significant winner is claimed. No configuration was promoted.

## Comparators

All arms use the same B70, 275 W cap, target checkpoint, FP16-compute GPTQ INT4/G128 target, FP8 target KV, server C1, prefix caching off, thinking off and strict rejection. Client concurrency measures **queuing against max-num-seqs=1**, not active multi-sequence decoding.

The matched DFlash arms share image `7a558f63…`, vLLM `73029d424`, XPU kernels 0.1.14.1, legacy V1, graph sizes `[1,2,4,8]`, K7 and a 32768 context limit. Only the drafter checkpoint/quantization changes. The candidate is **partial, calibration-free RTN INT4/G128**; QKV, conditioning, selector and other retained tensors remain BF16. It is neither calibrated GPTQ nor a fully INT4 drafter.

The deployed MTP4 reference uses image `f01e24f6…`, vLLM `ac7509e2b`, XPU kernels 0.1.12.3, the original five MTP patches, S+M1 INT4 and a 212992 context limit, with benchmark-only cold-cache/non-thinking/local-interface settings and the previously validated prefill guard/graph settings. Its different image, algorithm, patches and context ceiling make it a **stack/configuration bundle**. Full image digests, versions and source snapshots are retained below.

## Summary measurements

Cold decode entries are median **post-first-stream decode proxies**, with inclusive IQR in parentheses, six valid measured trials and 128 output tokens. The proxy is `(128-1)/(stream end-first nonempty chunk)`: speculative streaming emits bursts, so this is not a measurement of individual token latency. Warmups are excluded; all successful measured rows are retained.

| Measurement | MTP4 reference | BF16 DFlash | RTN INT4 DFlash |
|---|---:|---:|---:|
| BetterBench weighted category-median decode, tok/s | 94.05 | 100.79 | 109.51 |
| Cold 512-token decode, tok/s (IQR) | 70.78 (1.16) | 68.90 (3.09) | 73.96 (5.77) |
| Cold 8192-token decode, tok/s (IQR) | 63.28 (1.05) | 53.20 (3.03) | 56.30 (4.79) |
| Cold 16384-token decode, tok/s (IQR) | 62.96 (3.83) | 53.12 (4.13) | 54.27 (6.29) |
| Cold 32000-token decode, tok/s (IQR) | not selected | 53.26 (11.17) | 49.33 (10.52) |
| Cold 32768-token decode, tok/s (IQR) | 62.02 (1.46) | HTTP 400 | HTTP 400 |
| Cold 65536-token decode, tok/s (IQR) | 55.75 (2.39) | HTTP 400 | HTTP 400 |
| Cold 120000-token decode, tok/s (IQR) | 56.88 (1.25) | HTTP 400 | HTTP 400 |
| Cold 160000-token decode, tok/s (IQR) | 46.39 (1.87) | HTTP 400 | HTTP 400 |
| Cold 190000-token decode, tok/s (IQR) | 37.91 (3.51) | not attempted | not attempted |
| Mean speculative acceptance length, entire BetterBench phase¹ | 3.301 | 4.092 | 4.052 |
| Draft acceptance, entire BetterBench phase¹ | 57.53% | 44.17% | 43.61% |
| Model loading memory, target + speculator | 16.80 GiB | 20.34 GiB | 18.34 GiB |
| Allocated KV pool reported at startup | 8.23 GiB | 4.69 GiB | 6.65 GiB |
| Runtime-reported KV capacity, **not tested context** | 226397 tokens | 52535 tokens | 74440 tokens |
| Peak allocated device memory during workload | not measured | not measured | not measured |
| Largest successful tested prompt, plus 128 output tokens | 190000 | 32000 | 32000 |
| Exact 32640 + 128 boundary | not selected | engine crash | engine crash |

¹ Counter deltas include BetterBench warmups and all decode/prefill/concurrency phases; workloads are not identical across algorithms because MTP4 also runs the longer prefill point. Acceptance length includes the target/bonus token: `1 + accepted/draft_steps`. Proposed/accepted totals and zero-based position counts are in [analysis.json](analysis.json). Cold-point acceptance aggregates exclude warmups. No unavailable acceptance histogram is invented.

### vLLM serving workload

The same pinned **new-image benchmark client** ran against all three servers: random dataset, seed 42, temperature 0, ignored EOS, 512 input/128 output tokens, 48 measured requests and three warmups per concurrency. Every arm completed **48/48 with zero failures** at each level, with exact measured token lengths. These throughput values are aggregate totals/time, not per-request medians.

| Client concurrency | MTP4 output tok/s | BF16 output tok/s | INT4 output tok/s |
|---|---:|---:|---:|
| 1 | 70.05 | 63.00 | 65.82 |
| 2 | 70.45 | 61.98 | 66.66 |
| 4 | 70.52 | 64.68 | 66.33 |
| 8 | 70.19 | 62.60 | 67.98 |

At C1, INT4 is **6.04% slower than MTP4** on this workload. Request/total-token throughput, TTFT, TPOT, stream-update ITL, E2E latency distributions and speculative metrics are retained for every level in the [derived analysis](analysis.json) and raw vLLM JSON. High-percentile estimates have small sample counts; do not overinterpret them.

### BetterBench scope and dispersion

Unmodified full BetterBench 0.4.0, corpus v1.0: all eight categories, **20 measured passes/category and three warmups**, all phases, and C1/2/4/8/16 with 48 requests/level. All three arms completed 160/160 measured decode rows and 48/48 requests at every concurrency level. Every supported prefill depth has eight measured samples.

The headline is BetterBench's weighted sum of category medians, not a median over pooled tokens/requests. Category medians, inclusive IQR, sample CV and sample-size warnings are in the original offline HTML reports and `analysis.json`. All categories run, including chat/math, which have zero weight in the default combined score.

BetterBench prefill labels are approximate: nominal 2000/8000/16000/32000 depths rendered medians of **1516/5919.5/11795.5/23545** prompt tokens. DFlash explicitly skipped nominal 64000 as too large; MTP4 ran it at median **47057.5** actual prompt tokens. Thus BetterBench's nominal 32K success did **not** test the crashing exact boundary.

## Preserved failure and conservative follow-up

The original [INT4](int4-long/long-context/summary.json) and [BF16](bf16-long/long-context/summary.json) cold sweeps passed 512/8192/16384, then crashed during the 32640-token warmup. Both scheduler dumps show 32760 computed tokens, 121 output tokens and a shortened seven-token verification block against an eight-slot state layout:

```text
RuntimeError: Expected spec_token == num_spec_decodes * (num_speculative_tokens + 1) to be true, but got false.
```

Their [INT4 server log](int4-long/server.log) and [BF16 server log](bf16-long/server.log) preserve the traceback. `/health` then returned 503; those sweeps aborted. Longer points in the original failed cells were **not executed**, not silently counted as supported or rejected. This was not an OOM.

Separate [INT4 32000](int4-long-32000/long-context/summary.json) and [BF16 32000](bf16-long-32000/long-context/summary.json) follow-ups each completed one warmup and six valid measurements at 512/8192/16384/32000, plus real context-limit HTTP 400 probes at 32768/65536/120000/160000. Their 24 measured request payloads match exactly. The practical 32000 point neither erases the failed boundary nor proves the largest possible working context.

The [subsequently requested boundary fix](../../docs/qwen38-dflash2-boundary-fix-20260909.md) is a separate configuration and validation campaign. Do not relabel the numbers here as fixed-runtime measurements.

## Artifact index and reproduction

- [Adopted standard](../../BENCHMARKING_STANDARDS.md), committed as `64baade`; [predeclared process and primary sources](../../docs/qwen38-benchmark-process-20260909.md).
- [Metadata](metadata.json), [derived machine-readable analysis](analysis.json), [analysis source](analyze.py), [actual executed shell scripts](executed-scripts/), [follow-up exit codes](followup-cell-exits.tsv), [19 client tests on the host](benchmark-client-tests-v3.txt).
- **MTP4:** [offline HTML](mtp4-clients/betterbench/results.html), [BetterBench JSON](mtp4-clients/betterbench/results.json), [vLLM files](mtp4-clients/vllm-bench/), [speculation](mtp4-clients/spec-metrics/), [launcher](mtp4-clients/launcher.sh), [environment](mtp4-clients/collect_env.txt), [cold sweep](mtp4-long/long-context/summary.json).
- **BF16:** [offline HTML](bf16-clients/betterbench/results.html), [BetterBench JSON](bf16-clients/betterbench/results.json), [vLLM files](bf16-clients/vllm-bench/), [speculation](bf16-clients/spec-metrics/), [launcher](bf16-clients/launcher.sh), [environment](bf16-clients/collect_env.txt).
- **INT4:** [offline HTML](int4-clients/betterbench/results.html), [BetterBench JSON](int4-clients/betterbench/results.json), [vLLM files](int4-clients/vllm-bench/), [speculation](int4-clients/spec-metrics/), [launcher](int4-clients/launcher.sh), [environment](int4-clients/collect_env.txt).
- Canonical host directory: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard/`. Each cell contains its source snapshot, raw requests/responses, exact launcher and client `.command.txt` files.
- BetterBench is pinned to [`GGZ14/BetterBench` commit `1de941d256ddd633a8c117963ba72aebe4b4d5e4`](https://github.com/GGZ14/BetterBench/tree/1de941d256ddd633a8c117963ba72aebe4b4d5e4). Duplicate source checkouts and Python caches are omitted; its [license](BetterBench-LICENSE.txt) and all generated outputs are retained.

Recompute analysis locally: `python3 analyze.py > analysis.json` in this directory (standard library only). To repeat inference, use the archived **pre-fix** runner/source and its recorded command on the same isolated host/model layout with a **new output directory** and the pinned BetterBench checkout. Do not blindly rerun the shell scripts against current HEAD: newer code adds the boundary fix, and existing output directories must not be overwritten. MTP4 uses its recorded old server image while the vLLM benchmark client remains pinned to the newer image.

## Limitations and publication checklist

Greedy output was not universally bit-identical: **12/160** matched BetterBench rows differed in output length. Fixed-output vLLM exact text matches across DFlash arms were **40/48, 42/48, 40/48, 39/48** at C1/2/4/8. Text also varied within unchanged arms across client loads (BF16 C1 vs C2/4/8: 45/48, 41/48, 41/48; INT4: 45/48, 42/48, 43/48). The cause was not established. Strict rejection and unchanged target weights do not prove quality or bitwise invariance. Existing 3-canary/131-boundary/8-functional checks passed per cell but are **not** substitutes for Section 9's suite.

Hardware: one Arc Pro B70, Ryzen 7 5800XT, 33563914240 bytes RAM, CachyOS kernel `7.2.0-1-cachyos`, 275 W cap (**not measured draw**). Both sysfs and privileged lspci reported **2.5 GT/s x1**; [sysfs](pcie.json) and [lspci](pcie-lspci.txt) evidence is retained without asserting a different physical link or tuning it. [Model download metadata](model-download-revisions.json) records target revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` and DFlash2 revision `dedf8df68adfb1afeaf7b7480c0a0243108177b4`.

- [x] Same host/power cap; matched drafter comparison and separately declared current-best bundle reference.
- [x] Model/runtime revisions, complete launch arguments, environment and relevant source snapshots retained.
- [x] Full BetterBench/20-pass and vLLM serving results retained for all three arms.
- [ ] Interleaved A/B: not run. Upstream requires two resident endpoints, which cannot coexist here; no restart hook is provided. Sequential order is disclosed.
- [x] Cold sweeps executed with failures retained: original exact-boundary DFlash cells fail; conservative follow-ups and MTP4 complete. This check means the evidence is recorded, **not that the boundary is healthy**.
- [ ] Peak workload device memory: not measured. Loading/KV estimates and largest successfully tested context are recorded separately.
- [x] Exposed speculative counters and position data retained; no invented histogram or per-token timing.
- [ ] Quality sensitivity unresolved: Section 9's >=500-prompt divergence test and IFEval/GSM8K/HumanEval+/MBPP+ suite **not run**. No quality-preserving optimization claim.
- [x] Raw data, all failures and median/dispersion analysis are included in this result directory for canonical Git storage.
- [x] Capacity/stability tradeoffs are stated alongside the performance differences.

All completed cells verified the persistent launcher and power cap unchanged and Glimmer stopped; original model mounts were read-only. Final pre-fix pipeline exit was 0; the intentional BF16 exact-boundary control retained exit 1, as shown in the per-cell exit file. The earlier INT4 exact-boundary failure is retained separately. This package records a failed publication gate honestly rather than labeling it green.
