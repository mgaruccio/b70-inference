# Qwen3.8 DFlash2 cache grouping and prefill-budget experiments

Status: **six cells completed; two startup capacity failures retained**. The largest successful request was **180096 input + 128 output = 180224 total tokens**. The 212992 and native 262144 limits still do not fit. **Development tier**, not a standard-publishable throughput claim. Production launcher unchanged.

## Request and scope

User: “ok proceed with those suggested fixes”: test eight-layer cache grouping to remove target-layer padding, and smaller prefill token batches to recover cache headroom. Keep target/draft checkpoints, FP8 target KV, BF16 draft KV, FP32 recurrent state, strict K7 acceptance, graph policy and 0.95 utilization unchanged. Do not promote a persistent configuration.

The paired baseline is the current partial RTN INT4 DFlash configuration on the same pinned image and host. MTP4's previously committed results remain a historical whole-stack reference, not an isolated control for these allocator changes.

## Fresh external research (before implementation)

- https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ — groups must be homogeneous in cache specification and share a uniform physical page size. The documented minimum-layer-count heuristic can introduce excessive padding. Preserve these invariants; do not remove recurrent rollback slots or alter cache layouts. The documentation explicitly warns it describes an older commit; the implementation source of truth is pinned commit `73029d42441321b631779db3475031f5ec26dd6c`.
- https://docs.vllm.ai/en/latest/configuration/optimization/ — smaller `max_num_batched_tokens` changes chunked-prefill scheduling and can trade TTFT/throughput against decode latency and memory. Re-profile and measure; do not assume activation reservations scale linearly.
- Pinned allocator: https://github.com/vllm-project/vllm/blob/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/core/kv_cache_utils.py . The stopped-image source was inspected before modification. Existing grouping has 48 Mamba / 16 full-attention / 5 sliding-window layers, grouped by five. The candidate uses groups of eight with the original striding and shared-page allocation unchanged.

## Defined end-to-end process

- **Environment/preconditions:** idle `inference-host`, Arc Pro B70, 275 W; no running containers; Glimmer stopped. Persistent launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`. Image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`; vLLM `73029d424`, XPU kernels 0.1.14.1. Reuse read-only target and partial RTN INT4 drafter. Runtime dependencies stay on the inference host, outside interactive Pi.
- **Matched 64K matrix:** automatic grouping / prefill 8192 (baseline), group8 / 8192 (grouping only), automatic / 2048 (prefill only), group8 / 2048 (combined), group8 / 4096 (tradeoff). Same max model length 65536, C1, graph sizes `[1,2,4,8]`, thinking/prefix cache/audit off.
- **Public boundary:** use `scripts/experiments/qwen38_standard_bench.py --long-context-only --context 65536 --lengths 512 32768`, with explicit candidate switches. Existing startup canaries, finite-logprob checks and executable functional tests must pass. `/v1/models` must report the requested limit. The cold client tokenizes exact deterministic prompts through `/tokenize` and streams `/v1/completions` with greedy seed 42, ignored EOS and 128 output tokens.
- **Data/repetition:** 512, 32768, and the automatically added near-limit prompt 65408; one warmup and six measured requests per supported point. Exact-limit `65408 + 128` must finish, one-token-over `65409 + 128` must return a context-limit HTTP 400. Preserve request/SSE/usage/health and per-request speculative counters. Compare identical payloads across cells; keep every failure/outlier.
- **Long context:** after matched-cell validation, use the combined smaller-batch configuration for a 212992-token launch and the standard 512/8192/16384/32768/65536/120000/160000 sweep plus 190000 and exact near-limit points. Attempt native 262144 separately. Startup failure is a failed capacity attempt, not a measured maximum; a successful startup must be followed by actual long streaming requests. Do not silently lower requested context or change unrelated knobs.
- **Predeclared capacity fallback (after measured prefill-only result):** the 2048-token batch recovers only about 0.54 GiB of allocated cache, not a linear 4x reduction in activation memory. If the 212992-token launch is cleanly rejected for insufficient cache, retain that failure, attempt 262144 separately, then measure a separately named `g8-2048-176k` cell at 180224 total tokens through the standard points and `180096 + 128` exact boundary. This is not a successful 212992/262144 result. `run-candidates.sh` stops on any non-capacity or cleanup failure.
- **Expected results:** no layout/assertion/kernel failures, finite logprobs, successful functional checks, exact prompt and 128-output usage, `[DONE]`, `finish_reason=length`, and healthy server after requests. Actual group8 warning should describe only three padded sliding-window layers, not padded target attention/Mamba layers. Smaller prefill settings must appear in frozen launch arguments and runtime configuration.
- **Evidence/analysis:** exact commands, executed source snapshots, environment and model configs, raw API data/counters, server memory logs, cell exits. Report median/IQR/CV, TTFT, post-first-chunk decode proxy and end-to-end throughput. Separate profiled activation reservations and allocated cache from workload peak memory (not measured by this harness). No exact GPU-kernel timing attribution or significance claim from sequential cells.
- **Cleanup:** serial owned disposable containers only; verify no remaining workload containers and unchanged launcher/power/stopped Glimmer after every cell. Retain failed attempts. Commit and push the verified implementation and bounded artifacts.

## Publication limitations

This scoped development comparison does not rerun full BetterBench, `vllm bench serve`, a broad quality suite, or an independent workload-peak memory sampler. It is not community-comparable/standard-publishable, does not prove a mathematical maximum context, and does not authorize production promotion.

## Measured results

All throughput numbers below are medians of six valid measured requests, excluding one warmup. Decode is the existing post-first-stream-chunk proxy, not per-token ITL: `(128 - 1) / (stream_end - first_nonempty_chunk)`. Speculative bursts make that distinction important. Sequential cells are not an interleaved significance test.

### Matched 65536-total-context comparison

| Cell | 512 decode tok/s | 32768 decode tok/s | 65408 decode tok/s (IQR) | 65408 TTFT seconds | Allocated cache GiB |
|---|---:|---:|---:|---:|---:|
| Automatic groups, prefill 8192 (baseline) | 71.09 | 49.47 | 43.63 (2.78) | 51.36 | 6.633 |
| Automatic groups, prefill 2048 | 74.98 | 50.21 | 43.78 (2.24) | 52.45 | 7.173 |
| Group8, prefill 8192 | 72.26 | 50.19 | 44.27 (2.15) | 51.37 | 6.627 |
| Group8, prefill 2048 | 75.19 | 49.69 | 44.39 (2.06) | 52.45 | 7.160 |
| Group8, prefill 4096 | 74.32 | 50.78 | 44.53 (3.08) | 51.68 | 6.982 |

Group8 + prefill2048 changes decode by **+5.76% / +0.45% / +1.74%** at 512 / 32768 / 65408. The near-64K TTFT increases **2.12%** and end-to-end throughput falls **1.88%**. Small decode improvements are not established as significant. Group8 + 4096 gives a smaller TTFT penalty, but less cache capacity. This is primarily a capacity improvement, not a demonstrated long-context speed breakthrough.

### Longer-context candidate: group8, prefill2048, limit180224

| Exact input tokens | Decode tok/s | Decode IQR | TTFT seconds | End-to-end output tok/s | Mean acceptance length including bonus |
|---:|---:|---:|---:|---:|---:|
| 512 | 76.24 | 3.65 | 0.28 | 65.712 | 3.410 |
| 8192 | 56.89 | 6.97 | 4.35 | 19.481 | 2.861 |
| 16384 | 56.92 | 7.69 | 9.31 | 11.099 | 2.913 |
| 32768 | 49.72 | 1.82 | 21.17 | 5.396 | 2.835 |
| 65536 | 45.11 | 5.10 | 52.49 | 2.314 | 2.946 |
| 120000 | 38.80 | 1.75 | 127.54 | 0.979 | 3.081 |
| 160000 | 36.17 | 1.44 | 200.47 | 0.628 | 3.310 |
| 180096 | 29.03 | 4.26 | 242.83 | 0.518 | 2.669 |

Each row adds 128 output tokens. The last row exercises the exact configured boundary, not a proven maximum. Its near-boundary speculation counters can be affected by clipped final proposals. Acceptance length is `1 + accepted / draft_steps`; full proposed/accepted/depth counters, CV and latency dispersion are in [analysis.json](analysis.json).

### Historical MTP4 reference, not an isolated A/B

The earlier [MTP4 cold sweep](../20260909-qwen38-dflash2-rtn-standard/mtp4-long/long-context/summary.json) used the same host and identical request payloads at the three points below (all seven requests match at each length). However, MTP4 uses the older pinned vLLM/XPU-kernel stack, a different speculative method, and a 212992 configured limit versus 180224 here. Treat this as a historical whole-stack comparison.

| Input tokens | Current DFlash decode | Historical MTP4 decode | Current DFlash TTFT | Historical MTP4 TTFT |
|---:|---:|---:|---:|---:|
| 65536 | 45.11 tok/s | 55.75 tok/s | 52.49 s | 53.88 s |
| 120000 | 38.80 tok/s | 56.88 tok/s | 127.54 s | 134.33 s |
| 160000 | 36.17 tok/s | 46.39 tok/s | 200.47 s | 213.90 s |

MTP4 remains faster in long-context decode. DFlash has lower TTFT in these cold 128-output-token requests, so lower decode speed alone does not mean worse end-to-end latency. Within DFlash, acceptance length rises from 2.946 at 65536 to 3.310 at 160000 while decode speed falls; an acceptance collapse is not sufficient to explain the slowdown. No component/kernel profile was collected, so these results do not assign exact costs to draft generation, verification or attention kernels.

## Memory and capacity findings

- Model-loading memory is unchanged at **18.34 GiB**. Reducing prefill8192 to2048 lowers profiled activation reservation only **2.84 → 2.49 GiB**; logged consumed memory **including weights and runtime** changes **19.30 → 19.12 GiB**. Graph memory remains about **0.16 GiB**. The activation reservation does not scale linearly with prefill budget.
- Automatic grouping + 2048 grows the allocated pool **6.633 → 7.173 GiB**. Group8 + 2048 allocates **7.160 GiB** because its shared pool pages are larger and allocation rounds down to whole pages.
- Group8 removes all padded target attention/Mamba layers: six Mamba groups of8, two full-attention groups of8, and one draft sliding-window group of5 padded to8. The log's **60% padding warning applies to the five draft layers**, not total cache or total VRAM. This exchanges bounded draft padding for removal of growing full-attention padding.
- The source-derived cache-allocation reconstruction reproduces the runtime's reported token-capacity values for every successful cell. Those values are concurrency-normalized estimates, not experimentally proven C1 context ceilings.
- At the successful 180224 limit, the reconstructed per-request admission requirement is **6.982 GiB** within the **7.160 GiB** allocated pool. Largest actually successful prompt: **180096 + 128 output**.
- **212992 startup failed:** **7.95 GiB needed versus 7.15 GiB available for admission**. **262144 startup failed:** **9.47 GiB needed versus 7.15 GiB available**. Both errors estimate 186368 maximum, but that limit was **not tested**. Initial profiling logs report 7.17 GiB before downstream budgeting/page allocation; do not substitute that rounded number for the admission error.
- Full native context remains unmet. The requested grouping/prefill changes are verified, but the remaining shortfall is about **0.80 GiB at 212992**, or **2.32 GiB at 262144**, under these pinned settings. No cache dtype, target weights, acceptance policy or utilization changes were added to conceal it.

## Verification and reproduction

- **161 valid forced-length streams**, including **138 measured requests**, across six completed cells; **six correct context-limit HTTP 400s**. All 18 canaries, 786 finite-logprob boundaries and 48 executable functional checks pass. The two capacity failures occur during startup and are not counted as successful streams.
- **88 paired request payloads** match across the four 64K candidate/control comparisons against baseline, including warmups and rejection probes. Historical MTP payload matching is separately noted above.
- **48 focused CPU tests pass; one optional stopped-image XPU-source test is skipped.** Cache-group, launcher and cold-client tests run without skips. The actual grouping function is exercised with the retained [pinned source](reference-source/kv_cache_utils.py); native allocation and speculative boundary behavior are exercised by the real serving image and API tests.
- Exact executed code snapshots: [code/](code/); each cell also freezes its applied overlays. Per-cell `*.command.txt`, `launch-argv.json`, `launcher.sh`, `collect_env.txt`, `packages.txt`, target/draft configs, `server.log`, raw request/result/SSE files and speculative metrics are retained.
- [cell-exits.tsv](cell-exits.tsv) preserves all eight exits. Candidate campaign exit0 means the declared sequence finished, **not that both requested larger contexts fit**. Failed cells remain failed in the machine-readable analysis.
- [final-cleanup.txt](final-cleanup.txt) and per-cell cleanup files confirm no remaining GPU containers, the original launcher hash, 275 W power cap and stopped Glimmer. The persistent launcher was not promoted or modified.

Exact host entrypoints (the scripts refuse to overwrite existing cells; reproduction requires a fresh copied artifact directory):

```bash
ssh -n inference-host 'bash /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dflash2-cache-efficiency/run-cells.sh auto-8192-64k auto-2048-64k'
ssh -n inference-host 'bash /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dflash2-cache-efficiency/run-candidates.sh'
```

Reproduce the report without a GPU:

```bash
python3 results/20260910-qwen38-dflash2-cache-efficiency/analyze.py > /tmp/qwen-cache-analysis.json
cmp /tmp/qwen-cache-analysis.json results/20260910-qwen38-dflash2-cache-efficiency/analysis.json
```

Focused verification, from the repository root:

```python
import unittest
patterns = ['test_qwen38_standard_bench.py', 'test_qwen38_dflash2_probe.py',
            'test_qwen38_dflash2_cache_groups.py', 'test_qwen38_long_context_bench.py',
            'test_qwen38_xpu_boundary.py']
suite = unittest.TestSuite(unittest.defaultTestLoader.discover('tests', pattern=p) for p in patterns)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
```

The raw JSON/JSONL files parse successfully, and the credential-pattern scan found no candidates. Workload peak device memory, a new quality-equivalence suite, interleaved confidence intervals and a component timing profile remain **unmeasured**. No broader performance or fidelity claim is made.
