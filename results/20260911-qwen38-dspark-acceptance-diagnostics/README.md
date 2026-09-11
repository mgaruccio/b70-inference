# Qwen3.8 DSpark acceptance diagnostics (experiment only)

## Final bounded outcome

Native Markov precision was corrected and upstream probabilistic drafting enabled, but **neither materially improved acceptance**. The initial hypotheses are not established root causes. Keep corrected greedy as the simpler development candidate; probabilistic drafting is workload-dependent, not universally faster.

The graph-configured corrected-greedy run is the largest observed improvement: short-matrix medians **32.771–44.489 tok/s**, versus corrected eager **22.206–28.451 tok/s** (about48–61% higher in corresponding groups). Acceptance remains similar. These are small ordered development comparisons, not interleaved standard benchmarks. The real64K boundary run completed all seven **65536-input +128-output** requests; six measured median decode **16.9763 tok/s**, inclusive IQR **0.27767**, median TTFT **57.3856s**. Prior uncorrected eager64K was14.7631tok/s; that comparison is a precision+graph bundle, not isolated graph attribution.

Task `b306338ff` completed exit0 in20m26s. Shared canaries/131finite boundaries/eight functional checks passed in both graph cells; the short cell additionally passed36 forced512-token matrix requests. Installed overlay replay and host invariants passed. Server logs explicitly show **both target FULL and DSpark FULL graph captures**. Caveat: this V2 path emitted no per-dispatch graph tables/counters despite `--cudagraph-metrics`, so capture is confirmed but replay counts/padding are not independently quantified; do not claim those metrics were collected. `native-graph-greedy/` and `native-graph-64k/` retain logs, exact launch/client argv, requests, SSE, metrics and executed scripts. `comparison.json`/`summarize.py` include all180 short matrix requests.

Production remains unchanged. No160K, concurrency, broad quality, full target-distribution fidelity or standard benchmark claim. Identical greedy prose is not bitwise repeatable even target-only. GPTQ target-weight mismatch remains plausible but unproven; a full same-input reference-draft numerical comparison and alternative target-weight A/B were not performed. Graph execution improves throughput without solving the low-acceptance diagnosis.

This directory contains a bounded diagnostic driver, not a launcher, runtime fix,
benchmark harness, or production configuration.  It does not change the host
launcher, download models, alter repository runtime sources, or claim a speedup.

## What is measured

`run-acceptance-diagnostics.py` has one switch for the public comparison cell:

- `--cell target`: target-only FP16-compute GPTQ Int4 symmetric G128 with FP8 KV;
- `--cell dspark`: the current DSpark candidate using the same target and cache,
  fixed `K=7`, BF16 draft/cache, standard rejection, greedy draft sampling, and
  `enable_adaptive_verification=false`.

Both use the pinned B70 image, V2 eager/C1, context `8192`, one sequence, and
prefix caching disabled.  The candidate mounts the previous campaign's frozen
`draft/` snapshot and replays its existing prefill, BF16-draft, and C1 boundary
overlays in that order.  The driver mounts the previous campaign as
`/experiment:ro`; it does not copy or modify those assets. `--kv-cache-dtype auto`
is available only for a later isolated cache-selector diagnostic; the default and
acceptance cell are `fp8`.

Before the experiment matrix, the existing shared smoke code runs its 3 canaries,
131 finite boundaries, 8 functional checks, and prior 19-request greedy parity
smoke.  The diagnostic matrix is exactly 36 requests per cell:

- code and prose prompts from the previous probe, plus one fixed deterministic
  arithmetic prompt;
- thinking disabled and enabled;
- temperature 0 (`top_p=1`, `top_k=-1`) and temperature 1 (`top_p=.95`,
  `top_k=20`);
- seeds 42, 43, and 44;
- `ignore_eos=true`, `max_tokens=512` on every matrix request.

The request JSON and cache salts are identical in the target and candidate cells.
Every stream records raw request bytes/JSON, raw SSE bytes, parsed SSE events, rendered prompt token IDs,
output token IDs, usage, raw before/after metrics, metric deltas, full per-position
accepted counters, draft-step/proposal/accept deltas, emitted tokens per step,
and transport checks.  The driver deliberately does **not** compare stochastic
output-token parity between cells; output IDs are retained for diagnosis only.

## Fresh primary-source note

The pinned primary model card was read for this diagnostic:

<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/README.md>

It documents DSpark evaluation with SGLang and an NVFP4 target, with the
published acceptance/throughput settings using thinking enabled, temperature
1.0, top-p 0.95, top-k 20.  That is useful context, but it is not equivalent to
this experiment's vLLM B70 GPTQ target + FP8 KV cell, so this driver does not
present the two as a benchmark match.

## Real end-to-end process (lead runs on inference-host only)

Preconditions:

1. On the B70 `inference-host`, stage only the two tracked files in this directory
   into a new sibling campaign (the existing campaign process keeps large draft
   weights outside Git):
   ```bash
   root=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4
   mkdir -p "$root/20260911-qwen38-dspark-acceptance-diagnostics"
   cp results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py "$root/20260911-qwen38-dspark-acceptance-diagnostics/"
   cp results/20260911-qwen38-dspark-acceptance-diagnostics/README.md "$root/20260911-qwen38-dspark-acceptance-diagnostics/"
   ```
   Run from that sibling. The driver resolves the previous frozen campaign from
   either the results-name sibling or the existing host name
   `20260910-dspark-v2-feasibility`; it never downloads the draft.
2. Have no running containers, leave the Glimmer container stopped, keep the
   power cap at `275000000`, keep the pinned image already present (the driver
   uses `--pull=never`), and keep
   `/home/mike/inference/launchers/start-qwen38.sh` at SHA-256
   `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.
3. Keep the previous campaign sibling and its frozen `draft/config.json` and
   `draft/model.safetensors` in place.  The config hash and expected model hash
   are recorded in `dependencies.json`; the draft is mounted read-only.
4. Ensure the new sibling's `target-only` and `current-dspark` output paths do
   not already exist.  Output directories are refused rather than overwritten.

Run the real public API journey, one cell at a time (port 8000 is intentionally
serialized):

```bash
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-acceptance-diagnostics
python3 -u run-acceptance-diagnostics.py \
  --cell target \
  --out target-only

python3 -u run-acceptance-diagnostics.py \
  --cell dspark \
  --out current-dspark
```

The target-only cell is the attribution control; the current DSpark cell is the
old/current candidate baseline for optimization comparisons.  The expected
successful journey is: pinned image inspection, disposable server startup,
model/context confirmation, source replay checks, all shared gates, then 36
finite 512-token SSE streams with passing transport checks.  For DSpark, the
metrics should expose draft steps, proposals, accepts, and per-position
acceptance counters; target-only should not speculate.  These are diagnostic
expectations, not a stochastic parity or quality claim.

Artifacts to retain are `summary.json`, `dependencies.json`,
`launch-argv.json`, `launch-metadata.json`, `launcher.sh`, `image-inspect.json`,
`server.log`, `host-before.json`, `host-after.json`, `container-cleanup.json`,
the source replay records, the shared gate artifacts, and each matrix request,
raw/parsed SSE, prompt/output ID, usage/metrics, and result file.  On success or
failure the driver removes only its own named container and checks host
invariants.  If a run fails, inspect `failure.txt` and `server.log`; do not
reuse an existing output directory.

## Local checks

No torch or inference host is needed for static checks:

```bash
python3 -m py_compile results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py
python3 results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py --help
```

These checks supplement, but do not replace, the real inference-host API
journey above.

## Completed first matrix and next diagnostic

Paired task `b288c7a4e` completed in 39m18s, exit 0. Both `current-dspark/` and `target-only/` passed shared gates and all 36 forced 512-token streams. Host invariants matched and owned containers were removed. Executed drivers are frozen separately in each cell. Lead fixed the worker's missing `cell.rows.append` in the shared-chat adapter, added shell `set -e`, and skipped container cleanup when no launch was made, before running. Static row-accounting/metric regression checks passed.

Aggregated by thinking/temperature across the three workload families and three seeds (development-only; small, ordered diagnostic sample):

| Thinking | Temperature | DSpark emitted/step | First-position acceptance | DSpark median decode tok/s | Target median decode tok/s |
|---|---|---|---|---|---|
| off | 0 | 2.348 | 0.600 | 26.474 | 15.229 |
| off | 1 | 2.296 | 0.597 | 26.666 | 15.090 |
| on | 0 | 2.108 | 0.573 | 22.156 | 15.278 |
| on | 1 | 2.005 | 0.520 | 21.098 | 15.108 |

Thinking did not recover acceptance on these prompts. All paired request payloads and rendered prompt IDs match. However only **10/18 greedy 512-token outputs match exactly**: divergence starts at positions148–497. Target-only itself produced multiple outputs for identical greedy prompt/settings with different seeds (which should not affect argmax). This blocks attributing divergence to speculation or claiming full greedy identity; successful transport is not proof of distribution fidelity. Sampled output equality was not required.

The source audits found no clear tap/residual, FC packing, anchor/position, RoPE/mask or Markov conditioning discrepancy. Quantization mismatch remains unproven. A follow-up corrected the initial audit's sampler-equivalence assumption: [SGLang v0.5.17 sampler](https://github.com/sgl-project/sglang/blob/v0.5.17/python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py), commit `29481685462732237d80d86076d6563e1f658102`, uses stochastic proposals with folded sampling/top_k20, draft temperature but no draft top-k/p filtering, and probability-ratio rejection. Our matrix changes target sampling but keeps **greedy drafts**. [vLLM's native Gumbel path](https://github.com/vllm-project/vllm/blob/73029d424/vllm/v1/worker/gpu/sample/gumbel.py) supports probabilistic drafts; testing it requires extending only the current opt-in guard, not inventing a sampler or relaxing rejection. The published score still cannot be directly attributed to this difference.

Next real API diagnostic: `run-repeatability.py --cell target --out target-repeatability`, then `--cell dspark --out dspark-repeatability`. Same image/8192/eagerC1/runtime guards and shared gates; four exact-payload repetitions each of code/thinking-off and prose/thinking-on, seed42/temp0,512forced tokens. Within each family, even cache salt is identical; prefix caching remains disabled. Expected outcome is eight complete streams with explicit distinct-output/first-divergence reporting, not assumed bitwise stability. Raw request/SSE/metrics plus `repeatability.json` retained; no overwrites. Driver outer timeout1500s per cell and standard owned-container cleanup. This separates baseline repeatability from speculator-specific divergence before any optimization claim.

Prime Lab preview/show-patch tools remain unavailable (`ENOENT` local socket); previews were stated in chat under the user's explicit optimization authorization. No production launcher was modified.

## Native precision and probabilistic comparison results

Fixed-seed task `b2eabe337` passed all streams and invariants. Code had one distinct output in both cells; identical-payload prose had **two target-only outputs** (first difference273) and **three DSpark outputs** (first difference267). These establish baseline nondeterminism, not its numerical cause. Retained `target-repeatability/` and `dspark-repeatability/` contain exact repeated payloads and results.

Deeper installed-source inspection found a concrete precision discrepancy missed by the first audit: the draft-local [LogitsProcessor](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/model_executor/layers/logits_processor.py) inherits the target FP16 head dtype, casting BF16 Markov operands to FP16. Original mocks concealed that cast. Worker `8c19fe4`, integrated as `2ae0e55`, corrects draft-local `head_dtype=None` in both modes while retaining the explicit FP16 shared-head activation cast, and preserves the combined logits in FP32 for native probabilistic cache/rejection. Native Gumbel and standard rejection remain unchanged. All new helpers are pinned. Apply from pristine installed sources; old overlays are not upgraded in place.

Predeclared execution: `run-native-sampling.py --draft-sample-method greedy --cell dspark --out native-markov-greedy`, then `--draft-sample-method probabilistic --cell dspark --out native-probabilistic`, unchanged 36-request matrix/shared gates/8192/eagerC1. Nested read-only overlay bind leaves original campaign assets intact. Source replay and effective config are recorded; outer timeout2700s per arm, standard cleanup. Task `baac5f6ff` ran the full installed-image CPU suite first (**23/23, zero skips**, `cpu-native-sampling-tests.log`) and then both API arms; all passed, exit0 in30m46s, both host invariants unchanged. Selective probability/cache review found no blocking contract break.

`python3 -B summarize.py > comparison.json` reproduces all 144 matrix-stream validations, identical paired payloads/prompt IDs, acceptance by position, medians and greedy divergence. Each candidate's executed driver/wrapper/overlay is frozen in its cell.

| Thinking / target temp | Old greedy tok/s | Corrected greedy tok/s | Probabilistic tok/s | Old / corrected / probabilistic emitted per step |
|---|---:|---:|---:|---|
| off / 0 | 26.474 | 27.700 | 27.985 | 2.348 / 2.329 / 2.347 |
| off / 1 | 26.666 | 28.451 | 25.378 | 2.296 / 2.326 / 2.188 |
| on / 0 | 22.156 | 23.264 | 23.458 | 2.108 / 2.074 / 2.095 |
| on / 1 | 21.098 | 22.206 | 24.405 | 2.005 / 1.987 / 2.059 |

**Neither Markov precision nor sampler choice materially recovered acceptance.** The precision-corrected greedy arm is modestly faster here (roughly5–7%), but this small ordered matrix has no interleaved confidence interval, so it is not a robust speedup claim. Probabilistic mode is a supported diagnostic option, not a universal recommendation. Long greedy exact matches against target-only remain10/18 (old),10/18 (corrected),11/18 (probabilistic), with baseline nondeterminism unresolved. No target-distribution equivalence or production readiness is asserted. Weight-quantization causality and full identical-input reference draft numerical parity remain unestablished; no compatible alternative target was installed on this B70.

## Predeclared native graph check

Read-only audit of the installed pin confirmed native XPU graph support. Fresh primary source review: [XPU runner adapter](https://github.com/vllm-project/vllm/blob/73029d424/vllm/v1/worker/xpu_model_runner.py) redirects CUDA graph APIs to `torch.xpu.graph`/`XPUGraph`; [V2 graph manager](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/gpu/cudagraph_utils.py) otherwise appears CUDA-specific. The adapter is essential; the manager alone does not prove incompatibility. Torch2.13 satisfies the pinned graph-support gate. An attempted official XPU docs URL returned404; installed/pinned sources were used rather than assuming missing documentation meant missing support.

Smallest native route: retain corrected-greedy overlay, remove `--enforce-eager`, set `VLLM_XPU_ENABLE_XPU_GRAPH=1`, and pass `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[7,8]}' --cudagraph-metrics`. No torch.compile/Inductor port or new kernel. Keep all other settings unchanged. Requested sizes reflect seven draft queries and eight target verification positions; inspect actual capture/replay dispatch instead of trusting flags.

Task `b306338ff` executes `run-native-sampling.py --draft-sample-method greedy --graphs --cell dspark --out native-graph-greedy` (same shared gates +36request8192 matrix), then only on success `--graphs --long-context --cell dspark --out native-graph-64k` (maxlen65664, same shared gates then exact65536input+128output, one warmup+six measured requests through the existing byte-identical long client). Each driver is bounded by2100s with owned-container cleanup; the long client has1200s. Preserve source replay, actual graph-stat tables, request/SSE/metrics and failures. Baseline for graph attribution is `native-markov-greedy/`; prior64K eager result is contextual until matched within this corrected stack. No graph/64K success is assumed by launch. Production launcher/power/Glimmer invariants remain mandatory.
