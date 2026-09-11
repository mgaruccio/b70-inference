# Real target-weight / native-execution acceptance control

**Development, quality-sensitive diagnostic: both actual GPU arms completed successfully.** See [observed-results.md](observed-results.md) for measured acceptance, native kernels, offload differences and cleanup. The sections below preserve the pre-launch protocol and its then-pending caveats. Not a standard-publishable benchmark, throughput comparison, production promotion, or isolated bit-precision experiment. No persistent launcher changes.

## Declared comparison and real end-to-end process

Baseline A: existing GPTQ Int4 symmetric G128 target, native XPU kernel, **8 GiB CPU offload budget**. Candidate B: official `Qwen/Qwen3.8-27B-FP8` revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, native block-FP8 kernel, **the same 8 GiB budget**. Intentional difference is target weights/quantization **plus native execution bundle**. FP8 outputs may differ; cross-target token identity is not a gate.

Both arms: one B70, 275 W, pinned image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`, context 8192, target compute float16 and KV fp8, standalone corrected BF16 DSpark K7, greedy draft, standard rejection, adaptive verification off, C1, eager, prefix cache off. The execution-supplied canonical overlay is copied into each run and its required SHA256 recorded; both arms must use the same overlay. No grouped-normalization fix is implemented here.

Environment/preconditions: run on the isolated inference host, no other running containers, Glimmer stopped, unchanged launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; public FP8 download completed and verified by the lead; frozen prior draft present; all runtimes/workloads remain outside interactive Pi. No new dependency installation or paid resources.

Executable journey: lead runs `run-quant-control.py` first for GPTQ and then FP8, using the commands below. Each run uses a new child output directory, validates asset/model identities, starts only its own container, requires exact source replay and measured post-load native kernel/parameter/offload metadata, waits for `/health` and `/v1/models`, then reuses the original **3 canaries + 131 finite boundaries + 8 functional checks + 19-request greedy smoke**, and only then the original **36 requests** (code/math/prose × thinking off/on × temp0/temp1 × seeds 42/43/44; 512 forced output tokens). The driver's existing SSE, metrics, sandbox and transport checks remain authoritative. Pair exact request payloads, rendered prompt IDs and sampling settings, not generated output IDs. Retain launch commands, environment, server log, source hashes, loaded kernel/layout/dtype records, memory/offload measurements, raw requests/SSE/metrics, all failures and cleanup. Stop on OOM, unsupported native kernels, numerics, changed source or pairing failures—never relax guards or force a fallback. Cleanup removes only the run-owned container and verifies unchanged host invariants; it never restarts/stops production services or deletes model snapshots/results.

The existing `current-dspark` **same-target no-offload** result is recorded as an offload diagnostic, not a speed baseline. Its older overlay means it is not an isolated offload control when a new canonical correction is supplied. Pairing incompatibilities are reported, not silently ignored. No performance conclusion is allowed from offloaded timing.

## Fresh primary-source findings (2026-09-11)

- Model metadata: https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8 and https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/revision/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a . Public pinned revision confirmed.
- Exact config: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/config.json . SHA256 `74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc`; quant_method=fp8, dynamic activations, e4m3, **weight_block_size=[128,128]**, 882 modules_to_not_convert. This corrects the initial no-block-size assumption.
- Exact index: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/model.safetensors.index.json . SHA256 `f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2`.
- Pinned-image CPU source inspection: `Fp8LinearMethod` uses GroupShape(1,128) activations and GroupShape(128,128) weights; `init_fp8_linear_kernel` dispatches per-group activations through `_POSSIBLE_FP8_BLOCK_KERNELS`. XPU's first native candidate is `XPUFp8BlockScaledMMKernel`, whose inherited `apply_weights` quantizes input dynamically and calls `_xpu_C.fp8_gemm` with FP8 activations and block scales. This is **W8A8 block-FP8 with FP16 output**, unlike per-tensor `XPUW8A16FP8LinearKernel` (which genuinely takes raw FP16 input into `fp8_gemm_w8a16`). No kernel is forced here; any unexpected selection fails the post-load gate.
- Native block path requires K divisible by 128; ragged N uses gcd(N,128), a multiple of 16. Scales are exposed as [N-groups,K/128] transposed view backed by contiguous [K/128,N-groups]. Source admissibility is not GPU execution evidence.
- UVA only offloads parameters until its budget is reached (may overshoot by one parameter); it does **not** put all model weights in RAM. `device_loading_context` re-offloads replacement Parameters whose UVA marker was dropped, while in-place `.data` repacking can leave stale markers on device storage. Both initial offload and postprocessing re-offload backing addresses are recorded without retaining tensors, then compared with post-load parameters. Report these matches, not marker flags or an assumed 8-GiB device relief. XPU allocator/device free memory and process RSS are sampled after target and target+draft load. This bounded load inspection is not a profiler or complete driver-owned-memory accounting.

## Lead commands

Run on the inference host, with `R` set to the directory containing these copied assets and the lead's completed `target-fp8/` + `download-result.json`. `D` is the existing acceptance-diagnostics directory; `H` is the frozen feasibility directory. Use the same supplied canonical bytes for A and B. Shell logs are new paths (`noclobber`) and output directories must not already exist.

```bash
set -euo pipefail
set -o noclobber
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-dspark-quant-control
D=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-dspark-acceptance-diagnostics
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
# Point this to the lead-approved canonical candidate copied to the host:
: "${CANONICAL_OVERLAY:?set the canonical overlay path}"
: "${CANONICAL_SHA256:?set the reviewed exact SHA256}"
printf '%s  %s\n' "$CANONICAL_SHA256" "$CANONICAL_OVERLAY" | sha256sum -c -

PYTHONDONTWRITEBYTECODE=1 python3 "$R/run-quant-control.py" \
  --arm gptq --out "$R/gptq-offload8" \
  --driver "$D/run-acceptance-diagnostics.py" --previous-campaign "$H" --draft-dir "$H/draft" \
  --canonical-overlay "$CANONICAL_OVERLAY" --canonical-sha256 "$CANONICAL_SHA256" \
  --no-offload-reference "$D/current-dspark" \
  > "$R/gptq-offload8-driver.log" 2>&1

PYTHONDONTWRITEBYTECODE=1 python3 "$R/run-quant-control.py" \
  --arm fp8 --out "$R/fp8-offload8" --target "$R/target-fp8" \
  --target-manifest "$R/download-result.json" --paired-with "$R/gptq-offload8" \
  --driver "$D/run-acceptance-diagnostics.py" --previous-campaign "$H" --draft-dir "$H/draft" \
  --canonical-overlay "$CANONICAL_OVERLAY" --canonical-sha256 "$CANONICAL_SHA256" \
  --no-offload-reference "$D/current-dspark" \
  > "$R/fp8-offload8-driver.log" 2>&1
```

Launch these as a lead-owned background shell job, not in foreground Pi. If the lead staged `R`/`D` under different names, only those path variables change. `--startup-timeout` and `--request-timeout` default to 1800 seconds; failures retain traces, not retries with altered numerics. There is no implicit image pull. `summary.json` reports the truthful target dtype, offload accounting, source identity and paired inputs; `paired-gptq-offload.json` contains only paired acceptance/counters, never speedups. `historical-gptq-no-offload.json` reports the same-target historical input pairing and its explicit confounding limitation.

## CPU verification and source replay

From the repository root, using only stdlib Python and the existing exported pinned-image sources:

```bash
PYTHONDONTWRITEBYTECODE=1 VLLM_SOURCE_ROOT=/tmp/vllm-pinned-73029d424 \
  python3 results/20260911-qwen38-dspark-quant-control/test-quant-control.py
```

Optional `CANONICAL_OVERLAY=/absolute/candidate.py` tests a supplied canonical candidate. No test imports torch/vLLM or opens a GPU. CPU environment: Python 3.14.7, Linux 7.2.0-1-cachyos x86_64. Observed: **13 tests passed**, including full pristine → composed overlay → exact replay over **38 pinned source files**, rejection of damaged/mixed native and rejection-sampler sources, guard preservation, model-identity refusal, native-kernel refusal, original 36-case generation, exact input pairing allowing different target outputs, real driver launch composition and pre-matrix native-report failure. The apply/replay test uses a disposable `/tmp/quant-control-source-*` copy and removes it. `cpu-tests-final.txt` is the final raw passing output; `cpu-tests.txt` is the earlier passing output before the loader re-offload capture was added. `cpu-tests-initial-failure.txt` retains an initial test-only failure (a string test confused `==` with assignment; corrected to distinguish them).

Container replay remains the original driver's `source_check`: `runpy.run_path('/experiment/patch_dspark_bf16.py')` resolves this extension, verifies the mounted canonical SHA, and requires `prepare(current_sources) == current_sources` across the complete pin set, alongside the original prefill/boundary checks. Native kernels are pinned but never edited. The only extra runtime edits are the explicit FP8 config guard and four one-shot load/offload capture sites; no profiler framework, new service, sampler, target remapping or fallback is installed.

**Not yet verified:** the B70 load, physical memory headroom after all runtime allocations, XPU block-FP8/offloaded kernel execution, or API/36-matrix acceptance results. The lead must run the above real-system process. A source-supported kernel is not evidence it fits or runs. Download manifest full-weight hashes are trusted from the lead's completed public downloader; this wrapper rechecks exact config/index hashes and every indexed shard's size, rather than rereading 31 GB. The historical no-offload run may use older DSpark numerics. Publication checklist is intentionally incomplete (no full BetterBench, `vllm bench serve`, long-context sweep, or quality suite); report this campaign only as development acceptance diagnostics.
