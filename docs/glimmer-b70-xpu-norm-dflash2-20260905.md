# Glimmer B70: XPU norm fix, working DFlash2, and kernel profile

## Retained result

The existing concurrent launcher now inherits an XPU-only per-layer context-K
RMSNorm fallback from `scripts/patch-vllm-dflash-gptq-context-kv.py`.
No launcher flags, model weights, or kernel wheel versions changed.
**350 aggregate tok/s is not achieved.**

Pinned `vllm-xpu-kernels==0.1.13.2` accepts stacked `[L, head_dim]` norm weights
but applies row zero to every layer. A two-layer reproduction with a zero second
weight row produced maximum output **4.035 FP16 / 4.031 BF16**, instead of zero.
The fallback supplies each layer's actual 1-D weight. CUDA and 1-D weight calls
retain the existing grouped path. Shape/source mismatches fail closed.

Matched public SSE screen: eight clients, sky explanation prompt, temperature 1,
top_p .95, top_k 64, seed 42, 256 output tokens, one C8 warmup, prefix caching off.
Aggregate e2e tok/s is completion-token sum divided by concurrent wave wall time,
not per-stream decode. Context remains 131072; this is not eight reserved 128k slots.

- Same-session baseline: **299.726** median, 299.720–300.127 (3 waves).
- Initial norm fix: **320.396**, 318.436–322.453 (3 waves).
- Independent cold restart, exact repo launcher/overlay: **320.318**,
  318.626–322.819 (5 waves), **+6.9%** over the same-session baseline.
- Completed-answer smoke: **8/8** for baseline, initial fix, and cold restart
  (existing GSM8K Janet, HumanEval-0 doctests, and MT-Bench Hawaii checks).
- Initial mean emitted tokens per draft step: **2.177 → 2.262**, computed as
  `1 + accepted_tokens_delta / draft_steps_delta` from Prometheus counters.
- All short-screen requests returned 256 tokens and visible text without errors.
  Capped throughput probes are not completed-answer quality scores.

## Numerical and context verification

The actual overlaid `_normalize_context_k` method was extracted from the running
container and invoked with real XPU tensors `[5,32,8,128]`, distinct per-layer
weights, including a zero row. FP32 reference comparison passed in both dtypes;
maximum absolute errors were 0.00521 FP16 and 0.03659 BF16, within dtype-appropriate
mixed relative/absolute tolerances. The zero row was exactly zero in both.
Six CPU-only regression tests cover layer dispatch, non-XPU/1-D behavior,
shape mismatch, patch replay, and unsupported-source rejection.

With the same corrected runtime (profiler stopped):

- 12 staggered 2042-token requests: peak 8 active, queue and replacement streaming
  verified, clean drain, zero preemptions; 26.76 s.
- **128994 prompt tokens**: beginning/middle/end checkpoint retrieval correct,
  natural stop, clean drain, zero preemptions; 93.34 s.
- **8 × 65532 prompt + 2048 output tokens**: all successful, peak 8 active,
  84.18% KV usage, clean drain, zero preemptions; 413.65 s.

These repeat the prior retained workload, not a broad model-quality evaluation.
The prior six-request near-native empty-response failure is not resolved here.

## DFlash2: works, but not promoted

An isolated newer image loaded the native BF16 draft successfully:

- Image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`
  (`nightly-73029d42441321b631779db3475031f5ec26dd6c`).
- vLLM `0.28.1rc1.dev278+g73029d424.xpu`, kernels 0.1.14.1,
  torch 2.13.0+xpu, triton-xpu3.7.2.
- Draft `z-lab/Muse-Glimmer-30B-DFlash2`, revision
  `b54ffdd11fa9cfe2af370012e5763d492c904128`, native BF16, no draft GPTQ flag.
- Same GPTQ target; server `--dtype bfloat16`; same C8/K3, FP8 KV, 131072 limit,
  prefix-cache-off, and corrected XPU norm overlay.

C1 accepted 148 draft tokens during a 256-token request (51.03 client decode tok/s).
C8 median was **290.695 aggregate tok/s** (290.491–291.644), mean emitted tokens
per draft step 2.404, and 8/8 answer checks passed. Thus the old zero-acceptance
failure is avoided, but greater acceptance did not compensate for this setup's
cost. This comparison changes runtime, dtype and draft together; it does not
isolate which change causes the regression. No direct all-logit finiteness trace
or larger-K sweep was performed.

Loading consumed 22.42 GiB; reported KV capacity was only **157950 tokens total**.
The 131k per-request limit remained, but this is a major resident-capacity loss.
The native DFlash2 setup was not promoted or subjected to the 8×64k stress test.
Its exact generated launcher and logs remain in the remote artifact directory.

## Kernel evidence, not a speedup claim

Used vLLM's Torch CPU/XPU profiler via `/start_profile` and `/stop_profile` on
real C8 SSE requests, separately from clean throughput measurements.

- Graph-replay trace (20 decode iterations): captured
  `execute_context_0(0)_generation_8(32)` averaged **34.697 ms/step**.
  Inner graph kernels are absent from this trace. Visible GEMMs outside the
  graph total 5.765 ms/step. Layerwise context-K norm is only 0.012 ms/step.
- Eager diagnostic (4 decode iterations): GEMMs 74.47% of device-kernel duration.
  Eager also disables compilation, so its extra native normalization/copy costs
  are not evidence of an equivalent opportunity in normal compiled serving.
- **Compiled, no graph replay** (4 decode iterations): GEMMs **84.13%** of
  device-kernel duration. External-id correlation maps these to:
  - INT4 MLP gate/up `[32,6656] → 39936`: 16.007 ms/step.
  - INT4 MLP down `[32,19968] → 6656`: 8.645 ms/step.
  - Dense vocabulary projections, M32 and M24, K6656, N202048: 4.656 + 4.628 ms/step.
  - Other INT4 projections: approximately 6.63 ms/step combined.

MLP totals include 52 target and 5 draft layers (228 calls over 4 iterations per
MLP shape). The draft context QKV path is 5 calls/step, not 100 calls/step.
Profiled device-duration sums are not clean end-to-end latency or additive
speedup guarantees. Any alternative GEMM must pass numerical comparison and
then improve the original graph-enabled public C8 workload.

## Experimental M32 Triton filter: rejected for serving

`scripts/experimental/glimmer_xpu_w4a16.py` is a standalone experiment, not a
serving patch. It implements the actual K-packed GPTQ layout, six bounded
configurations, FP32 accumulation and deterministic split-K reduction. It loads
the pinned oneDNN operator through `vllm._xpu_ops`, not bare Torch.

Executed on an otherwise idle B70 in a temporary container from the original
image, with kernels 0.1.13.2. All six configurations passed the full awkward
`M13/K384/N77` reference and selected-column checks at the observed shapes.
Observed maximum absolute FP32-reference errors were approximately 0.00023
on the tail and 0.00082–0.00104 on the large shapes. References sample columns,
not every large output element; the theoretical error bound is conservative,
so these measured errors are reported separately. This is not model equivalence.

Best original kernel, warm / three-bank rotating-weight median microseconds:

- Gate/up: oneDNN **281.8 / 285.2**, Triton **1452.1 / 1458.5**.
- Down: oneDNN **143.3 / 142.8**, Triton **810.0 / 814.7**.

The oneDNN microbenchmark reproduces the profiled kernel cost reasonably well.
The prototype is 5–6x slower and fails its 1.10x improvement filter. A remote-only
algebraically equivalent transposed-dot variant also passed numerical checks
but remained 4.2–4.7x slower. Neither was integrated into serving.

Both drivers completed their checks/sweeps in about 11 seconds, excluding
container/package startup, using this CLI (script mounted at `/experiment`):

```bash
python /experiment/glimmer_xpu_w4a16.py --check --bench \
  --check-observed down --bench-shapes both --rotate-bank 3 \
  --max-seconds 285 --pretty
```

Artifacts: `kernel-micro/` and `kernel-micro-transposed/` under the same campaign
root, including exact source variants, results, stderr and Triton JIT caches.
The next credible kernel investigation is exact-version oneDNN/Xe2 GEMM strategy
specialization. Local source reconnaissance found `GEMM_KERNEL` selection gated
by `DNNL_DEV_MODE`; release builds ignore it. That requires matching the shipped
oneDNN revision and an isolated dev build, neither established by this prototype.
Do not infer that a generic Triton replacement will beat the current kernel.

## Commands and artifacts

Host: `mike@100.75.79.54` (`inference-host`, Intel Arc Pro B70); original image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
GPU was idle initially. Only session-owned `muse-b70-350-next` was restarted.
No driver, power, persistent-service, or production-provider configuration changed.
At cleanup both `muse-b70-350-next` and `muse-b70-kernel-micro` were absent,
`docker ps` was empty, and the integrated writing worktree was removed.

Artifacts on that host:
`~/b70-evals/muse-glimmer/350-next-20260905T150730Z/`.
Subdirectories: `norm-fix/`, `norm-context/`, `final-clean/`, `dflash2/`,
`profile/`, `eager-profile/`, `compiled-profile/`. Includes exact launchers,
server logs, raw SSE, JSON timings, metrics and gzip Torch traces. Raw logs stay
outside Git. Downloaded public image/model assets remain reusable local caches.

Core executed commands (`R` is the relevant artifact subdirectory and `OLD` is
`~/b70-evals/muse-glimmer/20260905-concurrency`):

```bash
NAME=muse-b70-350-next bash "$R/start-muse-vllm-concurrent.sh"
bash "$R/wait-vllm-health.sh" 480 18080
# One --reps 1 warmup first; 3 initial or 5 independent clean measured waves:
python3 "$OLD/instrument.py" --base http://127.0.0.1:18080/v1 \
  --model muse-glimmer-gptq --concurrency 8 --reps 5 --max-tokens 256 \
  --label final-clean --out "$R/measured.json" --log ''
python3 "$OLD/quality.py" 8 "$R/quality.json"
# Existing context client copied with only its artifact ROOT redirected:
python3 "$R/context-probe.py" queue 2048 12 norm-queue12
python3 "$R/context-probe.py" retrieval 129000 1 norm-native-retrieval
python3 "$R/context-probe.py" capacity 65536 8 norm-c8-64k 2048
# Local targeted regression suite:
python -m unittest discover -s tests -p test_dflash_xpu_context_norm.py -v
```

## Fresh primary sources used

- [XPU stacked RMSNorm indexing](https://github.com/vllm-project/vllm-xpu-kernels/issues/573)
- [DFlash2 FP16 overflow / BF16 workaround](https://github.com/vllm-project/vllm/issues/55250)
- [DFlash2 integration](https://github.com/vllm-project/vllm/pull/52816) and
  [decoder-layer load fix](https://github.com/vllm-project/vllm/pull/53435):
  pinned original runtime lacks the required integration; use an isolated newer image.
- [Muse DFlash2 checkpoint](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2)
- [DFlash2 transient vocabulary allocation](https://github.com/vllm-project/vllm/issues/53612)
- [vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/):
  profiling perturbs timing; never use profiled throughput for promotion.
- [Intel Triton matmul tutorial](https://raw.githubusercontent.com/intel/intel-xpu-backend-for-triton/main/python/tutorials/03-matrix-multiplication.py):
  FP16 tiles, FP32 accumulation, masked tails and reference comparison for a
  bounded M32 GEMM experiment. A prototype is not yet a serving improvement.
