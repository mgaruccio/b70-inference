# Observed results: 64K speculative-step cost

Development measurements only. No production promotion or publishable benchmark claim.

## Fresh baseline pair

Same B70, 275 W, GPTQ target, FP16 compute, FP8 target KV, C1, 65,664 configured context, 2,048-token batching, prefix caching disabled, thinking off. The unchanged long-context client issued one warmup and six measured requests: exactly 65,536 input tokens, 128 forced output tokens, greedy, seed 42. All seven parsed request payloads match between cells. Different immutable images/runner/patch stacks remain a disclosed **bundle** difference, not an isolated algorithm comparison.

| Metric | MTP4 | Corrected DSpark K7 |
|---|---:|---:|
| Decode median (tok/s) | 50.620571 | 26.211995 |
| Inclusive IQR (tok/s) | 4.684143 | 1.219920 |
| TTFT median (s) | 54.612411 | 57.360794 |
| Measured draft steps | 265 | 282 |
| Proposed tokens | 1,005 | 1,850 |
| Accepted tokens | 485 | 473 |
| Output tokens / measured draft step | 2.8981 | 2.7234 |

Raw data: `mtp4-baseline/` and `dspark-baseline/`. Six valid measurements each; no outliers removed. DSpark trial 5 is faster (33.851 tok/s) and retained. DSpark also passed the existing 3 canaries, 131 finite boundaries, 8 functional code checks, and repeatability smoke. Both launchers/power/Glimmer/running-container invariants were unchanged after cleanup.

These MTP4 settings intentionally match DSpark's smaller serving configuration. They do **not** replace the starting production-capacity MTP4 baseline for judging a useful optimization. The subsequent MTP4/MTP2 A/B restores the original 212,992 context capacity and 8,192-token batching in both arms.

## Profiling support and failures

Installed images both expose PyTorch 2.13 CPU+XPU profiling. A real B70 matmul smoke showed graph replay traces contain driver/host events but **no kernel events**; eager execution contains kernel events. `torch.xpu.Event(enable_timing=True)` successfully times actual replay regions. The disposable overlay also passed five real-XPU event samples without errors and eight CPU fixture tests.

- First support command failed before execution because the background shell is fish and rejected a heredoc. Explicit `bash` script invocation fixed it; retain `profiler-support/b0eb7575e.output`.
- DSpark `dspark-graph-profile/` failed with HTTP 500: vLLM 73029d424 leaves `AsyncLLM.profiler` unset with `ignore_frontend=true`. `dspark-graph-profile-02/` enables native frontend profiling and passes. No runtime source fix was added for this issue.
- Native Lab `preview_action` and `show_patch` calls failed with `connect ENOENT /tmp/prime-lab-1000/2c31a91a094d7226455d297a/lab.sock`. Disposable experiments proceeded under the user's explicit authorization.

Support scripts, traces and failures are in `profiler-support/`. No GPU activity overlapped between benchmark/profile cells. A short CPU-only immutable-image source inspection occurred during DSpark profiling startup; no GPU was exposed to it, and the cell's before/after host checks passed.

## Measured stage attribution

Each profile used a separate real 65,536/128 HTTP request. Profiling began only after the first nonempty output event; native traces confirm decode annotations `execute_context_0(0)_generation_1(5)` for MTP and `(8)` for DSpark. Native profiling is limited to five recorded worker steps (inspect actual trace); overlay duration follows the HTTP start/stop window and can extend beyond native auto-stop.

- **MTP4 target replay:** four samples, 41.2790 / 41.1923 / 41.1483 / 41.1692 ms. Raw role is conservatively `unknown`; call stacks identify `CUDAGraphWrapper -> XPUModelRunner._model_forward -> execute_model`.
- **MTP draft pieces:** two repeated graph identities, 14 samples each, approximately 0.14–0.18 ms and 0.29–0.30 ms. Call stacks explicitly contain `Qwen3_5MTP.forward` and `EagleProposer.propose`. These are **piecewise fragments**, not complete draft-step times. The native five-step trace also records 43.351 ms of eager GEMM kernels and 10.109 ms of eager attention kernels in total; these cannot be assigned wholly to the drafter without deeper attribution.
- **DSpark target replay:** seven samples, 54.420 / 54.380 / 54.333 / 54.365 / 54.378 / 58.990 / 54.334 ms. Exact descriptor: full graph, 8 tokens, 1 request.
- **DSpark draft-query replay:** seven samples, 33.449 / 33.450 / 33.460 / 33.447 / 33.422 / 33.467 / 33.422 ms. Exact descriptor: full graph, 7 tokens, 1 request. Includes backbone and sequential vocabulary/Markov sampling; excludes eager context-KV preparation. Native eager kernels outside replay include 24.659 ms of GEMM total over five steps.

Raw replay timings/caller stacks: `mtp4-graph-profile/step-timing/` and `dspark-graph-profile-02/step-timing/`. Native compressed trace files remain on inference-host under the corresponding campaign `profile/` paths, listed with sizes in each `summary.json`.

These are **profiled current-stream event intervals**, not isolated per-kernel times or unprofiled throughput. First-sample provenance collection, host enqueue delay, and any concurrent-stream work are caveats. MTP/DSpark graph counts are not directly comparable. Host-scope intervals overlap asynchronous device execution and must not be added to graph event intervals. Native graph replay hides internal kernels; do not infer target attention/GDN versus linear-layer fractions from these data.

## Evidence-selected experiment

Target verification is the largest measured region in both bundles. Test only **MTP2 vs MTP4**, reducing the target verification block from five tokens to three, rather than guessing at further DSpark acceptance or quantization changes. The pinned V1 runtime rounds graph capture sizes to multiples of the verification length; this is **not** a claim that MTP4 needlessly pads five tokens to eight.

Both arms retain the original 212,992 context capacity and 8,192 batching, same MTP image and patch stack, no prefix caching, and identical 64K workload. No acceptance-policy, target/draft precision, model, power or persistent launcher changes. The wrapper runs existing canary/finite/functional/repeatability gates before the unprofiled one-warmup/six-measurement point. Output comparisons must acknowledge normal batch-shape/numerical nondeterminism; this is normal lossless speculative decoding, not relaxed acceptance.

Fresh official guidance: <https://docs.vllm.ai/en/latest/features/speculative_decoding/> documents `num_speculative_tokens`, workload-dependent performance, and numerical/output equivalence caveats. Profiling sources are recorded in `README.md`; additional graph timing references: <https://docs.pytorch.org/docs/2.13/generated/torch.xpu.Event.html>, <https://github.com/intel/pti-gpu/issues/60>.

Exact baseline/profile commands (inference-host; all runtime dependencies remain there):

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-step-profile-64k
N=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm
(cd "$N" && python3 -u run-native-sampling.py --draft-sample-method greedy --graphs --long-context --cell dspark --out step-profile-baseline-01)
python3 -u "$R/run-step-profile.py" --cell mtp4 --out "$R/mtp4-baseline"
python3 -u "$R/run-timed-profile.py" --cell mtp4 --profile --out "$R/mtp4-graph-profile"
python3 -u "$R/run-timed-profile.py" --cell dspark --profile --out "$R/dspark-graph-profile-02"
```

The failed DSpark `dspark-graph-profile` attempt used the wrapper before its documented `ignore_frontend=false` workaround. Per-cell `launch-argv.json` is authoritative for each executed version. Before timing runs, the canonical `scripts/experiments/qwen38_step_timing_{overlay,patch}.py` files were copied into remote `$R/timing/`; the wrapper mounts that directory read-only. SHA256 values are in each timing cell's launch metadata. The copied DSpark baseline's original remote path is `$N/step-profile-baseline-01`, not `$R/dspark-baseline`.


Candidate execution command (inference-host):

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-step-profile-64k
python3 -u "$R/run-mtp-depth.py" --depth 4 --cell mtp4 --out "$R/mtp4-production-shape"
python3 -u "$R/run-mtp-depth.py" --depth 2 --cell mtp4 --out "$R/mtp2-production-shape"
```

## Candidate result: reject MTP2

Both cells passed all three canaries, 131 finite boundaries, eight functional checks, and six valid long-context measurements. All seven long request payloads and all 19 short request payloads match. Both arms are unprofiled, using the same immutable image and patch stack; launch metadata confirms the sole experimental configuration change is native MTP depth 4 → 2.

| Metric | Starting MTP4 | MTP2 candidate |
|---|---:|---:|
| Decode median (tok/s) | 55.698172 | 52.233087 |
| Inclusive IQR (tok/s) | 3.235782 | 0.647484 |
| TTFT median (s) | 53.845255 | 53.826043 |
| Measured draft steps | 254 | 318 |
| Proposed / accepted tokens | 1,016 / 518 | 636 / 449 |
| Actual output tokens / step | 3.02362 | 2.41509 |
| Median wall decode ms / counter step (proxy) | 54.88283 | 45.87484 |

MTP2 is **6.22% slower** by the ratio of decode medians. The approximate per-step interval falls 16.4%, but actual emitted tokens per step fall 20.1%; the cheaper steps do not repay the loss of accepted continuation. This is a sequential development A/B, not an interleaved confidence-interval claim. Reject the candidate rather than promoting a marginal or unproven result. The fresh 55.70 tok/s MTP4 result also reproduces the historical ~55.75 tok/s starting configuration.

Output identity: 18/19 short output-ID arrays match. The differing `parity-code-0` sample is within a family already non-repeatable in MTP4 (two distinct code outputs over four repeats; MTP2 has one). Four of six measured long-context output texts match exactly; this client did not request long output token IDs, so do not call that token-ID parity. All functional checks pass, but no universal output/distribution fidelity claim is made.

Raw data: `mtp4-production-shape/`, `mtp2-production-shape/`, and machine-readable `comparison.json`. Runtime draft counters can include terminal over-generation subsequently truncated to the requested 128 tokens; actual emitted/step above uses 768 observed output tokens across the six measurements.

**Decision:** retain MTP4. No performance gain over the starting configuration was achieved by this candidate. The next performance target is the dominant target-verification path at fixed MTP4 depth—first distinguish its attention/GDN state work from linear layers, then optimize the measured operation without sacrificing accepted tokens per step. No second configuration sweep or upstream-submission detour was undertaken. Production remains unchanged.
