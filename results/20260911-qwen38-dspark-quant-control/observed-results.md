# Actual B70 target-quantization control results

Development diagnostic, not a standard performance benchmark or isolated weight-precision experiment. Both requested real controls executed; this is not a source-only feasibility claim.

## Executed public-boundary journey

On inference-host, ran `bash run-paired-control.sh` after the corrected DSpark performance campaign released the GPU. Exact paths, immutable image and canonical overlay SHA are in that script and each cell's launch/dependency records. Canonical DSpark overlay SHA256 `0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643` includes the confirmed per-layer context normalization fix.

Baseline: existing GPTQ Int4 symmetric G128 target. Candidate: genuinely distinct official `Qwen/Qwen3.8-27B-FP8` snapshot at revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, downloaded from public Hugging Face and verified against all LFS SHA256 values. Full download manifest retained; no model weights in Git.

Both used the same 8 GiB CPU-offload budget, FP16 target compute/output, FP8 target KV, BF16 DSpark weights/cache, K7 greedy draft, standard rejection, C1 eager 8192, prefix caching disabled and unchanged 275 W cap. Native kernels were inspected, not forced. Both cells passed source/native-kernel gates, the existing finite/functional/API smoke gates, and all **36 requests × 512 output tokens** (code/math/prose × thinking off/on × target temperature 0/1 × seeds 42/43/44).

`fp8-offload8/paired-gptq-offload.json` confirms **all 36 request payloads, rendered prompt IDs and sampling settings matched**, with no pairing errors. Cross-target generated-token identity is not expected or used as a gate. `gptq-offload8/historical-gptq-no-offload.json` also matches all inputs to the freshly corrected `norm-eager` cell, despite the legacy 'historical' file label.

Both summaries report `status: passed` and `host_unchanged: true`. Run-owned containers were removed. Production launcher hash, power cap and stopped Glimmer state remained unchanged. Raw API/SSE/metrics, source checks, launch commands, effective configurations, native kernel/offload records and logs are under `gptq-offload8/` and `fp8-offload8/`. No service promotion was performed.

## Observed acceptance

From matrix-only native counters:

- **GPTQ Int4:** 13,139 accepted / 37,247 proposed = **35.2753%**; 5,321 steps; **3.46927 emitted tokens/step**.
- **Official FP8:** 13,247 accepted / 36,736 proposed = **36.0600%**; 5,248 steps; **3.52420 emitted tokens/step**.
- FP8 aggregate change: **+0.785 percentage points** acceptance, approximately **+1.58%** emitted tokens/step. Subgroups vary; this is not a statistical-significance claim.
- Corrected GPTQ with no offload had **35.6683%**, **3.49678 emitted tokens/step** in the same-input matrix. Its proximity is useful bounded evidence, not universal numerical reproducibility.

Group emitted tokens/step (nine requests per group), GPTQ → FP8:

- Thinking off, temperature 0: **3.93686 → 3.83844**.
- Thinking off, temperature 1: **3.53681 → 3.50759**.
- Thinking on, temperature 0: **3.49811 → 3.63365**.
- Thinking on, temperature 1: **3.02690 → 3.18194**.

On this workload, replacing the target with official FP8 does **not** produce a large acceptance recovery. The separately measured normalization correction improved graph acceptance from **16.54% to 35.45%** and median short-context graph decode from **40.21 to 62.77 tok/s**. That confirmed bug is the major demonstrated cause of the original poor result. This experiment does not prove quantization has no effect on other workloads or later trajectories.

## Native execution and offload caveats

Actual post-load classes:

- GPTQ: `vllm.model_executor.kernels.linear.mixed_precision.xpu.XPUwNa16LinearKernel`, native W4A16 symmetric G128.
- FP8: `vllm.model_executor.kernels.linear.scaled_mm.xpu.XPUFp8BlockScaledMMKernel`, native dynamic W8A8 block128×128 with FP16 output.

The intended comparison is target weights/quantization **plus native execution bundle**. It changes activation quantization and kernels as well as weight representation. Do not present it as pure weight-bit causality, an exact BF16 target control, or an offloaded speedup comparison.

The equal configured offload budgets did **not** yield equal retained UVA-backed storage. Native weight repacking matters:

- GPTQ offloader accounted 8,591,404,576 bytes, but only **292,708,045 bytes** of final target storage matched observed initial/re-offloaded UVA backing. Marker flags alone misleadingly covered 8,521,408,205 bytes.
- FP8 accounted 8,682,943,592 bytes (one-parameter budget overshoot); **8,682,943,552 bytes** of final target storage matched observed UVA backing, including 8,663,308,800 bytes re-offloaded after weight postprocessing.
- After target+draft load, before KV allocation, XPU allocated bytes were **21,431,036,928 GPTQ** and **24,830,112,768 FP8**. These are allocator snapshots, not peak full-serving memory or comprehensive physical-memory accounting.

These measured asymmetries prohibit an apples-to-apples offloaded throughput interpretation. They are retained explicitly rather than silently equating the 8-GiB flags or repairing unrelated offloader behavior. Both actual native paths nevertheless loaded and completed the full API acceptance journey without fallback or altered guards.

## Verification and scope closure

The integrated runner passed **13 stdlib tests** against the corrected canonical overlay and exact 38-source composition/replay (`lead-corrected-overlay-tests.txt`). A read-only review of guards, hooks, native dispatch and pairing found no blocker/high findings before launch. Real GPU success, not that review, establishes execution feasibility.

The requested numerical reference check and genuine target-quantization control are now completed, alongside the bounded normalization fix and eager/graph/64K performance reruns. DSpark still trails the historical MTP4 64K result; this is not a claim of parity with MTP or an exhaustive tuning campaign. The historical MTP comparison uses a different image/execution stack and is not contemporaneous. Persistent production configuration remains untouched.
