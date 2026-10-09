# Native MTP4 GEMM / attention investigation — development only

Continuation of the completed 64K GDN-locality cell; no GDN fusion and no
production launcher promotion. Hotspot percentages are shares of target replay
elapsed time, **not measured improvements** or fractions of full request latency.

## Fixed baseline

- B70 / 275 W; same boot `c93aea98-bb22-4cf6-879f-42d2980cb726`.
- CPU boost off, all 16 caps 3801000 kHz. Persistence now installed separately:
  [`../../docs/b70-cpu-boost-off.md`](../../docs/b70-cpu-boost-off.md).
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- vLLM ac7509e2b, torch 2.13.0+xpu, kernels 0.1.12.3.
- Qwen3.8-27B GPTQ symmetric G128, FP16 target, FP8 KV, native MTP4 (five rows).
- Context 212992; batch-token limit 8192; concurrency 1; utilization .95;
  graph capture sizes `[1,2,4,8]`; prefix cache and thinking disabled.

## Fresh research and actual installed-source findings

Sources consulted 2026-10-09:

- <https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/flash_attn/flash_api.cpp>
- <https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp>
- <https://github.com/vllm-project/vllm-xpu-kernels/pull/257>
- <https://github.com/vllm-project/vllm/pull/37844>
- <https://github.com/vllm-project/vllm/pull/41426>

Upstream tags are contextual evidence, not byte identity with installed 0.1.12.3.
Read-only source inspection in temporary CPU-only containers from the pinned
image established the actual Python interfaces. No package upgrade is proposed.

Installed `vllm_xpu_kernels/flash_attn_interface.py` SHA256:
`2a8ce07e2839232bc9f0e9cc9969a4410099c0c75ce616e72d4583e950864942`.
Its `_spec_decode_varlen_fwd` expands five query rows into five pseudo-sequences,
retains causal per-row KV lengths, and calls native `_vllm_fa2_C.varlen_fwd` with
`num_splits=None`. The candidate overrides only that argument for batch1,
`q.shape==(5,24,256)`. Other shapes, ordinary decode, prefill and draft paths
retain automatic selection. No replacement attention implementation is used.

`metrics-before.raw` in the archived native64K run confirms literal KV block
size1664 (not16). Its attention launch is `[SIMD16 {1;1;640} {64;1;1}]` and
reduction grid `{1;24;5}`. Combined with the nearest public legacy scheduler,
640 is consistent with5 pseudo-sequences ×4 KV heads ×32 splits. This is a
trace-backed interpretation, not proof of wheel/C++ source parity or a nominal
core-count heuristic. Select explicit16 as the first candidate, reducing split
workspace and reduction fan-in while retaining the native kernel. Relevant
scheduler reference:
<https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/attn/xe_2/paged_decode.hpp>.

`installed-python-sources.tar.gz` freezes the inspected image Python files and
existing serving lifecycle source (SHA256
`c4ab027e03e0d4b6fafe97b88007e5b10b9857715cb1745a88fdded19e93cccd`).

Installed `XPUwNa16LinearKernel.apply_weights` dispatches to
`_xpu_C.int4_gemm_w4a16`; transpose/contiguous weight preparation is in
`process_weights_after_loading`, not each forward. The traced generic GEMM name
alone does not prove individual projection ownership. Model config establishes
hidden5120, MLP intermediate17408, 48 linear-attention layers and16 full-attention
layers; gate+up logical N34816, down K17408, full qkv N14336, output K6144.
These are source-derived shapes, not new measured per-projection attribution.

Earlier repo work already tried a Triton grouped-query attention kernel and
reported a slowdown; do not repeat it or transplant a newer DSpark/noncausal
result into this older native-MTP cell. oneDNN-selected dense GPTQ GEMM and
small-M primitive selection remain distinct from MoE grouped-GEMM tuning.

## Reproduce the offline hotspot accounting

The raw inputs are already durable in the committed
`../20261009-qwen38-gdn-locality/native-public-fill64k-01.tar.gz`; `/tmp` is only
an extraction location. Archive-member SHA256 values were independently checked
against both `hotspots.json` inputs on 2026-10-09 and match exactly.

From the repository root, extract that trusted repository archive into a fresh
directory and run:

```sh
archive=results/20261009-qwen38-gdn-locality/native-public-fill64k-01.tar.gz
scratch=$(mktemp -d)
tar -xzf "$archive" -C "$scratch"
native="$scratch/native-boundary64k-20261009-100052-ff9f86/native"
python3 -B results/20261009-qwen38-gemm-attention/analyze-target.py \
  --trace "$native/unitrace/python3.285.json" \
  --step-timing "$native/step-timing/step-timing-session-rank0-session1.json" \
  --output "$scratch/hotspots.json"
python3 -B results/20261009-qwen38-gemm-attention/test-analyze-target.py
```

Observed: 42 target replays ×977 operations, True1913.950520 ms, linked
work1827.095850 ms; five focused tests pass. Zero-duration copy rows retain
ownership/counts but contribute zero work. Reduction names are classified
before the broader FMHA substring; measured API suffix ordinals start at zero.

## Defined end-to-end experiment

Run tier: **development**, not standard publishable. Intentional candidate
change: native split count only; no 5% gate inherited from GDN work.

1. Verify pinned lifecycle, golden client and production-launcher hashes; same
   boot, boost0/caps,275W; no competing containers; no guarded kernel errors.
2. Start an owned disposable server using the pinned native launch recipe.
3. Golden client exercises the real loopback HTTP serving API: one warmup and
   six measured deterministic, independently nonced prompts of exactly65536
   tokens, greedy seed42, ignored EOS, exactly128 output tokens.
4. Execute auto baseline, one explicit-count candidate, then auto repeat control.
   The same deterministic prompt set across cells permits paired output checks.
   No profiling overhead in these timing cells.
5. Retain raw requests/SSE, speculative counters, rendered token counts,
   request/TTFT/decode timings, launch argv, software/host configuration and all
   failures. Compare generated outputs and acceptance as well as timings.
   Reject a numerical/output regression; split accumulation order may differ.
6. Cleanup only owned containers; repeat host/hash/kernel checks. Production
   launcher remains untouched. Source overrides disappear with each container.

Commands (on `inference-host`, campaign root
`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gemm-attention`):

```sh
python3 -B run-split-cell.py --splits auto --out "$PWD/auto-before-01"
python3 -B run-split-cell.py --splits 16 --out "$PWD/splits16-01"
python3 -B run-split-cell.py --splits auto --out "$PWD/auto-after-01"
```

The adapter reuses the existing `20260911-qwen38-step-profile-64k` lifecycle and
`20260909-standard/mtp4-long` client, plus completed GDN-run host guards; their
hashes are checked before launch. Frozen old guards still require the existing
unitrace install but no tracer is launched by this experiment.

Read-only review found no blocking issues in the adapter, scope guard, native
argument position, cleanup or CPU unit. The3600-second benchmark timeout is an
intentional cell bound (not seven1800-second per-request budgets); a timeout
would be retained as a failed/incomplete development cell, not a valid sample.
`precision_context_acceptance_unchanged` in launch metadata asserts the requested
recipe only, not output identity or measured acceptance equality.

Local preliminary checks passed: auto leaves installed source bytes identical;
all three supported candidates change exactly one line; native argument24 is
changed only for the intended shape; source mismatch rejected; syntax compiled.
Auto-before completed with exit0: one warmup, six valid measured65536→128
requests, fourteen valid metric snapshots, no counter/parse errors, unchanged
host, and successful cleanup. Median decode54.137949625 tok/s (IQR2.194097961),
median TTFT53.857404263 s, median total56.182853754 s. Candidate16 and the repeat
auto control also completed with exit0, six valid measured requests, valid
speculative counters, unchanged host and owned-container cleanup.

| Cell | Median decode tok/s | Median total request seconds |
|---|---:|---:|
| Auto before | 54.137950 | 56.182854 |
| Explicit16 | 54.821617 | 56.209898 |
| Auto after | 55.576311 | 56.187051 |

**Do not promote split16.** Its apparent +1.26% versus the first control does not
survive the repeat control (−1.36% versus the second). Total request time is
essentially unchanged. Five of six measured candidate outputs differ from the
first control, but three of six outputs also differ between unchanged controls.
Thus this run does not isolate a candidate-induced correctness regression or a
performance gain; baseline output/acceptance variability is itself observed.
No precision, model, acceptance algorithm, context or production settings were
changed. Keep the raw failures/variation rather than selecting the faster arm.

The six paired decode-rate deltas have median +1.28% versus auto-before and
−3.66% versus auto-after; unchanged controls drift +3.75% by the same paired
statistic. Decode IQRs are2.19/2.22/3.34 tok/s for before/candidate/after.
All cells report226397 KV-cache tokens. `split-comparison.json` preserves the
individual paired timing/output hashes and acceptance-counter deltas. Exact
output text is checked; the API did not return token IDs, so this is not a
bitwise token-ID equality assertion.

Reproduce the comparison from this directory:

```sh
python3 -B compare-cells.py auto-before-01.tar.gz splits16-01.tar.gz \
  auto-after-01.tar.gz > split-comparison.json
```

Raw archives were retrieved using small-buffer SFTP (`-B512 -R4`, `reget`),
without rerunning any GPU cell because of transfer delays. Remote and local
SHA256 matched:
- `auto-before-01.tar.gz` (268440 bytes):
  `67395fc0a6a08df336dbc1bf847738bc587a89f0cb03cb2f47069f1d9da8611f`
- `splits16-01.tar.gz` (268692 bytes):
  `18a920ac18c151986e9aa0170504a01bf7f78b176d105479ec8b995b215a56b2`
- `auto-after-01.tar.gz` (267683 bytes):
  `f9a6d885c8b075ed60ffdf812737162815286e38ecf0ea453d4dfee5daaca83f`

Container environment scans found no nonempty credential-named fields.
CPU service execution passed; reboot untested.

## Bounded GEMM dispatch screen

Next development-only screen: the actual installed `_xpu_C.int4_gemm_w4a16`
operator on synthetic fixed-seed data with source-derived projection sizes:
`(K,N)=(5120,34816),(17408,5120),(5120,16384),(6144,5120),(5120,14336)`.
Compare nativeM5 with identical first five FP16 rows zero-padded toM8/M16;
retain GPTQ-G128 packed weight layout, FP16 scales and symmetric zero-point8.
This tests a dispatch/tiling hypothesis without changing the serving recipe.

Expected process: after serving control cleanup, verify the same host guards;
run only one owned pinned-image GPU container; warm and capture each shape/M;
use five interleaved rounds of25 graph replays with timing-enabled XPU events,
synchronizing outside replay loops. Save raw timings, dimensions/strides,
allocations, actual device/runtime details and first-five-row output comparisons.
Exact equality and finite outputs gate any follow-up; mismatch timings remain
diagnostic and cannot justify adoption. Include failure JSON and nonzero exit,
then cleanup and repeat host guards. A win would still require real public-API
serving validation; microseconds are not an end-to-end speedup.

Fresh official PyTorch source confirms `XPUGraph` capture/replay and timing-enabled
`Event.record/elapsed_time/synchronize` APIs:
- <https://raw.githubusercontent.com/pytorch/pytorch/main/torch/xpu/graphs.py>
- <https://raw.githubusercontent.com/pytorch/pytorch/main/torch/xpu/streams.py>
These current API references do not establish source identity with installed
Torch2.13.0+xpu; the actual pinned runtime probe remains required.

### GEMM screen outcome

Executed on `inference-host` after all serving cells had cleaned up:

```sh
python3 -B run-gemm-probe.py --out "$PWD/gemm-padding-01"
```

The owned container invoked `python -B /probe.py --out /output/probe.json` with
seed20261009, three warmups, five rotated rounds and25 replays per timed block.
Graph timings include the candidate's zero-fill/copy input-padding work. The
runner exited1 because the M16 routes failed exact-output checks, **not** because
of a crash or host instability. All five cases and timing rounds were retained.
No new guarded kernel errors; postconditions and owned-container cleanup passed.

| Logical projection shape | M5 µs | M8 µs | M16 µs |
|---|---:|---:|---:|
| Gate+up, K5120/N34816 | 181.19 | 184.69 | 190.55 |
| Down, K17408/N5120 | 106.46 | 103.71 | 111.10 |
| Linear-attention input, K5120/N16384 | 93.48 | 97.22 | 100.77 |
| Output, K6144/N5120 | 40.80 | 39.50 | 46.04 |
| Full-attention QKV, K5120/N14336 | 83.63 | 86.65 | 90.50 |

These are medians of **synthetic operator graph-block event timings**, not model
projection timings or serving gains. M8 first-five outputs matched M5 exactly on
all shapes, both direct and graph replay. M16 differed on all shapes (maximum
absolute difference0.00048828125–0.0009765625) and was slower everywhere. All
outputs were finite. This is an exact-output screening rejection, not evidence
that the M16 mathematical operation is incorrect.

M8 slowed three shapes; small reductions on down/output (about2.75/1.30µs)
do not establish a useful serving gain. No padded GEMM implementation was applied
to the model, and no universal row-padding change is justified. Any future
shape-specific candidate would need stronger repeated evidence and serving
validation under the observed baseline variability.

Runtime readback: B70 `max_compute_units=256`, driver1.15.39122+11,
Torch2.13.0+xpu, vLLM ac7509e2b.xpu, kernels0.1.12.3. Do not substitute nominal
Xe-core counts for this runtime property in attention split heuristics.

`gemm-padding-01.tar.gz` is37912 bytes, SHA256
`4f620b079b33ecd991ac769a331234ea63b8574c7b9bad59e992c3254e0cd07a`.
Remote/local hashes match. It contains the executed source files, raw log,
`probe.json` (all five cases and ten direct/graph M16 mismatch records), command,
container exit/cleanup and pre/post host/kernel checks. Frozen executed source
bytes match the final repository scripts. Host preconditions and postconditions
are identical; cleanup returned0. All fifteen shape/route timing sets contain
five positive finite event measurements; all graph output dtypes are FP16.

## Decision and remaining direction

This bounded pass produced **no demonstrated safe serving improvement**. Leave
native automatic attention splitting and native five-row GEMM unchanged.
CPU boot persistence is the only host behavior change.

The next useful work is native kernel dispatch/layout/weight-traffic tuning,
starting with the large gate+up GEMMs, or a native shared-KV design for the five
verification rows—not a blanket M8/M16 padding recipe or re-running the known
slow Triton grouped-attention prototype. Before claiming small serving gains,
control the output/acceptance variability observed in identical baseline runs.
No standard-compliant or quality-sensitive publication claim is made here.

Native Lab preview unavailable (`ENOENT` workspace socket); ordinary authorized
file/SSH operations used, without adding a substitute Lab service.
