# Qwen prefill regression: native XPU profile results

**Development evidence; reject both budget changes. Production remains guarded 8192.**

## What the profiling establishes

1. **Configured budget is not the executed shape.** With cache on / Mamba align,
   8192 and 6656 execute identical chunks at both lengths. The timing differences
   between them are +0.016% / +0.002%: effectively no measured benefit.
2. **8320 is worse even where it reduces the number of chunks.** At cache-on 32K
   it uses five steps rather than six, but INT4 GEMMs add 61.832 ms of device work
   and K64 attention adds 57.379 ms; dense GEMMs save only 6.852 ms. Cold median
   response latency increases 0.497%.
3. **The prior cache-off regressions reproduce in device execution.** At 16K / 32K,
   8320 is +0.890% / +1.255%; 6656 is +0.518% / +0.547%. For 8320, INT4 GEMM and
   attention account for nearly all the extra device activity. For 6656, an extra
   scheduled step adds attention, dense-GEMM and GDN work rather than improving it.
4. **CPU launch scopes were not the bottleneck diagnosis.** Native XPU kernel
   timings explain the deltas; CPU external/correlation IDs identify their launch
   operators, but CPU scope durations are never counted as GPU time.

## Matched cold response medians

Five unprofiled, exact-ID, distinct-prefix requests per cell; max output one,
non-streaming response elapsed time. Deltas use the initial 8192 cell with the
same cache policy and length. These are not production prefix-hit or decode tests.

| Cache policy | Budget | 16K seconds / delta | 32K seconds / delta |
|---|---:|---:|---:|
| off / Mamba none | 8192 | 8.717025 / baseline | 19.966986 / baseline |
| off / Mamba none | 8320 | 8.794642 / +0.890% | 20.217641 / +1.255% |
| off / Mamba none | 6656 | 8.762147 / +0.518% | 20.076221 / +0.547% |
| on / Mamba align | 8192 | 8.763334 / baseline | 20.154967 / baseline |
| on / Mamba align | 8320 | 8.814440 / +0.583% | 20.255158 / +0.497% |
| on / Mamba align | 6656 | 8.764711 / +0.016% | 20.155433 / +0.002% |

Cache-off baseline-after drift: **−0.052% at 16K, −0.008% at 32K**. Single-profile
response overhead versus each cell's unprofiled median: **+0.097% to +0.585%**.
Do not subtract that overhead from individual kernel measurements or call it a
performance improvement. One trace per cell and five timings are not a
statistically publishable one-percent claim.

## Executed chunks and the source-level reason

| Cache | Budget | 16K context tokens per step | 32K context tokens per step |
|---|---:|---|---|
| off | 8192 | 8192, 8192 | 8192 × 4 |
| off | 8320 | 8320, 8064 | 8320 × 3, 7808 |
| off | 6656 | 6656, 6656, 3072 | 6656 × 4, 6144 |
| on | 8192 or 6656 | 6656, 6656, 3072 | 6656 × 4, 3328, 2816 |
| on | 8320 | 8320, 4992, 3072 | 8320 × 3, 4992, 2816 |

The installed [scheduler's `_mamba_block_aligned_split`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/core/sched/scheduler.py)
rounds intermediate ends down to the effective 1664-token block boundary. Thus
8192 becomes 6656. It also stops at the last cacheable position and backs that
position off one block when `use_eagle` is true. The installed
[`SpeculativeConfig.use_eagle`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/config/speculative.py)
explicitly includes **mtp**: this is shared cache logic, not an EAGLE draft-model
change. The boundaries are `floor(16384/1664)*1664 - 1664 = 13312` and
`floor(32768/1664)*1664 - 1664 = 29952`, explaining the extra tail stops.
`installed-alignment-source.txt` retains the decisive installed-source excerpt;
iteration logs and native trace annotations agree on every executed chunk.

8320's actual chunks create partial **Q256** tiles: theoretical tile counts rise
64→65 at 16K and 128→130 at 32K, under both cache policies. This is consistent
with the measured attention penalty; it does not alone prove the entire penalty.
INT4 GEMM device time also increases. Shapes/stacks/memory recording were disabled,
so the exact oneDNN GEMM selection or microarchitectural reason is **not established**.
No fabricated per-GEMM dimensions or CPU-as-GPU attribution is used.

## Device attribution and interpretation

`analyze.py` separates the generic `gemm_kernel` events using CPU launch external
IDs and XPU runtime correlation IDs. Every generic GEMM in all 14 cells maps to
`_xpu_C::int4_gemm_w4a16` or `aten::mm`; none are left ambiguously attributed.
For example, the pilot's INT4 GEMMs account for 5.899451 s of kernel activity,
versus 0.033447 s for dense GEMMs. K64 attention contributes 1.572140 s; named
GDN kernels 0.620386 s. Separately named/fused Triton kernels remain in "other".
Kernel sums may overlap and exclude copies; they are not elapsed response latency.

Per-phase attribution uses the launch/operator timestamp, not an asynchronous
GPU start. Kernels outside the non-empty target context annotations remain
explicitly unassigned per phase but included in aggregate groups. This avoids
silently attributing speculative/other work to the wrong context scope.

## Execution, artifacts and cleanup

Host `inference-host`, boot `b46543b7-a459-4c7d-aeb4-7ed41a62ecd7`; pinned image,
Torch 2.13.0+xpu, vLLM 0.27.2rc1.dev77+gac7509e2b. Every cell retained C1,
context 212992, .95 utilization, FP8 KV, native MTP4/INT4 S+M1, mixed GDN v5,
K64, XPU graphs and 275W. No unitrace, 9984, context reduction or promotion.

Executed on the host:

```text
python3 -u /home/mike/b70-evals/20261012-qwen38-prefill-profiling/run-campaign.py --output /home/mike/b70-evals/20261012-qwen38-prefill-profiling/matrix-01
```

Seeded 12-cell order plus two final cache-off baseline anchors completed.
**112 requests** (warmups, unprofiled, profiled and restored probes) had HTTP200,
exact input/output usage and finite logprobs. All scheduler ledgers summed to the
requested lengths; **85,126 actual XPU kernel events** were retained. All cells
used the same boot, boost off, all 16 CPU caps at 3801000 kHz and 275W. Pre/post/
restored focused kernel checks found no matches. This does not diagnose the prior
SSH fault or declare the historical CO/MCA issue resolved.

The original launcher SHA256 stayed
`5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d`.
After the campaign, the externally accessed `/v1/models` advertised qwen38 with
context 212992; `/v1/completions` returned a finite 5+1 response in **0.121984 s**
(`public-after-matrix.json`). Production remains running on unchanged guarded 8192.

Complete matrix archive: **21,737,384 bytes**, SHA256
`b6abd59e597e34b19d2f985a165cd912b604a969d80f2b23418e4ea2455a112d`;
private R2 readback verified before extraction (424 members). Raw traces, logs,
responses, metrics, frozen controllers and host checks are retained in `matrix-01/`
and on the host. Temporary R2 object and private transfer/signing files were deleted.

Reproduce local analysis and targeted checks:

```text
python3 -B analyze.py matrix-01 > matrix-correlated-analysis.json
python3 -B test-pilot.py       # 6 passed
python3 -B test-analysis.py    # 1 passed
```

`comparison.json` retains medians, chunk shapes, per-category device deltas,
baseline drift and explicit limitations. `matrix-correlated-analysis.json` retains
all per-cell correlated groups and per-phase coverage. No candidate improves the
matched cold path; no cache policy or budget change is adopted from this experiment.


## Two distinct next options — proposals, not new improvements claimed here

- **Prefill: shape-specific INT4 GEMM tuning/replacement.** The correlated pilot
  puts 5.899 s in INT4 GEMMs versus only 0.033 s in dense GEMMs, making INT4 the
  strongest measured target. No achievable speedup or exact oneDNN bottleneck is
  established. First collect exact dimensions in one bounded diagnostic; then
  test one supported candidate with the same weights/precision/context/graphs,
  real 16K/32K API requests, baseline-after timing and finite/semantic canaries.
  Tradeoffs: implementation effort, graph compatibility and numerical correctness.
- **Decode: quality-gated shared-KV.** [Existing separate 230W research](../../docs/qwen38-shared-kv-long-context.md)
  measured +7.92% around 128K and +10.08% at the full window. It is not a prefill
  result or a measured gain on this 275W/K64 baseline. HumanEval+ was 136/164
  versus native 139/164, and quality neutrality remains unproven. A bounded next
  validation is the existing full API quality evaluation in paired restarted
  arms, followed only if cleared by matched long-context decode and explicit
  K64-composition checks. Tradeoff: known quality risk; keep it unpromoted.

Neither proposal authorizes another run or a production change in this campaign.
