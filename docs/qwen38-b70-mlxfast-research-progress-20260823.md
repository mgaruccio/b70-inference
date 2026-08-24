# Qwen3.8 B70 performance research progress — 2026-08-23

## Purpose

This is a handoff snapshot for external reviewers. It distinguishes measured B70 results from hypotheses and keeps the 98,304-token Pi-agent context / 32,768-token output contract intact.

## System under test

- GPU: Intel Arc Pro B70, 32 GB GDDR6, reported 608 GB/s bandwidth.
- Runtime image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- vLLM: `0.27.2rc1.dev77+gac7509e2b`; `vllm-xpu-kernels` `0.1.12.3`.
- Model: `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` at `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- Baseline serving: GPTQ INT4 symmetric G128 target, preserved BF16 native MTP draft tensors, MTP-4, FP8 KV, 131,072 max model length, `gpu_memory_utilization=0.88`, 8,192 max batched tokens, no prefix cache, `qwen3_xml` tool parser.
- Evaluation contract: Pi harness `0.83.0`, Harbor Terminal-Bench 2.1 pinned five-task cohort, Docker runtime, high thinking, 98,304 context window, 32,768 max output, C1.

The completed baseline five-task agent evaluation solved 3/5. Its OpenAI-call trace measured 28.54 aggregate completion tok/s and 17.10 median per-call tok/s. Those are agent-workload figures, not pure decode numbers.

## What has been measured

### Initial service controls

The original BF16-draft server was relaunched with logs persisted outside the disposable container. It passed LAN and Tailnet health plus public chat-completions tests.

| Control | BF16 draft | Draft-INT4 overlay | Result |
| --- | ---: | ---: | --- |
| Repetitive short, 575 prompt tokens, g128, n=5 median | 38.99 tok/s | 53.20 tok/s | +36.4%; synthetic prompt, not a model-card/agent claim |
| Medium coding, 3,138 prompt tokens, g128 | 24.52 tok/s | 30.56 tok/s | +24.6% |
| Long, 80,076 prompt tokens, g128 | 1.69 tok/s, completed | HTTP 500 | overlay initially OOMed |

The failed first overlay run reported `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` in `async_tensor_h2d`. vLLM reported 5.31 GiB graph-capture memory under XPU graph mode `FULL_AND_PIECEWISE`, with 37 captured sizes. The issue was not attributed to model quality.

### Memory-safe graph configuration

The draft-INT4 overlay quantizes the draft LM head and five MTP linears only; target verification remains unchanged. It was tested with XPU graph capture constrained while retaining the 131,072-token ceiling.

| Variant | Warm natural short median | Medium median | ~98k median | Stability |
| --- | ---: | ---: | ---: | --- |
| BF16 MTP-4 baseline | 44.03 tok/s | 16.91 tok/s | 0.639 tok/s | passed |
| Draft-INT4, graph capture cap 64, MTP-4 | **54.67 tok/s** | **17.83 tok/s** | **0.644 tok/s** | passed five tokenizer-verified 97,962-token controls |
| Draft-INT4, graph capture cap 128 | — | — | — | failed during startup |
| Draft-INT4, graphs disabled | 17.37 tok/s in the initial cold sample | 9.46 tok/s | 0.32 tok/s | passed long control, but not competitive |

The cap-64 configuration captured sizes through 64 and kept 158,959 KV-cache tokens. It is the currently measured throughput/stability winner, but is **not deployed** as the normal server because the agent rollout quality smoke is incomplete.

Sustained public-API controls on draft-INT4/cap64/MTP-4:

- 8,031-token prompt, forced 2,048-token completion: 23.56 s, 86.93 tok/s.
- 97,963-token prompt, forced 2,048-token completion: 125.20 s, 16.36 tok/s.
- Both completed by length without an engine failure.

### MTP-depth result

At cap64 with the same contract:

| Variant | Short median | Medium median | ~98k median | KV cache |
| --- | ---: | ---: | ---: | ---: |
| Draft-INT4 MTP-2 | 47.84 tok/s | 16.96 tok/s | 0.643 tok/s | 169,961 |
| Draft-INT4 MTP-4 | **54.67 tok/s** | **17.83 tok/s** | **0.644 tok/s** | 158,959 |
| MTP-3 | startup failure | — | — | — |

Greedy short-prompt output content matched between MTP-2 and MTP-4. MTP-5 was skipped after MTP-3 failed the sequential safety gate. MTP-4 is therefore the current best static native depth; lower depth did not improve B70 throughput.

### Mixed-batch C2 correctness

The cookbook `patch_gdn_mixed_split_v5.py` was applied after the existing MTP/boundary patches. It fixed the previously untested mixed speculative/non-speculative C2 path.

| C2 variant | 8k-prefill request | Speculative-decode request | Result |
| --- | ---: | ---: | --- |
| Draft-INT4/cap64/MTP-4 + v5 | 13.81 tok/s | 17.18 tok/s | both HTTP 200; engine survived |
| Same + metadata overlay | 12.63 tok/s | 16.29 tok/s | both HTTP 200; stable, but slower in one sample |

These are one mixed-smoke sample per row, not throughput claims. The service kept a 158,959-token KV cache and observed mean acceptance lengths around 2.65–2.82.

## Metadata overlay now in the repository

A narrow, reversible Python overlay was added and pushed:

- `c479140 Add versioned B70 GDN metadata overlay`
- `a3415d7 Guard B70 GDN metadata arange capacity`
- `818e8f9 docs(qwen38-b70-gdn): correct runtime target path for GDN metadata overlay`

Files:

- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py`
- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/README.md`
- `tests/test_qwen38_b70_gdn_metadata_patch.py`

The overlay selectively backports the static metadata-buffer half of unmerged vLLM PR #43955:

- reuse a preallocated speculative-token `arange` buffer;
- avoid zero-length non-spec copies;
- avoid redundant copies when the source already has graph-stable storage;
- add a capacity guard rather than silently truncating future oversized batches.

It deliberately does **not** alter `GPUModelRunner`, accepted-token bookkeeping, GDN state semantics, `_xpu_C.gdn_attention`, mixed-batch logic, or compiled XPU kernels. Unit/fixture validation passes 4/4, including dry-run, idempotence, reverse, and capacity-guard checks.

Runtime application must use the actual imported site-packages module, not `/workspace`:

```bash
TARGET="$(python -c 'import vllm.v1.attention.backends.gdn_attn as m; print(m.__file__)')"
python /patch_mtp.py
python /patch_boundary.py
python /patch_gdn_metadata.py --dry-run --path "$TARGET"
python /patch_gdn_metadata.py --path "$TARGET"
```

The metadata overlay applied successfully in the composed runtime sequence and was stable at C1 p98k and in the v5 C2 smoke. Its short C1 result, 54.94 tok/s versus 54.67 tok/s without the overlay, is within noise; no speed claim is made yet.

## MLXFast research: why it matters

The relevant external project is [MLXFast](https://www.yukon.org/mlxfast), specifically the [Qwen3.8 MTP challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge), not generic Apple/MLX anecdotes.

The leading public promoted change (`0863b06`) reached a reported 97.8 tok/s median decode on the challenge benchmark. The final diff did not change the custom proposal-head tensors or schedule. It reused a chunk-sum table already generated by fused residual+RMSNorm in MTP verification widths 4–9, removing 127 of 257 standalone `xsums` fills per verify round.

That exact Metal/QMV code is not directly portable:

- MLXFast uses an Apple M5/MLX/Metal stack, affine group-64 4-bit weights, 512-token decode fixtures, and greedy target-token equality.
- B70 uses XPU/vLLM, GPTQ symmetric group-128 W4A16, FP8 KV, up to 131k serving context, and a stochastic agent workload.
- The MLX `xsums` correction likely relates to its affine zero-point layout, so B70 may have no one-to-one `xsums` operation.

But the optimization principle is directly relevant: inspect the B70 MTP verification path for redundant temporary buffers, producer/consumer handoffs, reductions, kernel launches, and state/memory copies; reuse fused producer outputs where possible while preserving target verification.

## Actual installed B70 source audit

In the pinned image, unmerged vLLM PR #43955 is absent.

Relevant installed paths:

- `/workspace/vllm/vllm/v1/attention/backends/gdn_attn.py` — source/build-tree audit path.
- `/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py` — runtime module that must be patched.
- `/workspace/vllm/vllm/model_executor/models/qwen3_5_mtp.py` — native Qwen MTP model implementation.
- `/workspace/vllm/vllm/v1/worker/gpu_model_runner.py` — proposal dispatch.
- `/workspace/vllm/vllm/_xpu_ops.py` — Python XPU GDN wrapper; one `_xpu_C.gdn_attention(...)` call, no native mixed split.
- `/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels` — compiled XPU package (`0.1.12.3`) with `libgdn_attn_kernels_xe_2.so`; no editable source or headers were found in the running environment.

The durable higher-effort work is therefore in upstream `vllm-xpu-kernels` SYCL GDN/W4A16 code, not a direct Metal-source translation. Candidate upstream surfaces include the GDN attention interface and Xe2 chunk-gated-delta-rule kernels. The immediate Python metadata patch is a low-risk probe, not that kernel project.

## Current live state and open blockers

At this snapshot, the original BF16 launcher is the active fallback:

- `~/inference/launchers/start-qwen38.sh`
- LAN and Tailnet health have passed after each experiment.
- Experimental launchers/containers are removed after every run.

A one-task Pi/Harbor smoke was prepared but blocked before rollout by Prime: `Payment required. Insufficient balance.` Artifact:

```text
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T193000Z-cap64-agent-smoke.json
```

The draft-INT4/cap64/MTP-4 candidate is rejected for deployment despite its service-level speed. A final greedy parity gate (`temperature=0`, `seed=12345`) matched BF16 exactly on fixed short and coding prompts, but mismatched at ~8k context; both responses ended by `length`. The candidate must not replace BF16 until that divergence is explained and eliminated. See the final parity artifacts under `~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-{bf16,gdn,comparison.json}`. This supersedes the stale "currently running greedy parity comparison" item below.

## Priority questions for external experts

1. **W4A16/RMSNorm fusion:** Does the symmetric GPTQ G128 XPU path already compute reusable reductions/statistics analogous to MLXFast's affine-G64 `xsums`? If yes, which XPU kernel interface should expose/reuse them for MTP verify widths 4–9 without changing target numerics?
2. **MTP runner scheduling:** Can static Qwen MTP depth be made adaptive safely in vLLM while preserving GDN recurrent-state alignment, FP8 KV, and non-greedy target sampling? MLXFast uses variable 0–8 draft depth; the pinned vLLM MTP path is static.
3. **Proposal head:** Is there a tractable way to train a B70-compatible proposal-only MTP head for this exact GPTQ target, with target verification and agent-quality gates? The MLXFast head format is not drop-in compatible.
4. **Mixed XPU GDN batches:** v5 is a Python workaround that proved C2 correctness. What is the correct SYCL kernel-level partition/merge design for concurrent spec + prefill/decode, and can it be upstreamed?
5. **Graph memory:** Can XPU graph captures be shaped more efficiently than cap64 without crossing the cap128 startup-failure boundary? Cap64 is currently stable; full capture used 5.31 GiB and long-context overlay runs OOMed.
6. **Acceptance and long context:** Which model/runtime changes increase MTP acceptance at natural long contexts without trading away target-token parity or the 98k/32k agent contract?

## Recommended next work

1. Finish the currently running greedy parity comparison between BF16 and draft-INT4/cap64/MTP-4 + metadata overlay.
2. Resolve Prime balance, then run the planned one-task Pi/Harbor smoke before any full cohort or deployment.
3. Repeat C1 and C2 rows enough times for confidence intervals; do not use one-sample C2 results as performance conclusions.
4. Profile the exact XPU MTP verification round at widths 4–9. Attribute time to W4A16 GEMMs, RMSNorm/reductions, GDN state movement, metadata preparation, graph replay, and CPU synchronization.
5. Prototype only a measured redundant-buffer/fusion target. Do not port MLX Metal source or its custom head blindly.
6. Move mixed-batch handling from the Python v5 workaround toward an upstreamed SYCL/XPU kernel solution.

## Artifact index

```text
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T163646Z-c1-service-controls/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T165919Z-c1-draft-int4-service-controls/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T181000Z-draft-nograph/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T181000Z-draft-cap32/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T183500Z-draft-cap64/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T190500Z-draft-cap64-mtp2/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T195000Z-draft-cap64-gdn/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T203500Z-gdn-v5-c2/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T204500Z-gdn-v5-meta-c2/
```

## References

- [MLXFast leaderboard](https://www.yukon.org/mlxfast)
- [MLXFast Qwen3.8 MTP challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)
- [Promoted winner diff](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge/compare/eb5eadc7a165047d4321ce883b9ff30894d8bd19...0863b06ac16e26e48fc06e97444095b00feb66d4.diff)
- [B70 GPTQ/MTP model card](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16)
- [B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/main/docs/qwen38-27/QWEN38-VLLM-XPU.md)
- [vLLM PR #43955](https://github.com/vllm-project/vllm/pull/43955)
- [vLLM Qwen MTP source](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)
- [vLLM XPU kernels](https://github.com/vllm-project/vllm-xpu-kernels)
