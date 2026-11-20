# Flint on vLLM 0.34 XPU (single B70): non-speculative decode hill-climb — 2026-11-19

Tier: **development results** (same limits as
[the correctness/perf doc](flint-vllm-xpu-correctness-perf-20261119.md): 6 prompts × 96 tokens,
concurrency 1, short context, no task-quality eval; BENCHMARKING_STANDARDS checklist not completed).
Setup, launcher, and llama.cpp baseline (46.82 tok/s) are unchanged from that doc.

## Result

| Build | Decode tok/s (graph, median) | TTFT | Kernels / decode step | Kernel busy / step (eager profile) |
|---|---|---|---|---|
| `build-moe` (previous best) | 138.5 | 39 ms | 1086 | 6.87 ms |
| `build-dec3` (router + top-k, MoE GEMV rework) | 187.6 | 42 ms | 722 | 5.19 ms |
| `build-dec4` (+ shared expert folded into the MoE kernels) | 201.4 (repeat 201.5) | 41 ms | — | 4.92 ms |
| **`build-dec5`** (+ permuted norms, Triton qkv GEMV) | **204.3** (repeat 204.8) | 41 ms | — | — |

Same command as before: `EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800
EXTRA_ARGS="--max-num-batched-tokens 4096" bash maintenance.sh --bench` (raw rows:
`results/20261119-flint-vllm-xpu/vllm-graph-dec{3,4}-bench*.jsonl`). 4.4× llama.cpp decode (dec5).

## Where the time went (eager torch-profiler, per decode token)

`build-moe` (`hillclimb/profile-build-moe-eager.txt`): router `_router_kernel` (tl.dot, BM=16 tiles
for M=1) 15.6 µs + `TopKGating` 6.3 µs + six id-sanitization micro-kernels ≈ 29 µs/layer
(≈1.3 ms/token); MoE down GEMV 20.6 µs/layer at ~150 GB/s; oneDNN W4A16 linears 2.88 ms
(lm_head 1.03 ms at ~590 GB/s, already at bandwidth).

Measured floors (cold weights, XPU graph replay): DRAM copy 537 GB/s; a Triton streaming read
of 6.3 MB takes 11.7 µs (535 GB/s) and of 3.1 MB 6.4 µs; an empty kernel costs 0.65 µs in a graph.

## Changes (all in `ops/xpu_decode.py` + `11-flint-xpu-decode-router-shared-fold.patch`)

1. **Router + top-k** (`router()`): `_router_partial` is a split-K GEMV (BN=8, KS=4) writing fp32
   partial logits; each row runs identical code (fixed tile and reduction, runtime row loop of
   ≤16 rows/program), so logits are bitwise batch-invariant (tested M=1/3/8/17/37/100).
   `_router_topk` (1 warp) sums partials in fixed order, picks top-6 by logit (ties → lower id;
   non-finite logits rank last; ids always valid and distinct) and renormalized sigmoid weights.
   Replaces router + TopKGating + the `torch.where` sanitization. Cold: 45.8 → 10 µs/layer
   (profile: 6.2 µs). Also used by `FlintXpuGateLinear` for M>1, so logits match across shapes.
2. **MoE GEMVs**: int32 word loads (8 nibbles), gate_up reads x as a contiguous `[NW, 8]` tile
   (`warp_size=16`, BN=2), down processes one output row per program; shared output added in
   fp32 before the single bf16 rounding. Cold: gate_up 19.8 → 16.4 µs, down 20.6 → 9.6 µs.
3. **Shared expert folded into the MoE kernels** (`moe_decode_sh`): at the first forward (the
   memory-profiling run, after the expert layout is final, before graph capture) the GPTQ
   uint4b8 shared MLP (I=1024) is repacked into two signed-nibble pseudo-experts of I=512
   (`u ^ 8`), gate_up reads a nibble-major permuted x (contiguous loads, FMA chain), and the down
   kernel sums 6 routed + 2 shared experts in fp32. Removes two oneDNN GEMMs, the SiLU kernel and
   the add. Real weights, cold, whole MoE block: 43.6 → 35.1 µs/layer. Shared-expert error vs fp64:
   1.2e-3 (oneDNN path 4.5e-3).
4. **Permuted norms + Triton qkv** (`rms_norm_perm`, `gptq_linear_perm`, patch 12): at M=1 the
   input RMSNorm writes only the nibble-major permuted output (same arithmetic as `xpu_fused`,
   bitwise-identical values) and qkv runs through a GPTQ uint4b8 GEMV reading it (BN=4, NW=64,
   `warp_size=16`; synthetic cold 11.7 µs vs oneDNN 14.2 µs; relerr vs fp64 2.8e-3). The
   post-attention RMSNorm writes both orders, removing the MoE `_permute_x` kernel. o_proj stays on
   oneDNN (its input comes from the attention kernel; a separate permute cancels the gain).
   Eligibility (`gptq_ok`, uses `.item()`) is decided in the eager profiling forward, never during
   capture.

Kill switches (`SERVE_ENV=`): `FLINT_XPU_PERM=0` (dec4 behaviour), `FLINT_XPU_SHARED_FOLD=0`
(dec3), `FLINT_XPU_DECODE_V3=0` (build-moe).

## Correctness gate

| Check | dec3 | dec4 | dec5 |
|---|---|---|---|
| Unit gate `decfinal.py` (router fp64 err <2e-6, batch invariance, determinism, top-k order 100/100, MoE / shared-MoE vs fp64 <5e-3; dec5 adds permuted norm bitwise == `xpu_fused`, GPTQ GEMV vs fp64 <5e-3, invariance, determinism) | PASS | PASS | PASS |
| Graph == eager, 8 prompts × 128 tokens (tokens and logprobs bitwise) | 8/8 | 8/8 | 8/8 |
| Repeat graph runs bitwise identical | — | 8/8 | 8/8 |

FP32 CPU reference (`cpuref2.py`). Free-running trajectories differ between builds, so the
per-build trajectory metric mixes accuracy with *which* positions get sampled (dec4 trajectory
median 0.0071 vs 0.0026). Each build was therefore also replayed through the real graph decode
path on the **identical** build-moe token sequence with a teacher-forcing logits processor
(`hillclimb/forcetok.py`, `tf.py`; raw logprobs; validated: build-moe reproduces its own server
logprobs bitwise on 1024/1024 positions):

| Teacher-forced, same 1024 positions | median \|Δlp\| | mean | p95 | p99 | max | # > 0.5 | # > 1.0 |
|---|---|---|---|---|---|---|---|
| build-moe | 0.00259 | 0.0412 | 0.180 | 0.544 | 3.74 | 13 | 3 |
| dec3 | 0.00261 | 0.0372 | 0.177 | 0.423 | 1.52 | 9 | 5 |
| dec4 | **0.00226** | 0.0437 | **0.170** | 0.674 | 5.36 | 14 | 4 |
| dec5 | 0.00231 | 0.0403 | 0.183 | 0.456 | 4.70 | **8** | 3 |

Bulk accuracy is equal or better; the tails are isolated near-tie expert flips that every build
has at different positions (build-moe's worst: case 2 pos 112, −0.47 vs fp32 −4.21; dec4's: case 2
pos 35). Exceedance counts are within ±1–4 of the baseline.

## Tried and rejected (raw numbers in `results/.../hillclimb/dec*test.json`)

- Byte-granular / 3D-gathered expert loads, 8×-strided x gathers: 17–80 µs, config-erratic.
- "v4" GEMV core with a per-lane [BN, NW] accumulator: ~2× slower (cause not investigated).
- Triton GPTQ GEMV with an in-kernel x tile (normal order) for qkv/o/shared: more accurate but not
  faster than oneDNN in the real decode loop. With nibble-major x it wins for qkv (adopted, change 4).
- MoE down variants (`dec9test.json`): all-experts 2D gather, pre-loaded ids/weights,
  warp sweeps: none beat the current kernel (12.7–14.4 µs run-to-run).
- 2D block loads via tensor descriptors (`dec10test.json`; block pointers are removed in this
  Triton): same speed as plain loads for qkv, o and MoE gate_up.
- Permuted-input down kernel: no gain (9.7 vs 9.4 µs).
- int8 lm_head: not pursued (user decision: keep the exact bf16 lm_head).
- 2D-only unified attention for decode (env-gated `seq_threshold_3D = 0` in `triton_attn.py`,
  reverted): 186.7 tok/s vs 204.5 — 4 KV heads give the 2D kernel too few programs at batch 1.
  `results/.../vllm-graph-dec5-attn2d-bench.jsonl`.

## Remaining budget and next steps (non-speculative)

dec4 eager profile (`hillclimb/profile-build-dec4-eager.txt`, 24 merged decode steps): 4.92 ms
kernel busy per token — oneDNN GEMMs 2.40 ms (lm_head 1.03 ms at bandwidth, kept bf16), MoE
gate_up+shared 0.88 ms (18.8 µs/layer ≈ 450 GB/s), MoE down+shared 0.57 ms (12.1 µs, ≈ 350 GB/s),
attention + `reduce_segments` + cache + QK-norm 0.56 ms, router 0.29 ms, norms + permute 0.18 ms.
dec5 moves qkv to Triton and drops the permute. Remaining accuracy-neutral candidates are each
≈1–3%: Triton dense-layer MLPs (layers 0–1), fusing top-k into the router GEMV, 2D
(non-segmented) attention for short contexts, and fusing QK-norm/RoPE with the KV-cache write.
