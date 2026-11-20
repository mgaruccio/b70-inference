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

## Speculative decoding (DFlash 2) on the dec5 kernels — spec1/spec2

Development-tier results (not standard-compliant). The setup matches the rest of this document: single B70, graph replay
(`FULL_DECODE_ONLY`, `-cc.mode=none`, `--max-num-batched-tokens 4096`), concurrency 1, and drafter
`incoai/Flint-DFlash2@39b0a7f7`. Raw results are in `results/20261119-flint-vllm-xpu/spec/`.

**Problem on dec5.** Verify (M = k+1) still ran CUTLASS MoE plus oneDNN int4 GEMMs, so spec was
slower than non-spec at every k: 123–159 tok/s at k = 1–7, against 204 for non-spec. The cycle cost
was mostly fixed: 13.6 ms at k = 1, 17.8 ms at k = 7. In the eager profile at k = 7
(`spec/profile-dec5-k7-eager.txt`), a verify forward spends 12.4 ms in kernels, 7.5 ms of it in
CUTLASS MoE (≈159 µs/layer). Target lm_head plus drafter take 4.0 ms: 5 bf16 layers 1.08 ms, the
drafter's full-vocab bf16 lm_head pass 1.04 ms, and the target lm_head 1.04 ms. None of the 8
parity prompts stayed token-identical to non-spec.

**spec1 (patch 13 + `xpu_decode` TOKF).** Rows with M ≤ `FLINT_XPU_DECODE_MAXM` (default 8) take the
same permuted-norm → Triton qkv → router → MoE-GEMV path as M = 1. For M > 1 the GEMVs launch with
the token axis fastest in the grid. Rows routed to the same expert then read its tiles from cache,
and the per-row arithmetic is unchanged. Gate: `spec1test.py` checks M = 2/3/4/5/8 with random and
overlapping routing. Each row is bitwise equal to the M = 1 call and to the dec5 kernels, and runs
are deterministic. `FLINT_XPU_DECODE_MAXM=1` restores dec5.

**spec2.** For M > 1 there is now an expert-grouped gate/up (`_moe_gate_up_shg`: the leader pair
dequantizes the tile once for every row routed to its expert) and a row-shared qkv GEMV
(`_gptq_gemv_permg`). Both have the same per-row arithmetic and pass the same bitwise gate. The
gains are small: per-row cross-lane reductions dominate, not dequantization. Measured in µs per
layer, cold, in graph:

| M | MoE, overlapping routing (`half`) | qkv, row-shared vs per-row |
|---|---|---|
| 2 | 43 | 14.9 vs 20.5 |
| 4 | 73 | 24.5 vs 26.8 |
| 8 | 138 | 43 vs 43 |

A grouped down kernel was slower (serial per n-block) and is off (`MULTI_GROUPED_DOWN`).

| Workload (median tok/s) | non-spec | k=2 | k=3 | k=4 | k=5 | k=7 |
|---|---|---|---|---|---|---|
| raw-completion bench (`--bench*`, 96 tok, greedy), dec5 | 204.3 | 149.0 | 155.6 | 159.4 | 158.8 | 156.0 |
| raw-completion bench, spec2 | 204.3 | — | 200.9 | — | — | 160.4 (spec1) |
| chat bench (`chat-bench.py`, 256 tok), greedy, spec2 | 199.8 | 219.8 | **242.6** / 242.8 | 238.8 | 228.3 | 211.8 |
| chat bench, service sampling (T=1, top-p 0.95, top-k 20) | 188.5 | 207.3 | 223.4 / 223.7 | 222.5 | 227.0 | 191.8 |
| mean acceptance length, chat greedy / service | — | 2.23 / 2.16 | 2.65 / 2.49 | 2.93 / 2.76 | 3.11 / 2.97 | 3.27 / 2.99 |

Acceptance depends on the workload. With the chat template, k = 3 is +21% (greedy) and +18.5%
(sampling) over non-spec. The raw-completion prompts give lower acceptance (2.40 at k = 3) and spec
only breaks even there. k = 3–5 are within noise under sampling; k = 3 is best greedy. The drafter
taps are correct: vLLM reports aux layers (6, 16, 27, 37, 47), which is `target_layer_ids + 1`, the
DFlash `hidden_states[i+1]` convention.

**Correctness.**
- Non-spec spec2 parity is bitwise identical to dec5 (8/8: tokens, logprobs, top-5), because the
  M = 1 kernels are unchanged.
- Spec vs non-spec tokens are identical on 8/8 prompts at k = 3 and k = 7. Logprobs are bitwise
  identical on 8/8 at k = 7 and 7/8 at k = 3.
- The remaining difference is attention. The Triton unified-attention 3D path sizes its softmax
  segments from the step's `seq_len` (`cdiv(seq_len, 16 × 16)`). Near every 256-token boundary,
  verify rows therefore get a different key partition than M = 1 decode: a 1-ulp reduction-order
  difference, amplified through MoE routing. It was confirmed with `parity2.py` at `max_tokens` =
  128/131/140: the difference starts at absolute position 768 of the long prompt and persists, and
  short prompts never cross a boundary.
- Fixing this requires per-query segmentation in the vendored kernel. Not done.

**Verify-cycle budget at k = 3** (spec1 eager profile, `spec/profile-spec1-k3-eager.txt`): verify
forward 7.2 ms (MoE gate_up 2.2, down 1.56, qkv 1.3, o_proj + dense oneDNN 0.67, router 0.43,
attention 0.54, norms 0.34); target lm_head + drafter 4.0 ms. The next levers do not affect output
correctness, since spec decoding stays lossless, but they need a user decision because they quantize
drafter-side weights:
- an int8 copy of the lm_head used only for drafting candidates (≈ −0.45 ms/cycle, ≈ +4%);
- int8 drafter layers (≈ −0.5 ms/cycle).
A W4A16 XMX (`tl.dot`, 8-row tile) MoE kernel shared by M = 1 and verify would remove the per-row
cost scaling, but it changes M = 1 numerics and needs the full accuracy gate.

Commands (guest, `~/flint-vllm-spike`; `NO_RESTORE=1` leaves llama.cpp stopped):

```
NO_RESTORE=1 NUM_SPEC=3 EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 \
  EXTRA_ARGS="--max-num-batched-tokens 4096 -cc.mode=none" bash maintenance.sh --bench-dflash2
PROBE_SCRIPT=ktest3/chat-bench.py <same env> bash maintenance.sh --bench-dflash2   # chat bench
OFFLINE_SCRIPT=ktest3/spec1test.py bash maintenance.sh --offline                   # kernel gate
```

### spec3–spec5: drafter-side int8 (user-approved; target lm_head stays bf16)

- **Int8 draft lm_head** (`ops/xpu_draft_head.py`, patch 14 on `qwen3_dflash2.py`). DFlash 2
  chooses its top-16 candidates with a full pass over the target's bf16 lm_head (1.04 ms). spec3
  keeps a per-row symmetric int8 copy (+307 MB, built in the profiling run before graph capture
  and KV sizing) and computes those logits with a `tl.dot` kernel.
  - Measured against bf16 in µs: 561 vs 1036, which is 548 GB/s. The error is at the level of
    bf16 rounding, top-1 agreement is 100% and top-16 overlap is about 98% on the real weights
    (`spec/drafthead_test.json`).
  - Verification and the target logits are unchanged, so output stays lossless. Kill switch:
    `FLINT_XPU_DRAFT_HEAD_INT8=0`.
- **Int8 drafter linears** (spec4, 32 layers, M ≤ 64). No clear gain: chat k = 3 was 256.0 vs
  253.5 greedy and 227.9 vs 233.2 under sampling. They are now opt-in via
  `FLINT_XPU_DRAFT_LINEAR_INT8=1`.

| chat bench (median tok/s), greedy / service sampling | k=3 | k=4 | k=5 |
|---|---|---|---|
| spec2 (bf16 drafter) | 242.6 / 223.4 | 238.8 / 222.5 | 228.3 / 227.0 |
| spec3 (int8 draft head) | **253.5** / 233.2 | 246.0 / **238.3** | 236.0 / 221.2 |
| spec5 (= spec3, final install) | — | 245.6 / 237.6 | — |
| non-spec | 199.8 / 188.5 | | |

The raw-completion bench with spec3 at k = 3 runs at 210.0 tok/s (non-spec 204.3). With spec5 at
k = 4 it runs at 202.4. Parity in every spec3–spec5 run: tokens identical to non-spec on 8/8
prompts. The remaining logprob difference is the attention segmentation case above.

**Recommendation.** Use k = 4 for the live service's sampling (+26% over non-spec: 237.6 vs 188.5).
Use k = 3 for greedy (+27%: 253.5 vs 199.8). The guest venv is `build-spec5`
(`model.py`, `ops/xpu_decode.py`, `ops/xpu_draft_head.py`, `qwen3_dflash2.py`; the original is kept
as `~/flint-vllm-spike/qwen3_dflash2.py.orig`). Note that `swap-build.sh` does not yet handle
`xpu_draft_head.py` or `qwen3_dflash2.py`.

**Before a vLLM cutover** (next session):
- quality eval of GPTQ vs the GGUF baseline;
- long-context runs (tokens > 4096 need `--max-model-len` and `--max-num-batched-tokens` changes,
  and every 256-token attention boundary is a spec/non-spec divergence point);
- concurrency (all results here use `--max-num-seqs 1`; at concurrency > 2 with k = 4, verify
  M > 8 leaves the batch-invariant path);
- a persistent unit with the API key handled like llama.cpp's `--api-key-file`;
- a rollback path to `flint-inference.service`.
