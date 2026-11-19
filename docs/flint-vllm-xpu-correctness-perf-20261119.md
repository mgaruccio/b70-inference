# Flint on vLLM 0.34 XPU (single B70): correctness fixes and decode kernels — 2026-11-19

Tier: **development results** (BENCHMARKING_STANDARDS publication checklist not completed: short
single-stream workload, n=6 prompts, no long-context or concurrency sweep, no task-quality eval).

## Setup

- Guest `b70-dev` (KVM/VFIO, one Arc Pro B70 32 GiB, `xe`), isolated Intel Compute Runtime
  26.40.40196.8 + IGC 2.42.6, `NEOReadDebugKeys=1 ForcePreemptionMode=3`, vLLM `0.34.0+xpu`,
  PyTorch `2.15.0+xpu`, `vllm_xpu_kernels 0.1.17`.
- Target: `urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ@76a12a04` (sym int4, group 64),
  BF16 activations, `TRITON_ATTN`, `--max-num-seqs 1`, prefix caching off,
  `VLLM_XPU_INC_WNA16_BACKEND=w4a16`. Drafter (spec runs): `incoai/Flint-DFlash2@39b0a7f7`.
- Launcher: `scripts/flint-vllm-xpu/guest-maintenance.sh` (stops the live llama.cpp, runs one
  bounded transient unit on 127.0.0.1:18080, restores llama.cpp on exit).
- Baseline: live llama.cpp b12499 Vulkan, `Flint-Q4_K_M.gguf`, 46.74 decode tok/s (same prompts,
  96 tokens, concurrency 1).

## Correctness

| # | Finding | Fix |
|---|---|---|
| 1 | DFlash 2 verifier is correct: at the reported greedy divergences the emitted token equals the target argmax under batched (verify-shaped) scoring. | none needed |
| 2 | Router ran in BF16 (weights, logits, and sigmoid inside `topk_sigmoid`). The HF reference computes `F.linear(x.float(), w.float())` + fp32 sigmoid. BF16 created exact ties and picked a *lower-logit* expert (layer 24 tok 19: −0.7539 over −0.75). Batched MoE error vs fp32 dequant reference was 1.6e-2–3.1e-2 vs 6e-3 at decode. | fp32 router logits |
| 3 | Plain fp32 `F.linear` / `mm(out_dtype=fp32)` on XPU are run-to-run nondeterministic and batch-variant (~3e-6); with Flint's dense router near-ties this flipped routing between identical requests (1.7 nats). | Triton bf16×bf16→fp32 router (`ops/xpu_router.py`): deterministic, batch-invariant, 5e-7 vs fp64 |
| 4 | Single-token MoE path saw all-`-1` top-k ids on padding/dummy rows (47 events per forward); FusedMoE masks these, a raw kernel faults (DEVICE_LOST). | sanitize ids → expert 0, weight 0 |
| 5 | Chunked prefill attention (chunk with prior context) is 1-ulp nondeterministic in the Triton unified-attention kernel (not TD-specific, not stale KV). | avoided by `--max-num-batched-tokens 4096` (≥ max-model-len) |
| 6 | Passing a row stride as an extra runtime int argument to a Triton kernel page-faulted the GPU on this stack (`ccs` CAT error, DEVICE_LOST). | kernels index with constexpr strides |

Evidence (FP32 CPU ground truth = HF `modeling_flint` layers with fp32-dequantized GPTQ weights):

- All XPU ops vs fp32 dequant reference at BF16 level: W4A16 linears 2–4e-3, norms 1.7e-3,
  attention (sinks, sliding window, 643-token prefill) 1.4–2.3e-3 flat across positions.
- Final build, generated continuations (8 prompts × 128): median |Δlogprob| 0.0026, p95 0.18,
  served token == fp32 argmax 990/1024. Graph replay is bitwise identical to eager on all 8.
- Teacher-forced natural text (prompts, 951 positions, no selection bias): vLLM↔fp32 median 0.017;
  vLLM↔llama.cpp 0.111; fp32↔llama.cpp 0.124. vLLM is as close to llama.cpp as its own fp32
  reference; the large residual disagreements come from the **checkpoint** (GPTQ int4 vs GGUF
  Q4_K/Q8_0) and Flint's high sensitivity to perturbation, not the port. All unquantized tensors
  are bit-identical between the two checkpoints; experts differ uniformly (~0.12 rel, no broken
  expert in layers 2/3/24/47).
- **Spec vs no-spec parity is resolved**: with the fixes above, DFlash 2 output is token-identical
  to non-spec greedy decode on all 8 prompts × 128 tokens when both use the CUTLASS MoE path
  (max logprob difference 0.026; the session started at 1/6 exact matches). DFlash 2 graph replay
  is bitwise identical to DFlash 2 eager. With the faster M=1 GEMV MoE enabled for no-spec decode,
  the two paths use different (both fp32-accumulating) MoE kernels and diverge at near-ties again;
  set `FLINT_XPU_MOE_DECODE=0` when exact spec/no-spec equivalence matters.

## Decode performance (no speculation, 6 prompts × 96 tokens, greedy, median)

| Build | decode tok/s | TTFT ms | GPU kernel ms/token |
|---|---|---|---|
| llama.cpp Q4_K_M (Vulkan) | 46.74 | — | — |
| vLLM eager (start of session) | 20.82 | 105 | 13.7 (≈3,900 launches) |
| + XPU graph replay (`FULL_DECODE_ONLY`) | 68.84 | 58 | 13.7 |
| + Triton RMSNorm / QK-norm+RoPE / SiLU·mul (replaces eager fp32 bring-up ops) | 105.26 | 40 | 9.0 (≈1,200 launches) |
| + M=1 W4A16 MoE GEMV kernels (`ops/xpu_moe_decode.py`) | **138.51** | **39** | ≈6.8 |
| DFlash 2 (7 draft tokens), eager | 60.16 | 115 | — |
| **DFlash 2 + graph replay** (`-cc.mode=none`, no Inductor) | **168.32** | 52 | — |

DFlash 2 mean acceptance length 2.73 (24.7% per-position) on this workload. `-cc.mode=none` keeps
full decode-graph capture for target and drafter without Inductor (which needs the missing `ocloc`).
DFlash 2 outputs vs fp32: median |Δlogprob| 0.0052, fp32-argmax agreement 96.4%.

The decode MoE path replaces remap → CUTLASS grouped GEMM ×2 → act → gather (~87 µs/layer) with
two kernels (gate_up+SiLU 27.5 µs, down+weighted sum 13.2 µs under graph replay; bandwidth floor
12.2 + 6.1 µs). FP32 accumulation makes it ~2.5× closer to the fp32 reference than CUTLASS.
Batched MoE (prefill, spec verify) still uses CUTLASS, which wins at M≥8.

Remaining per-token budget (from the fused-build profile): lm_head BF16 1.03 ms (at bandwidth),
other oneDNN W4A16 GEMMs 1.82 ms (~290 GB/s), router 0.70 ms, top-k/attention/norms ~1 ms.

## Reproduce

Patches: `patches/flint-vllm-0.34-xpu/` (apply `00`–`02` bring-up, then `10`, and copy `ops/*.py`
into `vllm/models/flint/ops/`). Harnesses: `scripts/flint-vllm-xpu/` (`parity.py`, `probe*.py`,
`cpuref*.py`, `oplevel.py`, `repeat.py`, `moetest.py`, `moebench*.py`, `trace_breakdown.py`).
Graph bench: `EAGER=0 READY_WAIT=900 SERVE_MAXSEC=1800 EXTRA_ARGS="--max-num-batched-tokens 4096"
bash guest-maintenance.sh --bench`. Kill-switch for the MoE decode path: `FLINT_XPU_MOE_DECODE=0`.

## Open items

- Quality A/B before any cutover: unbiased perplexity/task eval of GPTQ-int4 (vLLM) vs GGUF
  Q4_K_M (llama.cpp); the checkpoints disagree substantially on teacher-forced text.
- DFlash 2: tune draft width (acceptance 24.7%/position); Inductor fusion would need `ocloc`;
  verify at M=8 uses CUTLASS MoE (~93 µs/layer).
- Further decode headroom: M=1 router GEMV (0.7 ms), Triton W4A16 GEMV for qkv/o/shared expert.
- Long-context and concurrency measurements per BENCHMARKING_STANDARDS.
