# Inkling 2 Mini on one B70: external kernel contributions, lossless-only comparison

**Tier: development.** Same host and session, but not standard-compliant: no BetterBench, no `vllm bench serve`, and no
512→max long-context sweep. Decode uses 6 prompts × 256 greedy tokens (3 prompts at depth), with two interleaved rounds
(A B C A B C). Do not compare these numbers with standard-tier results.

## Question

Two community contributions were posted after our Hugging Face post (discussion #29):

- **B — SergiioB's opt-in CUTLASS Xe2 expert kernel**, from cookbook `intel-arc-pro-b70-inference-cookbook@f10f667`, `patches/inkling2-mini-vllm-xpu/`, `INK2_CUTLASS_MOE=1`:
  - layers 0–19: a native NVFP4 grouped GEMM, `nvfp4_xe2_ext.cpp`
  - layers 20–23: vllm-xpu-kernels' MXFP8 CUTLASS grouped GEMM
- **C — F3zz1k's vLLM fork** `F3zz1k/vllm@edd722c73517b43680483187afca93fb6ab41958` (Lumnus `b70-flash-next` v0.32.0-b70.7 base), which brings:
  - SYCL relative-bias attention
  - its own NVFP4 grouped GEMM (`_inkling2_C`)

Which of them are worth incorporating if we accept **no quality loss**?

## Configurations

| | Stack | Precision settings |
|---|---|---|
| **A** (baseline) | Our current tree `hc-s1`: vLLM 0.34.0+xpu, torch 2.15.0+xpu, Triton 3.9.0, compute-runtime 26.40 (local driver dir), `-cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY` | Checkpoint NVFP4/MXFP8 experts as-is, BF16 dense, FP32-output head (`torch.mm(out_dtype=fp32)`, vLLM PR #69186 behaviour) |
| **B** | A + the cookbook CUTLASS kernel ported into `hc-s1` (`scripts/hc-ck1-moe.diff`), `INK2_CUTLASS_MOE=1` | Same as A |
| **C** | F3zz1k image built from `edd722c7`, per their README (`scripts/f3-build-and-pr69186.txt`); torch 2.14.0+xpu, XPU graphs | `VLLM_INKLING2_EXPERTS=nvfp4`, `MXFP8=bf16`, `DENSE=bf16`, `HEAD=fp32`, `PREFILL_EXPERTS=nvfp4`, plus vLLM PR #69186 bind-mounted (`logits_processor.py`) so the FP32 head does not recast its weight every step |
| C0 | C without the PR #69186 mount | Same as C; quality gate and one decode run only |

The fork's defaults are lossy: `EXPERTS=fp8`, `MXFP8=fp8` (one scale per expert instead of the checkpoint's per-32 block scales), and `HEAD=bf16`. Its published 107.7 tok/s also used `DENSE=int8` and `HEAD=int8`. All of these were overridden here.

### Shared serving settings

- `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`
- 1× Arc Pro B70, power cap 275 W
- `--max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 1 --block-size 64 --no-enable-prefix-caching`, BF16 KV
- DSpark `{"method":"dspark","num_speculative_tokens":7}` with the bundled drafter

### Intentional difference

`--max-num-batched-tokens` is 2048 for A and B and 4096 for C (the fork's recommended value). A matched-chunk prefill run for C at 2048 is reported separately. A and B **cannot start** at 4096: vLLM requires 7.94 GiB of KV for one 16K request and only 7.16–7.35 GiB is available at 0.92.

## Quality gate (`parity`, `tf`)

- **CPU FP32 parity:** teacher-forced prompt log-probs on the same 4,174 positions as the HF post, compared with CPU FP32 Transformers (experts dequantized to fp32). The CPU BF16-vs-FP32 floor is 0.9938 top-1 and p95 |Δlogprob| 0.0285.
- **Agreement with A:** each candidate is teacher-forced on A's greedy outputs (6 × 256 tokens), and its top-1 is compared with A's tokens.

| | top-1 vs FP32 | median / p95 \|Δlogprob\| | top-1 on A's greedy tokens |
|---|---:|---:|---:|
| A | 0.9933 | 0.000168 / 0.0235 | (reference) |
| B | 0.9935 | 0.000149 / 0.0219 | 0.9837 |
| C | 0.9931 | 0.000175 / 0.0231 | 0.9857 |
| C0 | 0.9943 | 0.000171 / 0.0230 | 0.9876 |

**All configurations sit at the CPU BF16 noise floor.** None of them loses quality relative to A.

B's greedy outputs drift from A's after 8–159 tokens (6 prompts; `runs/*-B-s0-qgate` vs `runs/*-A-s0-qgate`), which is the expected BF16 summation-order divergence. Smoke tests (37+58 → `95`) passed everywhere.

## Speed (medians; A and B bit-identical across rounds; C shown as r1 / r2)

### Decode without speculation, tok/s

| context | A | B | C |
|---|---:|---:|---:|
| short | **57.6** | 56.0 (−3%) | 55.3 (−4%) |
| +6K | **54.9** | 53.4 (−3%) | 51.3 / 51.2 (−7%) |
| +30K | **51.3** | 50.0 (−3%) | 48.0 (−6%) |

### Decode with DSpark k=7, tok/s (acceptance length)

| context | A | B | C |
|---|---:|---:|---:|
| short | 83.0 (2.99) | **99.0 (3.16), +19%** | 91.0 (3.04) / 88.4 (2.95) |
| +6K | 82.2 (3.22) | **103.6 (3.13), +26%** | 85.8 (3.01) / 77.7 (2.98) |
| +30K | 76.4 (3.11) | **100.5 (3.57), +32%** | 78.4 (3.36) / 78.4 (3.29) |

Acceptance length depends on the greedy trajectory, which differs between stacks, so part of these deltas is content. An approximate per-step cost (median tok/s ÷ acceptance length) isolates the kernels:

| context | A | B | C |
|---|---:|---:|---:|
| short | 36.0 ms | 31.9 ms (−11%) | 33.4 ms (−7%) |
| +6K | 39.2 ms | 30.3 ms (−23%) | 35–38 ms |
| +30K | 40.7 ms | 35.5 ms (−13%) | 42–43 ms |

### Prefill and memory

| | A | B | C |
|---|---:|---:|---:|
| 8K cold prefill, tok/s (3 trials) | 3,613 | 4,085 (+13%) | 4,935 (chunk 4096); **4,683 at chunk 2048 (+30%)** |
| TTFT, ~30K prompt | 8.65 s | 7.76 s | 6.28 s (chunk 4096) |
| Weights loaded | 17.61 GiB | 17.61 GiB | 21.39 GiB (BF16 layers 20–23, +3.8 GiB) |
| KV cache tokens, no spec | 56,775 | 57,068 | 69,791 (chunk 4096) / 78,035 (chunk 2048, max-len 16K) |
| KV cache tokens, DSpark | 43,200 | 43,642 | 40,023 |

### Determinism

- **A and B** produce bit-identical greedy outputs across rounds, with and without DSpark.
- **C is not run-to-run deterministic, even without speculation.** Over 12 prompt cases (6 short, 3 at +6K, 3 at +30K), r1 and r2 greedy outputs diverge in 9 of 12: after 3–133 tokens at short and +6K context, while +30K matched. DSpark speed also varies between rounds (77.7–85.8 at 6K).
- **Greedy DSpark output differs from each stack's own no-spec output**, as the known batched-verify numerics effect:
  - A: after 8–233 tokens, in 12 of 12 cases
  - B: after 32–242 tokens, in 10 of 12 cases (2 identical)
  - C: after 3–133 tokens, in 12 of 12 cases

  See `derive-output.txt`.

## Conclusions

1. **B is worth incorporating.**
   - It has no measurable quality cost (top-1 vs FP32 at the noise floor, slightly better than A).
   - It gives +19% to +32% DSpark decode (−11% to −23% per step) and +13% prefill.
   - It costs −3% non-spec decode. A hybrid with our ≤4-token GEMV was tried and closed (see "Follow-up" below): B as-is is the best lossless finisher.
   - The cookbook build script did not work as published. Four fixes are in `scripts/build_ck1.sh`:
     - add the `vllm-xpu-kernels` root include
     - add the `sycl-tla/tools/util/include` include
     - add `-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate` for icpx 2026.1.1
     - add `-DTORCH_EXTENSION_NAME=_inkling_nvfp4_xe2`

     It also needs `ocloc` (compute-runtime 26.40 `intel-ocloc`) on `PATH`.
2. **C as a whole stack is not worth switching to** under the no-quality-loss constraint.
   - Lossless decode is slower than A (−4% to −7% non-spec). DSpark gains +3% to +10%, but less than B.
   - Its headline speed depends on int8 and FP8 requantization.
   - Two parts are worth studying or porting:
     - **prefill path:** +30% vs A and +15% vs B at matched chunk 2048. B and C both use grouped-GEMM experts, so the remaining gap points at their SYCL chunk-prefill attention.
     - **KV/workspace memory layout:** more KV tokens at the same utilization, and it fits chunk 4096 where ours cannot.
3. **C's decode attention advantage did not show up end to end** in its lossless configuration. C's 30K decode is slower than A, but C also streams 3.8 GiB more expert weights in layers 20–23, so this is not a clean attention-only comparison.

## Follow-up: CUTLASS for verify/prefill + our decode GEMV for ≤4 tokens (closed)

**Change.** `scripts/apply_hy1.py` builds tree `hc-hy1` on top of B:

- NVFP4 layers with ≤4 tokens (`INK2_HYBRID_DEC_MAX=4`) use our decode GEMV; larger batches use CUTLASS.
- The GEMV gained a checkpoint-nibble-layout mode (`split_half=False`), because CUTLASS needs the checkpoint layout and a second weight copy would cost ~9 GB.
- Kernel check (`scripts/test_dec_layout.py`): bit-identical to the split-half GEMV at 1/2/4 tokens.

**Variants.**

- **H1:** MXFP8 layers stay on CUTLASS.
- **H2:** MXFP8 layers fully on our kernels (`INK2_CUTLASS_MXFP8=0`).

The same session reran A and B, ordered A B H1 H2 × 2. A and B reproduced their earlier values exactly.

| | A | B | H1 | H2 |
|---|---:|---:|---:|---:|
| top-1 vs FP32 / p95 \|Δlogprob\| | 0.9933 / 0.0235 | 0.9935 / 0.0219 | 0.9940 / 0.0198 | 0.9935 / 0.0229 |
| decode, no spec: short / +6K / +30K | **57.7 / 54.9 / 51.3** | 56.0 / 53.4 / 50.0 | 52.4 / 50.1 / 47.1 | 52.8 / 50.4 / 47.4 |
| DSpark k=7: short / +6K / +30K | 83.0 / 82.2 / 76.4 | **98.9 / 103.6 / 100.5** | 99.0 / 103.6 / 100.5 | 97.9 / 102.4 / 99.6 |
| 8K prefill | 3,613 | 4,080 | 4,080 | 4,030 |

**Result.** The hybrid made non-spec decode slower than both A and B.

**Cause.** Microbench (`scripts/bench_dec_layout.py`, 128 experts, top-8, 1 token): the checkpoint-layout GEMV takes 143 µs per call versus 110 µs on split-half weights.

In the checkpoint layout, a byte's two nibbles are adjacent k values. A 16-byte chunk therefore spans two scale groups (bytes 0–7 and 8–15), and the kernel has to split every chunk's products by byte half before scaling. In the split-half layout, all low nibbles belong to one scale group and all high nibbles to the other, so each nibble plane reduces directly under a single scale.

A reshape/split variant (`scripts/apply_hy2.py`) was slower still: 214 µs.

**Revisit when** the CUTLASS kernel can read split-half NVFP4, which would let one weight copy serve both kernels.

## Plugin path: B integrated into `vllm-inkling2-xpu` (`8e32652`)

The winning option, B, was integrated into the public plugin as opt-in `INK2_CUTLASS_MOE=1`:

- the kernel and dispatcher are vendored from the cookbook (MIT)
- a fixed build script ships as `scripts/build-cutlass-xe2.sh`
- a new FP64-reference test, `test_xpu_cutlass_experts`, covers NVFP4 and MXFP8 at 1–300 tokens

**Validation setup.** The plugin was validated on the guest through its own `vllm.general_plugins` entry point:

- vLLM main `3146a9df41` source tree, plus #69186 (`gh pr diff`)
- the plugin installed to a target dir on `PYTHONPATH`
- configs **P** (default kernels) and **PC** (`INK2_CUTLASS_MOE=1`)

**Unit tests.** `tests/test_moe_xpu.py`: 29 passed. The 1 deselected test is pre-existing and needs vLLM's `tests.conftest` (`tblib`), which isn't installed.

The first run had all GPU tests failing, because the shell had sourced oneAPI `setvars.sh`. Its SYCL runtime (2026.1.1) shadowed torch's bundled 2026.1.2. The plugin README now warns about this.

| | P | PC |
|---|---:|---:|
| top-1 vs FP32 / p95 \|Δlogprob\| | 0.9938 / 0.0227 | 0.9938 / 0.0230 |
| decode, no spec: short / +6K / +30K | **57.6 / 54.9 / 51.3** | 56.0 / 53.4 / 50.0 |
| DSpark k=7: short / +6K / +30K | 83.0 / 82.2 / 76.4 | **99.0 / 103.7 / 100.7** |
| 8K prefill | 3,616 | 4,085 |

P reproduces A and PC reproduces B (two rounds each, P PC P PC). So the published plugin on vLLM main matches our development tree. The HF post's 75.9 tok/s DSpark figure came from an earlier environment.

## Live switch (2026-11-23 03:06 UTC)

`scripts/inkling-vllm-xpu/serve/serve-inkling.sh` now serves tree `hc-ck1` with `INK2_CUTLASS_MOE=1`; the previous launcher is kept on the guest as `serve-inkling.sh.pre-cutlass`. Otherwise the live configuration is unchanged:

- 262,144 context, `--max-num-batched-tokens 256`, utilization 0.98
- prefix caching on
- DSpark k=7 with probabilistic draft sampling

**Before/after check.** Same live service on `:8080` with its API key, using `live/live_check.sh` (`live_bench.py` is `decode_bench.py` plus the auth header). Smoke (`95`) and tool call (`echo_value`) pass both times.

| live :8080 | before (hc-s1) | after (hc-ck1 + CUTLASS) |
|---|---:|---:|
| KV cache tokens | 270,133 | 272,796 |
| DSpark decode, short (6 prompts) | 83.3 (AL 2.99) | **99.0 (AL 3.16), +19%** |
| DSpark decode, +30K (3 prompts) | 97.8 (AL 3.57) | **100.1 (AL 3.39), +2%** |
| approx. ms per DSpark step at +30K | 36.5 | 33.9 (−7%) |

At +30K, both runs hit the live prefix cache: TTFT was 0.12–0.13 s. Acceptance length differs because the greedy trajectories differ, so the per-step cost is the cleaner comparison.

## Reproduction

All scripts are in `scripts/`; run on the guest.

- `cmp.sh CONFIG SPEC LABEL MODES...` starts a loopback server on :18080 and runs the bench modes. The production `inkling-inference` service was stopped during the runs.
- `rounds.sh` is the timed ABC×2 schedule.
- `collect.py` and `summarize.py` are the CPU FP32 parity collector and summarizer, adapted from `.pi/upstream/pr-b-inkling2-xpu/parity/`. Reference captures are not committed here.
- Bench clients are `~/inkling-vllm-xpu/hc-scripts/{decode_bench,tf_bench,xpu_prefill_bench}.py`, unchanged from the hill-climb harness.
- `derive.py` holds the divergence and determinism checks; its output is `derive-output.txt`.
- `build_ck1.sh` (fixed build) and `apply_ck1.py` (hook port) build B.
- C's image is built by `f3-build-and-pr69186.txt`. It needed `MAX_JOBS=4` and a temporary 48 GB swapfile on the 46 GB guest; the build's own RAM guard aborted three times at 6 jobs.

Per-run directories are in `runs/`. They hold `launch.txt`, `environment.txt`, `server.log`, `memory.txt`, per-case JSON (with token ids and log-probs), and summaries. Raw parity prompt-logprob captures (~12 MB per run) were left on the guest under `~/cmp/runs/*/parity/`; only `comparison.{json,md}` is committed.

### Failed or excluded runs (kept)

- `runs/*-chunk4096-ml16k` for A and B: engine refused to start (KV requirement at chunk 4096).
- `20261122-221845-B-s0-qgate`: extension built without `TORCH_EXTENSION_NAME`, so the import failed. Not copied.
- Three earlier attempts at C's image build hit the RAM guard.

## Environment

| Item | Value |
|---|---|
| Host | Guest `b70-dev`: 12 vCPU, 46 GB RAM; Ubuntu 26.04, kernel 7.0.0-43 |
| GPU | Arc Pro B70 passthrough, power cap 275 W |
| A/B runtime | `~/flint-vllm-spike/.venv`, vLLM 0.34.0+xpu tree `hc-s1`/`hc-ck1`, torch 2.15.0+xpu, vllm_xpu_kernels 0.1.17, Level Zero driver 26.40.40196.8 |
| B toolchain | icpx 2026.1.1 (`intel-oneapi-compiler-dpcpp-cpp-2026.1=2026.1.1-325`), sycl-tla `7b56b936`, vllm-xpu-kernels `e9b6e4b2` (headers) |
| C image | `vllm-xpu:b70-inkling2` `sha256:4c93bc516ae23659d69fcb0e700f07f49c2589228df379b76ef8ffa3f0c3e2f8`, vLLM `0.32.0+gedd722c735`, torch 2.14.0+xpu |
