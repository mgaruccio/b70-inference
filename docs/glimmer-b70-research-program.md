# Muse Glimmer on B70 — research program

Date: 2026-08-24
Status: ready to execute
Host: `inference-host` / Intel Arc Pro B70 (Battlemage G31, 32 GB, `xe`, ReBAR 32G)
Prior campaign: `docs/glimmer-b70-handoff-20260824.md`
Phase 0 results: `docs/glimmer-b70-phase0-20260824.md`

This is a two-track program. **C1** and **C4+** are different bottlenecks
and must not share launch flags, success metrics, or kill criteria.
**C2 and C3 are out of scope.**

The goal is not another ±10% flag sweep. The goal is to close a large
fraction of the gap to Qwen3.8-27B on the same card, then make four-plus
live streams usable.

---

## 0. Why we are slower than we “should” be

### Live numbers (same card, 128-token SSE unless noted)

| Cell | C1 | C4 each / agg |
|---|---:|---:|
| Glimmer Vulkan + DFlash `n_max=2` | **27–32 tok/s** | **3 / 12** (dead) |
| Glimmer Vulkan, no spec, 8 slots | 18 | **10.6 / 42** |
| Glimmer SYCL F16, no oneDNN, DFlash | 26.6 | 6 / 24 |
| OpenVINO GenAI 2026.4 `VLMPipeline` | 31.7 warm | serialized; unlocked C4 drops 3/4 |
| **Qwen3.8-27B vLLM-XPU + MTP-4** | **57–62 tok/s** | different engine; not a Glimmer C4 proxy |

Cookbook SYCL+F16+DFlash `n_max=2` at 128k is 26.8 tok/s (p512/g128,
acceptance 0.85). We already match that ceiling. The missing 2× is not
“we failed to apply the public recipe.”

### The KV hypothesis is the wrong C1 story

Glimmer **does** write less KV per token than a naive 27B transformer:

| | Glimmer 30B | Qwen3.8-27B |
|---|---:|---:|
| Shape | 52 × 6656, SwiGLU 19968 | 64 × 5120, FFN 17408 |
| Attention | 39 sliding (2048) + 13 full; 32 Q / **2 KV** / dim 128 | 48 Gated DeltaNet + 16 full; full attn 24 Q / 4 KV / dim 256 |
| KV write BF16 | **52 KiB/token** | **64 KiB/token** + fixed DeltaNet state |
| Our live KV | q8_0-K / q4_1-V ≈ **22 KiB/token** | FP8 ≈ **32 KiB/token** |
| Spec | 5-layer DFlash, trained **16-token** block | MTP-4 on a specialized XPU path |
| Think | **cannot disable**; template always opens thinking | officially disable-able |

That KV delta is real and modest (~1.2–1.5× fewer bytes written). It is
**not** an order-of-magnitude advantage, and it is not the short-decode
bottleneck.

Qwen only does full KV attention on **16 of 64** layers. At long
established context, Qwen’s *read* traffic can be the smaller of the two.
Glimmer’s 13 full-attention layers plus 39 local windows help capacity
and long-context reads. They do not make 128-token C1 faster.

### What actually taxes Glimmer on B70

Ranked from our measurements + architecture, not from blogs:

**C1 (short decode)**

1. **Target Q4_K_XL decode / weight traffic (~40–55% of the gap).**
   19.7 GB of Dynamic Q4, hidden 6656, 52 layers. Every accepted token
   streams almost the whole weight set through a generic Vulkan
   dequant+GEMV path. Qwen rides a vLLM-XPU kernel stack built for that
   card. This is why “less attention data” does not predict C1 tok/s.
2. **DFlash under-used (~20–30%).** No-spec 18 → DFlash 27–32 proves
   speculation is real. `n_max=2` drafts 2 of a 16-token block. Intel
   day-0 used OpenVINO + **assistant tokens = 8**. Cookbook `n_max`
   screen: n2 wins, n5–7 acceptance collapses to 0.30–0.37, **n8 aborts**.
3. **Unfused small ops / launch overhead (~15–25%).** 52 layers of
   RMSNorm, QK norm, RoPE, residual, SwiGLU, sampling. Batch-1 dispatch
   tax on Xe.
4. **Attention/KV kernel (<10% at 128 tokens; grows with context).**
   `-fa on` already covers the easy part.

**C4+**

1. **No continuous batching (~35–50%).** llama.cpp `-np` slots are not
   vLLM paged-KV + scheduler + prefix cache.
2. **DFlash multi-sequence corruption (~20–35%).** Our C4 DFlash
   (12 agg) is *slower than no-spec* (42 agg). This matches
   [llama.cpp#27117](https://github.com/ggml-org/llama.cpp/issues/27117):
   batched DFlash noise-block decode corrupts per-sequence draft state
   once several sequences draft together. Not a model-KV limit.
3. **Batched Q4_K / XMX under-use (~15–25%).** C4 work is not turning
   into efficient matrices on Vulkan.
4. **KV layout, long-context (5–20%, workload-dependent).** A strength
   once the serving path is fixed.

Secondary but user-visible: Glimmer **always thinks**. Engine tok/s
counts reasoning tokens that never appear as `content`. Qwen can turn
thinking off. Perceived speed will always look worse than the engine
number unless we cap the reasoning budget.

---

## 1. Success criteria

Measure on the public boundary: `POST /v1/chat/completions` `stream=true`.
Script: `~/inference/launchers/glimmer-stream-sweep.py` (extend it; do
not invent a parallel harness). Split **TTFT / prefill tok/s / decode
tok/s / acceptance / mean accept length**. Never report
`completion_tokens / wall` as decode.

### C1 — single stream

| Gate | Metric | Pass |
|---|---|---|
| Floor | warm 128-token SSE decode | **≥ 40 tok/s** (we are 27–32) |
| Target | same | **≥ 55 tok/s** (Qwen neighborhood) |
| Stretch | same, plus p8192/g128 | **≥ 50 tok/s** at 8k established context |
| Ambition | C1 decode | **≥ 70 tok/s** (needs kernels + real DFlash depth) |

Acceptance must stay ≥ 0.70. A 70 tok/s cell with 0.3 acceptance is a
lie (verify is doing the work, draft is noise).

### C4+ — four or more live streams

| Gate | Metric | Pass |
|---|---|---|
| Correctness | 4 identical seeded streams | per-slot acceptance spread **< 2×**, no HTTP 500, no garbage |
| Floor | C4 decode, each / agg | **≥ 12 / 48** (beat today’s no-spec 10.6 / 42) |
| Target | C4 with working spec **or** real continuous batch | **≥ 18 / 72** |
| Stretch | C8 | **≥ 10 / 80**, acceptance healthy |
| Ambition | C4 with DFlash that does not invert | spec **faster** than no-spec at the same `-np`, not slower |

C4 DFlash that is slower than no-spec is a **fail**, even if C1 looks
great.

### Non-goals

- C2 / C3 campaigns
- Shadeform 5090 / SGLang NVFP4 / NIM as a B70 proxy
- Loading mmproj unless a vision task is explicitly in scope
- Treating cookbook 1301 tok/s `pp4096` as a serve number
- Claiming GenAI unlocked-wrapper “concurrency”
- q4-er KV as a short-C1 speed project
- Training a new base model

---

## 2. What we steal vs what we invent

Nobody has published a **correct and fast B70 C4+** Glimmer cell.
That is the white space. C1 is less empty: the public SYCL recipe
already sits on our measured ceiling.

| Steal now | Invent / own | Ignore |
|---|---|---|
| llama.cpp `draft-dflash` + official Meta Dynamic Q4 + DFlash GGUF | Xe-specific Q4_K decode kernels / fusion | NVIDIA NVFP4, NIM, TensorRT, RadixArk Blackwell |
| [SergiioB B70 cookbook](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/muse-glimmer/MUSE-GLIMMER-B70.md) harness, F16 flag, `n_max` screen | Batched DFlash mask/KV fix on Vulkan/SYCL (`#27117` class) | vLLM XPU as current C1 baseline (21–25 tok/s, reasoning leak) |
| `#27117` repro + `n_max=1` high-concurrency workaround (AMD: 80 vs 37 agg at C16) | Arc-native continuous batching / scheduler | OpenVINO GPU outputs until [parity bug #37419](https://github.com/openvinotoolkit/openvino/issues/37419) is gated |
| SGLang batching *ideas* (not CUDA kernels) | OpenVINO DFlash `assistant_tokens=8` actually wired on this IR | Ollama / LM Studio as optimizers |
| AIwork4me ROCm C4+/C16 test matrix | SYCL+oneDNN on BMG G31 that does not pool-assert | Community AWQ/GPTQ/Marlin (no Xe path) |
| llama.cpp SPEED-Bench + `draft acceptance` log lines | Optional later: Glimmer EAGLE / MTP / DSpark | 5090 “20K tok/s/GPU” vendor slides |

---

## 3. Phase 0 — instrument before changing anything

**Duration:** 1–2 host sessions. **Mandatory.** No kernel or flag work
until these artifacts exist under `~/b70-evals/muse-glimmer/202608-program/`.

1. **Split the token.** For C1 and C4: TTFT, prefill tok/s, decode tok/s,
   thinking tokens vs content tokens, `draft acceptance`, mean accept
   length, draft-ms vs verify-ms. The server log is the instrument
   (`#27117`); `/metrics` spec counters are 0 on current builds.
2. **Reproduce `#27117` on this Vulkan B70**, not on AMD.
   - Servers: `-np 4` and `-np 8` (not 16; we care about C4+).
   - Loads: 4-in-flight and 8-in-flight on the *same* `-np 8` server.
   - `n_max` ∈ {1, 2, 4} at **safe** `-ub 512` / 32k–64k slot ctx.
   - Same prompt, same seed, 4–8 identical streams.
   - Record per-slot acceptance from the first speculative tick.
3. **Attribute C1 time.** GPA / VTune / `intel_gpu_top` + a single
   decode: fraction in Q4_K matvec/dequant vs attention vs elementwise
   vs host/launch. This decides whether Track A spends weeks on kernels
   or days on DFlash depth.
4. **Bytes/token at the working set.** Confirm KV write vs weight
   stream for p128, p2k, p8k, p32k. Predicted: weight-dominated at p128,
   attention rising by p8k–p32k (DFlash gain already grows with context
   in the cookbook: +19% → +52%).
5. **Thinking tax.** Same prompt with default template vs lowest
   reasoning-strength / budget the template accepts. Report engine
   tok/s **and** content tok/s.

Kill: if we cannot get per-slot acceptance out of the live Vulkan
binary, patch logging first. Blind `n_max` sweeps are how we already
wasted time.

---

## 4. Track A — C1: make one stream Qwen-fast

Work these in order. Each step has a kill.

### A1. Cheap DFlash depth, safely (days)

Cookbook already says n2 is the local max on SYCL at 128k. Still do a
**Vulkan** screen at **32k and 64k**, `-ub 512` only:

- `n_max` = 1, 2, 3, 4
- Never 128k + DFlash + `-ub 8192` (hard reboot)
- Never `n_max=8` at 128k (cookbook abort)

Pass: any cell ≥ 36 tok/s with acceptance ≥ 0.70.
Kill: if n3/n4 are ≤ n2 + 5%, stop sweeping depth on llama.cpp and
move. The trained block is 16 tokens; llama.cpp is not using it, and
raising `n_max` on this backend is the wrong shape of bet.

### A2. OpenVINO DFlash assistant-tokens = 8 (days, highest leverage low-code)

Intel’s 2026-08-14 day-0 config is **OpenVINO + DFlash + assistant
token = 8** on a B70 (Windows) and a B390 iGPU. We have the INT4 IR
and GenAI 2026.4 reaching 31.7 tok/s **without** that flag.

Ambition: if assistant=8 is real, C1 can jump without new kernels.

Work:

1. Find the actual GenAI / OVMS knob. Hunt source, nightlies, and
   Intel samples (`intel-samples/agentic-demos`). Do not take the
   article’s footnote as an API.
2. Gate **output parity** against llama.cpp greedy on a fixed prompt
   ([OpenVINO GPU corruption #37419](https://github.com/openvinotoolkit/openvino/issues/37419)
   is open on Xe2).
3. Measure C1 only. Do not claim C4 from `VLMPipeline`.

Pass: C1 ≥ 45 tok/s, greedy-parity, no vision compile tax.
Kill: no exposed flag after a bounded source hunt, or GPU outputs
diverge. Then OpenVINO is a C4 batching investigation (B4), not a C1
speed path.

### A3. SYCL + oneDNN bisect (days–1 week)

Cookbook winner is SYCL F16 at `d2f83055d` with oneDNN implied by the
prefill numbers. Our `b10588` + from-source oneDNN **aborts** in
`ggml_sycl_op_mul_mat_sycl`. AOT `bmg-g31` is unsafe.

Work: bisect llama.cpp `d2f83055d` ↔ `b10588` **and** oneDNN revs
inside the DLE image. JIT only. `--no-warmup` until warmup is proven.

Pass: a binary that loads, decodes, and beats Vulkan C1 by ≥ 15%
**or** unlocks a fused matmul path we can keep.
Kill: still aborting after the bisect window. Do not ship the
allocator-assert binary. Do not spend a second week here if A4
instrumentation already says Vulkan GEMV is the whole tax — then
write kernels against the production backend instead.

### A4. Custom Battlemage Q4_K decode (the ambitious C1 bet)

This is the program-defining C1 project **if** Phase 0 shows ≥ 40% of
decode in dequant+matvec.

Scope:

- Fused Q4_K dequant + XMX-friendly matvec for hidden=6656, GQA 32/2
- Fuse RMSNorm + QK norm / RoPE + residual / SwiGLU edges across the
  52-layer graph (dispatch tax)
- Graph capture / Level Zero command-list reuse for the decode loop
- Keep it in-tree (llama.cpp Vulkan or SYCL) so we can still serve

Ambition target: **no-spec C1 ≥ 35 tok/s**, then DFlash on top to
**≥ 55**. If we only make DFlash better and leave the target path
generic, we cap out around the current 32.

This is weeks, not a weekend. Do not start it until Phase 0 names the
kernel. Do not start it on SYCL until A3 is either green or killed.

### A5. Perceived speed (hours, parallel)

Cannot disable thinking. Can cap reasoning strength / budget in the
template and measure **content tok/s**. This does not move the engine
number; it moves whether a Pi turn *feels* like 30 or like 12.

Ship a default “agent, short think” template next to the fast cell.

---

## 5. Track B — C4+: make four streams real

Do **not** carry C1-winning DFlash `n_max=2` into C4. That cell is
already known-bad here (3 tok/s/stream).

### B1. Prove the defect on Vulkan B70 (Phase 0 item 2)

Hypothesis: `#27117` is in the **core** DFlash batch/mask/KV-inject
path (`common_speculative_impl_draft_dflash::draft()` packs
`n_seq × (n_max+1)` tokens into one non-causal noise-block decode).
If Vulkan C4 shows the same bimodal per-slot acceptance from tick 0,
it is not an AMD-HIP bug.

Pass: a receipt that says “core” or “backend” with logs.
This decides whether we patch llama.cpp ourselves or wait on HIP.

### B2. Ship the known workaround as a real C4 cell (days)

AMD at C16: `n_max=1` → acceptance 0.83–0.92, **80 agg vs 37 no-spec**.
That is the opposite of our C4 `n_max=2` disaster.

Build a **production C4+ cell**:

```text
DFLASH=1 PARALLEL_SLOTS=8 SLOT_CTX=32768 SPEC_N_MAX=1
# keep -ub 512; do not raise ub with DFlash
```

Sweep `n_max=1` at C4 and C8. Compare to no-spec 8-slot (today 42 agg).

Pass: spec ≥ 1.3× no-spec aggregate, per-slot acceptance ≥ 0.70,
spread < 2×.
This can be the **first useful C4+ Glimmer cell on this card**, and
nobody has published it.

### B3. Fix batched DFlash (the ambitious C4 bet)

If B1 says core:

- KV injection / attention mask / sampler alignment across sequences
  in the noise-block decode
- Stop calling `common_speculative_process()` in a way that interleaves
  encode+inject with concurrent drafts
- Related class: [#26741](https://github.com/ggml-org/llama.cpp/issues/26741)
  / [PR #26756](https://github.com/ggml-org/llama.cpp/pull/26756)
  (DeepSeek multi-seq spec KV rollback)

Ambition: `n_max=2..4` at C4/C8 **faster than no-spec**, acceptance
stable. That is an upstreamable llama.cpp patch and a unique B70
result.

Kill: if the bug is backend-private and we cannot see a Vulkan analog
after one focused week, keep B2 as the C4 spec cell and spend the
rest of Track B on batching the **target**.

### B4. Continuous batching that names `muse_glimmer` (week+)

llama.cpp slots will not become vLLM. Parallel bets, in order:

1. **OVMS `VLM_CB`** on a build that actually lists `muse_glimmer`.
   Weekly images do not. Nightlies / 2026.4+ only. Parity gate first.
2. **vLLM XPU Muse** only after
   [vllm-xpu-kernels#524](https://github.com/vllm-project/vllm-xpu-kernels/pull/524)
   (paged-decode tuple `16,128,64,...`) is in a pullable image **and**
   chat stops leaking reasoning into `content`. Today it is slower
   than llama.cpp DFlash C1. Revisit as a **C4 scheduler**, not a C1
   speedup.
3. **Intel llm-scaler vLLM** (`vllm-0.21.0-b3` lists Muse + DFlash).
   Correctness first; no public B70 number.

Pass: four in-flight generations that share a prefix, decode ≥ 15
tok/s each, no dropped connections.
Kill: any path that cannot name the architecture or cannot batch.

### B5. No-spec C4+ throughput (parallel to B2)

Today’s honest C4 cell is **no-spec 10.6 / 42**. Raise that even if
DFlash stays cursed:

- Batched Q4_K (same kernels as A4, C4-weighted)
- Prefix cache for agent system/tool text (Qwen already showed 36%
  TTFT on a 7.6k warm prefix; Glimmer Vulkan has no equivalent)
- Slot ctx 32k × 8 is the memory envelope; do not chase 128k × 4

Pass: no-spec C4 ≥ 15 / 60. This is the floor under every spec cell.

### B6. Alternative speculation — only after B1–B3

Do **not** train EAGLE / Medusa / MTP / DSpark until we know whether
native DFlash can be batched. If B3 dies:

- llama.cpp `ngram-*` as a zero-train C4 spec (cheap negative result)
- EAGLE-3 or DSpark only if a Glimmer backbone exists or we accept
  a training slice
- Tree spec only if a DFlash-compatible verifier exists (it does not
  today)

---

## 6. Suggested sequence

Two people or two interleaved host windows. The card is single-model.

```text
Week 0   Phase 0 instrumentation + B1 (#27117 on Vulkan)
         A5 thinking-budget template (does not need exclusive GPU long)

Week 1   B2  n_max=1 C4/C8 cell          → first shippable C4+
         A1  safe n_max screen on C1
         A2  OpenVINO assistant=8 hunt

Week 2   B3  batched DFlash patch if B1 says core
         A3  SYCL+oneDNN bisect
         B5  no-spec C4 floor

Week 3–5 A4  Xe Q4_K + fusion, only if Phase 0 ≥40% GEMV
         B4  OVMS VLM_CB / llm-scaler only if A2/B3 stall

Park     EAGLE/MTP/DSpark, vLLM XPU C1, NVFP4, 5090, mmproj
```

Do not run A4 and a reboot-class SYCL experiment on the same evening.

---

## 7. Measurement contract

Every claimed cell writes a directory:

```text
~/b70-evals/muse-glimmer/<date>-<track>-<cell>/
  launch.txt          # exact argv, binary path, git sha, driver
  manifest.sha256     # weights + draft
  c1.json c4.json     # sweep receipts
  server.log          # includes draft acceptance lines
  notes.md            # one paragraph: what changed, kill or keep
```

Fixed prompts: the existing 128-token SSE plus cookbook p512 / p8192
/ p32768 shapes. C1 and C4 only. Label **per-stream vs aggregate**.
Record acceptance. Record thinking vs content tokens.

Public boundary only. No `llama-bench` decode as a serve number.

---

## 8. Hard kill list (do not repeat)

- SYCL 262k unified, or 128k + DFlash + `-ub 8192` (hard reboot)
- `GGML_SYCL_DEVICE_ARCH=bmg-g31` AOT
- Serving `build-sycl-dnnl` while the pool assert remains
- OVMS weekly (`Unsupported 'muse_glimmer'`)
- Unlocked GenAI wrapper as “C4”
- vLLM XPU as the C1 champion (slower, leaks reasoning)
- NVIDIA NVFP4 / NIM / Shadeform 5090 as B70 evidence
- Multi-slot DFlash with `n_max≥2` without a per-slot acceptance gate
- Spending the C1 budget on KV quant
- C2 / C3 “just to see”

---

## 9. Why this is ambitious, not another tune

A successful program produces **three artifacts the ecosystem does
not have**:

1. **A B70 C1 cell in the Qwen 55–70 tok/s band**, from target-decode
   kernels and/or a real DFlash-8 path — not from hoping KV is smaller.
2. **The first correct C4+ Glimmer spec cell on Arc**, either `n_max=1`
   shipped this week or a batched-DFlash patch upstreamed.
3. **A written proof** that the 2× gap was runtime/kernel stack +
   unused 16-token drafter + a multi-seq DFlash bug — so we stop
   chasing attention-byte folklore.

If Phase 0 shows the time is *not* in Q4_K GEMV, we kill A4 early and
pour the ambition into B3 + B4. That is still a unique result.
If Phase 0 confirms GEMV, we stop pretending flags will get us to 60
tok/s and we write the kernels.

---

## Sources

- This host’s campaign: `docs/glimmer-b70-handoff-20260824.md`
- Meta config: https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/config.json
- Meta GGUF: https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
- DFlash semantics: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Multi-seq DFlash corruption: https://github.com/ggml-org/llama.cpp/issues/27117
- B70 cookbook: https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/muse-glimmer/MUSE-GLIMMER-B70.md
- Intel day-0 (assistant token = 8): https://www.intel.com/content/www/us/en/developer/articles/community/day-0-local-agentic-ai-with-metas-muse-glimmer.html
- OpenVINO IR: https://huggingface.co/OpenVINO/Muse-Glimmer-30B-int4-ov
- OpenVINO GPU parity: https://github.com/openvinotoolkit/openvino/issues/37419
- vLLM Muse: https://github.com/vllm-project/vllm/pull/51655
- vLLM XPU paged-decode tuple: https://github.com/vllm-project/vllm-xpu-kernels/pull/524
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer
