# Glimmer B70 — vLLM-XPU GPTQ + DFlash handoff — 2026-08-26

Status: **current research cell**. This is the authoritative handoff for the
Muse GPTQ requant and vLLM-XPU C1 DFlash work. The Aug 25 native SYCL C8
document (`glimmer-b70-native-result-handoff-20260825.md`) is a parked
llama.cpp restore recipe, not the live research path.

Transcript (intact, not compacted):
`~/.pi/agent/sessions/--home-mike-code-local-dev-model--/2026-08-25T20-33-39-091Z_01a03aa0-e913-7f4b-a8e8-ed8d8462270c.jsonl`

Gitignored Pi copy:
`~/code/local-dev-model/.pi/handoffs/handoff-2026-08-26T04-10-34-709Z.md`

## Why this was "missing"

The session lived in `local-dev-model` Pi transcripts. Its handoff was written
under gitignored `.pi/handoffs/`. Follow-up sessions started with no
`parentSession` and treated the Aug 25 SYCL pause doc as current, then scouted
Shadeform 5090 SGLang / reset recovery instead of vLLM-XPU C1.

## Goal

Two missing Muse capabilities vs Qwen3.8-27B on one Arc Pro B70:

1. XMX-quality W4A16/GPTQ target path instead of mixed GGUF K-quant MMVQ.
2. A deeper, cheaper speculative path approaching ~5 emitted tokens per target
   verify.

## Live research cell (vLLM-XPU C1)

- **Host:** `inference-host` / Tailscale `100.75.79.54` / LAN `192.168.8.172`
  (LAN often down after reset).
- **Container:** `muse-vllm-xpu-c1`, port **8000**, served id `muse-glimmer-gptq`.
- **Image:** `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vllm `0.27.2rc1.dev77+gac7509e2b`).
- **Kernel overlay:** `pip install vllm-xpu-kernels==0.1.13.2` at container
  start. Stock **0.1.12.3** is insufficient: missing Muse GQA 32/2 paged-decode
  tuples `16,128,64,false,true,false` and `16,128,16,false,false,false`.
- **Target:** `/home/mike/inference/models/Muse-Glimmer-30B-GPTQ-Int4-sym-G128`
  (~21 GiB). Produced on desktop RTX 5080 with gptqmodel **7.3.2** / torch
  `2.11.0+cu128`. Contract: 4-bit, G128, symmetric, `desc_act=false`,
  `pack_dtype=int32`, `lm_head=false`, format `gptq`. 416 text-decoder linears;
  vision / lm_head / embeddings stay BF16. Desktop leftover:
  `/tmp/muse-gptq-venv`, `/tmp/muse_quantize.py`.
- **Draft:** `/home/mike/inference/models/Muse-Glimmer-30B-assistant` (4.8 GiB).
- **Spec:** `{"method":"dflash","model":"/draft","num_speculative_tokens":15}`.
- **Flags:** `--quantization gptq --dtype float16 --max-model-len 8192
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --enforce-eager
  --max-num-seqs 1 --max-num-batched-tokens 2048 --no-enable-prefix-caching
  --language-model-only --reasoning-parser muse_glimmer`.
- **Kernel path:** Muse arch resolved; `XPUwNa16LinearKernel` /
  `torch.ops._xpu_C.int4_gemm_w4a16` selected. Model load 17.13 GiB.
- 32k / 0.85 KV OOM'd to 0 bytes (BF16 draft ate headroom). C1 is 8k for that
  reason.

Launchers were left in host `/tmp` (`start-muse-vllm-dflash-c1.sh`,
`start-muse-vllm-xpu-c1.sh`, `wait-vllm-health.sh`) and may not survive reboot.

## Numbers (do not mix labels)

| Cell | Metric | Value |
|---|---|---:|
| vLLM GPTQ no-spec C1 | public-boundary e2e, 5×128 | **31.46** tok/s (`chunks=0`, not decode) |
| vLLM GPTQ + official DFlash n=15 C1 | client decode (post-first `delta.reasoning`, median of 5) | **42.14** (41.56–42.81) |
| llama.cpp official DFlash n2 | client decode | **42.42** |
| llama.cpp DFlash2 SYCL n2 | client decode | **29.9** (killed) |
| llama.cpp DFlash2 SYCL n8 | client decode | **11.0** (killed) |
| Qwen no-spec, same image neighborhood | e2e | **32.9** |
| Qwen MTP4 | decode | **83.7** BF16-MTP4 / **112.65** draft-INT4 |

Receipts:

- vLLM no-spec: `~/b70-evals/muse-glimmer/20260826T-vllm-gptq-c1/`
- vLLM DFlash: `~/b70-evals/muse-glimmer/20260826T-vllm-gptq-dflash-c1/`
- DFlash2 SYCL: `~/b70-evals/muse-glimmer/20260826T-dflash2-sycl-c1/`
- Decode script on host: `/tmp/vllm-decode-reps.py` (reads `delta.reasoning`)

`glimmer-phase0-instrument.py` reports `median_client_decode_tok_s=null` and
`chunks=0` against vLLM because Muse streams `delta.reasoning`, not
`delta.reasoning_content`.

## Decisions

- **vLLM-XPU GPTQ-G128 is the Muse target path.** Do not write a llama.cpp
  Q4_K GEMM first.
- **llama.cpp DFlash2-SYCL depth bet is dead.** Do not spend another B70 window
  widening `n_max` on that port.
- **Official vLLM DFlash (`num_speculative_tokens=15`) is the speculation
  vehicle.** It matched llama.cpp official n2 (~42.1 vs 42.4) and did **not**
  reach Qwen MTP4. The remaining gap is speculation, not target W4A16.
- Park MMVQ cap-12 and Vulkan. Restore known-good SYCL C8 only if the user
  wants the llama.cpp cell back (recipe in the Aug 25 native handoff).

## Next

1. Confirm `muse-vllm-xpu-c1` still answers `http://127.0.0.1:8000/v1/models`
   on `mike@100.75.79.54`.
2. Instrument vLLM DFlash: tokens/verify, acceptance, draft vs target time.
   Fix or bypass `delta.reasoning`.
3. One slice: **draft-side INT4** of Muse assistant internals on this same
   cell, C1 decode vs **42.14**, kill if <5% or acceptance collapse. Keep
   shared embeddings / lm_head BF16 on the first INT4 draft slice.
4. Do not go back to SYCL DFlash2, Vulkan cap tuning, or Shadeform 5090.

---

# Addendum — 2026-08-26 night session (depth + graph sweep)

Status update: items 1–2 done; item 3 re-ranked but not yet run. The cell is
restored and faster than the recorded baseline. Receipts:
host `~/b70-evals/muse-glimmer/20260826T-vllm-dflash-depth-graph/`
(`sweep-summary.md`, `dflash-instrument-*.json`). Launchers + instrument now
versioned in repo `scripts/` (`start-muse-vllm-dflash-c1-graph.sh`,
`start-muse-vllm-dflash-c1.sh`, `vllm-dflash-instrument.py`; instrument reads
`delta.reasoning` correctly).

## What was found

- Host had rebooted before the session; `/tmp` launchers wiped. Cell restored
  from lead-machine script copies. Baseline eager n=15 reproduced at **43.72**
  tok/s client decode (prior receipt 42.14 ✓).
- Instrumentation (`/metrics` deltas): acceptance collapses after position ~3;
  positions 4–14 contribute ~1 extra token/step. In **eager** mode per-step
  wall was flat (~70–75 ms) across n∈{6,15,24} — fixed overhead dominated, so
  depth barely mattered.
- **XPU graphs are the lever**: `--enforce-eager` removed +
  `VLLM_XPU_ENABLE_XPU_GRAPH=1` cut step wall to ~48 ms and lifted decode
  ~+60–85%. Under graphs the depth optimum is **n=20** (deeper loses: serial,
  uncaptured draft forwards dominate marginal cost).

| Config | 5×128 window | 8×256 window |
|---|---:|---:|
| eager n=15 / n=24 | 43.72 / 44.95 | — |
| graph n=16 / n=32 | 64.27 / 53.11 | — |
| graph n=24 | 70.48 | 44.61 |
| **graph n=20 (final)** | **81.56** | **48.03** |

- Caveat: acceptance is strongly content-dependent (λ≈3.27 on a 128-token
  window vs λ≈1.58 on 256 tokens of the same prompt+seed). Use ≥256-token
  windows for decisions; short windows overstate.
- A mid-session host hard reset was investigated and is **pre-existing
  platform instability** (5 dirty power events 00:56–02:13 EDT, no shutdown
  wtmp rows / pstore / panic trace), not graph-mode-related: the identical
  graph config relaunched cleanly end-to-end. Watch PSU/cables if resets
  continue.

## New decisions

- Final C1 flags = prior flags minus `--enforce-eager`, plus
  `VLLM_XPU_ENABLE_XPU_GRAPH=1`, spec `num_speculative_tokens=20`. Cell left
  running in this state.
- Remaining gap to Qwen MTP4 draft-INT4 (112.65) is draft-forward cost →
  **draft-side INT4 slice is next** (handoff item 3 stands, kill rule
  unchanged).

---

# Addendum — 2026-08-26 full GPTQ DFlash assistant

Status: **passed and active**. The assistant GPTQ artifact and a candidate-only
vLLM DFlash overlay are now required for C1. BF16 assistant source remains
untouched.

Research gate (2026-08-26): vLLM GPTQ/XPU support and draft quantization
configuration were checked in
`https://docs.vllm.ai/en/stable/features/quantization/`,
`https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/`, and
`https://docs.vllm.ai/en/stable/features/speculative_decoding/draft_model/`;
GPTQModel 4-bit/G128 configuration and module overrides were checked against
`https://github.com/ModelCloud/GPTQModel`. These supported the GPTQ artifact
contract and explicit draft `quantization:gptq`; vLLM source was authoritative
for the DFlash packed-QKV compatibility patch.

## Artifact

- `Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128` on desktop + host: 1.6 GiB
  vs 4.8 GiB BF16 (-67.6%), SHA256 model file
  `25ab1e1d04508159b44b833c31b1f74625bc1dc0486d6a9fcaeead577091fcee`.
- GPTQ: 4-bit, G128, symmetric, `desc_act=false`, int32; 35 decoder linears
  (five × q/k/v/o + gate/up/down); `encoder.fc`, encoder norm, final norm
  BF16. Embeddings + lm_head remain target-shared BF16.
- Quantization uses deterministic shape-correct DFlash calibration (256
  batches of `noise_embeds` + target-context hidden states).
- `quantize-muse-dflash-assistant-gptq.py` writes/validates vLLM fused-module
  metadata: `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`. This is needed
  because DFlash offsets draft layer prefixes by target depth.

## vLLM full-QKV patch

- Full GPTQ first failed because DFlash fast context-KV precompute directly
  reads `qkv_proj.weight`; packed GPTQ exposes `qweight` only.
- `patch-vllm-dflash-gptq-context-kv.py` is a guarded candidate-only overlay.
  In packed-QKV mode it invokes each existing quantized QKV projection on
  normalized context, retains K/V, then reuses original K-norm/RoPE/cache
  storage. BF16 retains its original single-F.linear fused fast path.
- Candidate launcher: `start-muse-vllm-dflash-c1-graph-draft-gptq.sh`; it
  explicitly sets draft `quantization:gptq`, applies the overlay at startup,
  and does not alter the BF16 launcher or image.

## E2E acceptance result

Same real streaming OpenAI API journey, graph mode with `num_speculative_tokens=20`
(depth) and 8 replicates × 256 completion tokens (workload), fixed prompt/seed;
`vllm-dflash-instrument.py` reads `delta.reasoning` and `/metrics`.

| Metric | BF16 draft | GPTQ draft | Delta |
|---|---:|---:|---:|
| client decode | 48.03 tok/s | **52.90 tok/s** | **+10.1%** |
| e2e | 46.65 tok/s | **51.39 tok/s** | **+10.2%** |
| TTFT | 158.7 ms | **142.6 ms** | -10.1% |
| accepted / verify | 1.58 | **1.56** | -1.3% |

The candidate clears the >=5% gain gate with no acceptance collapse and is
the **active resting C1 cell**. Receipt:
`~/b70-evals/muse-glimmer/20260826T-vllm-dflash-draft-gptq-c1/`.

## Next

Draft INT4 is no longer the remaining first-order lever. Next only after a
multi-prompt long-window acceptance characterization; do not widen DFlash
depth based on 128-token windows.
