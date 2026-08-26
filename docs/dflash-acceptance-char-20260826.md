# DFlash acceptance characterization — 2026-08-26

Status: **measured on the clean C1 cell**. This is not a keep/kill of any candidate.

## Setup

- Server: `muse-vllm-xpu-c1` on `inference-host:8000`, GPTQ target + GPTQ DFlash draft, XPU graph, `num_speculative_tokens=20`, `DFLASH_KV_MODE=none`.
- Boundary: public OpenAI streaming API + `/metrics` (`scripts/vllm-dflash-acceptance-char.py`).
- Window: 8 prompts × 3 measured reps × 256 completion tokens, seed 42, temperature 1.0, top_p 0.95. One warmup per prompt. All reps emitted exactly 256 tokens.
- Single-prompt gate baseline (sky, 8×256) reproduced first: **52.938** decode tok/s, **51.428** e2e, TTFT **142.4** ms, **1.560** accepted/draft run (1248/800).

Receipts: `~/b70-evals/muse-glimmer/20260826T-vllm-dflash-next-opt-baseline/`.

## Results

| Prompt | Kind | Decode tok/s | E2E tok/s | Acc / draft run | Draft runs (3×256) |
|---|---|---:|---:|---:|---:|
| sky-rayleigh | reasoning | 52.937 | 51.426 | **1.560** | 300 |
| python-heap | coding | 52.344 | 50.843 | **1.554** | 303 |
| debug-race | coding | 52.900 | 51.375 | 1.590 | 300 |
| es-history | multilingual | 58.614 | 56.011 | 1.844 | 270 |
| kv-cache-systems | long-form | 60.714 | 58.465 | 1.931 | 261 |
| agent-plan | agentic | 61.370 | 58.761 | 1.977 | 258 |
| tool-json | tool-like | 68.832 | 66.257 | 2.325 | 231 |
| **aime-style** | math | **158.090** | **141.648** | **6.765** | **102** |

Across-prompt median decode **59.664** tok/s; median acc/run **1.888**; min **1.554**; max **6.765**. Draft tokens/run stayed 20.0 on every prompt (n=20 always proposed).

## Conclusions

1. The resting 1.56 accepted/run figure is the **sky/coding cluster**, not a global draft ceiling. Math on the same cell is 4.3× that acceptance and ~3× throughput. 158 tok/s at 6.77 accept is consistent with the ~44 ms verify step: `6.77 / 0.044 ≈ 154`.
2. Do **not** choose adaptive depth from the sky prompt. n=20 is wasteful on the 1.55–1.59 cluster and is paying rent on math.
3. Real-target GPTQ recalibration must mix **hard** (sky/coding) and **easy** (math/tool) hidden-state trajectories. Calibrating only the gate prompt would miss the distribution that already works.
4. Capture must use the same aux layers the live DFlash cell uses: Eagle3 IDs `(2, 14, 26, 38, 50)` from config `target_layer_ids [1, 13, 25, 37, 49]` plus vLLM's documented +1 conversion. Do not add a second +1.

## Next

vLLM `extract_hidden_states` on this Muse GPTQ/XPU cell is **blocked**: after the aux-layer config is accepted, engine init asserts in `HiddenStateCacheSpec.page_size_bytes` (`page_size_padded >= unpadded_page_size_bytes`). Reproduced with fp8 and default KV. Do not keep bouncing C1 on that path.

Capture next: a **dedicated** DFlash dump overlay of `aux_hidden_states` (no `torch.xpu.synchronize()`, not used for acceptance numbers), or offline target forwards. Then GPTQ into a new draft directory. Restore the known-good DFlash cell afterwards.

Scripts retained: `start-muse-vllm-extract-hidden-c1.sh`, `vllm-extract-hidden-dump.py`, `run-muse-hidden-capture-then-restore.sh` (restore-on-EXIT works; extract serve does not on this model/kernel).
