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

## Prompt set (updated)

Ad-hoc prompts (sky, fake contest math, Spanish history, heap textbook) are retired from characterization. Shared list: `scripts/dflash_char_prompts.py`.

* **eval** — Spec-Bench / DFlash-paper families: GSM8K test[0], HumanEval/0, MT-Bench q81 writing.
* **work** — this C1: failing-test fix, patch review, Harbor-style debug, tool call, vLLM-XPU serve triage.

The sky prompt remains the frozen **gate** in `vllm-dflash-instrument.py` only. Report `by_suite` medians; do not average eval math with work coding.

### Work+eval 3×256 (same cell, after prompt fix)

| Prompt | Suite | Decode tok/s | Acc / draft run |
|---|---|---:|---:|
| mtbench-81 | eval | 55.167 | **1.656** |
| review-race-patch | work | 67.253 | 2.283 |
| gsm8k-janet | eval | 86.529 | 3.151 |
| tool-pytest | work | 99.822 | 3.925 |
| fail-test-fix | work | 100.876 | 3.935 |
| harbor-debug | work | 110.061 | 4.354 |
| humaneval-0 | eval | 117.498 | 4.558 |
| serve-oom-triage | work | 152.325 | 6.168 |

Across-prompt median **100.3** tok/s / **3.93** acc/run. Suite medians: eval **86.5 / 3.15**, work **100.9 / 3.94**. Receipt: `acceptance-char-work-eval-3x256.json`.

Chat/writing (MT-Bench, and the frozen sky gate at 1.56) is the hard cluster. Real GSM8K is ~3.2, not 6.8. HumanEval and Harbor-style coding sit ~4.3–4.6. The fake AIME 6.8 was an easy-algebra artifact.

## Next

vLLM `extract_hidden_states` on this Muse GPTQ/XPU cell is **blocked**: after the aux-layer config is accepted, engine init asserts in `HiddenStateCacheSpec.page_size_bytes` (`page_size_padded >= unpadded_page_size_bytes`). Reproduced with fp8 and default KV. Do not keep bouncing C1 on that path.

Capture next: a **dedicated** DFlash dump overlay of `aux_hidden_states` (no `torch.xpu.synchronize()`, not used for acceptance numbers), or offline target forwards. Then GPTQ into a new draft directory. Restore the known-good DFlash cell afterwards.

Scripts retained: `start-muse-vllm-extract-hidden-c1.sh`, `vllm-extract-hidden-dump.py`, `run-muse-hidden-capture-then-restore.sh` (restore-on-EXIT works; extract serve does not on this model/kernel).
