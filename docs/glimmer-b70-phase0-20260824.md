# Glimmer Phase 0 — 2026-08-24

Host: inference-host, Intel Arc Pro B70, llama.cpp Vulkan `70adb1b` (reported as
b10588-class `0.2.0-dev`). Weights Dynamic Q4_K_XL + DFlash Q4_K_M.
Public boundary: `POST /v1/chat/completions` `stream=true`.
Receipts on the host: `~/b70-evals/muse-glimmer/202608-program/20260824T114731/`.

Production cell was restored at the end: DFlash `n_max=2`, `-np 2`, 131k,
`-ub 512`.

`/metrics` is **501** on this binary. Acceptance comes from
`slot print_timing` in the server log. The first parser dropped those lines
(it looked for `draft acceptance =` in the *remainder* after already
consuming that prefix). Numbers below are re-parsed from the saved log
slices. The instrumenter is fixed.

No `intel_gpu_top` / `iaprof` on this box. GPU attribution is Xe hwmon
card energy + GT0 `cur_freq`. Card cap is 275 W.

## Headline

1. **C1 engine decode is ~38 tok/s**, not the old 27–32 e2e figure, at
   **273 W / 2800 MHz** (power-capped). The card is full. Short C1 is not
   waiting on an idle GPU.
2. **A 128-token default generation never reaches `content`.** All 128
   tokens are `reasoning_content`. Perceived speed is zero until thinking
   finishes. `reasoning_strength=low` is the first time content appears
   (~25% of chars, content TTFT 1.0–2.1 s).
3. **C4 DFlash `n_max=1` is the first useful 4-wide cell:** 15.5 tok/s
   each / **49 agg**, acceptance 0.70–0.81, spread 1.16, **244 W**.
   Beats the old no-spec C4 of 10.6 / 42.
4. **This is not llama.cpp #27117 on Vulkan.** At C4 and C8, per-slot
   acceptance stays healthy (spread ≤ 1.25). When concurrency dies, **power
   falls to ~132 W** and decode falls to 2–3 tok/s. Drafts are still being
   accepted. The tax is scheduling / verify / idle, not corrupted drafts.

## C1 — live production cell (`n_max=2`, `-np 2`, 131k)

Prompt: “explain why the sky is blue”, `max_tokens=128`, seed 42.

| Cell | Prompt toks | Decode | E2E | Acc | Mean accept | Card W | Content? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default ×3 | 83 (78 cached) | **38.2–38.3** | 36.9–37.0 | 0.64 | 2.27 | 273 | none |
| `reasoning_strength=low` #1 | 83 (cold) | 37.5 | 27.1 | 0.62 | 2.23 | 237 | yes, TTFT 2.14 s |
| `reasoning_strength=low` #2 | 83 (cached) | **42.9** | 41.1 | 0.79 | 2.57 | 273 | yes, TTFT 0.96 s |
| pad ~2k (actual 1569) | 1569 | 30.6 | 16.7 | 0.68 | 2.35 | 244 | none |
| pad ~8k (actual 6024) | 6024 | 26.6 | 8.0 | **0.77** | 2.54 | 254 | none |

Prefill on the padded rows is ~400–430 tok/s. Do not report e2e as decode.

Acceptance **rises** with context; decode **falls**. That is the expected
attention/KV term growing. At 6k it is already a 30% decode tax vs the
cached 83-token row. Short C1 is still weight/dequant + launch, not KV.

`mean accept length` 2.2–2.6 with `n_max=2` means we are using the
2-token cap, not the trained 16-token block. Raising `n_max` on this
backend is still a C1 experiment; it is not how we got to 38 tok/s.

Restore smoke (32 tokens) printed 48 tok/s. Treat as a short-gen artifact,
not a new ceiling.

## C4+ — 8 slots × 32k, `-ub 512`, DFlash on

Same prompt, identical seed, 128 tokens.

| Cell | Each | Agg | Acc range | Spread | Mean len | Card W |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| **C4 `n_max=1`** | **15.5** | **48.9** | 0.70–0.81 | 1.16 | 1.70–1.81 | **244** |
| C4 `n_max=2` | 3.11 | 11.8 | 0.70–0.78 | 1.10 | 2.40–2.54 | 134 |
| C4 `n_max=4` | 3.45 | 12.9 | 0.40–0.46 | 1.14 | 2.57–2.80 | 136 |
| C4 `n_max=2` on `-np 4` | 3.11 | 11.8 | 0.70–0.78 | 1.10 | 2.40–2.54 | 135 |
| C8 `n_max=1` | 2.25 | 16.5 | 0.72–0.81 | 1.14 | 1.72–1.81 | 132 |
| C8 `n_max=2` | 2.93 | 21.0 | 0.59–0.73 | 1.22 | 2.19–2.44 | 133 |
| C8 `n_max=4` | 3.67 | 25.3 | 0.41–0.52 | 1.25 | 2.62–3.02 | 140 |
| Old no-spec C4 (prior campaign) | 10.6 | 42 | n/a | n/a | n/a | n/a |

Gates from the research program:

- C4 correctness (spread < 2×): **pass** on every cell.
- C4 floor 12 / 48: **pass only for `n_max=1`**.
- Spec faster than no-spec at C4: **pass for `n_max=1`** (15.5 vs 10.6).
- C8 stretch: **fail**. Acceptance is fine; throughput is not.

So B2 (`n_max=1` at 8 slots) is a real C4 cell and should be the
production 4-wide recipe. Do not carry C1’s `n_max=2` into C4.

C8 is a different problem from C4 `n_max=2`. `n_max=1` does not save C8.
Whatever serializes the decode loop (likely
`common_speculative_process()` per sequence plus graph recapture) shows
up as a half-idle GPU, not as bad drafts.

## Thinking tax

Glimmer cannot disable thinking. Default template
(`reasoning_strength=high`) + `max_tokens=128` = 100% reasoning channel.
`chat_template_kwargs.reasoning_strength=low` is a real, working knob
via llama.cpp `--jinja`. Ship that on the agent cell.

## What Phase 0 did *not* get

- Kernel split (Q4_K matvec vs attention vs elementwise). Need `iaprof`
  or a Vulkan timestamp path. Circumstantial: C1 sits on the 275 W cap
  at 2800 MHz, which is consistent with a weight-stream bound, not an
  idle-dispatch bound.
- Draft-ms vs verify-ms as separate clocks. The log has no
  `spec statistics` lines on this build. We only have acceptance and
  wall decode.
- p32k. p8k already shows the attention slope.

## What to do next

1. **Ship C4 as DFlash `n_max=1`, 8×32k.** That is the first cell that
   beats no-spec four-wide on this card.
2. **Ship `reasoning_strength=low` on the Pi/agent template.** Otherwise
   C1 “38 tok/s” is invisible.
3. **Track B3 changes shape.** Do not hunt #27117-style KV corruption
   first. Profile why `n_max≥2` at C4, and any C8, drops the card to
   132 W with healthy acceptance. Sequential draft process / graph
   recapture is the lead hypothesis.
4. **Track A4 (custom Q4_K kernels) is still the C1 ambition**, and
   Phase 0 did not falsify it: C1 is power-capped. Install `iaprof`
   before writing kernels so we do not fuse the wrong op.
5. OpenVINO assistant-tokens=8 (A2) remains the highest-leverage
   low-code C1 bet. Not run this session.

## Sources

- Receipts: `~/b70-evals/muse-glimmer/202608-program/20260824T114731/`
- Reparse: `acceptance-reparse.json` in that directory
- Program: `docs/glimmer-b70-research-program.md`
- AMD contrast: https://github.com/ggml-org/llama.cpp/issues/27117
