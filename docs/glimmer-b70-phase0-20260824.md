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

Do not report `completion_tokens / wall` as decode. That is how the old
27–32 C1 figure and the C4 “49 agg” figure were blended.

1. **C1 decode is 38 tok/s** on a cached 83-token prompt (128 new tokens).
   Prefill on that row is only 5 new tokens and is not a prefill number.
   At **273 W / 2800 MHz** the card is **power-capped** (275 W limit), not
   VRAM-full. Short decode is using the whole power budget.
2. **A 128-token default generation never reaches `content`.** All 128
   tokens are `reasoning_content`. `reasoning_strength=low` is the first
   time content appears (~25% of chars, content TTFT 1.0–2.1 s).
3. **C4 DFlash `n_max=1` decode is 15.5 tok/s per stream** (~62 concurrent
   decode if all four stay in generate). Acceptance 0.70–0.81, spread 1.16,
   **244 W**. C4 `n_max=2` decode is 3.1 tok/s/stream at **134 W**.
4. **This is not llama.cpp #27117 on Vulkan.** Acceptance stays healthy.
   When concurrency dies, power falls to ~132 W. Drafts are still accepted.
   The tax is scheduling / verify / idle, not corrupted drafts.
5. **No-spec C4 was not re-measured this run.** The old 10.6 / 42 row is
   prior-campaign e2e. Do not treat 15.5 decode as a proven win over it
   until we split a no-spec cell the same way.

## C1 — live production cell (`n_max=2`, `-np 2`, 131k)

Prompt: “explain why the sky is blue”, `max_tokens=128`, seed 42.
Decode = llama.cpp `eval time` / client `predicted_*`. Prefill = `prompt eval`.

| Cell | Prompt | Prefill | Decode | Acc | Mean accept | Card W | Content? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| default ×3 | 83, 78 cached / 5 new | 5 new @ 44–45 (not a prefill row) | **38.2–38.3** | 0.64 | 2.27 | 273 | none |
| `reasoning_strength=low` #1 | 83 cold | **64** | 37.5 | 0.62 | 2.23 | 237 | yes, content TTFT 2.14 s |
| `reasoning_strength=low` #2 | 83, 78 cached | 5 new @ 44 | **42.9** | 0.79 | 2.57 | 273 | yes, content TTFT 0.96 s |
| pad ~2k (actual 1569) | 1569 | **432** | 30.6 | 0.68 | 2.35 | 244 | none |
| pad ~8k (actual 6024) | 6024 | **402** | 26.6 | 0.77 | 2.54 | 254 | none |

Acceptance **rises** with context; decode **falls**. At 6k that is a 30%
decode tax vs the cached 83-token row. Short C1 is still weight/dequant,
not KV. Long-prompt e2e (8 tok/s at 6k) is prefill wall, not decode.

`mean accept length` 2.2–2.6 with `n_max=2` means we use the 2-token cap,
not the trained 16-token block.

Restore smoke (32 tokens) printed 48 tok/s decode. Short-gen artifact,
not a new ceiling.

## C4+ — 8 slots × 32k, `-ub 512`, DFlash on

Same prompt, identical seed, 128 new tokens. Prefill is **not isolated**
here: concurrent TTFT includes queueing. Numbers below are **decode only**
(log `eval time`). Concurrent decode agg is `n × per-stream decode`, not
`total_tokens / wall`.

| Cell | Decode each | Decode agg (n×each) | Acc range | Spread | Mean len | Card W |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| **C4 `n_max=1`** | **15.5** | **~62** | 0.70–0.81 | 1.16 | 1.70–1.81 | **244** |
| C4 `n_max=2` | 3.11 | ~12 | 0.70–0.78 | 1.10 | 2.40–2.54 | 134 |
| C4 `n_max=4` | 3.45 | ~14 | 0.40–0.46 | 1.14 | 2.57–2.80 | 136 |
| C4 `n_max=2` on `-np 4` | 3.11 | ~12 | 0.70–0.78 | 1.10 | 2.40–2.54 | 135 |
| C8 `n_max=1` | 2.25 | ~18 | 0.72–0.81 | 1.14 | 1.72–1.81 | 132 |
| C8 `n_max=2` | 2.93 | ~23 | 0.59–0.73 | 1.22 | 2.19–2.44 | 133 |
| C8 `n_max=4` | 3.67 | ~29 | 0.41–0.52 | 1.25 | 2.62–3.02 | 140 |

Prior-campaign no-spec C4 (10.6 / 42) was e2e from the old sweep. Not in
this table.

Gates, decode-only:

- C4 acceptance spread < 2×: **pass** on every cell.
- C4 decode floor 12 tok/s each: **pass only for `n_max=1`** (15.5).
- Spec vs no-spec: **not scored**. Need a split no-spec C4.
- C8: **fail** on decode. Acceptance is fine; the GPU is half-idle.

B2 (`n_max=1` at 8 slots) is still the only C4 cell that keeps the GPU
busy (244 W vs 134 W). Do not carry C1’s `n_max=2` into C4.

C8 is a different problem from C4 `n_max=2`. `n_max=1` does not save C8.
Lead hypothesis: sequential draft process / graph recapture, visible as a
half-idle GPU, not as bad drafts.

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
