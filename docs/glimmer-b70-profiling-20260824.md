# Glimmer B70 — C4 n_max=2 tick profile — 2026-08-24

Host: `inference-host`, Intel Arc Pro B70, public `POST /v1/chat/completions` `stream=true`.
Instrument: `~/inference/launchers/glimmer-phase0-instrument.py`.
Decode = log `eval time`. Card W = Xe hwmon. Never `completion_tokens / wall`.
Receipts: `~/b70-evals/muse-glimmer/20260824T145826-profile/`.

## What this session added

- Pacman: `intel-gpu-tools 2.5-1`, `perf 7.1.8-1`.
- `intel_gpu_top` **refuses Xe** (`Detected Xe device which is not supported`). The replacement is `gputop` from the same package (`-d 0.1 -n N`). Engine that moves is **ccs**, not rcs.
- Temporary `ggml_time_us()` clocks on the `64f765f` side tree (`llama.cpp-dflash2`). No algorithm change.
  - `common_speculative_draft()`
  - `llama_decode(ctx_tgt) + llama_synchronize` (30B verify)
  - `common_speculative_process()`, split inside packed DFlash `process()`: layer readback / `llama_encode` / inject `llama_decode`
  - `ctx_tgt_seq_rm_type` / `ctx_dft_seq_rm_type` once at init (`SRV_INF`)
- Binary: `~/inference/src/llama.cpp-dflash2/build/bin/llama-server` (timers). Packed-only copy kept as `llama-server.packed-64f765f`. Unpatched kept.

`iaprof` was not installed. It is Battlemage-capable but Level Zero USDT in a *patched* `libze_intel_gpu`. This cell is Vulkan.

## Same-workload matrix

Sky prompt, seed 42, official DFlash Q4_K_M, `-ub 512`, `--log-timestamps --perf`, one warmup then 128 new tokens.

| Cell | Load | Decode each (log eval) | Acc | Card W | CCS p50 (resident) | Accounted tick p50 |
|---|---|---:|---:|---:|---:|---:|
| C1 n2, `-np 2` | 1 | **38.7 tok/s** | 0.66 | **273** | 96.8% | 59 ms |
| C4 n1, 8×32k | 4 | **15.7 tok/s** | 0.74–0.79 | **239** | 95.9% | 110 ms |
| C4 n2, 8×32k | 4 | **3.05 tok/s** | 0.70–0.73 | **130** | **100%** | **793 ms** |

C1 / C4 n1 stay on the busy control signature. C4 n2 still dies. Acceptance stays healthy. `seq_rm` is **PART / PART** on both target and draft — not FULL checkpoints.

## Where the C4 n2 tick goes

Generate tick, p50, four live sequences (`unique_seq=4`, `n_tokens=12` = 4 × (1 sampled + 2 draft)):

| Bucket | p50 | Share of 793 ms | What it is |
|---|---:|---:|---|
| 30B `llama_decode` verify | **694 ms** | **83%** | `llama_decode(ctx_tgt, batch_view)` + sync |
| `draft()` noise block | 90 ms | 11% | 4 sequences drafting |
| packed `process()` | 9.4 ms | 6% | readback 8.9 / encode 0.28 / inject 0.21 |

Pass: one bucket ≥50% of C4 n2 tick wall, controls stay busy.

Typical C4 n2 generate lines:

```
b70tick draft ms=90.09 n_drafting=4
b70tick verify ms=698.17 n_tokens=12 unique_seq=4 has_output=1 ret=0
b70tick dflash process unique_seq=4 packed=12 chunks=1 readback_ms=8.94 encode_ms=0.27 inject_ms=0.21
```

Healthy C1 n2 generate tick for scale:

```
b70tick draft ms=6.59 n_drafting=1
b70tick verify ms=52.28 n_tokens=3 unique_seq=1 has_output=1 ret=0
b70tick dflash process unique_seq=1 packed=3 chunks=1 readback_ms=0.40 encode_ms=0.09 inject_ms=0.16
```

ms / verified token:

| Cell | Verify p50 | Tokens | ms/tok |
|---|---:|---:|---:|
| C1 n2 | 51.7 ms | 3 | 17.2 |
| C4 n1 | 95.9 ms | 8 | 12.0 |
| C4 n2 | 694 ms | 12 | **57.8** |

C4 n1 batches *better* than C1 per token. Adding one more draft token per stream (8 → 12 tokens) makes verify **7.2×** slower, not 1.5×.

## GPU vs host

`gputop` resident llama-server row (the 22–23G client; ignore the 0B twin):

- C1 n2: CCS p50 97%, 273 W — real Q4 work.
- C4 n1: CCS p50 96%, 239 W — still real work.
- C4 n2: CCS p50 **100%** (98% of samples ≥90%), **130 W**.

This is not a host stall and not idle CCS. The compute engine is occupied by a **cheap path** for ~700 ms. Packed `process()` is 9 ms; serial inject is dead as a C4 n2 explanation. Checkpoint restore is dead (`seq_rm=PART`).

`draft()` at 90 ms (vs 6.6 ms C1 / 11.6 ms C4 n1) is a real secondary tax. It is not the ≥50% bucket.

## What this is not

- Not llama.cpp [#27117](https://github.com/ggml-org/llama.cpp/issues/27117) on this Vulkan cell. Acceptance holds.
- Not a reason to rewrite `process()` again.
- Not a reason to build patched NEO / `iaprof` next. The wait is inside a **Vulkan** `llama_decode` of a 4×3 speculative batch. L0 USDT would not see that graph.

## Restore

Production cell is back: official DFlash `n_max=2`, `-np 2`, 131k, port 18099, tmux `muse`. Timer binary left in place (log-quiet; no `--log-timestamps` / `--perf` on the live argv).

## Next

Name the **op inside the 12-token 30B verify**, not another host rewrite.

1. Vulkan timestamps / ggml Vulkan perf on one C4 n2 generate `llama_decode` (4 seq × 3 tokens) vs C4 n1 (4 × 2) and C1 n2 (1 × 3).
2. Only then decide: mask/graph recapture, unfused tiny dispatches, or a Q4 path that stops streaming weights.
3. Optional cheap check: `perf record -g` on one C4 n2 wave if the Vulkan split still leaves >50% unaccounted *inside* the submit. Still not `iaprof`.

## Sources

- Receipts: `~/b70-evals/muse-glimmer/20260824T145826-profile/`
- Tick rollup: `tick-summary.json`, `gputop-summary.json` in that directory
- Plan: `docs/glimmer-b70-profiling-plan.md`
- Prior kill: `docs/glimmer-b70-dflash2-process-20260824.md`
- `gputop` / Xe: intel-gpu-tools 2.5; `intel_gpu_top` is i915-only
- iaprof (not used): https://github.com/intel/iaprof
