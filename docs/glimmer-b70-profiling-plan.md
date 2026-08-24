# Glimmer B70 — profiling plan (next session)

Date: 2026-08-24
Host: `inference-host` (CachyOS, kernel `7.2.0-1-cachyos`, `xe` `8086:e223`)
Prior: `docs/glimmer-b70-dflash2-process-20260824.md`

Goal: name **where the ~950 ms C4 `n_max=2` tick goes**. Not another inject rewrite.

Public boundary stays `POST /v1/chat/completions` `stream=true`.
Instrument: `~/inference/launchers/glimmer-phase0-instrument.py`.
Decode = log `eval time`. Card W = existing Xe hwmon sampler. Never `completion_tokens / wall`.

## Why we are profiling (not patching)

| Fact | Number |
|---|---|
| Packed `process()` on C4 n2 | `unique_seq=4`, `packed=12`, `chunks=1` (20/34 calls) |
| C4 n2 after that pack | **3.05 tok/s / 130 W** (was 3.1 / 134) |
| C4 n1 on the same binary | **15.7 tok/s / 243 W** |
| C1 DFlash2 n8 (one stream) | **5.6 tok/s / 129 W** |

130 W at a pinned 2800 MHz is idle/wait, not a busy Q4 GEMV. The single-stream n8 collapse means this is **not** “Vulkan has the same multi-seq inject bug.” Serial `process()` is gone and the tax remains.

## Host reality (checked 2026-08-24)

Missing: `iaprof`, `vtune`, `intel_gpu_top`, `perf`.

Present and useful:

- Kernel 7.2 + `/sys/kernel/btf/vmlinux` **and** `/sys/kernel/btf/xe` (iaprof kernel prereqs are already satisfied)
- Xe sysfs: `hwmon2` energy, `tile0/gt0/freq0/{cur,act,max}`, `throttle`
- llama-server flags: `--verbose` / `--log-verbosity`, `--log-timestamps`, `--perf`
- Pacman (not apt): `extra/intel-gpu-tools 2.5-1` (`intel_gpu_top`), `extra/perf 7.1.8-1`

`iaprof` ([intel/iaprof](https://github.com/intel/iaprof)) **does** list Arc B-series / Battlemage. It is **not** the first tool:

- It profiles via **Level Zero USDT** in a *patched* `libze_intel_gpu`. Stock NEO has no probes.
- Our live cell is **Vulkan llama.cpp**, not L0/SYCL.
- Reliable CPU stacks also want frame pointers in the graphics stack.

Do **not** spend the session building patched NEO unless Phase A cannot name ≥50% of the tick.

## Phase A — cheap split (hours, do this first)

Install only the thin tools:

```bash
sudo pacman -S --needed intel-gpu-tools perf
```

Keep the live C1 n2 cell until the tools are in. Then swap **one** server at a time. Always restore official DFlash `n_max=2`, `-np 2` at the end.

### A0. Server clocks (highest leverage)

On the existing `64f765f` side tree, add **temporary** `ggml_time_us()` (or `CLOCK_MONOTONIC`) around the three already-known sites. Log once per tick, not per token:

1. `common_speculative_draft()` in `tools/server/server-context.cpp` (~2952)
2. `llama_decode(ctx_tgt, batch_view)` (~3588) — this is the 30B verify of ~12 tokens at C4 n2
3. `common_speculative_process()` (~3653) — further split inside `process()`: layer readback vs `llama_encode` vs inject `llama_decode`

Also log `batch_view.n_tokens`, unique seqs (we already know generate is 12/4), and `ctx_tgt_seq_rm_type` once at init (`SRV_INF`, not TRC). We still do not know if Vulkan `seq_rm` is `PART` / `RS` / `FULL`.

Rebuild **only** `llama-server`. Keep `llama-server.unpatched-64f765f`. Do not change algorithms.

### A1. Matched cells

Same sky prompt, seed 42, official DFlash Q4_K_M, `-ub 512`. Receipts under `~/b70-evals/muse-glimmer/<stamp>-profile/`.

| Cell | Why |
|---|---|
| C1 n2, `-np 2` | Busy baseline (~38 tok/s / 273 W) |
| C4 n1, 8×32k, load4 | Busy concurrent (~15.7 / 243 W) |
| C4 n2, 8×32k, load4 | Idle concurrent (~3 / 130 W) — the bug |
| Optional: C1 DFlash2 n8 | Idle single-stream twin (only if A1 C4 n2 is ambiguous) |

For each: `--log-timestamps --perf`, 64–128 new tokens, one warmup. Parallel `intel_gpu_top -s 100` (or `-J`) into the receipt dir. Keep the hwmon sampler in the instrumenter.

### A2. Rank the tick

A C4 n2 generate tick is ~950 ms wall at ~3.7 tok/s and ~2.3 mean accept. Budget it:

| Bucket | If this is most of the 950 ms | Next move |
|---|---|---|
| 30B `llama_decode` verify | GPU should be ~240 W; if it is 130 W, the wait is *inside* the submit | Vulkan timestamps / iaprof later |
| `process()` layer readback | `llama_get_embeddings_layer_inp` sync | On-device activations |
| `process()` encode+inject | Draft graphs, already packed | Draft-graph capture / fusion |
| `draft()` noise block | 4×3 non-causal draft decode | Draft kernel / mask |
| Checkpoint save/restore | Only if `seq_rm==FULL` and logs show restore | Avoid FULL snapshots |
| Unaccounted host | sampler / queue / `llama_synchronize` | More timers, not inject |

Pass: one bucket ≥50% of C4 n2 tick wall, and C4 n1 / C1 n2 stay busy as controls.
Kill: after A0+A1 we still cannot name half the tick. Write that. Do **not** invent a new patch.

## Phase B — only if A names the GPU and we still cannot see the op

1. Confirm `intel_gpu_top` busy% tracks card W (C4 n1 high, C4 n2 low). If busy% is high at 130 W, the card is on a cheap path; if busy% is low, the CPU is holding the GPU.
2. `perf record -g -p $(pgrep llama-server)` on one C4 n2 wave. Host is CachyOS; `perf` is the `linux-tools` package. This catches CPU-side sync, not EU stalls.
3. **Do not** start `iaprof` on this Vulkan cell unless A says the wait is inside a GPU submit **and** we have a reason L0 probes would see it. Building patched NEO + frame-pointer graphics stack is a different session.

`iaprof` install sketch (later, not this session): kernel/BTF already OK; needs `libelf-dev` equivalents, rustup, `git clone --recursive https://github.com/intel/iaprof`, `make deps && ./build.sh`, root, patched `libze_intel_gpu`. See https://github.com/intel/iaprof.

## Restore

```bash
# production 1–2 stream cell
DFLASH=1 PARALLEL_SLOTS=2 SLOT_CTX=131072 SPEC_N_MAX=2 \
  # but start the 64f765f binary, not 70adb1b
```

Equivalent argv is already in `~/inference/launchers/` via the last restore. Port 18099. Official DFlash n2, `-np 2`.

## Out of scope this session

- Rewriting `process()` again
- DFlash2 C4
- SYCL / oneDNN / OpenVINO
- Custom Q4_K kernels
- Upstreaming the pack as a speedup
- Shadeform / 5090

## Sources

- Last session numbers: `docs/glimmer-b70-dflash2-process-20260824.md`
- iaprof (Xe / Battlemage, L0 USDT): https://github.com/intel/iaprof
- llama-server `--verbose` / `--perf` / `--log-timestamps`: `llama-server -h`
- Pacman: `intel-gpu-tools`, `perf`
