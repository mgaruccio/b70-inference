# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one.
Last results: `docs/glimmer-b70-profiling-20260824.md`.
Prior kill: `docs/glimmer-b70-dflash2-process-20260824.md`.

**This session:** name the **Vulkan op** inside the 12-token C4 `n_max=2` 30B verify. Do not rewrite inject.

## Live cell (leave it up)

Host: `inference-host` (CachyOS, `xe` `8086:e223`). tmux `muse`, port 18099.
SSH: `inference-host` / `192.168.8.172` / tailnet `100.75.79.54`.

Vulkan llama.cpp **#27342** `64f765f5adefa4620dddda436ce56f1430435536` (PR still **open**).
Local packed `process()` + temporary `b70tick` host timers. Official DFlash, not DFlash2.

```
~/inference/src/llama.cpp-dflash2/build/bin/llama-server
  -m ~/inference/models/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
  -a muse-glimmer-30b
  -ngl 99 -c 262144 -np 2 --kv-unified -fa on
  -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja
  --temp 1.0 --top-p 0.95 --top-k 64
  --host 0.0.0.0 --port 18099
  -md ~/inference/models/Muse-Glimmer-30B-GGUF/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
  --spec-type draft-dflash --spec-draft-n-max 2 -ngld 99
```

Binaries next to it:

- `llama-server` — current (packed + `b70tick`)
- `llama-server.packed-64f765f` — packed, no timers
- `llama-server.unpatched-64f765f` — before pack

C4 production, when needed: official DFlash `n_max=1`, 8×32k, `-ub 512`. **Not** `n_max=2`.

## What is already true

Do not re-litigate these. Receipts: `~/b70-evals/muse-glimmer/20260824T145826-profile/`.

| Cell | Decode (log eval) | Acc | Card W | CCS p50 | Tick p50 |
|---|---:|---:|---:|---:|---:|
| C1 n2, `-np 2` | 38.7 tok/s | 0.66 | 273 | 97% | 59 ms |
| C4 n1, 8×32k load4 | 15.7 tok/s | 0.74–0.79 | 239 | 96% | 110 ms |
| C4 n2, 8×32k load4 | 3.05 tok/s | 0.70–0.73 | 130 | 100% | 793 ms |

C4 n2 generate tick, p50 (`unique_seq=4`, `n_tokens=12`):

| Bucket | p50 | Share |
|---|---:|---:|
| 30B `llama_decode` + sync | **694 ms** | **83%** |
| `draft()` | 90 ms | 11% |
| packed `process()` | 9.4 ms | 6% |

ms / verified token: C1 n2 **17.2**, C4 n1 **12.0**, C4 n2 **57.8**.
C4 n1 batches better than C1. One extra draft token per stream (8 → 12) makes verify **7.2×** slower.

Killed:

- Serial `process()` / inject rewrite — packed already, 9 ms
- `#27117` draft-state corruption — acceptance holds
- FULL checkpoint restore — `seq_rm` is **PART / PART**
- Host stall / idle CCS — `gputop` CCS pegged at 130 W (cheap GPU path)
- `iaprof` as the next tool — L0/USDT in patched NEO; this cell is Vulkan
- `intel_gpu_top` — Xe unsupported; use `gputop` (ccs, not rcs)

`draft()` at 90 ms is a real secondary tax. It is not this session.

## Host tools (already installed)

```bash
gputop -d 0.1 -n 50          # Xe top; parse the 22–23G llama-server row
perf --version               # 7.1.8-1; optional Phase B only
```

Launchers:

- `~/inference/launchers/glimmer-phase0-instrument.py` — public SSE client
- `~/inference/launchers/glimmer-profile-run.sh` — C1 n2 / C4 n1 / C4 n2 / restore
- `~/inference/launchers/b70-summarize-ticks.py` — parse `b70tick` logs

Public boundary stays `POST /v1/chat/completions` `stream=true`.
Decode = log `eval time`. Card W = Xe hwmon. Never `completion_tokens / wall`.

## Do this session

### 0. Keep the live cell until the first restart

No rebuild required. `vk_perf_logger` is a **runtime** env on this tree
(`ggml-vulkan.cpp` ~7416):

```bash
GGML_VK_PERF_LOGGER=1
GGML_VK_PERF_LOGGER_CONCURRENT=1   # group at existing sync points; try this first
# GGML_VK_PERF_LOGGER_FREQUENCY=1  # default; raise only if the log drowns
```

`GGML_VK_SYNC_LOGGER=1` only if the concurrent logger is unreadable.

### 1. Three matched verifies

Same sky prompt, seed 42, official DFlash Q4_K_M, `-ub 512`, one warmup, **64–128** new tokens.
Reuse `glimmer-profile-run.sh` (or the same `start_cell` / `run_wave` pattern). Add the env on the **tmux** server line. Keep `--log-timestamps --perf` so `b70tick` still ties wall to the graph dump.

| Cell | Why |
|---|---|
| C1 n2, `-np 2` | Healthy 3-token verify (~52 ms, 273 W) |
| C4 n1, 8×32k load4 | Healthy 8-token verify (~96 ms, 239 W) |
| C4 n2, 8×32k load4 | Sick 12-token verify (~694 ms, 130 W) |

Receipts: `~/b70-evals/muse-glimmer/<stamp>-vkperf/`.
Also capture `gputop -d 0.1` (ccs on the 23G row) and the existing hwmon sampler.

### 2. Rank ops

From the logger, for **one generate tick** of each cell, list the top kernels/nodes by GPU ms.

Pass: one named op (or a tight family: Q4 matvec vs flash-attn vs elementwise vs barrier) accounts for **≥50% of the 694 ms C4 n2 verify**, and the same op is *not* the C4 n1 / C1 n2 bottleneck.

Likely stories to confirm or kill:

| If the 694 ms is mostly… | Next move (later session) |
|---|---|
| Flash-attn / non-causal mask on 4×3 | Mask / FA path for speculative batch |
| Many tiny unfused nodes | Graph capture / fusion |
| Q4_K matvec that is long *and* 130 W | Cheap/wrong Q4 dispatch (not “needs more XMX”) |
| Barriers / empty submits | Submission batching, not a new kernel |

Kill: after one C4 n2 wave the logger cannot name half the verify. Write that. Do **not** invent a patch. Optional then: `perf record -g -p $(pgrep -n llama-server)` on one C4 n2 wave (CPU stacks only). Still not `iaprof`.

### 3. Restore

```bash
# production 1–2 stream cell, no GGML_VK_PERF_LOGGER, no --log-timestamps/--perf
# official DFlash n_max=2, -np 2, 131k, port 18099, tmux muse
# binary: ~/inference/src/llama.cpp-dflash2/build/bin/llama-server
```

Leave the `b70tick` binary in place unless it is in the way. Do not revert the pack.

## Do not

- Rewrite `process()` / inject
- DFlash2 on C4
- 128k + DFlash + `-ub 8192`
- `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- Score `completion_tokens / wall`
- Upstream the pack as a C4 speedup
- Build patched NEO / `iaprof`
- Custom Q4_K kernels before a named op
- C2 / C3
- Shadeform / 5090

## Sources

- This session’s numbers: `docs/glimmer-b70-profiling-20260824.md`
- Host receipts: `~/b70-evals/muse-glimmer/20260824T145826-profile/`
- Vulkan perf logger: `GGML_VK_PERF_LOGGER` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (runtime `getenv`)
- `gputop`: intel-gpu-tools 2.5; `intel_gpu_top` is i915-only
- Open PR: https://github.com/ggml-org/llama.cpp/pull/27342
