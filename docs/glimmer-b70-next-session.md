# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one.
Last results: `docs/glimmer-b70-vkperf-20260824.md`.
Prior: `docs/glimmer-b70-profiling-20260824.md`.

**This session:** keep the 12-token C4 `n_max=2` verify on `MUL_MAT_VEC`. Raise `mul_mat_vec_max_cols` to 12 or 16. Do not rewrite inject. Do not write a new Q4 GEMM first.

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

C4 production, when needed: official DFlash `n_max=1`, 8×32k, `-ub 512`. **Not** `n_max=2`.

## What is already true

Do not re-litigate these. Receipts: `~/b70-evals/muse-glimmer/20260824T153825-vkperf/`.

| Cell | Decode (log eval) | Acc | Card W | Path | Verify p50 |
|---|---:|---:|---:|---|---:|
| C1 n2, `-np 2` | 43.7 tok/s (64 tok) | 0.95 | 255 | MUL_MAT_VEC n=3 | 56 ms |
| C4 n1, 8×32k load4 | 16.4 tok/s | 0.88–0.94 | 216 | MUL_MAT_VEC n=8 | 98 ms |
| C4 n2, 8×32k load4 | 3.67 tok/s | 0.91–0.95 | 131 | **MUL_MAT GEMM n=12** | **696 ms** |

C4 n2 generate verify (20 full graphs):

| Family | p50 | Share |
|---|---:|---:|
| **MUL_MAT** (non-VEC) | **678.5 ms** | **98.2%** |
| FLASH_ATTN_EXT | 8.0 ms | 1.2% |

Named top op: `MUL_MAT q6_K m=6656 n=12 k=19968` (22%). The family is the pass.

Cause: `static constexpr uint32_t mul_mat_vec_max_cols = 8`. n=8 stays on `ggml_vk_mul_mat_vec_q_f16`. n=12 falls through to `ggml_vk_mul_mat_q_f16`.

Killed:

- Serial `process()` / inject rewrite — packed already, 9 ms
- `#27117` — acceptance holds
- Flash-attn / 4×3 mask — 1.2% of the sick graph
- Host stall / idle CCS — CCS 100% at 131 W (cheap GEMM)
- `iaprof` — this cell is Vulkan
- Custom Q4 GEMM before naming the path — named; still do not write GEMM first

## Do this session

### 1. One-line VEC cap

On `~/inference/src/llama.cpp-dflash2`, change only:

```text
static constexpr uint32_t mul_mat_vec_max_cols = 8;
```

to **12** first (exact C4 n2 width). Keep a copy of today's binary. Rebuild **only** `llama-server`.

If 12 builds and C4 n2 dumps show `MUL_MAT_VEC ... n=12`, stop. If the shader array / pipeline init asserts, try **16** (next power-of-two-ish slot) or stop and write the assert.

### 2. Two cells only

Same sky prompt, seed 42, official DFlash, `-ub 512`, `GGML_VK_PERF_LOGGER=1` `GGML_VK_PERF_LOGGER_CONCURRENT=1`, 64 new tokens, one warmup.

| Cell | Pass |
|---|---|
| C4 n1, 8×32k load4 | Still VEC n=8, still ~16 tok/s / ~220 W |
| C4 n2, 8×32k load4 | Dump is `MUL_MAT_VEC` n=12, verify << 696 ms, decode toward n1, watts toward 220 |

Kill: C4 n2 still `MUL_MAT` n=12, or VEC n=12 is *slower* than today's 3.7 tok/s, or C4 n1 regresses.

### 3. Restore

```bash
# production 1–2 stream cell, no GGML_VK_PERF_LOGGER, no --log-timestamps/--perf
# official DFlash n_max=2, -np 2, 131k, port 18099, tmux muse
```

If the VEC-cap binary is a win on C4 n2 and does not hurt C1, leave it. Otherwise put `64f765f` + pack back.

## Do not

- Rewrite `process()` / inject
- DFlash2 on C4
- 128k + DFlash + `-ub 8192`
- Custom Q4_K GEMM kernels
- Split the 4×3 batch unless VEC-cap fails
- `iaprof` / patched NEO
- Score `completion_tokens / wall`
- Shadeform / 5090

## Sources

- This session: `docs/glimmer-b70-vkperf-20260824.md`
- Host receipts: `~/b70-evals/muse-glimmer/20260824T153825-vkperf/`
- VEC cap / dispatch: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`mul_mat_vec_max_cols`, `ggml_vk_mul_mat`)
