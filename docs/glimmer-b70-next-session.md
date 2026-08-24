# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one.
Last results: `docs/glimmer-b70-vulkan-cap12-20260824.md`.
Prior: `docs/glimmer-b70-profiling-20260824.md`.
**This session result:** cap=12 was exercised on the real public stream path and removed the C4 n=12 GEMM cliff, but was **not promoted**: steady C4 n=2 verify was 130.8 ms versus the C4 n=1 guard at 98.2 ms (1.33×; decisive guard is ≤1.25×). Receipt: `~/b70-evals/muse-glimmer/20260824T163018-cap12/`.
**Next branch:** do not promote or re-run cap=12/cap=16 here. Commission the bounded split/native investigation; same-GGUF SYCL is the fair native fallback. Keep Vulkan production at cap=8 and official DFlash `n_max=2` until the next branch is measured.

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

## Result and next branch

Receipt: `~/b70-evals/muse-glimmer/20260824T163018-cap12/`; report: `docs/glimmer-b70-vulkan-cap12-20260824.md`. Cap=12 routed C3 n=9 and C4 n=12 through `MUL_MAT_VEC` with no assert/device loss, and C4 n=2 improved from 3.55 to 18.01 log-eval tok/s. It was not promoted because steady C4 n=2 verify was 1.33× the C4 n=1 guard, above the 1.25× decisive limit.

Next: commission the bounded split/native branch. Do not rewrite inject/process or start a custom GEMM. The fair native comparison is same-GGUF SYCL; OpenVINO remains a separate IR/single-flight comparison. Keep the restored cap=8 production cell up while the next branch is measured.
The cap=12 candidate remains only in the receipt; the host is restored to cap=8 with normal `muse` on port 18099.

## Do not

- Rewrite `process()` / inject
- DFlash2 on C4
- 128k + DFlash + `-ub 8192`
- Custom Q4_K GEMM kernels
- Do not promote cap=12; split/native is the next bounded branch
- `iaprof` / patched NEO
- Score `completion_tokens / wall`
- Shadeform / 5090

## Sources

- This session: `docs/glimmer-b70-vulkan-cap12-20260824.md`
- Host receipts: `~/b70-evals/muse-glimmer/20260824T163018-cap12/`
- VEC cap / dispatch: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`mul_mat_vec_max_cols`, `ggml_vk_mul_mat`)
