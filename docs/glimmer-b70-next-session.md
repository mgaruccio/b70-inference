# Glimmer B70 — next session

Date: 2026-08-26
Public repo: this one.
Current handoff: `docs/glimmer-b70-vllm-xpu-handoff-20260826.md`.
Parked llama.cpp restore: `docs/glimmer-b70-native-result-handoff-20260825.md`.
Prior Vulkan result: `docs/glimmer-b70-vulkan-cap12-20260824.md` (not live).
**Current authoritative state:** vLLM-XPU GPTQ-G128 + official DFlash C1 on `inference-host:8000` (`muse-vllm-xpu-c1`, served id `muse-glimmer-gptq`). Median client decode **42.14** tok/s vs llama.cpp official DFlash n2 **42.42**. Target W4A16 is in Qwen's no-spec band; remaining gap is speculation (Qwen MTP4 83.7–112).
**Next action:** confirm the vLLM cell is up, instrument DFlash acceptance / tokens-per-verify, then draft-side INT4 of Muse assistant internals vs 42.14. Do not treat the Aug 25 SYCL C8 pause as current. Do not widen DFlash2-SYCL `n_max`.

## Parked llama.cpp C8 restore (not the live research cell)

Host: `inference-host` (CachyOS, `xe` `8086:e223`). SSH: `inference-host` / `192.168.8.172` / tailnet `100.75.79.54`.

The last intended live cell is vLLM-XPU on port **8000**. Native SYCL + oneDNN C8 on tmux `muse` port 18099 is the restore recipe only: VMM0/DNN1, deferred backing off, graph/fusion/ESIMD disabled, no DFlash, llama.cpp `2fcb070cf4eeef907ea4d2e0abf76a8a0e740904`.

```text
~/inference/launchers/start-muse-glimmer-sycl.sh
```

Exact launch and two-wave warmup: `docs/glimmer-b70-native-result-handoff-20260825.md`. Do not relaunch SYCL unless the user wants the llama.cpp cell back.

## Prior Vulkan reference (not live)

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

## Historical llama.cpp result (2026-08-25, not current)

Superseded by `docs/glimmer-b70-vllm-xpu-handoff-20260826.md`. Kept for the parked C8 restore.

Native SYCL+oneDNN C8/`-np 8` no-DFlash cell: `-c 131072`, `-b 512 -ub 512`, commit `2fcb070cf`, matched baseline approximately 50.95 aggregate public-boundary e2e tok/s (`aggregate_e2e_tok_s`: completion tokens divided by concurrent wave/service wall time), not server-log decode; per-request log-eval is separate, after two shape-matched warmups. Fusion1 and graph1 regressed and ESIMD hard-reset at startup; all are reverted.

The MMVQ cap-12 candidate is remote commit `73dbcef870b1218bb806095c6036736e3beba24b` and is preserved as `patches/glimmer-b70-sycl-mmvq-cap12-73dbcef870b1218bb806095c6036736e3beba24b.patch`. It extends only Q4_K/Q5_K/Q6_K `n=9..12` cases; candidate C8/C10/C12 results were 50.762 / 53.570 / 54.858 aggregate public-boundary e2e tok/s (`aggregate_e2e_tok_s`: completion tokens divided by concurrent wave/service wall time), not server-log decode; per-request log-eval is separate. Public outputs were valid and source audit found no credible defect.

Do not promote it yet. The gate is the 12 CPU-reference `test-backend-ops` `MUL_MAT` cases (Q4/Q5/Q6 × n9..12); the concurrent text-hash comparator is invalid because of cache/slot mismatch and scheduling nondeterminism, and two retries reset the host. Regroup before any GPU promotion or optimization.

## Do not

- Rewrite `process()` / inject
- DFlash2 on C4
- 128k + DFlash + `-ub 8192`
- Custom Q4_K GEMM kernels
- Do not auto-promote MMVQ cap=12; CPU-reference gate and regroup are next
- `iaprof` / patched NEO
- Score `completion_tokens / wall`
- Shadeform / 5090

## Sources

- Current vLLM-XPU handoff: `docs/glimmer-b70-vllm-xpu-handoff-20260826.md`
- Native result/handoff (parked): `docs/glimmer-b70-native-result-handoff-20260825.md`
- MMVQ candidate patch: `patches/glimmer-b70-sycl-mmvq-cap12-73dbcef870b1218bb806095c6036736e3beba24b.patch`
- Prior Vulkan result: `docs/glimmer-b70-vulkan-cap12-20260824.md`
- Host receipts: `~/b70-evals/muse-glimmer/20260824T163018-cap12/` and native audit receipts under `~/b70-evals/muse-glimmer/`
- VEC cap / dispatch: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (`mul_mat_vec_max_cols`, `ggml_vk_mul_mat`)
