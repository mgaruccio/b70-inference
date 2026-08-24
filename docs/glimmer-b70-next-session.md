# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one.
Last results: `docs/glimmer-b70-dflash2-process-20260824.md`.
**This session:** `docs/glimmer-b70-profiling-plan.md`.

## Live cell (leave it up)

Vulkan llama.cpp **#27342** `64f765f` + local packed `process()`, tmux `muse`, port 18099.

- Target: Meta Dynamic Q4_K_XL
- Draft: official DFlash Q4_K_M, `--spec-type draft-dflash --spec-draft-n-max 2`
- `-ngl 99 -ngld 99 -c 262144 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja`
- Binary: `~/inference/src/llama.cpp-dflash2/build/bin/llama-server`
- C4 production, when needed: official DFlash `n_max=1`, 8×32k. Not `n_max=2`.

[#27342](https://github.com/ggml-org/llama.cpp/pull/27342) is still **open**.

## What is true

Packed `process()` already sees 4 seqs (`unique_seq=4`, `packed=12`) and **did not** move C4 `n_max=2` (3.05 tok/s / 130 W). Same idle signature on **C1 DFlash2 `n_max=8`** (5.6 tok/s / 129 W). Serial inject is not the tax. We have **not** kernel-profiled. Host has no `iaprof` / `intel_gpu_top` / `perf`.

## Do this session

Follow `docs/glimmer-b70-profiling-plan.md`. Attribute one C4 `n_max=2` tick. Do not rewrite inject.

## Do not

- 128k + DFlash + `-ub 8192`
- `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- Score `completion_tokens / wall`
- Upstream packed `process()` as a C4 speedup
- Start `iaprof` by rebuilding patched NEO / L0 before the cheap host-timer pass
- Custom Q4_K kernels before a named op
- C2 / C3
