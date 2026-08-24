# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one.
Last results: `docs/glimmer-b70-profiling-20260824.md`.

## Live cell (leave it up)

Vulkan llama.cpp **#27342** `64f765f` + local packed `process()` + temporary `b70tick` host timers, tmux `muse`, port 18099.

- Target: Meta Dynamic Q4_K_XL
- Draft: official DFlash Q4_K_M, `--spec-type draft-dflash --spec-draft-n-max 2`
- `-ngl 99 -ngld 99 -c 262144 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja`
- Binary: `~/inference/src/llama.cpp-dflash2/build/bin/llama-server`
- C4 production, when needed: official DFlash `n_max=1`, 8×32k. Not `n_max=2`.

[#27342](https://github.com/ggml-org/llama.cpp/pull/27342) is still **open**.

## What is true

C4 `n_max=2` generate tick is **793 ms p50**. **83% is 30B `llama_decode` verify** (694 ms for 12 tokens / 4 seqs). Packed `process()` is 9 ms. `seq_rm` is PART/PART. `gputop` CCS is pegged (~100%) at **130 W** — cheap GPU path, not host stall, not #27117. C4 `n_max=1` verify is 96 ms / 239 W / 15.7 tok/s.

Host now has `gputop` + `perf`. `intel_gpu_top` does not support Xe. Do not start `iaprof` (L0/USDT) on this Vulkan cell.

## Do this session

Name the op inside the 12-token C4 n2 verify. Vulkan timestamps / ggml Vulkan perf vs C4 n1 (4×2) and C1 n2 (1×3). Do not rewrite inject.

## Do not

- 128k + DFlash + `-ub 8192`
- `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- Score `completion_tokens / wall`
- Upstream packed `process()` as a C4 speedup
- Build patched NEO / `iaprof` for a Vulkan decode
- Custom Q4_K kernels before a named op
- C2 / C3
