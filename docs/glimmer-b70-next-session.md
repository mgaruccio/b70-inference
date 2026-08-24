# Glimmer B70 — next session

Date: 2026-08-24 (updated after DFlash2 + process() session)
Public repo: this one.
Results: `docs/glimmer-b70-dflash2-process-20260824.md`.
Phase 0: `docs/glimmer-b70-phase0-20260824.md`.

## Live cell (leave it up)

Vulkan llama.cpp **#27342** `64f765f` + local packed `process()`, tmux `muse`, port 18099.

- Target: Meta Dynamic Q4_K_XL
- Draft: official DFlash Q4_K_M, `--spec-type draft-dflash --spec-draft-n-max 2`
- `-ngl 99 -ngld 99 -c 262144 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja`
- Binary: `~/inference/src/llama.cpp-dflash2/build/bin/llama-server`
- Do **not** put `n_max=2` on 4+ streams. C4 production is `n_max=1`, 8×32k.

[#27342](https://github.com/ggml-org/llama.cpp/pull/27342) is still **open**. `master` cannot load DFlash2.

## What is true

| | Decode | Card |
|---|---:|---:|
| C1 official n2 on 64f765f | **38.6** | 273 W |
| C1 DFlash2 n2 | 38.4 | 273 W |
| C1 DFlash2 n15 coding | 10.8 (mean accept **8.47**) | 132 W |
| C4 n_max=1 | **15.7 /stream** | **243 W** |
| C4 n_max=2, packed `process()` | **3.05 /stream** | **130 W** |

Packed `process()` really does one encode+inject for 4 seqs (`unique_seq=4`, `packed=12`). C4 n_max=2 did not move. That tax is elsewhere.

## Next — do not invert this

1. **Do not spend another session rewriting inject.** The serial loop is gone and C4 is still 130 W.
2. **Attribute the 950 ms C4 n_max=2 tick** with server verbosity + `iaprof` / Xe timestamps. Rank: `llama_get_embeddings_layer_inp` sync, spec checkpoints if `seq_rm==FULL`, `draft()` noise block, 30B verify of 12 tokens.
3. **On-device 5-layer activations** only if (2) names the host readback.
4. **DFlash2 at C1 n_max≥8** only after (2). Quality is real; the cell is idle.
5. Cheap: ship `reasoning_strength=low` on the Pi/agent template.

## Do not

- 128k + DFlash + `-ub 8192`
- `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- Score `completion_tokens / wall`
- Upstream packed `process()` as a C4 speedup
- DSpark
- Custom Q4_K kernels before `iaprof`
- C2 / C3
