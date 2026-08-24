# Glimmer B70 — DFlash2 + batched `process()` — 2026-08-24

Host: `inference-host`, Intel Arc Pro B70, public `POST /v1/chat/completions` `stream=true`.
Instrument: `~/inference/launchers/glimmer-phase0-instrument.py`.
Decode = log `eval time`. Do not use `completion_tokens / wall`.

## What shipped on the host

- Side tree: `~/inference/src/llama.cpp-dflash2` at
  [llama.cpp#27342](https://github.com/ggml-org/llama.cpp/pull/27342) head
  `64f765f5adefa4620dddda436ce56f1430435536` (PR is **still open**; not on `master`).
- Vulkan Release, Ninja, `GGML_VULKAN=ON`, no SYCL AOT.
- Binary: `0.1.2-dev` build 10509.
- Draft GGUF: `incoai/Muse-Glimmer-30B-DFlash2-GGUF`
  `Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf`
  (`93dbfb6f88e4645dec1347cf93f9d6fc80b90d413038722385b2a8e53565c949`).
- Local patch on that tree: pack all live tokens in
  `common_speculative_impl_draft_dflash::process()` into one
  `llama_encode` + one inject `llama_decode` per ubatch chunk.
  Unpatched binary kept as `build/bin/llama-server.unpatched-64f765f`.
- Live cell restored: official DFlash `n_max=2`, `-np 2`, patched binary, port 18099.

Receipts: `~/b70-evals/muse-glimmer/20260825-dflash2/`.

## C1 — DFlash2 vs official (unpatched `process()`)

| Cell | Decode | Mean accept | Acc | Card W |
|---|---:|---:|---:|---:|
| Live `70adb1b` official n2 | 42.1 | — | 0.77 | 268 |
| PR official n2 | 38.6 | 2.31 | 0.66 | 263–273 |
| DFlash2 n2 | 38.4 | 2.31 | 0.66 | 273 |
| DFlash2 n4 | 36.6 | 3.23 | 0.57 | 253 |
| DFlash2 n8 | 5.6 | 4.38 | 0.43 | 129 |
| DFlash2 n15 sky | 6.3 | 4.54 | 0.26 | 132 |
| DFlash2 n15 coding | 10.8 | **8.47** | 0.52 | 132 |

DFlash2 loads with `--spec-type draft-dflash`. At `n_max=2` it is a wash
with official DFlash (kill gate was “worse than 38”; 38.4 is not). The
longer accepts are real on a coding prompt at `n_max=15` (8.47 vs Inco’s
5.70 block-16 claim) but the cell is GPU-idle. Do not take DFlash2 to C4.

## Batched `process()` — C4 kill

Patched C1 official n2 stayed 38.6 / 273 W.

| Cell | Decode each | Acc | Card W |
|---|---:|---|---:|
| C4 n_max=2 patched | **3.05** | 0.70–0.73 | **130** |
| C4 n_max=1 patched | **15.7** | 0.74–0.79 | **243** |

Unpatched C4 n_max=2 was 3.1 / 134 W. Kill hit.

## Instrumentation — the pack *does* see 4 sequences

Temporary `LOG_INF` on the patched `process()` during C4 n_max=2 load4
(64 new tokens):

```
calls 34
('4', '12', '1') 20   # unique_seq=4, packed=12, chunks=1
('1', '3', '1')   5
plus a few prefill rows (packed 29–151)
```

`12 = 4 slots × (1 sampled + 2 draft)`. One encode+inject per generate
tick. Server `decode()` already hands `process()` a multi-seq
`batch_view`. Serial-per-seq inject was real in the source and is gone.
It was **not** the 7× C4 n_max=2 tax.

## What this means

- C4 production stays official DFlash `n_max=1`, 8×32k.
- C1 production stays official DFlash `n_max=2` on the 64f765f tree.
- Do not upstream the pack as a C4 speedup. It is hygiene: same
  numerics, one graph instead of N. A follow-up would keep the 5
  layer activations on device (still a host `memcpy` of hidden 6656).
- Next C4 suspects: host sync around `llama_get_embeddings_layer_inp`,
  per-tick spec checkpoints when `seq_rm` is FULL, and `draft()` /
  verify of the n_max=2 noise block. Need verbosity / `iaprof`, not
  another inject rewrite.

## Sources

- PR (open): https://github.com/ggml-org/llama.cpp/pull/27342
- DFlash2 GGUF: https://huggingface.co/incoai/Muse-Glimmer-30B-DFlash2-GGUF
- DFlash2 blog: https://inco.ai/blog/dflash2/
- llama.cpp speculative docs (DFlash, not DFlash2):
  https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
