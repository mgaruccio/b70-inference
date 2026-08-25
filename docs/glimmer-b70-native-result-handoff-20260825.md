# Glimmer B70 — native result / handoff — 2026-08-25

Status: **paused for regroup**. This is the authoritative handoff for the native
SYCL+oneDNN audit. Keep the known-good B70 service live; do not auto-promote the
candidate or continue optimization from this document.

## Live service (leave it up)

- **Host:** `inference-host` / Intel Arc Pro B70 (`8086:e223`, `xe`), tmux
  `muse`.
- **Endpoint:** `http://inference-host:18099/v1` (local probe:
  `http://127.0.0.1:18099/v1`).
- **Path:** native SYCL + oneDNN, no DFlash, `C8` / `-np 8`, `-c 131072`,
  `-b 512 -ub 512`.
- **Source:** llama.cpp commit `2fcb070cf4eeef907ea4d2e0abf76a8a0e740904`
  (`2fcb070cf`).
- **Runtime switches:** VMM `0`, DNN `1`, graph `0`, fusion `0`, ESIMD `0`,
  deferred backing off (`EnableDeferBacking=0`).
- **Launcher:** `~/inference/launchers/start-muse-glimmer-sycl.sh` on the
  host. The B70 service should remain live on port `18099`.
- **Matched baseline:** approximately **50.95 aggregate public-boundary e2e tok/s**
  (`aggregate_e2e_tok_s`: completion tokens divided by concurrent wave/service wall time), not server-log decode; per-request log-eval is separate. Request-latency p95 was approximately **10.4 s** after two shape-matched warmup waves.

The host is persistently booted with `amd_iommu=off`; this is required to avoid
the fatal platform/data-fabric resets seen during this audit. Do not change host
boot settings or restart the live service during the pause.

### Exact launch recipe (reference only; do not restart now)

The launcher defaults are made explicit here so a future session does not
accidentally change the live cell:

```bash
# Run only on inference-host, with the persistent amd_iommu=off boot state.
DFLASH=0 \
PARALLEL_SLOTS=8 \
SLOT_CTX=16384 \
CONTEXT_SIZE=131072 \
PORT=18099 \
BATCH=512 \
UBATCH=512 \
BIN=/src/build-sycl-dnnl/bin/llama-server \
~/inference/launchers/start-muse-glimmer-sycl.sh
```

The launcher supplies the following container environment: `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`,
`ZE_AFFINITY_MASK=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:0`,
`SYCL_CACHE_PERSISTENT=0`, `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0`,
`NEOReadDebugKeys=1`, `EnableDeferBacking=0`, `VMM=0`,
`GGML_SYCL_ENABLE_VMM=0`, `GGML_SYCL_ENABLE_DNN=1`,
`GGML_SYCL_ENABLE_GRAPH=0`, `GGML_SYCL_ENABLE_FUSION=0`,
`GGML_SYCL_ENABLE_ESIMD=0`, and `LD_LIBRARY_PATH=/opt/onednn/lib`.
The server command includes `--no-mmap --no-warmup`; keep `--no-warmup`.

The matched warmup recipe was two explicit public-boundary waves, each using
the fixed sky prompt, seed `42`, `64` output tokens, eight concurrent
requests, and one repetition:

```bash
python3 ~/inference/launchers/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18099/v1 \
  --label baseline-fusion0-final-warmup1 \
  --out ~/b70-evals/muse-glimmer/20260825T-baseline-fusion0/baseline-fusion0-final-warmup1.json \
  --log ~/inference/logs/muse-glimmer.log \
  --concurrency 8 --reps 1 --max-tokens 64 --seed 42

python3 ~/inference/launchers/glimmer-phase0-instrument.py \
  --base http://127.0.0.1:18099/v1 \
  --label baseline-fusion0-final-warmup2 \
  --out ~/b70-evals/muse-glimmer/20260825T-baseline-fusion0/baseline-fusion0-final-warmup2.json \
  --log ~/inference/logs/muse-glimmer.log \
  --concurrency 8 --reps 1 --max-tokens 64 --seed 42
```

These are evaluation-level warmups; they do not replace the required server
`--no-warmup` safety setting. The two waves remove the cold JIT tail.

## Native scaling and screens

All throughput values below are **aggregate public-boundary e2e tok/s**—the instrumenter's
`aggregate_e2e_tok_s` (completion tokens divided by concurrent wave/service wall time). They are
not server-log decode; per-request log-eval is separate. Do not relabel `completion_tokens / wall` as decode.

| Cell | Aggregate public-boundary e2e tok/s (`aggregate_e2e_tok_s`) |
|---|---:|
| Native C4 | 40.78 |
| Native C6 | 46.82 |
| Native C8, matched wave / final baseline | 49.77 / **50.95** |
| C10, old fallback | 29.14 |
| C12, old fallback | 34.44 |
| C16 | 44.72 |
| DFlash C4, `n_max=2` | 26.34 |
| DFlash C4, `n_max=1` waves | 32.49 / 38.05 / 38.52 |

Runtime screens were all reverted: fusion1 was **-0.915%**, graph1 was
**-9.404%**, and ESIMD hit a startup hard reset. The known-good C8 baseline is
the live cell, not any of those screens.

## oneDNN unblock and reset boundary

- oneDNN PR **#27671** / commit `893a5e7` plus VMM `0` unblocked the native
  path.
- The old allocator assertion was a persistent-scratchpad allocation against
  a strict-LIFO VMM pool; it was not evidence that the model or output was
  invalid.
- `amd_iommu=off` contained the AMD data-fabric reset. `pci=realloc` and
  deferred-backing-only did not contain it.

## MMVQ cap-12 candidate (not promoted)

The exact remote candidate is commit
`73dbcef870b1218bb806095c6036736e3beba24b` (`73dbcef87`,
`sycl: extend K MMVQ batch cap to 12`). Its patch is preserved at
[`patches/glimmer-b70-sycl-mmvq-cap12-73dbcef870b1218bb806095c6036736e3beba24b.patch`](../patches/glimmer-b70-sycl-mmvq-cap12-73dbcef870b1218bb806095c6036736e3beba24b.patch).
The patch extends only the Q4_K, Q5_K, and Q6_K cases for batch columns
`9..12`; the remote tree was not modified.

| Candidate cell | Aggregate public-boundary e2e tok/s (`aggregate_e2e_tok_s`) |
|---|---:|
| C8 | 50.762 |
| C10 | 53.570 |
| C12 | 54.858 |

Debug traces proved K-MMVQ dispatch at `n=9` and `n=12`. Public outputs were
valid, and source audit found no credible defect. Promotion is nevertheless
blocked until the 12 CPU-reference `test-backend-ops` `MUL_MAT` cases pass:
Q4_K/Q5_K/Q6_K × `n=9,10,11,12`.

The concurrent text-hash comparator is invalid evidence because of cache/slot
mismatch and scheduling nondeterminism. Two comparator-container retries reset
the host. Do not use that comparator or auto-promote before regroup/testing;
the known-good native C8 baseline remains live.

## Parked DFlash2 candidate

[`z-lab/Muse-Glimmer-30B-DFlash2`](https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2)
is the same already-tested draft artifact. Keep it as a conditional **C1-only
SYCL compatibility** candidate; kill C4/C8 absent new evidence and require a
**≥5% C1** improvement before any future run. It is not scheduled.

## Vulkan cap-12 correction

The nominal Vulkan `n=12` dispatch improved, but the raw C4 n=2 stream also
contained an `n=13` GEMM fallback. The prior total-verify `≤1.25×` gate was
invalid: the audit mixed per-request server-log eval/verify observations with
aggregate public-boundary e2e behavior and used median acceptance where a
weighted acceptance would be required. See the corrected
[`docs/glimmer-b70-vulkan-cap12-20260824.md`](glimmer-b70-vulkan-cap12-20260824.md)
for the separated labels and no-promotion conclusion.

## Paused next action

Regroup, then run the 12 CPU-reference cases against the preserved patch before
considering any GPU promotion. Do not run GPU optimization, comparator retries,
host changes, service restarts, or the parked DFlash2 C4/C8 path during this
pause.
