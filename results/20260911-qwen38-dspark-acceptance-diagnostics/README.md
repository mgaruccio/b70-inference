# Qwen3.8 DSpark acceptance diagnostics (experiment only)

This directory contains a bounded diagnostic driver, not a launcher, runtime fix,
benchmark harness, or production configuration.  It does not change the host
launcher, download models, alter repository runtime sources, or claim a speedup.

## What is measured

`run-acceptance-diagnostics.py` has one switch for the public comparison cell:

- `--cell target`: target-only FP16-compute GPTQ Int4 symmetric G128 with FP8 KV;
- `--cell dspark`: the current DSpark candidate using the same target and cache,
  fixed `K=7`, BF16 draft/cache, standard rejection, greedy draft sampling, and
  `enable_adaptive_verification=false`.

Both use the pinned B70 image, V2 eager/C1, context `8192`, one sequence, and
prefix caching disabled.  The candidate mounts the previous campaign's frozen
`draft/` snapshot and replays its existing prefill, BF16-draft, and C1 boundary
overlays in that order.  The driver mounts the previous campaign as
`/experiment:ro`; it does not copy or modify those assets. `--kv-cache-dtype auto`
is available only for a later isolated cache-selector diagnostic; the default and
acceptance cell are `fp8`.

Before the experiment matrix, the existing shared smoke code runs its 3 canaries,
131 finite boundaries, 8 functional checks, and prior 19-request greedy parity
smoke.  The diagnostic matrix is exactly 36 requests per cell:

- code and prose prompts from the previous probe, plus one fixed deterministic
  arithmetic prompt;
- thinking disabled and enabled;
- temperature 0 (`top_p=1`, `top_k=-1`) and temperature 1 (`top_p=.95`,
  `top_k=20`);
- seeds 42, 43, and 44;
- `ignore_eos=true`, `max_tokens=512` on every matrix request.

The request JSON and cache salts are identical in the target and candidate cells.
Every stream records raw request bytes/JSON, raw SSE bytes, parsed SSE events, rendered prompt token IDs,
output token IDs, usage, raw before/after metrics, metric deltas, full per-position
accepted counters, draft-step/proposal/accept deltas, emitted tokens per step,
and transport checks.  The driver deliberately does **not** compare stochastic
output-token parity between cells; output IDs are retained for diagnosis only.

## Fresh primary-source note

The pinned primary model card was read for this diagnostic:

<https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/README.md>

It documents DSpark evaluation with SGLang and an NVFP4 target, with the
published acceptance/throughput settings using thinking enabled, temperature
1.0, top-p 0.95, top-k 20.  That is useful context, but it is not equivalent to
this experiment's vLLM B70 GPTQ target + FP8 KV cell, so this driver does not
present the two as a benchmark match.

## Real end-to-end process (lead runs on inference-host only)

Preconditions:

1. On the B70 `inference-host`, stage only the two tracked files in this directory
   into a new sibling campaign (the existing campaign process keeps large draft
   weights outside Git):
   ```bash
   root=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4
   mkdir -p "$root/20260911-qwen38-dspark-acceptance-diagnostics"
   cp results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py "$root/20260911-qwen38-dspark-acceptance-diagnostics/"
   cp results/20260911-qwen38-dspark-acceptance-diagnostics/README.md "$root/20260911-qwen38-dspark-acceptance-diagnostics/"
   ```
   Run from that sibling. The driver resolves the previous frozen campaign from
   either the results-name sibling or the existing host name
   `20260910-dspark-v2-feasibility`; it never downloads the draft.
2. Have no running containers, leave the Glimmer container stopped, keep the
   power cap at `275000000`, keep the pinned image already present (the driver
   uses `--pull=never`), and keep
   `/home/mike/inference/launchers/start-qwen38.sh` at SHA-256
   `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.
3. Keep the previous campaign sibling and its frozen `draft/config.json` and
   `draft/model.safetensors` in place.  The config hash and expected model hash
   are recorded in `dependencies.json`; the draft is mounted read-only.
4. Ensure the new sibling's `target-only` and `current-dspark` output paths do
   not already exist.  Output directories are refused rather than overwritten.

Run the real public API journey, one cell at a time (port 8000 is intentionally
serialized):

```bash
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-acceptance-diagnostics
python3 -u run-acceptance-diagnostics.py \
  --cell target \
  --out target-only

python3 -u run-acceptance-diagnostics.py \
  --cell dspark \
  --out current-dspark
```

The target-only cell is the attribution control; the current DSpark cell is the
old/current candidate baseline for optimization comparisons.  The expected
successful journey is: pinned image inspection, disposable server startup,
model/context confirmation, source replay checks, all shared gates, then 36
finite 512-token SSE streams with passing transport checks.  For DSpark, the
metrics should expose draft steps, proposals, accepts, and per-position
acceptance counters; target-only should not speculate.  These are diagnostic
expectations, not a stochastic parity or quality claim.

Artifacts to retain are `summary.json`, `dependencies.json`,
`launch-argv.json`, `launch-metadata.json`, `launcher.sh`, `image-inspect.json`,
`server.log`, `host-before.json`, `host-after.json`, `container-cleanup.json`,
the source replay records, the shared gate artifacts, and each matrix request,
raw/parsed SSE, prompt/output ID, usage/metrics, and result file.  On success or
failure the driver removes only its own named container and checks host
invariants.  If a run fails, inspect `failure.txt` and `server.log`; do not
reuse an existing output directory.

## Local checks

No torch or inference host is needed for static checks:

```bash
python3 -m py_compile results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py
python3 results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py --help
```

These checks supplement, but do not replace, the real inference-host API
journey above.
