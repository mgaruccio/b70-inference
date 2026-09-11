# Same-input DSpark / official HF reference experiment

**Development experiment assets, not a parity/acceptance result. No GPU/API run was performed by the worker.**
Only this new result directory is changed. Production launchers, delivered runtime source and earlier results are untouched. Lead must review before either GPU command below.

## Predeclared test and interpretation

Baseline: corrected canonical overlay at `8b512a3`, greedy K7, eager V2 C1, context 8192, FP16 GPTQ Int4 symmetric G128 target, FP8 target KV, BF16 draft/cache, standard rejection, no adaptive verification/top-k truncation/prefix caching. Hardware preconditions: pristine B70, 275 W, no other running containers, Glimmer stopped, persistent launcher SHA `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

1. Start the disposable baseline server using the existing acceptance driver's launch helper (including prefill, corrected canonical BF16 and XPU-boundary overlays), then add the temporary hook.
2. `/health` and `/v1/models` must identify qwen38/8192. Verify installed `/opt/venv/` sources by removing **only the exact diagnostic additions in memory**, replaying the canonical overlay checks and retaining the original XPU-boundary/GDN checks. Record both underlying and hooked source hashes plus hook-file identity.
3. Send the existing synthetic code prompt to real `/v1/chat/completions`: temperature 0, top_p 1, top_k -1, seed 42, thinking false, ignore_eos, stream and return_token_ids, 16 output tokens (32 is the only alternative). First request is unarmed. Its streamed prompt IDs arm the second identical request, not a profiling/warmup call. Both use the same cache salt and disabled prefix caching.
4. Capture only the second request's **first** proposal when its complete prompt IDs match, computed/prefill-computed counts and draft-token count are zero. Subsequent cached proposals are never captured. Check all visible context and query slots are newly written, disjoint and resident; full noncausal attention sees exactly N+7 tokens. Physical cache storage can be reused, but no old logical KV position is visible. Do not clear or mutate the cache to make a test pass.
5. Check both SSE transports/counts and capture-on/off emitted token identity. Retain their raw requests/SSE, IDs, usage, speculative counters and failures. This is an instrumentation-side-effect check, **not reference numerical parity or acceptance proof**. Stop the owned server, remove its temporary arm and check host invariants.
6. With the server stopped, replay the **exact captured ordered aux tensors and query embeddings**, absolute context+query positions and empty cache through the unchanged official `DSparkDraftModel.forward`. Reuse existing weights; selectively load only the dense target `lm_head.weight`, not the target model. Validate all 62 official/native mapped keys, shapes, effective dtypes and bytes; no missing/unexpected weights. Check the exact effective shared head bytes too.
7. Compare fc, shared context norm, per-layer context K/V (raw, normalized, RoPE), query norm/RoPE/attention/MLP/layer boundaries, final hidden, base logits, official sequential Markov-corrected logits and proposed IDs. Also hold the native sampled prefix fixed to distinguish arithmetic differences from argmax cascade. Run an actual second independent official forward for self-repeatability. Report max-abs/RMSE/cosine, top-1 agreement, mutual ranks/margins and first nonexact/decision-different stage. No pass tolerance is invented or relaxed after observing results. Native deferred residual addends are retained; the corresponding HF layer-output comparison explicitly uses a diagnostic CPU BF16 sum.
8. Retain `capture.pt`, `official-first.pt`, `official-repeat.pt` on inference-host, plus small JSON reports/metadata. Capture activation bound is 256 MiB and prompt bound 512 tokens. Effective weight hashing streams 8 MiB chunks; it does not save another copy of model weights. No request secrets are involved. The CPU copies/hashes intentionally add latency, so these are not throughput measurements.

A discrepancy should guide the lead's next fix/experiment. Do not expand to cached-step capture unless the first-step result requires it. Finite outputs, matching final API output or this small experiment alone cannot establish high acceptance or publishable performance. The repository standard publication checklist is **not completed**.

## Fresh primary research and dependency finding

Fetched on 2026-09-11:
- https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/dflash.py
- https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/dspark.py
- https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/raw/b9a5dbdf03bc999c6c73c426b19c2d9041cea393/config.json

Conclusion: official forward returns a tensor (despite its annotation), accepts noise embeddings + concatenated post-layer aux features `[5,19,33,47,61]`, projects fc/hidden_norm once, uses query-only Q and context-then-query K/V, Q/K norm, absolute YaRN RoPE, noncausal attention and final norm. The official Markov head implements the transition bias. The reference does not own embedding/head weights.

The pinned image contains PyTorch `2.13.0+xpu`, Transformers `5.15.0`, safetensors and all DFlash imports, **but not SpecForge** (`find_spec('specforge') is None`). This was reported before accommodating it. `dspark.py` imports `specforge.modeling.draft.dflash`. The fetch option puts the two byte-identical official files under that namespace, with no replacement/stub modules and no changed official code. Checksums are fixed in `parity_common.py`; replay rejects shadowed modules. No dependencies or weights are downloaded. All imports/model testing run inside the existing image, never the Pi Python environment.

## Exact lead commands

Execute long GPU work with the lead's background-task tool, not an unbounded foreground shell. Shell snippets on the host use **bash**, not its default fish shell.

### Stage only this new experiment directory (lead checkout)

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-reference-parity
ssh inference-host "mkdir -p '$R/_dependencies' '$R/official'"
rsync -a results/20260911-qwen38-dspark-reference-parity/ "inference-host:$R/"
rsync -a scripts/patch-vllm-qwen38-dspark-bf16.py \
  results/20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py \
  "inference-host:$R/_dependencies/"
```

### Source/import preflight, NO GPU and NO model weights

Run this and subsequent snippets on `inference-host` using `bash`:

```bash
set -euo pipefail
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-reference-parity
H=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
IMAGE=vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4
docker run --rm --pull=never --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$R:/parity:ro" -v "$R/official:/sources" \
  --entrypoint /opt/venv/bin/python "$IMAGE" /parity/replay-reference.py \
  --official-source /sources --fetch-source --check-imports --device cpu \
  > "$R/official-imports.json" 2> "$R/official-imports.stderr"
# On repeat, omit --fetch-source; it refuses to overwrite a source tree.
docker run --rm --pull=never --network=none --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 -v "$R:/parity:ro" \
  --entrypoint /opt/venv/bin/python "$IMAGE" /parity/test-reference-cpu.py \
  --official-source /parity/official > "$R/official-cpu-test.txt" 2>&1
```

### AFTER REVIEW: real API first-proposal capture

```bash
python3 "$R/run-capture.py" --approve-gpu-launch \
  --driver "$R/_dependencies/run-acceptance-diagnostics.py" \
  --previous "$H" --overlay "$R/_dependencies/patch-vllm-qwen38-dspark-bf16.py" \
  --out "$R/capture-01" --output-tokens 16 \
  > "$R/capture-driver.log" 2>&1
```

It refuses an existing output directory and unapproved launch; checks host invariants before/after; stops only its uniquely named disposable server. It does **not** execute the earlier matrix or functional smoke. Review `capture-01/summary.json`, `capture.json`, `effective-source.json`, both requests' usage/counters and any `failure.txt` before replay.

### AFTER CAPTURE REVIEW, server stopped: official XPU replay

```bash
# Repeat the existing host invariant check immediately before and after replay.
check_host() {
  python3 -c 'import json,runpy,sys; from pathlib import Path; d=runpy.run_path(sys.argv[1]); h=d["host_invariants"](); Path(sys.argv[2]).write_text(json.dumps(h,indent=2)); assert h==d["expected_host_invariants"](), h' \
    "$R/_dependencies/run-acceptance-diagnostics.py" "$R/$1.json"
}
check_host before-reference
trap 'check_host after-reference' EXIT
TARGET=/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16
docker run --rm --pull=never --network=none --read-only --tmpfs /tmp --ipc=host \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /dev/dri:/dev/dri:ro -v "$R:/parity:ro" -v "$R:/output" \
  -v "$H/draft:/draft:ro" -v "$TARGET:/target:ro" \
  --entrypoint /opt/venv/bin/python "$IMAGE" /parity/replay-reference.py \
  --official-source /parity/official --capture /parity/capture-01 \
  --draft /draft --target /target --out /output/reference-01 \
  --device xpu --approve-gpu-replay > "$R/reference-driver.log" 2>&1
```

Expected: `reference-01/comparison.json` has measured stages, independent self-repeatability, exact weight-key/value checks and draft-ID decisions. `measured-not-threshold-certified` is deliberately not a parity pass. Any import/weight/layout/runtime error is blocking and must be retained, not bypassed. Containers are disposable; mounts for prior assets/models are read-only. Keep host-local tensors; copy small reports to this result directory for lead interpretation. No persistent launcher/service needs restoration because none is changed.

## Worker verification (not the real GPU gate)

```bash
PYTHONDONTWRITEBYTECODE=1 python3 results/20260911-qwen38-dspark-reference-parity/test-parity.py -v
```

14 tests passed, including compilation, existing prompt identity, exact canonical source overlay + diagnostic apply/replay/reversal/tamper rejection using `/tmp/vllm-pinned-73029d424`, first-request exclusion, zero-cache visibility, native/HF fused weight mapping, distinct per-layer observations despite shared RoPE modules, output-object preserving callbacks (including failures without retrying the original proposer) and invariant-safe launch reuse. The CPU test above also passed against the unmodified official model with reduced dimensions and five layers/62 keys. This supplemental test does not replace the lead's real API/XPU gate. See `verification.txt` for observed commands/results and preflight failures.
