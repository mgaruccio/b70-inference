# MTP4 draft attribution — development diagnostic

This is a bounded measurement campaign, not a candidate optimization and not a
performance result.  The lead owns GPU execution and optimization selection.
No outcome, speedup, or positive gain is inferred here.

## Predeclared contract

- **Host gate:** `inference-host`, idle before launch, power cap `275000000`
  microwatts (275 W), no running containers, Glimmer stopped, and unchanged
  production launcher SHA256
  `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.
- **Bundle:** pinned MTP4 image/vLLM `ac7509e2b` with the original archived
  MTP nightly, boundary, GDN mixed-split v5, draft-LM-head INT4, draft-MTP
  INT4 S+M1, and prefill-guard stack.  Target is FP16/GPTQ-Int4 symmetric
  G128 with FP8 KV; C1, prefix caching off, MTP K4.
- **Capacity:** `--max-model-len 212992`,
  `--max-num-batched-tokens 8192`, `--gpu-memory-utilization 0.95`, graph
  capture sizes `[1,2,4,8]`, and `--mamba-cache-mode align`.  The wrapper
  does not add `--enforce-eager` or otherwise change graph mode.
- **Public journey:** start the disposable server, warm it through the real
  HTTP API, then send exactly 65,536 input tokens and 128 forced output
  tokens, greedy/temperature 0, seed 42, with a complete streamed response.
  `/start_profile` is launched only after the first non-empty SSE output;
  the bounded native window is predeclared as delay 3 / maximum 5 worker
  iterations, with the stream allowed to provide 24 non-empty events so the
  actual captured step count is retained rather than assumed.
- **Evidence:** request/rendering/SSE/metrics, launch and source hashes,
  runtime identity, native trace, canonical `step-timing` graph events,
  `draft-attribution` scopes/dispatcher/draft-span metadata, and server/host cleanup
  artifacts.  The lifecycle driver removes only its owned container on
  success, stop, or error.

## What is measured

`profile-draft.py` reuses
`../20260911-qwen38-step-profile-64k/run-step-profile.py` for lifecycle,
HTTP, metrics, finite-response validation, and cleanup.  `draft-patch.py`
composes the canonical `qwen38_step_timing_patch.py` rather than copying it.
`draft-annotations.py` is imported only after `/start_profile` returns and
wraps these narrow MTP boundaries when available:

- `EagleProposer.propose`;
- `Qwen3_5MTP.forward` and its `Qwen3_5MultiTokenPredictor` body;
- draft `compute_logits` and greedy/sampling helpers; and
- `CudagraphDispatcher.dispatch`.

The dispatcher artifact retains requested token count, returned
`BatchDescriptor` graph key, runtime mode, available keys, caller provenance,
and conservative first-five-token/later-one-token labels.  The canonical
XPU-event overlay is extended only through a wrapper to retain matching graph
context.  XPU events are recorded around each replay and around each complete
`EagleProposer.propose` span, then resolved once at stop; there is no per-step
synchronization.  The two event sources remain separate in the artifact.  CPU
scopes are inclusive annotations and are never added to graph-event or kernel
time.  Startup/capture paths are not wrapped, so instrumentation cannot perturb
compilation/capture.

`summarize-draft.py` maps eager trace kernels once through CPU `External id`
to the innermost draft phase, audits unmatched runtime correlations, and
reports graph replay attribution and complete-propose span timings separately.
`draft_span_attribution` retains deferred XPU durations without adding them to CPU
scopes or graph rows.  Graph replay often hides inner kernels; summed device work
is not critical-path latency.

## Exact inference-host execution

Run from a checkout containing this campaign and the existing sibling lifecycle
driver.  Keep all runtime dependencies and raw traces on `inference-host`.

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-mtp4-draft-attribution
C=/home/mike/code/b70-inference
mkdir -p "$R/timing"
cp "$C/scripts/experiments/qwen38_step_timing_overlay.py" "$R/timing/"
cp "$C/scripts/experiments/qwen38_step_timing_patch.py" "$R/timing/"
cp results/20260913-qwen38-mtp4-draft-attribution/{README.md,profile-draft.py,draft-patch.py,draft-annotations.py,summarize-draft.py} "$R/"
python3 -u "$R/profile-draft.py" --timing-dir "$R/timing" --out "$R/profile-01" \
  >"$R/profile-01.console.log" 2>&1
python3 "$R/summarize-draft.py" "$R/profile-01" \
  --out "$R/profile-01/draft-attribution-summary.json"
```

The profile command is graph-enabled and uses the original five-patch MTP4
launch.  Inspect `profile-01/summary.json` before accepting the attribution:
HTTP controls must be 200, the stream must finish at length with exact token
counts, metrics must be finite, and cleanup/host invariants must pass.  The
summary script must report the actual trace step count and any annotation or
runtime-correlation errors.

## Research and limitations

Fresh primary-source checks:

- [PyTorch profiler](https://docs.pytorch.org/docs/2.13/profiler.html):
  `record_function` provides CPU scopes; device timing and CPU scope timing
  are different measurements.
- [vLLM CUDA graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/):
  graph capture/replay is shape/mode dispatched and replay can hide inner
  operations from a normal profiler trace.
- [Pinned proposer source](https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/spec_decode/llm_base_proposer.py):
  `initialize_cudagraph_keys` maps FULL mixed mode to PIECEWISE for the
  drafter; setting FULL alone is not a valid MTP attribution fix.

This diagnostic does not optimize kernels, change acceptance, alter the
persistent launcher, or compare throughput.  The earlier ~54.88 ms eager
proxy is deliberately not reused: its configuration and timing window differ.
A separate unprofiled production-capacity baseline is required for any later
performance claim.

## CPU-only checks in this checkout

```bash
python3 -m py_compile \
  results/20260913-qwen38-mtp4-draft-attribution/{profile-draft.py,draft-patch.py,draft-annotations.py,summarize-draft.py,test-draft-attribution.py}
python3 results/20260913-qwen38-mtp4-draft-attribution/profile-draft.py --help
python3 results/20260913-qwen38-mtp4-draft-attribution/summarize-draft.py --help
python3 results/20260913-qwen38-mtp4-draft-attribution/test-draft-attribution.py
```

These checks do not import the ML runtime or exercise a GPU.

## Observed baseline and attribution

`baseline-01` passed: 3 canaries, 131 finite boundaries, 8 functional checks,
and one warmup plus six measured 65536/128 HTTP completions. Measured decode
tok/s: 52.604882, 56.389909, 55.041815, 53.837301, 57.823587, 55.061446.
Median 55.051631 tok/s; inclusive IQR 1.919364; TTFT median 53.877306 s.
No samples excluded. Driver exit 0 and host-unchanged checks passed.

Baseline command on inference-host (R is the campaign path above):
```bash
python3 -u "$R/../20260911-qwen38-step-profile-64k/run-mtp-depth.py" \
  --depth 4 --cell mtp4 --out "$R/baseline-01"
```

`profile-01` also exited 0, completed the real HTTP response and host checks,
and reported no annotation errors. Its native trace contains five draft
generation steps. Exclusive eager device attribution per captured step:
- draft vocabulary LM head: 4.390061 ms (20 GEMMs total);
- remaining propose work: 2.239318 ms, predominantly attention;
- greedy selection: 0.153519 ms.

The separate deferred-event window contains 23 complete propose spans, median
8.613854 ms. These intervals include current-stream scheduling effects and are
not unprofiled draft latency. Do not add different windows into a step budget.
Observed dispatcher calls: 23 target FULL 5-to-5; 23 draft PIECEWISE 5-to-5;
23 draft PIECEWISE 1-to-5. No NONE fallback observed. Continuation padding
exists in the draft; the target already uses exact-five full graphs.

Raw trace remains on inference-host under
`profile-01/profile/rank0.1789282253597902610.pt.trace.json.gz` (295152 bytes),
SHA256 `edefd5241e513d2df27b258312518cd2bfd5b30a1bf8feab65cbc313554b340c`.
The local summary retains full trace attribution, dispatcher and event samples.

Decision: no gain claimed. Device computation dominates observed draft work;
this does not establish a large recoverable host-launch gap. Continue only
with a separately identified equivalent native/runtime candidate. A candidate
must exceed 5% median 64K throughput gain, independently confirm via
interleaved baseline/candidate runs, pass correctness gates, and avoid >5%
regressions at 512/8K/32K. Precision, capacity, depth and acceptance stay fixed.

Lead reran all four CPU fixtures and both CLI help checks successfully.
Read-only review found no blocking defects before the real profile. Lab
preview/patch tools were unavailable (ENOENT); scoped CLI execution used the
user's explicit authorization. Production launcher and 275W cap are unchanged.
