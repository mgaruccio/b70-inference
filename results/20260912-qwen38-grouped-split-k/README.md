# Guarded grouped-query Split-K prototype

Development-only, numerically quality-sensitive experiment. User authorized a prototype sharing KV reads across verification queries while retaining Split-K parallelism. Production remains unchanged; no promotion or publishable speedup claim is authorized.

## Fixed contract and test process

Baseline: pinned MTP4 stack from `20260911-qwen38-step-profile-64k`, image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`. Target GPTQ Int4 sym G128, FP16 compute, FP8 KV, original model capacity 212992, batch budget 8192, MTP depth 4, original acceptance rules. B70 on `inference-host`, 275 W; Glimmer stopped, no unrelated GPU jobs.

Only candidate difference: an opt-in disposable worker import replaces eligible `_spec_decode_varlen_fwd` calls with the grouped-query kernel. Interception is **before** the native helper expands five queries into five pseudo-sequences. Shapes in a native-op trace describe its post-expansion inputs, not the serving caller's inputs. Unsupported calls retain the original path; supported kernel failures must surface.

Predeclared gates:
1. In the pinned serving image, compare native and candidate with an independent FP32 paged causal reference, including short/page-boundary lengths, nonunit scalar KV scales, permuted page tables, and serving-style strided HND cache views. Check causal future-key invariance, finite outputs, output-buffer semantics, and unsupported-call fallback. Forbid silent reference-attention fallback. Retain numerical errors and all failures.
2. Capture/replay XPU graphs with changing input/length/page-table contents; compare replay output against fresh reference. Warm compilation before native/candidate operator timing. This is qualification, not serving throughput.
3. At the application's real HTTP boundary, reuse existing canaries, finite-logprob boundary checks, sandboxed functional checks and repeatability smoke. Validate actual eligible candidate dispatch rather than accepting an import marker alone.
4. If qualified, fresh baseline/candidate graph-mode requests: C1, 65536 rendered input tokens, 128 output tokens, greedy seed 42, ignore EOS, thinking/prefix cache off; one warmup plus six measured requests. Retain full responses, usage, raw timings and speculative counters. Report median/IQR and output divergence; no benchmark-standard publication claim from this development subset.
5. Remove only owned experiment containers. Confirm launcher SHA `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, power cap `275000000`, stopped Glimmer, no running containers. Do not alter persistent launchers or services.

`run-grouped-split-k.py` reuses existing drivers and gates, not a second benchmark implementation. Stage it into the same-named remote campaign directory. Candidate runs additionally require canonical `scripts/experiments/qwen38_grouped_split_k.py` and existing `qwen38_step_timing_patch.py` staged there; the latter is only the import shim, not the prior timing overlay.

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-grouped-split-k
python3 -u "$R/run-grouped-split-k.py" --out "$R/baseline-01"
# Only after operator qualification:
python3 -u "$R/run-grouped-split-k.py" --candidate --out "$R/candidate-01"
```

Use fresh output directories for retries; preserve failures. ML dependencies and GPU work stay inside the existing serving image, outside Pi.

Operator invocation on inference-host (same command for `operator-01`/`02`, before the documented compiler/fixture/policy corrections):

```bash
set -euo pipefail
test -z "$(docker ps -q)"
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-grouped-split-k
RUN=operator-03
mkdir "$R/$RUN"
sha256sum "$R/qwen38_grouped_split_k.py" "$R/check-grouped-split-k.py" > "$R/$RUN/source-sha256.txt"
docker run --pull=never --rm --name b70-grouped-operator \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v "$R:/experiment:ro" -v "$R/$RUN:/output" \
  --entrypoint /opt/venv/bin/python \
  vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f \
  -P /experiment/check-grouped-split-k.py --out /output 2>&1 | tee "$R/$RUN/driver.log"
```

The output-stride diagnostic used the same image/device/env mounts with `python -P -`: load the check script with `runpy.run_path`, assign its `make_case.__globals__['torch']`, create `make_case(1665)` using the operator-01 fixture, fill `out` with NaN, call the unchanged `fa._spec_decode_varlen_fwd(*case.helper_args())`, and compare with `reference(case)`. Repeat with only `case.out=torch.empty(case.q.shape, device='xpu', dtype=case.q.dtype)` changed. Q/reference were finite in both; only contiguous native output was finite.

## Fresh research informing implementation

- <https://github.com/intel/intel-xpu-backend-for-triton/blob/main/.claude/reference/hardware-reference.md>: BMG/Xe2 FP16 DPAS and FP8 conversion; installed-backend compilation still requires real-device qualification.
- <https://triton-lang.org/main/python-api/generated/triton.language.dot.html>: FP16 inputs with FP32 accumulation. Ordinary FP8 KV is converted/descaled, not treated as microscaled `dot_scaled` input.
- <https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py>: grouped GQA, online softmax, paged addressing, and stable split reduction precedent.
- <https://docs.pytorch.org/docs/2.13/generated/torch.xpu.graphs.graph.html> and <https://docs.pytorch.org/docs/2.13/generated/torch.xpu.graphs.XPUGraph.html>: capture fixed shapes/allocations; update buffer contents in place, not graph addresses.
- <https://github.com/intel/intel-xpu-backend-for-triton/issues/6206>: avoid forced SIMD width overrides on BMG due masked-store miscompilation reports.

Proposed tile: five verification queries times six GQA heads gives 30 rows padded to 32, per KV head, with independent KV splits and stable FP32 partial reduction. Each row keeps its own causal endpoint. Reuse alone is insufficient: the previous native chunk-prefill selector was 11.2× slower in its isolated test.

## Results

- `baseline-01`: HTTP gates passed (3 canaries, 131 finite boundaries, 8 functional checks); six measured 64K requests yielded **55.722486 tok/s** median, **3.156728** inclusive IQR. Host unchanged.
- `operator-01`: failed. Triton 3.7.2 cannot cast integer `other=0` to masked FP8; changed the two load defaults to floating `0.0`. The fixture also used pitched output, unlike serving. An isolated native test (`native-output-strides.log`) showed nonfinite pitched output but finite contiguous output passing FP32 comparison; the fixture now keeps pitched Q/HND KV and uses contiguous output. These are compiler/fixture corrections, not tolerance changes.
- `operator-02`: kernel compiled; normal/page-boundary/64K, causal, dummy, fallback and fail-loud checks passed. The strict FP32 gate failed for both native and candidate at the same tiny short-context elements (including graph-replay length 5). Candidate/native length-4 comparison passed; FP32 absolute errors at the failing near-zero elements were about 1.31e-4 and 1.57e-4. No timing was allowed by that gate. Raw failures remain unchanged.

### Supplemental qualification policy (declared before operator-03)

The original FP32 `rtol=0.02, atol=1e-4` diagnostic is retained verbatim in every subsequent reference comparison and separately counted; it is not relabeled as a pass. It excludes the unchanged native FP16 computation on these cancellation-sensitive short examples. The revised development qualification therefore also requires:

- **Unchanged strict candidate/native comparison** (`rtol=0.02, atol=1e-4`).
- Independent FP32 comparison with `atol=max(1e-4, eps_fp16 * max(abs(descaled V)))`, retaining `rtol=0.02`. This test allowance derives from probability/output FP16 rounding scale, not the observed failing magnitude; it is not a universal error proof or permission to change model precision.
- Every graph replay state compared to a fresh native call on the same current inputs, as well as the independent reference. Future-key invariance and dummy-output checks remain exact.

Policy-v2 success is not original strict-FP32-gate success. `operator-03` passed all 32 revised qualification checks, including native/candidate mutable-input graph replay; **8 original strict FP32 diagnostics still fail** and are explicitly retained/countable in the JSON.

Operator graph medians (12 paired alternating samples, three warm replays; not serving speeds):
- Split16/stage1: native **0.9307295 ms**, candidate **23.355599 ms** (~25.1× slower).
- Split32/stage1: native **0.898724 ms**, candidate **11.9910155 ms** (~13.34× slower).

This implementation is rejected as a speedup candidate. The faster of its two tested variants (split32/stage1) was selected only for real HTTP integration validation, not promotion. `candidate-01` completed the existing API gates and 64K workload at unchanged graph-mode capacity/settings:

- Candidate decode: **5.837453 tok/s** median, **5.636706** IQR versus baseline **55.722486 / 3.156728**; observed median delta **-89.52%**. Candidate measured samples were approximately `[0.832, 2.422, 8.414, 3.551, 12.886, 8.123]` tok/s; warmup was 0.518 tok/s. All samples are retained. The large variability means this is not a steady-state or publishable benchmark estimate; even the best candidate sample was substantially slower.
- Both cells passed 3 canaries, 131 finite boundaries, and 8 functional checks. All **19/19 fixed-suite output token-ID files match** across cells.
- The six long request JSON files are pairwise identical, including rendered prompt token IDs. Only **2/6 full output texts match** (trials 3 and 5). No repeated-baseline control was run for those exact long requests, so this does not isolate kernel effects from runtime nondeterminism; the candidate is **not bitwise-output-qualified**.
- Pooled six-request speculation: baseline 518/1020 accepted drafts over 255 steps (50.784%, 3.0314 one-plus-accepted-per-step); candidate 520/1020 over 255 steps (50.980%, 3.0392). Nearly unchanged acceptance did not compensate for the kernel cost.
- Actual integration evidence: `candidate-01/server.log` shows eligible pre-expansion q5 dispatch during FULL graph capture, followed by real five-token FULL graph execution stats. This is more than an import-only marker; the operator harness also asserts dispatch counters.
- Both runtimes reported 8.23 GiB available KV memory and 226397 KV tokens, with model capacity 212992. Graph capture memory increased from 0.15 to 0.21 GiB. These are runtime reports, not peak-memory or near-limit-request measurements.

`comparison.json` links the summarized figures to raw artifacts. Final host check: launcher SHA and 275 W cap unchanged, Glimmer exited, no running containers and no render-device holders. The isolated worker worktree was integrated and removed. Selective source review found no blockers under the pinned contract; real-device tests, not that review, exposed the compiler and fixture issues above.

**Conclusion:** retain native attention. This experiment does not establish that grouped-query Split-K cannot work; it rejects this Triton implementation. No further kernel tuning, production change, DSpark/DFlash abandonment, or new experiment is implied.

Before serving, installed runner source was checked: `gpu_model_runner.py:2373-2377` uses `self.max_model_len` for `max_seq_len` during graph capture, rather than the dummy/live sequence length. Thus this pinned serving path captures the prototype's host KV bound at 212992; live `seqused_k` remains a device input. The prototype is not claimed compatible with other runners that capture smaller stale host bounds.

Native Lab preview is unavailable in this session (`connect ENOENT .../lab.sock`); no Lab-managed configuration is changed.
