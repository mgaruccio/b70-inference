# Singleton r2 results

**W3 qualifies for the unchanged public suite. W7 remains rejected.**
The A/W3 trace reproduced its public results and the compared first-frame raw
F32 score rows are byte-identical. No timing, performance, deployment, or promotion
claim is made. The original service was independently restored after each GPU capture.

## Correction and scope

The strided RHS staging branch in `ggml/src/ggml-sycl/ggml-sycl.cpp` now calls
`quantize_row_q8_1_sycl<quantize_f>` instead of hardcoding `quantize_q8_1`.
This honors the caller's reordered SoA format rather than feeding ordinary AoS
Q8_1 bytes to `MMVQ_REORDERED`. Copy, quantization, and computation retain their
existing queue/order. No routes, fixtures, case counts, tolerances, or required
byte comparisons were changed from r1.

The unchanged r1 test executable reproduced six padded-input NaNs against the
protected parent library. See
[`r1 RESULTS.md`](../20261018-nemotron35-bolt-lilicorr-verify-singletons/RESULTS.md).
From r1 to r2, **only `libggml-sycl.so.0` changed** among the six protected build
artifacts; the native test executable is byte-identical.

## Observed results

- **Build and CPU checks passed:** exact source/build checks, focused quantizer
  regression, 655 host guard cases, four native-parser tests, 13 public-controller
  tests, ten N9 tests, ten first-frame tests, and eight trace-controller tests. Frozen-code
  CPU checks also passed before public and trace execution.
- **N9 baseline equivalence passed:** 17,856 output bytes identical between parent
  and r2. Both still report `0.000000140` against the unchanged `1e-7` CPU-reference
  tolerance. These remain numerical failures, not waived or relabeled passes.
- **Native gates passed on every public arm:** inherited F32 gates, all 82 Q8 and
  34 IQ2 cases, including the six previously failing padded cases. All 54 Q8 and
  16 IQ2 required byte comparisons passed.
- **Public qualification completed all 75 requests:** A passed 25/25; W3 passed
  25/25; W7 failed two fidelity comparisons. Both speculative arms exercised
  rejected-draft behavior; there were no transport/functional failures.
- **W7 failures:** `long/ordinary` and `long/diagnostic`, with 8,270 prompt tokens
  and 128 generated tokens. Diagnostic IDs first differ at ordinal 19 against
  both frozen and fresh A. Finish reason and usage match, but IDs, pieces, and
  text do not. W7 is not qualified.
- **A/W3 trace completed all 46 requests:** 23 per arm, excluding the two long
  cases. Both trace arms reproduced their corresponding public-arm results.
- **First-frame comparison:** the complete 55-token prompt prefix and actual
  emitted IDs were checked, along with the required live routes. Prefill and
  first-verification sampled raw F32 score rows each contain 524,288 bytes;
  both comparisons have zero changed F32 words and zero byte differences.
  This does not prove equality of every intermediate operator input or all
  later frames, nor does it establish operator causality or compare probabilities.

## Execution and evidence

Environment: `inference-host`, the existing B70 device and pinned Intel oneAPI
2026.0.0 image. Candidate:
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`.
Remote experiment root:
`/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-verify-singletons`.

Equivalent commands from that root are below. Actual runs used absolute paths;
trace fixtures ran via `runpy` between the frozen-code checks described below.

```sh
bash build-candidate.sh
python3 -B test-guards.py \
  --parent /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1 \
  --candidate /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2
python3 -B test-native.py
python3 -B test-public-gate.py
python3 -B n9/test-control.py
python3 -B test-first-frame.py
python3 -B n9/run-control.py --lead-run-control
python3 -B n9/run-control.py --lead-after-restore
bash cpu-public-gate.sh
python3 -B public-gate.py --lead-run-qualification
python3 -B public-gate.py --lead-after-restore
python3 -B test-trace-probe.py
python3 -B trace-probe.py --lead-run-trace
python3 -B trace-probe.py --lead-after-restore
python3 -B analyze-first-frame.py
```

Host-only fixture runs used fresh scratch directories under `worker-scratch/`,
with `BOLT_VERIFY_SINGLETONS_CPU_SCRATCH`, `TMPDIR`, and
`PYTHONDONTWRITEBYTECODE=1`; the trace suite additionally requires
`BOLT_VERIFY_SINGLETONS_TRACE_CPU_SCRATCH`. Trace fixtures ran between checks of
the existing frozen tested-code record. The first trace CPU invocation omitted
that additional variable and failed before GPU/service actions; `trace-cpu-r1.log`
is retained. The corrected invocation passed in `trace-cpu-r2.log`; no source or
acceptance rule changed.

Key artifacts:

- `build-attempt-r1/`, `source-diff.json`, `candidate-binaries.json`
- `lead-cpu-tests.log`, `lead-cpu-exit.txt`, `cpu-public-tests.log`
- `preparation/public-tested-code.json`
- `n9/run-r1/`: raw control output, equivalence, and independent restoration
- `qualification-r1/`: all native logs, raw requests/responses, reports, and
  `analysis.json` (`qualified_widths: [3]`)
- `trace-short55-r1/`: raw captures, request evidence, `analysis.json`,
  `first-frame-comparison.json`, and independent restoration

All run containers and failed evidence were retained. The original service was
restored without launcher changes; the disposable integrated worker checkout
was removed. Oversized cumulative original-service log prefixes and fixture
scratch remain preserved remotely rather than committed. No model weights,
accepted baseline, production launcher, or frozen r1 experiment code were changed.
