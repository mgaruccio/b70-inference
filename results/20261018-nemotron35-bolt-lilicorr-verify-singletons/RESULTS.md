# Singleton experiment r1 — native startup rejected

Development-only and unpromoted. No public qualification or performance result.

## Observed results

- CPU-only native build succeeded. Exactly `libggml-sycl.so.0`,
  `libllama.so.0.7.0`, and `test-backend-ops` changed; protected ancestors passed.
- Host guard/parser/frame checks passed. Integrated public/N9 suites passed
  13 + 10 tests. The final CPU freeze passed in `public-cpu-8jEk6RBK`.
- Fresh N9 parent/candidate output was baseline-equivalent over 17,856 bytes.
  Both retained the numerical failure `0.000000140 > 0.000000100`.
  This is equivalence, not a numerical pass or a qualification waiver.
- Public startup passed the inherited F32 native controls, then Q8 finished
  **76/82**. All **54 required Q8 byte comparisons** had zero mismatches.
- The six `q8-p{0..5}-n4-m1-l2` padded-stride controls dispatched to
  `MMVQ_REORDERED` and produced NaNs against finite CPU results. Their final
  byte markers were absent because the reference comparison failed first.
- IQ2 did not run. No public arm completed and no qualification requests ran.
  The trace journey was not started. The later parent reproduction and source
  diagnosis below identify an inherited RHS-format mismatch; r1 remains rejected.
- Original service restoration and a separate independent live check **passed**.
  No tolerances, controls, persistent service settings, or promotion rules changed.

## Evidence and execution

Remote root: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons`.

Executed on `inference-host`: `bash build-candidate.sh`; host CPU guard/parser/frame
checks; `python3 -B n9/run-control.py --lead-run-control` and
`--lead-after-restore`; `bash cpu-public-gate.sh`; then
`python3 -B public-gate.py --lead-run-qualification`. After startup failure, the
existing `Runtime.verify_restored()` performed the independent live check.

- `build-attempt-r1/`, `source-diff.json`, binary manifests, and build/dependency logs.
- `n9/run-r1/equivalence.json`, `analysis.json`, raw outputs, and restoration records.
- `preparation/public-tested-code.json` records the successful exact CPU freeze.
- `qualification-r1/summary.json`: failed, completed `[]`, restored `true`.
- `qualification-r1/independent-restoration.json`: independently live, passed.
- `qualification-r1/VSA-A/restored-verify-singletons-q8.log`: complete failing native output.
- `public-capture-r1.log` and `public-failed-independent-restoration.log`.

The two cumulative original-service stdout prefixes exceed GitHub's per-file
limit. Those prefixes and their stderr companions remain unchanged remotely and
locally, excluded from Git. Synthetic CPU scratch is also excluded; its original
logs remain under remote `worker-scratch/`.

## Parent reproduction and identified correction

A separate parent-library run of the six padded controls, using the unchanged r1
`test-backend-ops` executable, also returned **0/6 with six NaNs**. Its diagnostic
adapter then rejected Q8 routes through an F32-only parser; it did not execute its
second arm. Correct offline parsing of the saved parent output and the earlier
r1 startup output confirms the same six `MMVQ_REORDERED` failures. This is not a
paired-operand or cross-run byte-equality claim. Independent restoration passed.
See `padded-parent-control-r1/offline-analysis.json` and preserved raw output.

Source diagnosis identifies the non-contiguous RHS branch's hardcoded
`quantize_row_q8_1_sycl<quantize_q8_1>`: it emits ordinary AoS data even when the
selected reordered kernel expects SoA. The contiguous branch uses `quantize_f`.
A fresh r2 is being prepared to use that selected quantizer in the strided branch
as well, without changing routes, fixtures, counts, tolerances, or byte gates.
