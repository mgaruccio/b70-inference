# Direct router callsite: private build passed; live validation pending

Development candidate only. No timing, promotion, or persistent launcher change.

## Build observed

On `inference-host`, using the pinned Intel 2026 image and unchanged CMake cache:

```sh
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
python3 -B "$ROOT/check-source.py"
python3 -B "$ROOT/cpu-inherited-r1/test-trace.py"
bash "$ROOT/build-candidate.sh"
```

- Source/predicate regression and inherited CPU fixtures passed.
- Full non-build source comparison found exactly one changed file:
  `src/models/nemotron-h.cpp`.
- Exactly **`libllama.so.0.7.0`** changed among six artifacts. SYCL, native test
  executable, server, server implementation, and common library stayed identical
  to the verify-phase parent.
- Offline dependency checks passed. All four protected source trees and 24
  protected artifacts remained unchanged, including the accepted service's build.
- Build used no GPU mounts and did not stop or change the accepted service.

Evidence: `build-attempt-r1/`, `cpu-inherited-r1/tests.log`, `source-check-*.log`,
`source-diff.json`, `{parent,candidate}-binaries.json`, `runtime-precheck.log`, and
`post-build-verification.log`.

## Fresh N9 control observed

After 10 remote CPU/controller checks, the lead ran:

```sh
python3 -B "$ROOT/n9/run-control.py" --lead-run-control
python3 -B "$ROOT/n9/run-control.py" --lead-after-restore
```

The fresh verify-phase parent/callsite candidate pair matched **all 17,856 device
output bytes** with default accumulation on N9. Both retained the inherited
CPU-reference error `0.000000140` against `1e-7`; it was not relabeled as a CPU pass
or used to block the equivalent unchanged path. Independent accepted-service
restoration and current source/binary/seeded-runner checks passed.

Evidence: `n9/cpu-control-r1.log`, `n9/run-r1/{equivalence,analysis,inputs-before,independent-restoration}.json`,
and each arm's raw stdout/stderr. The original stdout copy is compressed losslessly;
the uncompressed original remains on the host.

These checks do not establish an active router mark or correct wider verification.
Inherited marked/prefill native gates, real public qualification, actual
verification-route assertions, traced/untraced reproduction, and independent
restoration of those live runs remain required before a fidelity claim.

## Preserved pre-start validation failure

`qualification-r1` created but never started `RCA-A`. Its Config environment had
the exact same unique keys and values as the frozen baseline, in a different array
order. The full-list equality check rejected that serialization difference. The
original was restarted and its READY smoke passed; the controller still reported
restoration failure because its preserved-candidate check hit the same assertion.
All original records and the executed scripts remain preserved.

The retry compares unique environment key/value mappings, rejecting duplicates,
missing/extra keys and changed values without rewriting raw records. It uses fresh
`qualification-r2` output and `RCA2-*` names; no model or binary change is involved.
The same saved record also exposed reordered bind strings. Their exact contents
and multiplicity are checked independently of order, while complete Mounts and
all remaining HostConfig fields stay strict. Independent baseline restoration
passed with no errors; offline rechecking of the saved record confirmed the full
environment, bind and mount contracts without changing that record.

## Fresh qualification r2

All 75 real requests completed with no collection errors. Independent restoration
passed. A matched all 25 cases; W3 matched 23/25; W7 matched 21/25. Every arm passed
all 21 natural-stop functional cases and exercised both fixed-length cases and both
real-ID diagnostics. Each speculative arm observed rejected drafts in both diagnostics.

Against fresh A, the W3 short diagnostic first differs at emitted ID 40
(candidate 10533, A 1046); its long diagnostic matches all 128 IDs. W7 first differs
at short ID 40 (10533 versus 1046) and long ID 14 (8617 versus 1429). Thus neither
width is fidelity-qualified. These are measured outputs, not proof of which
operator causes the residual divergence. The completed live trace below verifies the router intervention.

Artifacts: `qualification-r2/analysis.json`, the three raw role directories,
`qualification-r2/independent-restoration.json`, and `qualification-r2*.log`.
The failed pre-start r1 and its independent baseline restoration are retained too.
Both archived original stdout prefixes were decompressed and checked against their
recorded byte counts and SHA-256 values (546,693,774 and 554,977,013 bytes).

## Live callsite intervention verified

`trace-short55-r1` completed and independently restored the original service.
Both traced roles reproduced their matching fresh untraced roles. The first-frame
checker verified the full 55-token prompt and common first emitted ID, unchanged
partial-output prefill, and all five target routers (blocks 1, 8, 21, 37, 48).
Actual W3 verification at positions 55..58 carried accumulator precision 10 and
four ordered singleton `MUL_MAT_SYCL` / `LIBRARY_MKL_F32` pairs per router; A and
prefill remained unmarked. This intervention is now proven active in the real model.

The sampled prefill row matched all 524,288 bytes. Verification still differed in
131,070 of 131,072 F32 words, with maximum finite absolute delta
0.18981361389160156. This is residual numerical divergence, not a fidelity pass or
proof that any particular remaining operator caused it. See
`trace-short55-r1/first-frame-comparison.json`, `analysis.json`, the raw captures,
and `independent-restoration.json`.

Read-only source inspection identified two existing singleton paths limited to
n=2: Q8_0 latent/attention DMMV and IQ2_XXS routed expert matmul. Their wider paths
use different arithmetic (MMVQ and grouped GEMM). The next bounded hypothesis is
to reuse these existing singleton paths for small full-output verification only,
retaining partial-output prefill and the original n=1/2 behavior. This is not yet
implemented or established as the cause of the remaining divergence.
