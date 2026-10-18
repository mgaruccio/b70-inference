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
