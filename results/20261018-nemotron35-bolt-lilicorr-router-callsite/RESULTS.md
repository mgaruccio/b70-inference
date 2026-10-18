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

This establishes the intended source/library delta, not an active router mark or
correct wider verification. Fresh native/N9 controls, real public qualification,
actual verification-route assertions, traced/untraced reproduction, and independent
service restoration remain required before any fidelity claim.
