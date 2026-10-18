# Seeded router N=9 control preparation

**Test-only development preparation; not a benchmark result and not a qualification.**

This case prepares a paired parent/candidate `test-backend-ops` executable for the
single known failure:

```text
router_columns8=1,router_single2=1,weight=blk.48.ffn_gate_inp.weight,
layout=0,type_a=f32,type_b=f32,m=496,n=9
```

The test-only source patch keeps the predeclared per-chunk engine seed
`0x5A17u + start` and, only for the selected contiguous f32 device tensor,
emits the full 17,856-byte SYCL output as one `BOLT_N9_OUTPUT_HEX=` line before
numeric verdict handling. No tolerance, shape, fixture, runtime, or backend source
is changed. The fixed seed makes parent and candidate operands replayable; it does
not reproduce the original `std::random_device` operands. Do not choose or retry
another seed after observing results.

## Lead CPU build

Run only on `inference-host`, using the existing Intel 2026.0.0 image and oneDNN
mount, with no host installs or GPU mounts:

```text
bash /home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control/build-test.sh
```

The script privately copies the router-columns8 parent with reflink into the fresh
`router-n9-control-r2` tree, applies `seeded-test.patch` at fuzz 0, builds only
`test-backend-ops -j6`, stores r2 manifests under `r2/`, performs the five-binary/
three-source protected checks, and runs an `ldd` dependency precheck. Failed
attempts are retained and destination/container name collisions abort.

## Lead-owned native comparison

The lead must run two fresh native-only containers, parent then candidate, with
identical fixed-seed binary and selected filter, no model load/API, and the same
DNN/OPT/VMM/ESIMD/FUSION/GRAPH settings. Mount the selected backend tree at
`/src:ro`, plus the candidate `build-sycl/bin/test-backend-ops` overlay at the
same path. Retain actual route, error/summary, raw logs, mounts, and dependency
output. A numerical failure is retained as data; infrastructure aborts are not
rerolled. Parent and candidate both failing does not qualify the runtime.

The public boundary is the native `test-backend-ops` CLI, not a mock or model
request. Prerequisites: a successful test-only build, frozen protected sources
and binaries, unchanged B70 hardware, fresh container names, and the accepted
service as sole DRM owner. `run-control.py` reuses the frozen restoration
lifecycle, stops that service for the two native arms, and requires no DRM owner
between arms. It preserves raw logs, actual container bindings, dependency
resolution, dispatch pairs and the numerical verdict. A valid numerical failure
is data, not an infrastructure success claim or a qualification waiver.

Run on `inference-host`:

```sh
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control
python3 -B "$R/test-control.py"
python3 -B "$R/run-control.py" --lead-run-control
python3 -B "$R/run-control.py" --lead-after-restore
```

Expected: exactly one M=496/N=9 fixture per backend, the same seeded executable,
unsplit `MUL_MAT_SYCL` + `LIBRARY_DNNL_F32` dispatch, and the unchanged `1e-7`
tolerance. Each native arm must emit exactly one well-formed 17,856-byte marker;
the parent and candidate device bytes are compared literally, not by rounded error.
The proof is bound to current parent/candidate backend hashes, the test source and
binary hashes, and the preserved raw logs. A missing, malformed, duplicate, or
different marker fails closed. Cleanup restores the exact accepted container's
identity, configuration, mounts, health, READY/stop smoke and exclusive DRM
ownership; it leaves the two completed diagnostic containers stopped. Evidence
stays in `run-r2/`, including `equivalence.json`, `analysis.json` and
`independent-restoration.json`. No timing or promotion is permitted.

## Preserved r1 result — blocked, not qualified

The lead completed the CPU-only build, dependency precheck, eight remote CPU
checks, and both native arms on `inference-host` / Intel Arc Pro B70. Only the
private `test-backend-ops` executable changed; protected runtime sources and
binaries remained unchanged.

Both parent and candidate exited 1 with reported NMSE **0.000000140 >
0.000000100**, using the same unsplit oneDNN route. The runner exited 0 because
it captured both numerical failures and restored the accepted service; that
is **not** a qualification pass. This establishes parent reproduction under the
predeclared seed, not byte-identical outputs or replay of the original random
operands. No seed retry or tolerance change was made.

- Paired results: `run-r1/analysis.json`; raw per-arm `stdout.log` / `stderr.log`.
- Exact commands and bindings: each arm's `create-argv.json` and `container-before.json`.
- Test executable and protected backend hashes: `candidate-binaries.json`,
  `run-r1/inputs-before.json`, and `run-r1/hash-checks-before.json`.
- Build and CPU checks: `build-attempt-r1/build.log`, `post-build-verification.log`,
  `runtime-precheck.log`, and `cpu-tests-r2.log`.
- Independent restoration: `run-r1/independent-restoration.json` passed with no
  errors. `final-container-state.txt` confirms the accepted service running and
  errors. `final-container-state.txt` confirms the accepted service running and
  both native diagnostic containers stopped.

The r1 numeric failures remain inherited diagnostic data, not passes. The r2
sequence above must complete the literal-byte proof before the public r3 gate is
staged; no oneDNN numerical remediation or tolerance relaxation is part of this
case.

The original router-columns8 R2 attempt remains **61/62**, with all 404 route
records validated and 35 split cases observed. Its sole numerical failure was
this M=496/N=9 control; no candidate model qualification requests ran. See
`../20261018-nemotron35-bolt-lilicorr-router-columns8/qualification-r2/failure-analysis.json`.

Public qualification, timing, and promotion remain blocked. Resolving the
inherited oneDNN failure or changing the out-of-scope control's acceptance
criterion requires a separate scope decision. These are diagnostic results,
not standard-compliant performance evidence.

Repository archival follows the existing gzip convention for the two large
`ORIGINAL/stopped-prefix.stdout.log` files (this control and router-columns8
R2). Their `.gz` files decompress to the bytes and SHA-256 recorded in each
`stopped-log-prefix.json`. The uncompressed originals remain on
`inference-host`; no runtime evidence was truncated or normalized.
