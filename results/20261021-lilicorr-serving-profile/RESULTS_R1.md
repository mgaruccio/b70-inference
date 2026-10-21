# Serving profiling — failed qualification; work paused and archived

**Development diagnostic only. No completed GPU profile, speedup, or promotion.**
The user paused this model investigation on 2026-10-21. No more profiling or model
experiments are queued. See the [model archive](../../docs/archives/20261021-nemotron35-bolt-lilicorr.md).

## What actually ran

### Clean controls passed; traced serving did not start

`bash lead-run.sh` reused the current qualified Q8-target/IQ2batch4 backend,
frozen oneAPI2026 image and original exclusive lifecycle. The planned sequence
was clean A1, clean W3a, traced W3b, traced A2. Only the first two finished.

- **50 public qualification jobs and 12 measured responses** from A1/W3a passed
  independent raw-artifact rereading using the unchanged native/public validators
  (`recheck-clean-controls.py`, `capture-r1/clean-controls-analysis.json`).
- W3 clean native gates retained Q8 **82/54 byte-exact** and IQ2 **34/16**, and
  measured W3 counts remained short129/84 and long133/82.
- unitrace 2.3 wrapped the actual native commands with
  `-d -h -s -v --chrome-kernel-logging`, not just the eventual server.
- Traced W3b's first 56-case NVFP4 native gate stopped progressing after the
  `LIBRARY_MKL_F32` marker for `m=37,n=9,k=64`. Readiness timed out. The process
  required forced termination after the normal Docker stop grace period:
  exit 137, `OOMKilled=false`.
- It never reached traced Q8/IQ2 checks or a traced public request. There is no
  qualified GPU breakdown or tracing-overhead comparison from this attempt.

The last marker establishes an observed boundary, **not a proven root cause**
in oneMKL, SYCL, the driver or the profiler.

### Installed unitrace 2.4 reproduced the blocker

Fresh primary-source comparison found no demonstrated fix for this exact hang;
see [NATIVE_PILOT_240.md](NATIVE_PILOT_240.md). No tool was installed or rebuilt.

`bash lead-native-pilot-240.sh` ran a bounded native-only pair on identical
sources, binaries and 56 cases:

- Clean: **56/56 passed**, including the n9 library path.
- Traced with the same flags: **180-second deadline exceeded**. The last marker
  was again `LIBRARY_MKL_F32`, `m=37,n=9,k=64`.
- Frozen SDK dependencies resolved, including the already-present XPTI library.
  No SDK, driver, queue, graph, arithmetic or adapter workaround was applied.
- The paired pilot failed qualification. It is not a serving profile or a reason
  to bypass a native gate. Partial traces are retained but not interpreted.

## Restoration

Both GPU attempts restored the original in finally and passed **independent new
SSH/Python** checks. Latest original Id:
`7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`.
The post-pilot observation was health `ok`, sole observed GPU owner PID 1154077.

Archive preparation performs another independent check, retaining it as
`archive-live-state.json` / `.json.gz`, summarized in [archive-status.json](archive-status.json).
Protected source/binary, original identity, stopped experiments, health and sole
observed GPU ownership are checked. Only the existing exact-stopped-native-Id
false/null OOM comparator is used for that native container; original comparisons
remain strict. No persistent service configuration was promoted or edited.

## Clean controls are not a replacement for profiling

Descriptive medians from **three measured samples per point/configuration**:

| Point | Clean NoDraft decode | Clean W3 decode |
|---|---:|---:|
| short55 |44.9159 tok/s|41.1774 tok/s|
| long8270 |43.1561 tok/s|38.8369 tok/s|

These are only the completed controls, not complete ABBA or profiler-overhead
results. All samples, including the slower first short W3 response, are retained.
`clean-controls-summary.json` also differences the existing cumulative
`impl->draft` host timer: roughly 304–307 ms short and 333 ms long per response.
That timer includes waits, excludes process callbacks, and is **neither pure GPU
time nor all drafter overhead**. It does not establish a full-serving bottleneck.

## Code, checks and failed attempts

- Profiling adapters reuse existing controllers; no application source/build or
  model weights changed. Existing source/binary proofs remained enabled.
- Local checks cover 28 tests: 9 adapter, 10 trace-analysis, 6 serving-analysis,
  and 3 native-pilot tests. Remote preflight ran 18 initial tests; the native pilot
  ran its 3 CPU tests remotely. These are supplementary checks, not GPU qualification.
- The trace analyzer streams the pinned JSON format, rejects incomplete traces,
  retains overlap and unattributed time, and requires calibrated request windows.
  The end-to-end trace postprocessor was **not successfully exercised on a new
  qualified capture** because neither instrumented native run qualified.
- The older profiler's epoch-like timestamps are not REALTIME; the retained
  BOOTTIME/MONOTONIC_RAW samples and pinned UniTimer source explain the correction.
- An initial SSH invocation used rsync-style IPv6 brackets and failed before
  connecting or changing service state (`launch-transport-attempt-r1.txt`).
  SSH/rsync argument forms were separated before the real capture.
- Review's predicted bound-method `__globals__` AttributeError did not reproduce
  in CPython; explicit `__func__.__globals__` and a direct bound-warmup-path test
  were nevertheless added before the GPU run. No traced warmup actually ran.
- Native wrapper rewriting and namespace binding were corrected/tested before
  execution; the qualified source/build namespace is not rebound to this package.
- NativeLab preview/patch tools were unavailable (`lab.sock` ENOENT); no action
  was presented as confirmed through that unavailable UI.

## Evidence and exact entry points

```bash
python3 -B run-profile.py --lead-preflight
bash lead-run.sh
python3 -B recheck-clean-controls.py
bash lead-native-pilot-240.sh
```

The launchers run capture and restoration in separate SSH/Python processes, even
on failure. They use strict host-key checking and no agent/control forwarding.
Original failed records are not rewritten as passes.

Raw originals remain at this local result directory and
`/home/mike/b70-evals/20261021-lilicorr-serving-profile/` on inference-host:
`capture-r1/`, `native-pilot-240-r1/`, `preflight-r1/`.
[archive-retention.json](archive-retention.json) records cross-host SHA256 checks,
[the manifest](archive-raw-remote-manifest.json) lists raw files, and selected
lossless gzip sidecars are committed. Large cumulative original-service logs are
retained outside Git, not deleted. Rehydrate a sidecar to its original filename
when an existing reader requires it.

**Final state: paused, evidence preserved, original service restored; full-serving
GPU attribution and the dramatic-speedup objective remain unresolved.**
