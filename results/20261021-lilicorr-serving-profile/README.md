# LiLiCorr serving profile (profiling only)

User request: **“ok please profile it properly this time”**.

This is a bounded development diagnostic, not an optimization, benchmark, speed
claim, or promotion.  Unitrace is intentionally perturbative: traced request
latencies must not be compared as performance wins.  The only paired numbers
are clean/traced durations used to estimate tracing overhead.

## Research and pinned inventory

Fresh primary research used for this adapter:

- <https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md>
  documents `-d -h -s -v`, `--chrome-kernel-logging`, named sessions,
  `--start-paused`, and external `--resume`/`--pause` control.  Device, host,
  and submission records are attribution evidence; the Chrome trace is for
  correlation, not an unchanged-performance measurement.

The remote preinstalled tool is reused, never built or replaced:

- `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-kernel-profile/tools/unitrace`
  SHA-256 `99ffdad3f3ae86aeee8e52f4b687a9d41e5e67491993952de37c112544a0ff30`.
- `libunitrace_tool.so` SHA-256
  `4bb8d93a64d9091867babd6e14d3b8ed7647b1f4c649802cc1e8570ad0aa2d04`.

The adapter reuses the qualified
`20261020-lilicorr-q8-target-batch4/run-performance.py` bind and its
IQ2batch4 ancestry, native/public validators, source and binary proofs, PID
proof, exclusive singleton lifecycle, and normal `docker stop -t 90` flush.
No application source, build, launcher, queue, hardware-metric, filter, or
persistent service is changed.  The exact original service identity is kept by
the inherited contract: `Id7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`.

## Defined real journey (before remote execution)

**Environment and preconditions:** `inference-host` with the existing B70
GPU, the unchanged qualified Q8 target plus IQ2batch4 sources/binaries and
model, the original service healthy and the sole GPU owner, strict protected
source identity, the pinned tool files above, and at least 20 GiB free on the
host filesystem.  A low-disk preflight aborts before the inherited lifecycle
stops the original service.  No deletion is performed.

1. Run the CPU-only adapter tests locally.
2. On the remote host, run the four cells in this exact order:
   `VSPF-A1` (NoDraft clean), `VSPF-W3a` (W3 clean), `VSPF-W3b` (the same W3
   traced), and `VSPF-A2` (the same NoDraft traced).
3. For every cell, retain the unchanged public 25-job qualification corpus.
   The ordinary timing payloads remain short55 and long8270, temperature 0,
   seed 42, cache off, ignore EOS, output 128; the inherited gate keeps the
   one warmup plus three measured requests per point.  The complete journey is
   100 public qualification jobs plus 8 warmups and 24 measured timing jobs.
   The inherited W3 checks remain 129/84 for short and 133/82 for long.
4. Only W3b and A2 use unitrace.  Their native `test-backend-ops` commands are
   themselves wrapped with the same `-d -h -s -v --chrome-kernel-logging` flags
   before the server starts; clean-cell native gates are not used to claim a
   traced native gate.  Native shell redirections, checks, and log filenames
   are preserved exactly.
5. The traced server starts with collection paused.  Every qualification,
   warmup, and measured inference request resumes its cell session immediately
   before the request and pauses it in `finally`.  Each traced job writes a
   sidecar containing REALTIME, MONOTONIC, BOOTTIME, MONOTONIC_RAW and the clock
   acquisition span before/after resume, request, and pause. The pinned profiler
   uses `(REALTIME - BOOTTIME at initialization) + MONOTONIC_RAW`, not REALTIME.
   Raw profiler timestamps must therefore be calibrated before request matching.
   The source is pinned at
   <https://github.com/intel/pti-gpu/blob/887bba6e28ce84cc0d3813ef876e24add107c318/tools/unitrace/src/unitimer.h>.
   No tool clock setting is changed. Clock-offset stability and matching
   uncertainty are retained with the final analysis.
6. Gracefully stop each server with the inherited 90-second Docker stop.  The
   server profile must retain nonempty complete device, host, submission, and
   Chrome outputs.  Partial outputs are preserved, not deleted, when any step
   fails.
7. Pair clean A1 with traced A2 and clean W3a with traced W3b by point and
   ordinal.  Report this only as tracing-overhead evidence; never interpret it
   as a speed benefit.
8. The inherited finally path restores the original service and checks health,
   identity, sole GPU ownership, hardware, telemetry, and protected source /
   binary proofs.  The launcher then performs an independent second SSH/Python
   restoration check even if capture fails.

The existing false/null OOM comparison remains scoped to the exact stopped
native container identity if the inherited path needs it; this adapter does
not broaden or normalize that comparator.

## Files and outputs

The adapter files are `run-profile.py`, `test-profile.py`, `lead-run.sh`,
`.gitignore`, and this README; `analyze-trace.py`, its tests and
`kernel-rules.json` perform bounded streaming analysis.  Capture evidence is written remotely under
`capture-r1/` (ignored here and retained by the lead): per-cell native/server
logs, public request artifacts, PID/TID/proc evidence, source proofs, sidecars,
trace validation, restoration evidence, and `paired-overhead.json`.

Traced cells mount the installed tool read-only at `/tools` and their own
profile directory read/write at `/profile`.  Native traces are separated into
`/profile/native/native-XX`; server traces are under `/profile/server`.  The
the four required output names are `device_timing.txt`, `host_timing.txt`,
`device_submission.txt`, and `chrome_trace.json`.  Clean cells receive neither
profiling mount and contain no unitrace wrapper.

## Commands

Local checks (no remote execution):

```bash
python3 -B test-profile.py -v
bash -n lead-run.sh
```

The lead capture and independent verification are separate CLI modes:

```bash
python3 -B run-profile.py --lead-preflight # source/hardware/lifecycle checks, no GPU workload or stop
python3 -B run-profile.py --lead-capture
python3 -B run-profile.py --lead-verify-restored
```

`lead-run.sh` uses explicit IPv6 `mike@[fd7a:115c:a1e0::c933:4f38]`,
`HostKeyAlias=inference-host`, strict host-key checking, batch mode, no control
socket, no TTY, no agent/X11/local forwarding, and syncs only these eight
adapter/analysis files before capture.  It always invokes the independent verification
as a separate SSH/Python process, then rsyncs only the bounded `capture-r1/`
result directory.  It does not start a remote run during local testing.

After evidence collection, stop any remaining profiling container through the
inherited restoration path, verify `independent-restoration-live.json`, and
retain the raw result directory for review.  Do not promote the trace or its
perturbed timings to an application configuration.
