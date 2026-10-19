# Bounded A/W3 kernel attribution (not a performance benchmark)

The Q8 batched candidate passed its native/public gates, but achieved only
33.1548/31.0822 server tok/s versus A 44.7843/43.1923 on short55/long8270.
No promotion. This case identifies remaining costs before another kernel change.

## Fresh primary research and inventory

- https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md
  documents Level Zero host/device timing, submission intervals, verbose shapes,
  Chrome kernel events, and CPU build options. Host/device durations overlap;
  do not add them or use instrumented latency as headline performance.
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/performance-tools-in-intel-profiling-tools.html
  describes external GPU profiling. Route/debug records are not GPU timers.
- https://intel.github.io/pti-gpu/knownissues.html records tracing/driver limits.
- Official release tag resolves to commit
  `887bba6e28ce84cc0d3813ef876e24add107c318` (PTI 1.0.0).

No unitrace/VTune executable was found on the host PATH or in the frozen SDK
image's `/opt/intel`, `/usr/local/bin`, or `/usr/bin` inventory. The research
suggestion that a profiler was already installed was not supported by inventory.
Build the official external tool in a disposable CPU-only SDK container, not
inside the application source and not as a host package. Its scratch is a bounded
2 GiB tmpfs; container CPU/memory limits are 6 cores/8 GiB. Disable MPI, ITT, XPTI,
and OpenCL integrations: first pass needs only Level Zero kernel/host timing.

## Defined real journey (pending)

Environment: inference-host/B70, unchanged validated Q8-r1 candidate and model,
frozen SDK image, original service initially healthy and sole GPU owner.

1. Build and dependency-check the profiler with `bash build-profiler.sh`; record
   actual tool/library hashes and supported CLI flags. No GPU access during build.
2. Reuse the existing native startup gates, container/PID/image/source checks,
   HTTP/SSE capture, and finally-path restoration. Launch fresh A and W3 cells,
   wrapping only the final server executable in unitrace. The native gate commands
   themselves remain unchanged. Explicitly validate the two extra tool/output
   mounts and wrapped command; never rewrite observed records.
3. Send one unchanged short55 greedy128 request to each public server. Compare
   exact output/finish/usage against frozen qualified A and each other. This is
   attribution with an already-qualified backend, not a fresh 25-case qualification
   claim and not a timing-eligible ABBA run.
4. Gracefully stop each server so the profiler flushes. Retain stdout/stderr,
   actual HTTP/SSE, kernel names/counts/shapes/durations, host API intervals,
   profiler errors, native logs, and raw restoration evidence. No hardware metric
   sampling or extra privileges. Do not interpret instrumented throughput as speed.
5. Restore original READY/health/identity and sole GPU ownership, recheck all
   protected source/binary proofs, then analyze. Stop all profiling containers;
   no persistent launcher/service change or application kernel edit.

Source hypotheses to measure: IQ2 width-four token-serial dispatch, NVFP4 MoE
ID-copy/wait and per-expert fallback, and draft callback cost. Existing acceptance
logs already show position rates short `(0.791, 0.674, 0.488)` and long
`(0.800, 0.667, 0.356)`; do not assume every third draft token is rejected.

## Executable capture and collection window

```bash
python3 -B test-profile.py -v
python3 -B run-profile.py --lead-capture
```

The official tool supports an external named-session control interface. Start
collection paused, then issue `unitrace --resume <session>` inside the existing
cell immediately before its real request and `--pause <session>` in `finally`.
This excludes model initialization/warmup from the collection window; the request
still includes short55 prefill and greedy128 generation. No application code,
ITT calls, queue properties, or benchmark-request parameters change. Unit tests
check resume/request/pause ordering, including request failure. See the pinned
unitrace README's temporal/out-of-application control section.

The CPU build succeeded and actual hashes are pinned in `run-profile.py`. Its
out-of-tree generated embedded Git-version headers emitted Git discovery warnings;
the explicit source checkout assertion passed and the tool/library linked with
all dependencies resolved. No missing embedded version is treated as a source pin.
The reviewed lifecycle/mount binding was subsequently narrowed with this official
collection-window control and tested separately before execution.
