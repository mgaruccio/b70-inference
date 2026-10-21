# Native-first unitrace 2.4 diagnostic — not a serving profile

The full `capture-r1` completed clean A1/W3a, then stalled during traced W3b's
first 56-case NVFP4 native gate, at the n9 `LIBRARY_MKL_F32` call. It did not
reach Q8/IQ2 gates or traced serving. The readiness timeout was followed by a
normal stop attempt and forced termination (exit137, OOMKilled=false). Original
restoration and independent new SSH/Python verification passed. No native gate
is bypassed and no attribution is inferred from that failed attempt.

## Fresh research

- https://raw.githubusercontent.com/intel/pti-gpu/pti-1.1.0/tools/unitrace/src/version.h
- https://raw.githubusercontent.com/intel/pti-gpu/pti-1.0.0/tools/unitrace/src/version.h
- https://github.com/intel/pti-gpu/compare/pti-1.0.0...pti-1.1.0.patch
- https://raw.githubusercontent.com/intel/pti-gpu/pti-1.1.0/tools/unitrace/README.md
- https://github.com/intel/pti-gpu/issues/77

PTI1.0.0/1.1.0 identify unitrace2.3/2.4. The relevant append/event collector path
is unchanged; there is **no demonstrated fix** for this exact hang. The installed
2.4 binary is only a bounded tool-version experiment, not an SDK/queue/adapter
workaround. Its executable SHA is
`8945080853cce50d87be4b73982c261fbe204475139cbd7ed0278666e51e5fcc`;
its library SHA is
`c55e5df84cbcd4215d0756e775cc71fdaeb762343a6c196d5e4dac02d05af3bb`.

## Defined real native journey

Preconditions: original healthy/sole owner, unchanged frozen oneAPI2026 image,
qualified Q8-target/IQ2batch4 sources and binaries, all original/protected proofs,
20GiB disk floor and read-only installed profiler240. No package installation.

1. Run `python3 -B test-native-pilot-240.py -v` locally and remotely.
2. `python3 -B native-pilot-240.py --lead-run` reuses the existing exclusive
   lifecycle. Fresh native-only container runs the exact first56-case native
   command once clean, then under installed2.4 with unchanged flags
   `-d -h -s -v --chrome-kernel-logging`.
3. Require both56/56 passes, original unsupported/skip/reorder checks, observed
   n9MKL route and nonempty actual GPU events. The docker attached command has a
   180-second deadline. A timeout is retained, not treated as numerical failure.
4. Existing finally restores the original, source/binary/hardware and GPU owner.
   Only the existing exact-stopped-native-Id false/null OOM comparator is used,
   scoped to stopped native comparisons and restored in finally.
5. A **new SSH/Python process** always runs
   `python3 -B native-pilot-240.py --lead-verify-restored`, even after failure.
6. Retain raw clean/traced logs, tool/dependency identity, command/deadline,
   native container metadata, partial/complete traces and both restorations in
   `native-pilot-240-r1/`. A pass here is **not** full native/public serving
   qualification. Only then consider another complete matched serving capture.

No queue, graph policy, math, model, application binary, driver, SDK, adapter or
persistent launcher change is authorized or performed by this pilot.
