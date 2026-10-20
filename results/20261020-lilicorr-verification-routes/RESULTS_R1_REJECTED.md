# Verification route capture r1 — rejected at startup

**This capture failed. It provides no public verification-route or performance result.**

The user approved a development diagnostic capture after the read-only verification
inspection. The original Q8 runner/controller was reused with private OUT/PREFIX;
source, binaries, model bytes and server process argv were unchanged. The private
prelude enabled `BOLT_VERIFY_CAPTURE_DIR` while keeping `LLAMA_TRACE` unset.

The configuration-only validation passed, but the first A1 startup was rejected by
the inherited diagnostics-off contract:

```text
AssertionError: ('serving diagnostic marker', 'BOLT_ROUTE')
```

This was a capture-policy mismatch, not a target or drafter speed/fidelity result.
No public prevalidation or warmup/measured jobs ran. The unchanged analyzer remains
`status=failed` with 29 completion/missing-evidence errors; its 24 row entries are
unfulfilled planned measurements, not 24 observed responses. Nothing was filtered,
relabeled as passing, or promoted.

## Restoration and retention

The existing controller performed cleanup. A separate process then ran the original
live restoration checks. `capture-r1/analysis-live-restoration.json` reports:

- `passed=true`, `errors=[]`, `independently_live=true`;
- health `ok`; sole observed GPU owner/service `nemotron-bolt-probe`;
- original identity, source/binaries, qualified proofs and stopped experiment checks.

The follow-up process exited nonzero only after restoration passed, when the
unchanged analyzer rejected the incomplete capture. Preserve both observations.
All 48 remote run files (780,065,219 bytes) were copied and SHA256-verified against
`capture-r1-transfer-sha256.json`. Raw originals remain locally and remotely.
Eighteen verified lossless text copies retain 779,786,961 raw bytes in 47,639,198
gzip bytes. Decompress the `.gz` files to recover original bytes and filenames.

Exact invocation: `run-capture-r1.sh`, modes `validate`, `capture`, then `after`.
Process output: `capture-r1-process.log` and `after-r1-process.log` (or `.gz`).
Remote run: `/home/mike/b70-evals/20261019-lilicorr-q8-drafter/verification-routes-20261020-r1`.
No persistent launcher, application source, binary or model was changed.

## Scope correction before any retry

`BOLT_VERIFY_CAPTURE_DIR` is not a route-only switch. Besides host route/server
records, `common/sampling.cpp:24–58,664–700` writes already-available CPU sampler
logits. The attempted path had no capture directory, so later sampling would also
have failed its file-creation observer; startup rejection prevented reaching that
point. There was no backend readback or new profiling installation.

The user subsequently approved the existing full diagnostic mode and autonomous
continuation. A retry must use the already-existing trace-aware startup/PID/log
validators in `trace-probe.py`, a proper capture mount, unchanged raw records,
the same native/public/exact-output/acceptance checks, and clean original-service
restoration. It must not disable the clean-benchmark guard or treat instrumented
timings as a speed comparison. `trace-storage-plan.json` records the storage plan.
