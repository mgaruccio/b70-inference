# R3 result: corrected control passed; model-output fidelity failed

The inherited M=496/N=9 CPU-reference error is no longer an accuracy prerequisite
for this unchanged path. The user-approved criterion is exact parent/candidate
device-output equivalence for that control. The changed N=2..8 paths retain their
original `1e-7` CPU-reference accuracy requirement. Public model-output fidelity
requirements are unchanged.

## Observed checks

- The private test-only build succeeded; protected runtime sources and binaries
  were unchanged. Nine remote control CPU tests passed, including a same-length
  one-byte mismatch rejection.
- The native N9 control compared all **17,856 device-output bytes** literally:
  parent and candidate matched exactly. The inherited CPU-reference error remains
  recorded as `0.000000140 > 0.000000100` for both; it was not relabeled a numerical
  accuracy pass. See the sibling `router-n9-control/run-r2/equivalence.json`.
- R3 CPU fixtures and the regenerated public CPU freeze passed before launch.
- The real candidate's 61-case native suite passed **61/61**, with its unchanged
  accuracy tolerance and route checks. The N9 control is covered separately by
  the exact comparison, not silently dropped.
- The no-draft candidate completed **22 public generation requests**. All 20
  natural-stop quality cases and the retrieval anchor passed. The next request,
  `short/ordinary` (55 prompt tokens, 128 generated tokens), passed its functional
  checks but did **not** reproduce the frozen baseline text.
- The first differing character is offset **197 (zero-based)**. After
  `A backup is only reliable if it can be restored. `, the candidate starts a new
  paragraph; the baseline continues `If the backup fails during a crisis, ...`.
  This is an output-equivalence failure, not evidence of worse semantic quality.
  No actual-emission token-ID comparison ran for this request.
- Qualification stopped at the no-draft mismatch. Long and diagnostic requests,
  W3 and W7, and timing were not run. No candidate was promoted.
- The exact accepted service was restored. Independent restoration passed with
  no errors; protected sources/binaries and ownership were checked.

## Execution and evidence

On `inference-host`, Intel Arc Pro B70 / xe, Intel oneAPI 2026 image:

```sh
N=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8
python3 -B "$N/test-control.py"
bash "$N/build-test.sh"
python3 -B "$N/run-control.py" --lead-run-control
python3 -B "$N/run-control.py" --lead-after-restore
# After staging the R3 public adapter:
python3 -B "$R/test-trace.py"
bash "$R/cpu-public-gate.sh"
python3 -B "$R/public-gate.py" --lead-run-qualification
```

The final capture command exited 1 on the baseline-output mismatch. Full public
post-capture analysis was not run because capture did not complete. Instead, the
existing `Runtime.verify_restored` was invoked independently and saved to
`qualification-r3/failure-independent-restoration.json`.

Evidence: `qualification-r3/R8R3-A/qualification.json`, the unchanged raw public
request/response artifacts in that directory, `ready-router-columns8-r3-native.log`,
`qualification-r3/summary.json`, `qualification-r3/failure-independent-restoration.json`,
`cpu-r3.log`, `public-capture-r3.log`, and the sibling
`../20261018-nemotron35-bolt-lilicorr-router-n9-control/run-r2/` paired records.
Large original-service prefix logs are archived losslessly as `.gz`, verified
against their existing `stopped-log-prefix.json` byte counts and SHA-256 values.
The uncompressed originals remain on `inference-host`.

**Conclusion:** the incorrect N9 blocker is fixed. This candidate still does not
meet the real model-output fidelity contract, even with drafting disabled.
There is no speed or promotion claim.
