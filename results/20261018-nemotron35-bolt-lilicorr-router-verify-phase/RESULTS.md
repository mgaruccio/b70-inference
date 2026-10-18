# Phase-scoped router: no-draft regression fixed; wide verification still differs

The private `router-verify-phase-r1` build restores the frozen no-draft output.
It does **not** qualify W3 or W7, and was not promoted or timed.

## Observed results

- Intel 2026.0.0 build passed. Only `libggml-sycl.so.0`, `libllama.so.0.7.0`,
  and `test-backend-ops` changed among the six recorded artifacts. Protected
  accepted, reference-parent, and earlier columns8 trees remained unchanged.
- Native/parser CPU suite: 11 passed. Integrated public CPU suite: 12 passed.
  Final log: `worker-scratch/public-cpu-3oSBVltf/tests.log`. The earlier failed
  fixture attempt remains recorded separately; it was not live model evidence.
- Existing N9 controller CPU suite: 9 passed (`../20261018-nemotron35-bolt-lilicorr-router-n9-control/cpu-control-r3-final.log`). The separate `cpu-equivalence-r3.log` records a mistaken invocation of a nonexistent file, not an executed test suite.
- Native router gates passed in every arm: old N2-only 30 cases/70 routes/5
  splits; marked N2–N8 61/402/35; unmarked N4/N5 prefill controls 2/4/0.
- Fresh N9 reference/candidate comparison matched **all 17,856 device-output
  bytes**. Both still reported the inherited CPU-reference error `0.000000140`
  against `1e-7`; numerical failure was not relabeled as a CPU pass. Evidence:
  `../20261018-nemotron35-bolt-lilicorr-router-n9-control/run-r3/equivalence.json`.
- All **75 real public requests** completed. `RVP-A` passed **25/25** against
  frozen no-draft results, including ordinary output and actual emission IDs.
- W3 and W7 each passed the 21 natural-stop quality/anchor comparisons, but
  failed short/long ordinary and diagnostic fidelity (four comparisons each).
  Both exercised both diagnostics and observed real draft rejection in each.
- Both widths first differ from fresh/frozen A at emitted-token ordinal **116**
  on the short prompt (`4618` versus `6833`) and **14** on the long prompt
  (`8617` versus `1429`). Short ordinary output differs at character 519
  (`original` versus `primary`); long output differs at character 73.
- Independent accepted-service restoration passed without errors after both
  the N9 pair and the full public run. Experimental containers remain stopped.

## Execution and evidence

On `inference-host`, with the existing B70/oneAPI image, model, cache types,
flash attention, batch/ubatch settings, and frozen request payloads unchanged:

```sh
bash "$PHASE/build-candidate.sh"
python3 -B "$N9/test-control.py"
python3 -B "$N9/run-control.py" --lead-run-control
python3 -B "$N9/run-control.py" --lead-after-restore
bash "$PHASE/cpu-public-gate.sh"
python3 -B "$PHASE/public-gate.py" --lead-run-qualification
python3 -B "$PHASE/public-gate.py" --lead-after-restore
```

`PHASE=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase`;
`N9=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-n9-control`.
Primary public evidence is `qualification-r1/analysis.json`, each arm's raw
HTTP/native/server records, and `qualification-r1/independent-restoration.json`.
`qualified_widths` is empty; timing and promotion remain disabled.

The no-draft checkpoint leak is corrected. The remaining W3/W7 divergences are
unchanged from the earlier width sweep; investigation continues into actual
prompt/verification shapes and the remaining width-dependent compute paths,
without broadening kernels blindly or relaxing fidelity.

The large original stdout prefixes are archived as `.log.gz` in the repository;
uncompressed originals remain on the host. Decompression must match their
existing `stopped-log-prefix.json` byte counts and SHA-256 values.
