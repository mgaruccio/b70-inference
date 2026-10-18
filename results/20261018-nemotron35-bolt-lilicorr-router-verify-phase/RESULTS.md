# Router phase candidate: no-draft restored, intended verification mark inactive

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

The no-draft outputs are restored, but the follow-up below shows that the intended
verification-router mark is inactive in the real Nemotron target graph. The
unchanged W3/W7 divergences therefore do not test the intended intervention.
No broader kernel change or relaxed fidelity criterion follows from this result.

The large original stdout prefixes are archived as `.log.gz` in the repository;
uncompressed originals remain on the host. Decompression must match their
existing `stopped-log-prefix.json` byte counts and SHA-256 values.

## Real A/W3 trace follow-up

`trace-short55-r3` completed **46 public requests**, 23 per arm, in the exact
same-role history specified in [TRACE.md](TRACE.md). Both arms reproduced their
untraced controls. Independent accepted-service restoration passed. CPU fixtures
passed 8/8 (`trace-probe-cpu-r5.log`). Earlier failed attempts remain preserved:
R1 stopped at a socket precheck; R2 created but never started its A container and
rejected Docker's unset/false OOM-disable representation. Both restored the original.

Commands on `inference-host`, after the documented preconditions:

```sh
python3 -B "$PHASE/trace-probe.py" --lead-run-trace
python3 -B "$PHASE/trace-probe.py" --lead-after-restore
python3 -B "$PHASE/analyze-first-frame.py"
```

Observed from the request-scoped raw records:

- Both arms processed the same complete 55-token prefix: batches 5 and 46 with
  no outputs, then batch 4 with output mask `[0,0,0,1]`.
- First post-prompt target batch: A has one token at position 55; W3 has four at
  positions 55–58, all outputs enabled. Both start with token `47789` and select
  `1302` next.
- Prefill vocabulary rows match all **524,288 bytes**. Verification row zero
  differs in **all 131,072 F32 words**, maximum absolute finite delta
  **0.9920768737792969**, with no nonfinite-value differences. First argmax agrees;
  these rows are not numerically equivalent.
- All five observed router weights (blocks 1, 8, 21, 37, 48) remain at
  `accum_prec=0` in W3, using `MUL_MAT_SYCL` / `LIBRARY_MKL_F32`, not the intended
  column-single intervention. Q8 latent/attention and grouped expert routes also
  differ by width; route differences alone do not establish causality.

Source inspection identifies the inactive integration point: the new guard in
`src/llama-graph.cpp` is inside `build_moe_ffn`'s `probs_in == nullptr` branch.
The actual target constructs `router_logits` in `src/models/nemotron-h.cpp:311`
and passes it into that helper at line 326, bypassing the guarded branch. The
similarly named `nemotron-h-moe.cpp` is the MTP path, not this target insertion
point. [Official source](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/nemotron-h.cpp)
independently confirms the direct-router structure. This explains why the intended
mark was inactive; it does **not** establish the cause of the full logit divergence.

Evidence: `trace-short55-r3/analysis.json`, `trace-short55-r3/first-frame-comparison.json`, each
`TRACE3-*` directory's raw HTTP/log/F32 records, and
`trace-short55-r3/independent-restoration.json`. Full raw captures remain on the
host. No throughput claim, promotion, launcher change, or qualified width results
from this diagnostic. The next correction is limited to the actual router call site
in a new private candidate; the frozen candidate and its evidence remain unchanged.
