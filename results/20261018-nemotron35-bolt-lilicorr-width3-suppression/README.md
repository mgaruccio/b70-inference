# W3 same-width suppression control — preparation only

**Development-tier isolation, not timing or promotion.** User goal: keep working toward faster speculation. The completed fixed-candidate width sweep had all 25 fresh-A checks pass; W3/W7 passed 20 quality checks plus the anchor but differed at actual diagnostic ID 116 (short) and 14 (long). Neither width is timing-eligible. This preparation supplies ONE control, not a runtime patch, new benchmark framework, or causal conclusion. No SSH, helper imports/tests, API, native, or GPU execution was performed locally; remote CPU and live gates remain pending with the lead.

Only this new directory may change. Reuse `../20261018-nemotron35-bolt-lilicorr-width-sweep/` at **74ebc785** (the two used controller/contract content hashes are enforced in `probe.py`) read-only, including its transport, public requests, lifecycle, source/binary guards and frozen helpers. Previous cases, containers, source maps, native/build output, launchers, weights and hardware settings remain untouched.

## Variable and interpretation

Derive exactly `old contracts.make_create('W3', h)`, changing only the container name and one **server-side** `--spec-draft-n-min 0` to `4`. Max stays **3**, p-min **0**; the native command prefix is byte-identical. Cell `w3-suppressed` uses the old prefix plus that exact name so inherited cleanup resolves the correct container. Only this new name is collision-checked; preserved A/W3/W7 names are allowed, never hidden/reused/removed. Live output is exclusive `run-r1/`, never resumed or overwritten.

Pinned host source `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1/common/speculative.cpp` runs LiLiCorr's full block 8, extracts `n_max` proposals, then clears a result shorter than `n_min`. Max3/min4 fits the trained limit 7 without a clamp. **Max7/min8 clamps and is NOT a valid suppression control; do not run it.** W3/min4 keeps target `n_rs_seq=3`, draft `n_rs_seq=0`, target `n_outputs_max=4`, drafter/features loaded and the same prefill routing. Target ncol4 is not draft max3. Router dimensions 240/304/368/432/496 are not draft depths.

Actual PID argv must equal the derived argv, including all existing diagnostic-off guards. Unaltered startup must pass shared native **385**, OUTPROD **190**, support **9**, IQ2/Q8/attention/view/router and model/off-log gates, then `[3,0]` recurrent sequence widths, max/min `(3,4)`, block8, output capacity4, F32 R/S, backend flags, and no clamp. The old min0 validator is not called on suppression evidence; its narrow validation is copied with min4 expected. Nothing normalizes live PID or startup evidence.

Suppression is **unproven until independent post-restoration analysis** finds real cumulative `#calls(b,g,a) ... #gen tokens ... #acc tokens ... dur(b,g,a)` records with zero generated/accepted tokens and positive generation calls AND generation duration. Missing HTTP draft fields are not zero proof. Present HTTP counts must also be zero. Raw logs before stop, after stop and final preserved logs remain available; the shared `log_duration_stats` parser is reused, not durations inferred from wall time. Durations establish work occurred, not performance.

If suppressed W3 matches frozen A, active verification/rollback is implicated, not proven causal. Otherwise changed state/prefill/drafter presence is also implicated, not router proof. Assistant-prefix replay would change floating-point state history; this same-width control is the less-confounded initial isolation. No raw F32 capture/equality, synthetic acceptance, retry, replacement, speed interpretation or deployment promotion is authorized.

## Research already completed before preparation

Fresh external research was supplied by the lead; pinned local source is authoritative for this candidate:

- <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md> — speculative work/acceptance must be observed, not inferred from absent HTTP counters.
- <https://github.com/ggml-org/llama.cpp/pull/30506> — LiLiCorr trained block/candidate behavior; depth is distinct from router output dimension.
- <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/domains/blas/gemm> and <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html> — numerical routing/state-history differences are hypotheses only, not evidence that the router caused these mismatches.

## Remote CPU gate — lead only

Stage the four preparation files at `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression/` on `inference-host`. Do not overwrite any prior case. Existing remote `width-sweep/qualification-r1/` must contain the real A/W3/W7 results and successful independent restoration/analysis. The local mirror may be called `remote-qualification-r1`; the probe consumes the original remote `qualification-r1`, not a substitute fixture.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no \
    -o ControlPath=none -o ForwardAgent=no mike@inference-host \
    'bash /home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression/cpu-checks.sh'
```

The CPU runner preserves its exact command/environment, stdout/stderr, exit status and before/after freeze under `worker-scratch/cpu-XXXXXXXX/`. Only a successful run writes `preparation/tested-code.json`. It hashes all new files/tests, used old helpers, actual frozen A and fresh A/W3 request/response files, recorded startup/PID fixtures, both protected source maps and both five-binary sets. Tests use real recorded W3 startup with an **in-memory synthetic min4 change**, explicitly not live evidence. They cover launch diff, actual PID/startup negatives (width/min/clamp/off guards), only-new-name collisions, four frozen requests and real reference divergences, missing/positive counter and zero-work rejection, mismatch continuation vs infrastructure abort, inherited cleanup/signals and independent-restoration ordering. Tests cannot execute subprocesses/API/native work or read hardware. `unittest.main(exit=False)` allows the runner to finish the post-test freeze. A failed gate blocks live work; retain the failed attempt.

## Predeclared live E2E — lead only, after CPU passes

Environment/preconditions: `inference-host`, the unchanged frozen runtime/target/drafter, exact accepted healthy `nemotron-bolt-probe`, original hardware/caps, sole live DRM owner, root ownership visibility, 4 GiB reserve plus 1 GiB evidence allowance, frozen code/reference/source/binary checks green, and absent new control name plus absent `run-r1/`. No other workload or container is started.

One fresh control executes exactly these four unchanged `old contracts.qualification_jobs(h)` entries, in order:

1. `short/ordinary` — streaming short55, 128 tokens, ignore EOS.
2. `short/diagnostic` — actual emitted IDs/pieces for the same frozen request.
3. `long/ordinary` — streaming long8270, 128 tokens, ignore EOS.
4. `long/diagnostic` — actual emitted IDs/pieces for the same frozen request.

All retain `cache_prompt=false`, temperature0, seed42 and original payload bytes/settings. Each uses inherited public `/apply-template` and `/tokenize` prompt-ID checks then `/v1/chat/completions`. Diagnostic output IDs come only from `choices[0].logprobs.content[*].id`, never retokenized text. Compare exact text/finish/usage and diagnostic IDs/pieces against frozen A, already-qualified fresh A, and actual active W3. Prior A was repeated exactly in the six-cell run and all 25 fresh-A qualification checks; no new A or 20-quality replay is needed for this bounded isolation.

On the remote host, run the two commands **separately**, inspecting successful restoration before analysis (use the lead's existing background-job mechanism, not a local Pi inference environment):

```bash
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression
python3 -B "$R/probe.py" --lead-run-suppression 2>&1 | tee -a "$R/controller.log"
# Only after the capture command succeeds and run-r1/summary.json reports restored:
python3 -B "$R/probe.py" --lead-after-restore 2>&1 | tee -a "$R/analysis.log"
```

Expected process: valid fidelity differences retain full responses and continue all four requests. Input/API/transport/native/source/binary/isolation failures abort immediately through inherited capture/finally cleanup. SIGINT/TERM/HUP use that corrected cleanup too. Always stop and preserve the new control and restore exact accepted ORIGINAL identity, `Config`, `HostConfig`, complete canonical mounts, READY/natural-stop smoke, health, sole live container/DRM owner, hardware/caps and source/five-binary hashes. No container deletion, retry/replacement or source/build mutation is cleanup.

Retain `run-r1/` in full: declaration and tested-code freeze; launch diff; raw native/startup/PID/libraries/buffers; request bodies and HTTP records; full output/comparison artifacts; `before-stop.*.log`, `server.*.log`, `restored-final.*.log`; cumulative statistics; hardware/ownership/capacity observations; failure records; preserved container; restoration smoke and live checks. Separate analysis performs a fresh live restoration check before `analysis.json` can assert suppression, and retains full comparisons/first actual-ID differences against all three references. No statistics or suppression proof means **inconclusive**, not success.

Preparation is not a standard-compliant performance publication. No benchmark/publication checklist is claimed complete, and no performance comparison will be made from this control.
