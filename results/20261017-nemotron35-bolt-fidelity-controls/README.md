# Bolt matched repeatability and draft-off controls

Status: completed six-launch development diagnostic; accepted runtime is not repeatable on this prompt, so speculative fidelity remains unresolved. Not a speed benchmark. User requested continuing after the isolated OUT_PROD fix passed numerical checks but LiLiCorr output differed after195 characters, with accepted cross-launch outputs also differing after346 characters. Prior evidence: `../20261017-nemotron35-bolt-lilicorr-outprod`, commit `852f6328`.

## Frozen controls and scope

No source edits, rebuilds, weights changes, hardware changes, persistent launch edits or promotion. Accepted source is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`; patched source is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1`. Same upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`, original target GGUF, pinned official LiLiCorr draft, original q8 target KV/F32 recurrent state, context65536,batch/ubatch1024,parallel1,ESIMD1/FUSION1/graphs0,boost0,16 CPU caps3801000kHz,275W. Use recorded hashes from prior evidence. Single client, no concurrent requests. Preserve original continuous-batching setting; do not silently introduce an additional runtime change.

All workloads, source inspection, tests and analysis execute on `mike@inference-host` with explicit Bash and SSH `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`. Remote artifacts: `/home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls`. Local repository receives scoped evidence only. Lab preview/patch sockets are already known unavailable; no retries or repair.

## Fresh external research and interface limits

- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-context.cpp — pre-sampling probabilities disable backend sampling when enabled; target defaults already disable it. Speculative acceptance sets probability1.0 and has `TODO: set result.probs`, so speculative logprob0/empty top-k is not confidence evidence.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-common.cpp — chat `logprobs` maps to `n_probs`; no tools+stream combination here.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-task.cpp — non-stream verbose response exposes actual generated token IDs; probability rows include IDs but speculative probabilities are unimplemented.
- https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/tools/server/server-schema.cpp — `return_tokens`, `verbose`, `backend_sampling` request fields.
- https://github.com/ggml-org/llama.cpp/issues/3014 and https://github.com/ggml-org/llama.cpp/issues/7228 — batching/backend operation order can affect greedy reproducibility; these reports are not B70-specific proof or a diagnosis of this case.

## Predeclared matrix and real API process

Exactly six fresh launches, in order: **A0,B0,C0,C1,B1,A1**. A=accepted non-speculative, B=patched non-speculative, C=patched LiLiCorr depth4/n_min0/p_min0/greedy/no draft-backend sampling. Bracketing reduces order confounding but two launches do not exhaust all sources of nondeterminism. All server launches are non-debug (`GGML_SYCL_DEBUG` unset); add identical `LLAMA_TRACE=1` logging to every arm solely for verification-event evidence. Native tests retain their inherited explicit debug settings.

For each launch:

1. Existing native385 passes before the server. Patched arms also run the unchanged prior `native-outprod.sh` (190 numerical plus nine support-contract checks).
2. Fresh real `/v1/chat/completions` gate-v2 API **20/20** must pass. A functional/native/quality failure stops the entire matrix; preserve it, no retry-until-lucky.
3. One full128-token decode warmup using the frozen short measured55-token payload. This diagnostic deliberately uses the same payload as the following probes, all cache_prompt=false. No output1-only warmup.
4. Three streamed requests with byte-equivalent JSON payload to the frozen short55/128 request: temperature0,seed42,thinking disabled,EOS ignored,cache_prompt=false. Verify exactly55 prompt tokens,128 outputs,cache_n=0 and finish reason length. Save every response and all events. **Output mismatches are diagnostic observations, not reasons to retry or truncate the fixed matrix.**
5. One separately labeled diagnostic request with the same messages/sampling/output limit, but stream=false,verbose=true,return_tokens=true,logprobs=true,top_logprobs5,backend_sampling=false. Save actual token IDs and complete body. This is a separate request variant, not silently substituted for the frozen stream. Compare token IDs only between matched diagnostic requests. A diagnostic response differing from the preceding streamed output must be reported, not equated. Speculative probability values are unimplemented and must never be interpreted as logits/confidence.
6. Stop and retain that container, verify unchanged accepted/candidate hashes and hardware before advancing. In nested finally, stop all owned experiments, restart the same original accepted container, and verify ID/config/all mount fields/hashes/hardware/health/READY.

Total planned ordinary probe responses:18, plus6 warmups,6 diagnostic responses and120 quality checks. No long-context probe, timed benchmark or fidelity promotion in this matrix. Incidental response timing fields are retained but not performance results.

## Analysis and stop boundary

Analyze within-launch and cross-launch identity before cross-arm comparisons. Record full rendered-output identity, common-prefix character length, diagnostic token-ID identity/first differing index and token pieces, prompt/output/cache counts, actual sampling settings, and draft proposals/accepts/rejection events where exposed. Preserve all outcomes. Baseline variability, patched-off differences, draft-on differences, and diagnostic-instrumentation differences are distinct findings; none alone proves recurrent rollback failure. Do not infer real logits from placeholder speculative probability rows.

The matrix always leaves timing disabled. Any first-divergence logit/replay instrumentation or source change is a separately declared next step based on these results, not an automatic patch or relaxed-fidelity path. Commit/push only scoped scripts, raw responses, logs (lossless gzip copies) and the analysis; keep original raw files remotely and accepted Bolt resident.

## Observed results

The fixed order A0,B0,C0,C1,B1,A1 completed without an inference failure or adaptive rerun. All six native385 gates and all four patched-arm focused gates (190 numerical +9 support checks each) passed. Every fresh API gate passed20/20: **120/120 total**. All18 ordinary responses,6 full-decode warmups and6 separate token-ID diagnostics completed with55 prompt tokens,128 outputs,zero cached tokens and length termination. Operator debug logging was absent from all model probe windows.

Accepted A produced **five distinct outputs across six identical ordinary requests**; patched/off B produced four, patched/on C produced three. Within-launch distinct-output counts, out of three requests, were A0=2,B0=2,C0=2,C1=3,B1=3,A1=3. Thus variability is not just a restart effect. Accepted A0 versus A1 diverged as early as character195, reproducing the boundary that previously triggered the speculative fidelity failure. Cross-arm output overlap exists, but these observations neither prove nor disprove an additional patch/speculation error. Do not describe the candidate as fidelity-passed or the cause as confirmed floating-point noise, rollback, or patch behavior.

Separate diagnostic token-ID comparisons found A0 versus A1 first differing at generated token41 (1-based); B0 and B1 matched all128 IDs; C0 versus C1 first differed at token40; C1 matched B0/B1 exactly. The probability-enabled diagnostic often differed from the ordinary streams: matches among each launch's three ordinary outputs were A0=0,B0=2,C0=0,C1=1,B1=0,A1=0. Therefore these diagnostic sequences are **not** reconstructions of the primary streams.

The four non-spec diagnostics shared the first39 generated IDs. At token40, A0/A1 selected `.` (1046), while B0/B1 selected ` quickly` (10533). Their saved, non-spec top-two logprob gaps were approximately0.08872,0.00529,0.04639,0.17772 for A0,A1,B0,B1 respectively. `verification.json:non_spec_probability_snapshot` retains the actual rows and scope. These separately instrumented observations do not isolate the origin of the variation. No speculative probability was treated as confidence.

Across each draft-on launch's five full-length probes (warmup +3 ordinary +1 diagnostic), C0 reported441/759 accepted/proposed tokens and192 verification events, including118 rejection events; C1 reported439/768 and194 events, including123 rejection events. Logged event totals exactly matched API counters. Histograms and per-request counts are in `verification.json`; these are short-prompt diagnostics, not a speed, quality-equivalence or production acceptance estimate.

**Timing and promotion remain disabled.** Next bounded investigation, not executed here: localize draft-off variability using an exact token-prefix one-token replay around token40 before attributing divergence to LiLiCorr or changing recurrent rollback. No long-context inference or timing benchmark ran.

## Execution, artifacts and restoration

Controller tests:9/9 (`controller-tests-r1.log.gz`). Comparison helper contracts:6/6 (`analysis-contract-tests.log.gz`). The first background launch failed in the local Fish parser before SSH or any model launch; `launch-shell-r1.log.gz` preserves it. Explicit Bash corrected the launcher without consuming or repeating any experiment cell.

Actual remote commands, delivered over the SSH options above with explicit local and remote Bash:

```bash
cd /home/mike/b70-evals/20261017-nemotron35-bolt-fidelity-controls
python3 -B test-controls.py 2>&1 | tee controller-tests-r1.log
python3 -B run-controls.py 2>&1 | tee controller-run-r1.log
python3 -B analyze-and-verify.py 2>&1 | tee analysis-r1.log
# After adding extraction of the already-saved target-only probability rows:
python3 -B analyze-and-verify.py > analysis-r2.log 2>&1
```

Do not rerun `run-controls.py` in place: it intentionally refuses the existing run directory and retained experiment names. The controller and original declaration are frozen in `run-r1/controller.py` and `run-r1/declaration.md`; each cell retains exact Docker argv, native/server logs,20 quality-check responses, ordinary requests/results/events, its diagnostic body and stopped-container inspection. `run-r1/summary.json` records completion and cleanup. `verification.json` contains the reproducible comparisons and independent restoration checks. `analysis-r1` succeeded; `analysis-r2` additionally extracts saved non-spec probability rows, with no new inference.

Logs were compressed using `gzip -n -c`, checked using `gzip -t` and decompressed `cmp` against remote originals, then retrieved without the duplicate uncompressed logs. Retrieved gzip files were checked again. Original raw files remain under the remote artifact root; the repository evidence directory is approximately11MiB.

Independent verification confirms the original accepted container `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e` running with identical configuration, all mount fields, accepted/candidate binary hashes, boot ID, boost0,16 CPU caps3801000kHz and275W. Health is `ok`, restoration smoke returned exactly `READY`, and all six experiment containers are stopped. No persistent launcher, accepted archive, model weights or hardware settings changed.
