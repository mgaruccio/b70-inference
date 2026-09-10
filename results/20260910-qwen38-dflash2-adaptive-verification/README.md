# DFlash2 verification-depth development experiment

Status: **implemented and validated**, opt-in only. **Development tier**, not a standard publishable benchmark or production promotion. Adaptive verification works, but its exploration cost gives no general speedup on128-token outputs. Draft generation remains fixed at7.

## Scope and baseline

Implement opt-in verification prefixes before adding an adaptive controller. Baseline source: commit `0a0ab2e`, DFlash generation K7 and verification K7. Candidate stage: generation stays K7; target verification caps 1, 3, 7. These are draft-token counts (target query includes one additional token). Strict standard rejection, checkpoint/target quantization, sampler and persistent launcher stay unchanged. Matched generation/verification depths remain a subsequent control, not a claimed benefit of verification-only truncation. Do not enable the upstream DSpark-only adaptive-verification flag.

Same-stack configuration: inference-host, one B70, 275 W; pinned image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`, vLLM `73029d42441321b631779db3475031f5ec26dd6c`, legacy V1 async C1. FP16 target compute, FP8 attention KV, BF16 draft compute/KV, existing partial RTN4/G128 draft. Group8, prefill budget2048, max length180224, graphs1/2/4/8, prefix/thinking off. Intentional variable: scheduled verification prefix only. Kmax7 recurrent-state allocation and drafter block8 remain unchanged.

## Fresh primary-source findings

- [Pinned async scheduler](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/core/sched/async_scheduler.py): next-step draft placeholders control scheduled verification length; current in-flight output placeholders are accounted using the actual scheduled prefix. Keep these two quantities separate.
- [Pinned LLM proposer](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/spec_decode/llm_base_proposer.py): `propose()` assigns the requested depth to the drafter before preparing inputs. Therefore leave `SchedulerOutput.num_spec_tokens_to_schedule` at7 for the verification-only experiment, rather than inadvertently changing generation width.
- Existing retained checkpoint configuration explicitly has `is_causal=false`, block8. Shorter generation blocks are not guaranteed to produce the same draft prefixes.
- [Pinned speculative config](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/config/speculative.py) restricts upstream adaptive verification to DSpark. Its [adaptive manager](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py) is design reference only, not the executed legacy runner. DSpark uses confidence survival probabilities and profiled costs; DFlash has no equivalent confidence head here.
- [Pinned GDN metadata](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/attention/backends/gdn_attn.py) requires CPU/device query-length agreement. Do not transplant V2's device-only ragged reallocation; the C1 CPU-scheduled prefix preserves this agreement.

## Defined end-to-end process

Preconditions: no running Docker workload, Glimmer stopped, original launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, power275000000 microwatts. Use explicit Bash over SSH; keep all serving dependencies and GPU execution on inference-host, outside Pi.

1. Run focused CPU overlay/routing tests; fail closed on pinned-source drift and replay tampering. These are supplemental, not the end-to-end gate.
2. In the pinned disposable XPU image, apply existing boundary patch and execute `scripts/check-qwen38-xpu-boundary-state.py` with the real target config, FP16 then BF16, extending its real native output/state oracle across alternating verification widths and accepted histories. Require exact output/z, conv-history and SSM-checkpoint comparisons with full-width references; no CPU numerical substitute.
3. Launch disposable baseline, then candidate cells through `scripts/experiments/qwen38_standard_bench.py --long-context-only --context 180224 --max-num-batched-tokens 2048 --cache-group-size 8 --lengths 512 65536 160000 --near-limit 180096`, plus explicit checkpoint/patch/guard/output paths. Candidates add `--verification-cap 1`, `3` or `7`.
4. Existing public API gates exercise `/health`, `/v1/models`, `/v1/completions` and `/v1/chat/completions`: streaming integrity, finite logprobs, canaries, functional tasks, EOS/length completion. Cold token-ID input sweeps use identical rendered payloads, greedy sampling, ignore EOS,128 output tokens, one warmup and six measured runs per point. Include exact total180224 and one-token-over rejection HTTP400.
5. Retain exact argv/environment, source snapshots, raw SSE/request/JSON outputs, metrics, failures and exit codes in this directory. Compare payload identity, token outputs and acceptance by position. Report medians/IQR and measured depth distribution, not just acceptance percentage. CPU/client time per speculative round is a proxy, not a GPU profile.
6. Stop each owned container; verify no containers, unchanged launcher/power, Glimmer still stopped. Never silently retry failures. Reproduce analysis before commit/push.

Controller work is conditional on safe fixed-cap behavior and useful throughput evidence. GDN scratch padding still computes eight native positions and may negate attention savings. No memory-reclamation, quality-invariance or speedup claim follows from shorter verification alone.

## Implementation checks (not native/API proof)

The pinned async overlay is integrated in `160d655` + `576c806`. The latter corrects the required deployed prefill-guard source hash and DFlash's normalized `parallel_drafting=True` configuration. Only next-step placeholders are capped; current-step counters, seven-wide draft generation/tensor stride, standard rejection and rollback allocation are unchanged. Explicit `--async-scheduling` is required.

Lead-observed focused checks:20 verification-overlay tests,24 runner tests,10 boundary-routing tests passed (the real-source boundary test used `QWEN38_BOUNDARY_SOURCE_ROOT=/tmp/qwen38-boundary-source-73029d424`). Read-only integration review found no blocking defect in scheduling, aliasing, cap7/off behavior, deployed pins or flag wiring. Review and CPU tests do not establish native recurrence or graph/API correctness.

## Native gate (completed)

`bash run-native.sh` on inference-host passed FP16 and BF16 against the real pinned `_xpu_C.gdn_attention` reference: **224 partial cases,896 continuations and5,408 alternating chained steps per dtype**. Exact zero-tolerance output/z, conv-history and SSM comparisons passed with independent zero/random dummy-suffix histories. Width cycles `(2,4,8,2)`, `(4,2,8,4)`, `(8,1,2)` cover every valid intermediate accepted count; physical-padding variants are rotated per link. Commands and raw outputs are in `native/`. Graph/API testing is separate.

## Unchanged baseline (completed)

All four supported points have six valid measured streams plus a valid warmup. Input180097 plus128 output was correctly rejected (HTTP400). Decode medians at inputs512/65536/160000/180096:77.26/46.14/35.70/29.78 tok/s. All three canaries,131 finite-logprob boundaries and eight functional tests passed. There are no cold-client errors; host cleanup passed. These are baseline observations, not optimization gains.


## Cap7 no-op and repeatability diagnosis

Cap7 passed the same public API gates, all28 supported cold streams and the over-limit rejection. Its payloads are identical to baseline, but only16/28 decoded outputs match exactly (including warmups). No byte-identical-output claim is made.

Before lower-cap benchmarks, `repeatability.py` replayed the two smallest mismatching512-token requests six times each in baseline-a/cap7/baseline-b order. Unique output counts for the two requests were `(1,2)`, `(3,2)`, `(3,1)` respectively. Thus the unchanged baseline itself varies under identical seed42/temperature0 inputs, both within a server and across restarts. This explains why an exact-output gate cannot isolate a cap7 regression here; it is not proof of broad quality equivalence. All36 replays and the repeated canary/finite-logprob/functional gates passed. Original driver and raw results are retained in `repeatability/`.

**Invalid diagnostic source captures retained:** `repeatability/*/effective-source-sha256.json` and `cap-smoke/*/effective-source-sha256.json` resolve the image's unused `/workspace/vllm` checkout, not the installed serving package. Even the login-shell Python can do this for `-c` through its working-directory search path. They must not be cited as active-source evidence. Corrected capture uses the launch venv Python with `-P`, enforces `/opt/venv/`, and validates every pinned dependency plus exact overlay replay before full lower-cap gates. Earlier driver versions are retained alongside their outputs.

## Lower-cap short API gates

`repeatability.py --smoke-caps` completed twelve replays each for cap1 and cap3. Every interval's actual speculative counter delta satisfies proposed=cap×rounds (all24 intervals), demonstrating the cap really affects scheduling. Each server also passed3 canaries,131 finite-logprob boundaries and8 functional tests. Production/cleanup invariants passed. These checks are not timing results.


## Verification comparison (completed)

Decode medians, tok/s (six measured streams per supported point):

| Input tokens | Baseline K7 | Cap7 | Cap1 | Cap3 | Adaptive3/7 |
|---:|---:|---:|---:|---:|---:|
|512|77.26|75.15|37.34|58.64|70.12|
|65536|46.14|45.59|32.31|42.01|43.50|
|160000|35.70|36.59|29.98|37.67|35.77|
|180096 (exact boundary)|29.78|32.06|28.58|35.18|32.46|

All four fixed/control cells passed the API gates,28 supported cold streams and one over-limit HTTP400 each; no cold-client errors. For cap1/cap3, `summary.json.verification_runtime_source` is now valid evidence: `/opt/venv/lib/python3.12/site-packages/vllm`, installed async SHA256 `685f2be18db0b625f1750639274b54987b5b3487babd4f1437ee89002a21d51c`, all dependency pins and exact overlay replay checked before tests. The raw earlier cwd-checkout captures remain invalid as noted above.

Cap1 loses at every point. Cap3 loses24.1% at512 and8.9% at64K, but gains5.5% at160K versus unchanged baseline (only3.0% versus cap7). This modest long-context result is noisy and not a significant-speedup claim. At the exact boundary, baseline/cap7 proposals are clipped to about6.2/round; its larger apparent cap3 gain is not a clean context-only comparison. All cells still generate7 proposals and reserve Kmax7 state. Client ms/round at160K: baseline90.20,cap7 90.17,cap1 60.75,cap3 71.97. Lower cost per round alone does not imply better throughput.

## Adaptive stage predeclaration

Implement a default-off per-request online controller over verification depths3 and7; retain fixed1 only as a diagnostic because it was dominated. Preserve generationK7 and strict rejection. Measure useful emitted tokens per CPU completion interval, not acceptance percentage; these intervals are scheduler-side elapsed proxies, not GPU profiles. Start7, measure bounded windows at7 and3 after switch settling, choose with hysteresis, periodically re-probe. Exclude prefill, terminal/clipped/stale samples and queued old-depth completions. No GPU synchronization or per-step logging; emit a compact per-request depth/score summary for the required adaptive-depth distribution.

Before the matched adaptive128-output long-context sweep, run real API smoke requests with1024 output tokens at512 and160000 input, two repetitions each, to exercise multiple depth changes, re-probing, and fresh-request cleanup. Preserve exact payloads, SSE, speculative counters and controller summaries. Existing native alternating-state oracle covers the relevant query-width transitions; run CPU controller scheduling/time/cleanup tests and the actual graph/API boundary gates as well. No production promotion, adaptive generation, new persistent services or broad quality-equivalence claim.


## Adaptive implementation and switching gate

Controller commit `cc0774c` preserves fixed/off paths and the pinned native interfaces. Start7, anchor a completion interval, settle two valid intervals, score8 rounds, probe3, and require a3% useful-token/time improvement to replace the incumbent. Exploit for24 scored rounds then re-probe the other arm. Pending schedule identity, request identity and preemption generation are checked; terminal/clipped/stale/prefill samples never teach the controller. Per-request state is removed on completion/abort/replacement. It has no checkpoint persistence or GPU synchronization.

Lead regression checks: **70 tests passed**, no skips in this environment (`cpu-verification.txt`:35 overlay/controller,10 probe runner,15 standard runner,10 boundary-routing). Selective read-only review of the adaptive feedback, hysteresis, async identity/lifecycle, placeholder aliasing and source replay found no blocking defect. The real smoke run independently demonstrated completion feedback and switching rather than merely testing the controller in isolation.

`repeatability.py --adaptive-smoke` passed four1024-output streams (two at512 input, two at160000), all three canaries,131 finite-logprob boundaries and eight functional tests, with correct token counts/length finish and no parse errors. Installed source checks passed; raw data is in `adaptive-smoke/adaptive/`. Across that cell,146 request summaries include scheduled decode depths `{7:857,3:832}` and73 arm switches (39 seven-to-three,34 three-to-seven). This includes gates and warmups, not only the four long-output streams; depth0 denotes prefill/non-spec scheduling. The four long-output requests each switched both directions repeatedly. Their short-context final incumbent was7; both160K requests ended with incumbent3. Prefill, queued-depth mismatch and settling samples are explicitly counted as ignored. These are functioning-adaptation observations, not a throughput gain claim.


## Final adaptive sweep and limitations

`bash run-cells.sh adaptive` passed the same28 supported cold streams (24 measured,4 warmups), three canaries,131 finite-logprob boundaries and eight functional tests; the over-limit request was correctly rejected. No cold-client errors; launcher,275W power, idle-host and stopped-Glimmer cleanup checks passed. All140 supported cold streams across the five main cells passed, alongside five expected HTTP400 rejections.

Adaptive decode deltas versus unchanged baseline at512/65536/160000/180096: **−9.24%,−5.73%,+0.18%,+9.00%**. Adaptive decode IQRs:2.99/9.90/1.56/2.66 tok/s. The last point is boundary-clipped and must not be promoted as a clean context-only gain. End-to-end deltas are−8.02%,−0.29%,+0.05%,+0.21% respectively; long prefill dominates these requests. Short-output exploration overhead is included, not removed. No general throughput win or statistical significance is claimed.

The final adaptive cell logged170 request summaries (including gates/warmups): scheduled depths7:785 and3:513, plus33 boundary-clipped schedules of depths1/2/4/5/6, and1,570 prefill/non-spec depth0 schedules. It made52 arm switches (30 seven-to-three,22 three-to-seven), with scored decisions favoring3 eight times and7 twenty times. Raw per-request scores, ignored intervals and depth histograms are retained in `server.log` and reproduced in `analysis.json`; scheduled counts are not mislabeled as executed-round counts. Measured speculative counters independently show mean executed draft counts5.70/5.49/5.48/4.49 per round at the four points.

Greedy decoded-output identity is15/28 for adaptive versus baseline, with identical payloads in all28 pairs. Unchanged-baseline variability was independently reproduced; this small development study is not a broad quality or bitwise-equivalence proof. Kernel timing was not profiled: all round-cost figures are CPU/client elapsed proxies. Native GDN still computes full eight-row scratch blocks, and Kmax7 state remains reserved. No adaptive generation, target/checkpoint quantization change, relaxed acceptance or persistent launcher modification was made.

## Reproduction and files

- Serving entry points: `scripts/experiments/qwen38_dflash2_probe.py --adaptive-verification` (smoke) and `qwen38_standard_bench.py --long-context-only --adaptive-verification` (development sweep), with the same explicit model/guard/patch paths as each retained command. `--verification-cap {1,3,7}` remains the mutually exclusive fixed diagnostic control. Both options are default-off and DFlash2-only.
- On inference-host, `run-cells.sh baseline cap7 cap1 cap3 adaptive` reproduces the main sequence in a **new** artifact directory. Drivers refuse to overwrite existing cells. `run-native.sh` executes the real state oracle; `repeatability.py --adaptive-smoke` reproduces long-output switching. All exact executed commands and launch argv are retained next to their cell outputs.
- `code/` is the final source snapshot. Each cell's copied scripts, patch files, recorded hashes and launch argv preserve its actual executed revision; early fixed runs predate the adaptive controller and corrected source check. Do not substitute the final snapshot for those historical executed files.
- Run `python3 results/20260910-qwen38-dflash2-adaptive-verification/analyze.py` from the repository to reproduce `analysis.json` without a server. Raw SSE, requests, metrics, logprobs, failures/diagnostic mistakes, environment captures, controller summaries and cleanup outputs remain unnormalized.

No configuration was promoted to production. Further tuning or matched adaptive generation is separate work, not part of this delivered verification experiment.
