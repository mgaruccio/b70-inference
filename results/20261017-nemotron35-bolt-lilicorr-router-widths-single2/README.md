# Five-width F32 router two-singleton control — preparation only

## Predeclaration

Development diagnostic, quality-sensitive; **not a performance comparison, fidelity fix, accepted runtime change, or promotion**. Timing/promotion remain false. Computational parent is `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1`, not any observer runtime. Candidate is `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`. Remote result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2`; worker scratch is confined to its `worker-scratch/`. Container prefix: `bolt-lilicorr-router-widths-single2-r1-`.

The measured block8 boundary and actual GGUF inventory motivate a closed width set `{240,304,368,432,496}` with K4096/n2 only. Only 240 and 304 were independently observed divergent before their singleton control. This does not establish divergence at the other widths. Route evidence covers representatives at blocks 1/8/21/37/48, not all 40 model routers.

Fresh official research was completed by the lead: https://github.khronos.org/SYCL_Reference/iface/queue.html and https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html . Queue/stride semantics support retaining existing execution behavior; they do not promise cross-shape bitwise equivalence.

Preparation uses explicit local and remote Bash over `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host`. All executable processing, source generation and CPU tests run remotely, outside Pi. Only bounded assets/logs are retained in Git: no full runtime source, binaries, weights, tensors, or raw logits. Every failed attempt is retained rather than overwritten.

## Lead-owned real E2E (not executed by worker)

Accepted `nemotron-bolt-probe`, ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, endpoint 18098, stays immutable. Before copy/capture require a 4 GiB reserve; no cleanup is authorized. Build only `ggml-sycl` and `test-backend-ops` with six jobs in disposable IntelLLVM2026 container, read-only oneDNN3.14; disposable-container apt bootstrap only. Protect both source trees and five artifact hashes; wrapper/implementation/common remain exact, only SYCL/native may differ.

Lead runs inherited native gates plus the five-width router family, API20 each arm, warmup ordinary, ordinary streaming, then separate diagnostic real IDs/full raw vectors. Frozen 55-token prompt, forced128, temperature0, seed42, cache false, ignore EOS; S0 max1/min2, C0 max1/min0; n_rs_seq1, K2 capacity2 reservation, q8 KV, F32 recurrent, ctx65536, batch/ubatch1024, parallel1, graphs0, callback verbose off. No agreement retries. Missing arms, if needed, use a separate namespace, lead only.

S0 must preserve all128 immediate-parent vectors, prompt/text/IDs/batches/positions/rows. C0 follows its actual newly generated proposals and acceptance history; parent comparison is report-only and limited to common prior emitted history, including the first different output. Within-arm text/ID, raw greedy argmax, before-rollback and strict route gates remain required. Parent S0 reference: `run-r1/S0` with `reanalysis-S0.json`; parent C0 reference: `continuation-c0/run-r1/C0/alignment.json`. Original failed S0 summary is never changed.

Finally restore the original ID/config/HostConfig/all mounts, five hashes, protected sources, boost0 / 16×3801000 / GPU275W, health READY, sole original container. Independently re-run the analyzer. Exact lead commands and raw results belong under the remote result root; no native/GPU/API/restoration success is claimed by this preparation.


## Prepared implementation and coverage

- `router-widths-single2.patch`: only the existing F32 router helper's first shape guard and four exact route selectors change in `ggml/src/ggml-sycl/ggml-sycl.cpp`. K4096, n2, exact suffix, F32, op/flags, dimensions/strides, owner/device/ranges/non-view/reorder/alias checks, both-column preflight, metadata copies and recursive singleton dispatch remain intact. Destination rows must equal the selected source width. No observer hooks or new knobs.
- Native source changes only the router fixture constructor (optional `m=240`) and adds four copies of the existing six profiles: ordinary n1/n2/n3, unrelated n2, offset-view n2, padded-column-view n2. The original six240 instantiations, fresh initialization, `1e-7` numerical tolerance and inherited canary machinery remain unchanged. Exact native gate: **30 cases, 35 operations, 70 outer/inner records**. New unrelated cases use `blk.1.width{m}.unrelated.weight`, so the existing block1 selector logs them without broadening production selectors.
- The inherited385 count is unchanged: the original shell selects `nvfp4_reorder=1` (56×2), `MOE_REDUCE_9` (20×2), `BOLT_ROUTER` (82×2), and `BOLT_SSM_PREFILL` (34×2+1), totaling385. New cases are `MUL_MAT(router_single2=1,...)`, outside those filters. OUT_PROD190/9, IQ2six, latent7, attention13 and attention-view6 use separate unchanged selectors/gates.
- `run-trace.py`, `analyze-trace.py`, `build-candidate.sh`: computational-parent harness, pinned five artifacts/two sources, strict native30 gate, asymmetric parent mapping, S0-only all128 preservation and C0 report-only policy. Every candidate frame requires all five representative routers in exact order and shape, including prefills. The independent direct-route/common-VERIFY FIFO parser is unchanged. Actual C0 capture determines proposal/acceptance/frame counts.
- `test-trace.py` retains70 computational tests; `test-widths.py` adds15 bounded tests. Saved-parent fixtures synthesize **only** the four absent representative router records, after block7 even in prefills without `output.weight`. Existing parent routes and verifier history are unchanged; synthetic route success is not candidate-runtime evidence.

## Remote CPU verification — final attempt r3

Environment: `mike@inference-host`, Linux `7.2.0-1-cachyos`, Python `3.14.7`, GCC `16.2.1 20260810`. Remote scratch retained about12GiB free; the CPU runner enforces4GiB reserve. No Docker, SYCL/native binary build, GPU, API, model load, service, hardware, or candidate-runtime creation occurred.

Executed preparation command (inside explicit remote Bash, using the SSH flags above):

```sh
python3 -B /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2/worker-scratch/inputs-r3/prepare.py prepare-r3
```

The lead's `bg_run` task `b62bbe9f4`, through explicit local and remote Bash, ran:

```sh
bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-widths-single2/worker-scratch/cpu-checks-r1.sh prepare-r3 r3
```

`cpu-checks.sh` is the identical retained runner. It changes into `worker-scratch/prepare-r3`, records hashes/environment, and executes `python3 -B test-trace.py` then `python3 -B test-widths.py`, retaining both exit codes even on failure.

Observed: **70/70 passed (127.053s), 15/15 passed (28.183s)**; `TRACE_EXIT=0 WIDTHS_EXIT=0 CPU_SUITE_EXIT=0` in `cpu-r3.log`. These are CPU-check durations, not inference timing. Actual patched helper/dispatch compiled with UBSan: all five positive widths, distinct columns/numerics/metadata/canaries, and **510 exclusions** (all85 inherited exclusions plus13 unsupported-width and4 mismatched-width negatives per positive width). Late second-column preflight failure launches no singleton work. The actual compiled route selector admits all10 native weight names and rejects16 off-scope names.

`prepare-r3.log` proves exact forward/reverse application using strict Git and GNU patch, including both native hunks. `source-proof.json` retains pinned parent and candidate source hashes. `final-pinned-proof-r1.log` rechecks both live parent sources and all five binaries after the full CPU suite; all match. The candidate runtime path remains absent. All seven final generated assets were byte-compared with the remote tested attempt. Patch SHA256: `a946f4af2d2597f90b3cd13013235837d7237bf2f02bbc256a7469555ccb1fcb`.

## Attempts and limitations

All attempts remain under remote `worker-scratch/`; logs are retained here, never overwritten:

- `inspection-r0.log`: initial read-only lookup used nonexistent `BOLT_TEST_ROUTER`; pins matched, grep exited1. `inspection-r1.log`: native/helper inspection succeeded but guessed selector-function name did not; later inspection used the actual `bolt_route_selected_weight` definition.
- `lead-launch-r0.log`: lead task `b141cc75d` failed in Fish parsing before SSH; no CPU/model execution. Corrected lead task `b5de4a2af` ran r1.
- `cpu-r1.log`: 70 tests had one failure/one error; all15 added tests passed. Fixes were only the stale expected container prefix and S0 fixture alignment path; no old parent fixture was edited. `targeted-r2.log` records both fixes passing and exact r1/r2 asset comparison.
- `cpu-r2.log`: 70+15 passed (lead task `b548fc5f6`), but final source review discovered unselected new native unrelated names. `native-selector-review-r1.log` records the actual selector; `selector-regression-red-r1.log` proves the new compiled-selector regression fails against r2. r3 changes only those four native-negative names, their exact parser map, and adds the selector regression. Production SYCL and analyzer remain unchanged from r2.
- `targeted-r1.log`, `targeted-r2.log`, `targeted-r3.log`, `prepare-r1.log`, `prepare-r2.log`, `prepare-r3.log`, and all three full CPU logs preserve the preparation history.

**Lead real E2E is pending**, including GPU numerical/canary/native30 execution, actual five-representative dispatch evidence, API20 each arm, actual newly generated C0 history, fidelity comparison, and restoration. The CPU spy is not oneMKL/GPU evidence. Coverage is five representatives, not all40 routers; unsupported/mismatched widths and n1/n3 remain ineligible. No fidelity, timing, promotion, or standard-publication claim is made.

## Lead-observed result — frozen S0/C0 fidelity passes

After the recorded remote70+15 CPU pass, the lead ran `bash build-candidate.sh`, `python3 -B run-trace.py 2>&1 | tee controller-r1.log`, then `python3 -B analyze-trace.py 2>&1 | tee analysis-r1.log`, fail-fast from this remote result root through explicit local `bash -c`, the declared SSH flags, and remote `bash -s`. Real build/dependency/source/five-binary gates passed. Both arms completed without continuation or inference retry.

Both arms passed inherited385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + view6, **native router30 with35 operations/70 outer-inner records**, fresh API20/20, ordinary warmup, ordinary stream and separate diagnostic. All five representative routers passed actual all-frame route gates. Representative coverage is not an assertion that all40 router activations were logged.

**All128 full131072-element F32 raw-logit vectors are byte-identical between S0 and C0. All128 emitted token IDs match; within-arm ordinary/diagnostic text gates pass.** `first_score_difference` and `first_output_difference_ordinal` are null, compared through128. This is the first full frozen-probe match after the prior output2 raw/output41 token discrepancy.

S0 preserves all128 immediate-parent raw vectors and prompt/IDs/text/batch inputs/selected positions/rows. C0 changed as allowed: against its old parent it first differs in raw scores at2 and emitted tokens at41, so that comparison stops at41. Its newly generated history has57 accepted out of69 proposed tokens and12 rejections;70 post-prompt frames,69 width2. No old proposal history or batch count was forced. These acceptance counts are not a speed claim.

The saved20 API quality cases also agree across arms: every request is identical, every output text/finish reason/completion-token count matches, and both functional gates pass (`cross-arm-quality-r1.json`). Those short API cases do not supply raw-vector or actual token-ID equality evidence; only the frozen128-token diagnostic does.

Candidate SHA256 values: SYCL `64d6a58f17cb92ea9fe6354d9c553f74724565a58fc018d2f4bbd013d090a72e`, native test `bdd86349fac5ab43fa32d465aad27cec7f31dcad0651f7611403d8a783f10c7d`. Wrapper/implementation/common remain identical to the parent. Controller and independent analyzer restoration checks passed for the original container ID/config/HostConfig/all mounts, five protected hashes, protected sources, unchanged hardware, live health, READY and sole original process. Accepted runtime unchanged.

Canonical remote artifacts: `run-r1/analysis.json`, per-arm `alignment.json`, raw `capture/`, API quality request/results, and `cross-arm-quality-r1.json`. Local logs and metadata JSON larger than1MB use lossless gzip with mtime0; raw vectors remain remote. Preparation failures remain retained. Two post-run inspection/archive mistakes (wrong comparison-field key; Fish expanding a remote shell variable) were corrected against saved artifacts; neither reran inference or changed original run summaries.

This passes the declared **suppressed-draft versus active width1** fidelity probe, not the whole optimization task. S0 still loads/executes LiLiCorr and discards proposals, so it is not a fair no-draft timing baseline. The next gate is a true non-spec/no-draft baseline on this same five-width runtime, compared with these saved vectors and API results. Larger draft windows, broader workloads and timing remain unvalidated. Timing and promotion remain blocked.
