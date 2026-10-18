# Bolt first-block SSM operand localization

Development quality-sensitive diagnostic only. No timing or promotion. User wants continued bounded diagnosis, not checkpoint stops. Parent: IQ2-single2 (implementation296dd379, completed evidence7a17b32d, archive cleanupafb752e5). Its S0/C0 all128 vectors exactly match attention-dmmv2-view parents, while cross-arm scores still differ at output2 before rollback and emitted IDs at40. The next question is WHERE the first-block computation diverges, not whether route labels match.

## Research and bounded scope

Fresh official research: https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html guarantees bitwise repeated execution of the SAME primitive on a given platform; it does not establish cross-shape equivalence. https://github.khronos.org/SYCL_Reference/iface/queue.html documents asynchronous USM memcpy, event wait_and_throw, and ordering only within an in-order queue. Verify actual consumer queue/pointers from the pinned source; do not infer dependencies across queues.

Read-only source/current captured routes establish blk.1.ffn_gate_inp.weight is F32[4096,240], with both n1/n2 using MUL_MAT_SYCL then LIBRARY_MKL_F32. Block0 SSM projections both use MMVQ_REORDERED. Conv uses the same short path; scan uses the original serial kernel with custom prefill disabled. Same route names do NOT establish numerical equivalence. Adding another route-only logger is less discriminating than checking actual operands at the earliest block.

Isolated parent runtime `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-single2-r1`; new runtime `...-lilicorr-ssm-operands-r1`. Parent evidence `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2/pair-r2/run-r1` (note nested pair-r2). Current root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-ssm-operands`.

Runtime scope: only ggml/src/ggml-sycl/ggml-sycl.cpp. Add temporary default-off capture around the EXISTING GGML_OP_MUL_MAT execution of exactly blk.0.ssm_in.weight and blk.0.ssm_out.weight, first target decode width1/2 in the diagnostic request. No tensor callbacks, graph changes, dispatch/math/precision/reorder changes, additional kernel invocations, or new queues. No new general-purpose tracing product. BOLT_ROUTE stays unchanged. Only SYCL binary may differ; server, implementation, common, native test, accepted and parent artifacts remain unchanged.

Capture complete F32 src1 BEFORE the existing call and complete F32 dst AFTER it, plus the actual consumed reordered NVFP4 weight bytes (remote only). Read from the actual consumer device pointers and prove the covered extent/layout; validate device-buffer bounds, types, expected dimensions and initialized reorder. Use the existing in-order queue, wait for each host copy to complete, propagate errors, and preserve host memory lifetime. No asynchronous host buffer destruction. If the actual consumed pointer/extent cannot be established, stop with the precise blocker instead of pretending public storage proves the operand identity. Capture weights after execution only if source confirms they were already reordered and unchanged during that call; otherwise explicitly capture the consumed representation at the correct point without changing computation.

Known shapes: ssm_in weight[4096,18560], src1[4096,n], dst[18560,n]; ssm_out weight[8192,4096], src1[8192,n], dst[4096,n], n1(S0)/2(C0), other axes1. F32 nb0=4 and columns contiguous. Preserve exact ne/nb metadata and source0 type/reorder. Compare full actual weight byte ranges and token0 activation/output columns between S0/C0; do not infer equality from printed sums or hashes alone.

Use a separate cell-local operand capture directory, not the raw-logit capture directory (its strict128-file gate remains intact). A private per-experiment capture directory environment setting and a simple arm marker owned by the controller may gate the first pair of captures only during the existing nonstream diagnostic request. No native, warmup, ordinary-stream or prefill captures. Marker lifecycle is within that request's try/finally; fresh exclusive paths, bounded once per weight, no overwrite or retry. Fail closed for incomplete/duplicate/conflicting records. Record backend context and files so the analyzer binds both captures to the actual first target post-prompt batch at position55 (raw output2), not the drafter. Parse separate capture records strictly; do not alter BOLT_ROUTE or count outer dispatch records as separate kernels. Existing successful route metadata and BOLT_VERIFY batch boundaries provide attribution. Full copied tensors/weights remain remote; commit only metadata/comparisons and losslessly compressed logs.

## Defined E2E and stopping rules

ALL executable inspection/analysis/Python/tests/build/GPU/API on mike@inference-host via explicit LOCAL Bash and REMOTE Bash; SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Local edits/git only. No host dependency, accepted source/weights, persistent launcher or hardware changes. Intel2026 image, oneDNN3.14, six build jobs, fuzz0 application, capture reserve at least4GiB. Preview/show_patch sockets unavailable; do not retry or repair.

Retain the existing S0(max1/min2), C0(max1/min0), n_rs_seq1/K2/block8,p_min0, greedy draft,no draft-backend sampling,ngl999. Frozen55-token prompt/128 outputs, warmup then ordinary streamed request then separate nonstream diagnostic at http://127.0.0.1:18098. Same ctx65536,batch/ubatch1024,parallel1,q8 KV,F32 recurrent state,graphs0,FUSION/ESIMD/OPT/VMM/DNN1,callbacks0,ONEDNN_VERBOSE0,temperature0/seed42/cachefalse/ignoreEOS. Boost0,16 CPU caps3801000kHz,275W. Current native385, OUT_PROD190/9 support, six IQ2 with actual singleton route gates, seven latent, thirteen attention, six attention-view cases, followed by API20/20, remain unchanged. Preserve all native numeric/failure/count/route gates, no unsupported/skipped numeric tests or duplicate summaries.

CPU regressions cover default-off/target/request/width bounds, pointer/span/shape checks and host-copy lifetime/order; controller marker cleanup and separate mount; strict capture records/first-target-frame attribution; missing/duplicate/truncated/nonfinite data; token0 strided extraction; byte-identical and differing input/output cases; unchanged inherited source/route/native/parent/restoration behavior. Use saved real logs for parser regressions where possible. One-use old run-pair-r2 history is not the new controller: pin the correct completed nested parent path. Retain real numeric-format regression against the original failed native log; do not manufacture formatter output. Report any retired historical-only test separately; do not silently weaken current gates.

This is an OBSERVER, not a computational control: require ALL128 raw vectors, emitted IDs/text, prompt, target batch inputs/positions/rows to remain identical to the respective parent in BOTH arms. Enforce explicitly in controller and analyzer with consistent return fields and a regression test; preserving S0 alone is insufficient. Keep legacy accepted-reference checks honest (C0 is not accepted-baseline-equivalent). If any observer invariance fails, preserve evidence, restore and stop this experiment; never retry until matching or interpret captures as the unobserved parent trajectory.

After both parity gates pass, compare the first SSM input projection src1/weight/dst token0. If input and weight match but output differs, localize to this projection's actual width-sensitive computation (not yet to a kernel subcomponent). If input/output match there but ssm_out src1 differs, localize to the intervening SSM subgraph, not automatically the scan. If both projections' inputs/outputs match, report block0 projections excluded in this fixture and choose the next boundary later. No automatic broadening of capture scope in this change.

Lead runs remote CPU tests, bash build-candidate.sh, python3 -B run-trace.py, python3 -B analyze-trace.py. Retain exact commands, configuration, failures, raw captures remote, small metadata/logs locally. Nested-finally restores original container identity/config/HostConfig/all mounts, five protected hashes, hardware, health and READY; independently verify only original service running. Worker prepares source patch/build/controller/analyzer/tests, runs remote CPU-only checks and commits; lead owns actual build and GPU pair. No timing, promotion, or production/fidelity success claim from this localization experiment.

## Worker delivery (CPU-only; not a completed GPU experiment)

The implementation is `ssm-operands.patch`, `build-candidate.sh`, `run-trace.py`, `analyze-trace.py`, and `test-trace.py`. The patch changes **only** `ggml/src/ggml-sycl/ggml-sycl.cpp`. No runtime has been created or built by the worker; no container, API, GPU, hardware-control or launcher actions were performed. C++ compilation and the declared real S0/C0 journey remain the lead's acceptance gate. These are development checks, not standard-publishable performance or fidelity results.

### Consumed-pointer / extent / queue proof

The pinned parent `ggml-sycl.cpp` SHA256 is `057ab53bd77e1cb943c110a9c414b6b4ac17fabfef9cbde3ed0db32bfdf849ed`. References below are to this original parent, not shifted patch line numbers:

- `ggml-sycl.cpp:3886–4185`: for non-split contiguous tensors on `ctx.device`, `dev[i].src0_dd = src0->data`, `src1_ddf = src1->data`, and `dst_dd = dst->data`. With axes2/3 equal to1, `i0=0`; the unsplit column stride is the entire width, so `src1_col_0=0`, `row_low=0`, and all three consumer offsets are zero. The observer validates exactly these layouts and the same-device regular SYCL buffers (`:873–875` distinguishes them from split buffers), with subtraction-based allocation bounds before copying.
- `ggml-sycl.cpp:5404–5434`: NVFP4 reorder writes **in place** at `src0->data`: all packed quants, followed by all scales. Each64-element block occupies32 packed bytes +4 scale bytes. The entire consumed span is **42,762,240 bytes** for ssm_in `[4096,18560]`, and **18,874,368 bytes** for ssm_out `[8192,4096]`, not the larger backend allocation or an assumed unreordered GGUF span. The input and destination spans are respectively `4096*n*4` / `18560*n*4` and `8192*n*4` / `4096*n*4` bytes.
- `ggml-sycl.cpp:5510–5541` returns immediately for an already-reordered weight. The observer requires that flag **before** the existing call; it does not initiate reorder. `mmvq.cpp:2577–2990`, specifically the NVFP4 case at2970, passes `src0_dd_i` unchanged to `reorder_mul_mat_vec_nvfp4_q8_1_sycl`; `vecdotq.hpp:856–884` reads packed bytes and scales through this pointer and never writes weights. Consequently copying weights before the call captures exactly its consumed representation. The existing n2 NVFP4 path invokes the vector kernel per column; a single projection route record is not a claim of a single kernel launch.
- `ggml-sycl.cpp:3995,4056` uses `ctx.stream(device,0)` for this unsplit path. `common.hpp:411–419` maps that to the default queue; `dpct/helper.hpp:727–795` makes the default queue in-order. The observer also checks `is_in_order()` at runtime. Every copy uses that existing queue and `.wait_and_throw()` while its host vector is alive; its exception path waits before releasing host storage. No queue, kernel or additional matmul is created.
- The hook brackets exactly the existing `GGML_OP_MUL_MAT` call at6688–6694, after its ne3 validation. Graph fusion/early returns at7074–7194, matmul selection/wrapper, IQ2 paths and the route logger remain byte-identical. Saved parent successful `MMVQ_REORDERED` records establish that these projections traverse this call, rather than a fused gate/up path. Missing future records fail closed; no fusion is disabled to obtain captures.

### Request and analysis contract

`BOLT_SSM_OPERAND_DIR=/ssm-operands` points to a **separate** cell-local `ssm-operands/` directory (0700). Its six exclusively created files are readable by the host controller (0644 inside the private directory); the original raw-logit `capture/` and strict128-file check are untouched. Only `operand_diagnostic()` creates the `armed` marker, enclosing the existing nonstream request in try/finally. S0 arms width1 and C0 width2. The two weights are captured once per process; subsequent decode/replay widths do not rearm or overwrite them.

Six strict `BOLT_SSM_OPERAND` records include backend context, queue, n, type, ne/nb, data/base/offset/allocation bounds, size and fixed filename. Analysis requires before-weight/before-src1, the existing selected successful route, then after-dst, all inside the actual successful first post-prompt target `BOLT_VERIFY` frame selected by raw output ordinal2/position55/row0. It rejects unarmed, drafter-context, outer/duplicate route, malformed, incomplete, stale, nonfinite and out-of-frame captures.

Both controller and analyzer use `observer_parent_invariant_passed`; C0 report-only acceptance has been removed. Each arm must preserve all128 parent vectors, IDs/text, prompt IDs, batch inputs and positions/rows against **IQ2-single2/pair-r2/run-r1**. Parent comparison covers all128 even when an emitted ID differs, solely to enforce observer rejection—not to interpret a changed trajectory. Cross-arm operand comparison occurs only after both invariants pass, using complete weight bytes and F32 token0 columns. Projection localization includes activation quantization; it does not identify a kernel subcomponent or blame the SSM scan.

### Lead-only build prerequisites

The Intel2026 image lacks `/usr/bin/ninja`, as the saved parent `build-attempt-r1/build.log:50–55` shows. Per explicit clarification, the script reproduces **only** the parent's disposable build-container bootstrap:

```bash
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install --no-upgrade -y -qq ninja-build libcurl4-openssl-dev pkg-config
```

This can update curl dependencies **inside that disposable container**, despite `--no-upgrade`; it is not zero container package changes. No host packages/software, installed Intel/oneDNN, accepted runtime or parent runtime are changed. There are no added packages or upgrade command. The script clones the current parent once, reuses its configured Ninja tree, builds only `ggml-sycl` with six jobs, and requires the other four protected binaries—including the native test—to remain identical. It checks all five accepted/parent hashes, protected source inventories and the one-file candidate source scope before reporting build success. oneDNN3.14 stays read-only at the inherited mount. A failed build is retained, not overwritten or automatically retried.

### CPU verification and retained failures

All executable research/source inspection, transformations, AST checks and tests ran on `mike@inference-host` through an explicit local Bash plus remote Bash, never local Python. Fresh official URL fetches reaffirmed the oneDNN same-primitive versus cross-shape distinction and SYCL in-order/asynchronous copy semantics above. Environment: Linux `7.2.0-1-cachyos`, Python `3.14.7`, GNU patch `2.8`.

Exact final suite invocation (from this worktree):

```bash
bash <<'LOCAL'
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-ssm-operands
python3 -B test-trace.py > worker-cpu/full-cpu-r2.log 2>&1
REMOTE
LOCAL
```

Observed: **66/66 tests passed**, 48.461s. `worker-cpu/final-close.log` (2026-10-17T23:44:28Z) records Python AST + `bash -n build-candidate.sh`, matching accepted/parent five-binary hashes, pinned original source, candidate-runtime absence, and fresh scratch `patch --fuzz=0 --dry-run --batch --forward -p1`, actual fuzz0 application and `cmp` against the prepared source: all passed. `test_syntax_and_patch_scope` reproduces the scratch application and checks unchanged consumer/dispatch/fusion/native code. Copy lifetime, once-only and eligibility coverage is source-contract testing, not a claim of executing SYCL. New operand fixtures are explicitly synthetic; saved real parent logs/raw vectors exercise actual parser/parity wiring. The real `amax=1.000000` regression remains pinned to the original failed **IQ2-single2/run-r1/S0/server.log**, not fabricated output or the nested successful log.

Final packaging removed two whitespace-only unified-diff context prefixes to satisfy `git diff --cached --check`. GNU fuzz0 dry-run/application still produces a byte-identical prepared source, and both source-contract tests pass (`worker-cpu/patch-whitespace-close.log`, 2/2). This changes patch encoding only, not the source tested by the full66 suite. Final staged diff whitespace check is clean.

Retained remote iteration logs under this root's `worker-cpu/`:

- `initial-source.log`: 3/5 passed; missing transferred build script and an inherited bool-only C++ symbol extractor caused two test errors. Fixed transfer stdin handling and bounded extractor.
- `observer-unit-r2.log`: 5/7 passed; a helper defined outside the scoped file and an ambiguous comment selector caused two source-test failures. Corrected the tests, not runtime gates.
- `observer-wiring-r3.log`: 3/3 passed, including both-arm saved-parent wiring.
- `full-cpu-r1.log`: 64/65 passed; the old pre-IQ2 parent-route expectation was stale. The current parent must pass, and deleting one actual singleton route must fail. `route-parent-r2.log`: targeted regression passed.
- `full-cpu-r2.log`: final full66 passed after the route fix, symmetric observer test, valid alternate-context/outer-route rejection and inherited build-bootstrap/source-protection test.

Historical-only `test_pair_r2_preserves_controller_and_analyzer_logic` is retired: no one-use retry controller is part of this experiment. The old IQ2 route-replacement wiring test is superseded by `test_both_arm_parent_parity_and_controller_return_fields`, using current real parent routes rather than synthesizing the already-integrated control. Native numerical/failure/count/route gates, API20 checks, raw128 checks and restoration behavior are retained. No source archive, raw tensors, raw logits or prior build logs are copied into Git.

Lead execution remains `bash build-candidate.sh`, `python3 -B run-trace.py`, then `python3 -B analyze-trace.py` **on inference-host** using the same SSH envelope. Preserve their logs and `run-r1/` evidence remotely. If any parity/attribution/native/API/restoration gate fails, retain the attempt, restore the original service and stop; no retry-until-match. No timing or promotion is authorized.

## Saved-S0 parser correction and missing-C0 continuation

Lead66 CPU tests, build, dependency checks and protected source/five-binary gates passed. S0 completed the native gates, API20, warmup, ordinary and diagnostic requests; all128 parent raw vectors and text/ID/batch identities passed. Analysis then rejected `copy/compute ordering`. The original failed run remains failed, with original service restored and no C0 run.

The failed assumption was ordering between different logging mechanisms, not a failed copy or observed arithmetic change. `BOLT_VERIFY` uses LOG_INF and the new operand records use GGML_LOG_INFO -> common_log_default_callback -> common_log_add(common_log_main()), an ordered asynchronous logger (common/log.cpp:187–190,240+,560+). Existing BOLT_ROUTE writes directly to stderr (ggml-sycl.cpp:231+). In the real S0 diagnostic, the first projection's route printed at line190 before its queued before-src1 message at191 (one-based), even though the unchanged C++ hook copies/waits before invoking the original matmul.

Correct analysis requires ordered before-weight/before-src1/after-dst records from the common logger, exactly one matching successful route inside the same actual target frame, all context/type/shape/stride/bounds metadata, and unchanged128-parent vectors. It no longer treats the direct-stderr record's printed position relative to queued operand text as GPU operation ordering. This preserves copy-phase and target-frame checks; runtime source/binary/capture behavior is unchanged. Regression tests use the real saved S0 log/captures, valid direct-route positions, and retain invalid same-logger phase-order/duplicate/missing/drafter/outside-frame negatives.

Run remote `python3 -B test-trace.py`, then `python3 -B continue-c0.py`. The one-use continuation validates the original specific failure, absent C0, frozen payload/create argv, native/API/128-row/operand gates using SAVED S0, protected sources/hashes and restored identity/hardware/health/READY. It does not rerun S0 or rebuild. Only the missing C0 executes in fresh continuation-c0 paths and a distinct container, with unchanged controller function bodies and pinned parent paths (AST regression). Both arms must independently preserve all128 parent vectors before comparing operands. The original failed summary is never rewritten; new reanalysis-S0.json and continuation-comparison.json identify their separate source cells. Any new failure remains failed; no retry-until-match or timing.
