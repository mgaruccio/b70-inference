# Bolt early complete-block output observer

## Predeclaration — development diagnostic; observation only

Prepare ONE bounded observation-only probe to continue Bolt LiLiCorr optimization without checkpoint approval pauses. Fidelity precedes timing/promotion; the accepted runtime remains immutable. This is not a standard-compliant performance comparison. No timing, promotion, full-fidelity, or internal-root-cause claim is authorized.

Computational parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1` (NOT the preceding observer). Candidate, created only by the lead: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-early-block-outputs-r1`. Result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-early-block-outputs`; container prefix: `bolt-lilicorr-early-block-outputs-r1-`.

Harness BASE: `../20261017-nemotron35-bolt-lilicorr-router-control-operands/`. Retain its corrected independent-FIFO/all-frame strict-cardinality parser and inherited native/API/restoration gates. Immediate-parent references remain explicitly asymmetric:

- `parentRoot=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2`
- S0 raw/request/response cell: `parentRoot/run-r1/S0`; alignment: `parentRoot/reanalysis-S0.json`.
- C0 cell: `parentRoot/continuation-c0/run-r1/C0`; alignment: that cell's `alignment.json`.

Never rewrite failed S0, inherit the MoE observer runtime, or precreate a continuation. Before interpreting either arm's captures, BOTH arms must preserve all 128 immediate-parent raw vectors byte-for-byte plus prompt IDs, emitted IDs, text, batch inputs, selected positions and rows. Latest preceding observer preserved both arms, and showed byte-exact block1 router full weights/input/240-F32 output and latent-down full weights/input/output at position55/token0. Final raw scores still first differ at output2 and emitted tokens at41. That excludes only those observed internal projections, not a complete block or full-model divergence.

## Source boundary and exclusions

The only allowed source diff is `ggml/src/ggml-sycl/ggml-sycl.cpp`; only `libggml-sycl.so.0` may change. The wrapper, implementation, native test, common library and all other source remain parent-identical. Parent five SHA256 values are fixed in `run-trace.py:PARENT_EXPECTED`.

One call in `ggml_backend_sycl_graph_compute_impl`, after view/noop and noncompute skips and before `ggml_sycl_fuse` or any fusion dispatch, selects consumer `GGML_OP_RMS_NORM` with exact `src[0]` names `nemotron_h_block_out-0` through `-7`. It copies **src[0], never the consumer output**. `nemotron-h.cpp` names each complete post-residual ADD before assigning `inpL`; next-block RMS norm consumes it. Complete ADD+ADD fusion materializes the final ADD; a compute-forward-only producer hook would miss that branch. Do not change arithmetic, launch choices, fusion, graph nodes, or callbacks.

First eight blocks of the current 88-layer model: 0/2/4/6 SSM, 7 attention, 1/3/5 FFN (block1 is proven IQ2 MoE, not dense). Consumer names are recorded, not guessed (`norm-1` need not be `attn_norm-1`). Exact source names/op ADD, F32 `[4096,n,1,1]`, token axis1, n1/S0 or n2/C0, tight full strides, nonview, aligned data, complete allocation bounds, and same-device SYCL buffer ownership must pass before backend metadata access. F32 extra is not needed. Renamed scheduler copies such as `SYCL0#nemotron_h_block_out-0#0` are not unwrapped. Missing outputs/layout/fusion/split support rejects the experiment.

The existing in-order `ctx.stream()` copies to a local host vector; completion/error handling must keep it alive through queue completion before file writing. No new queue or persistent host slot. Default-off arming, once-only ordered eight files and exclusive creation keep native/API/warmup/ordinary and prefill widths5/46/4 unobserved. Remaining decode steps cannot recapture. All eight records must be in the same first post-prompt position55 target frame, block order0..7, target SYCL backend and same queue. Complete exact-size finite F32 files only: S0 128KiB, C0 256KiB total. Same existing async common logger as `BOLT_VERIFY`; do not correlate direct-stderr `BOLT_ROUTE` text positions with common-logger positions. Retain other-backend/drafter routes and exact one-to-one target-group/VERIFY frame count, order and retained width; no unknown prefixes or leftovers.

Compare only token0's full 16,384 bytes for each of eight blocks after BOTH128 parity. Report all eight equal/different plus first differing byte/float and earliest unequal complete block. Never compare C0 future column1 with S0 token0. An unequal boundary only brackets the remaining difference; equality of these eight says nothing about blocks8..87.

## Fresh external research

Consulted from inference-host during preparation: https://github.khronos.org/SYCL_Reference/iface/queue.html (`in_order`, `memcpy`, `wait_and_throw`). Same-queue in-order commands have implicit dependency ordering; no cross-queue ordering guarantee. `memcpy` returns an event; `wait_and_throw` waits and delivers asynchronous errors. Host-vector lifetime is enforced by implementation, including an exception-path queue wait. These docs do NOT establish floating-point observer invariance or cross-shape GEMM bit identity: BOTH128 empirical parity remains mandatory. This matches the lead's fresh research for the ongoing task.

## Defined real E2E — lead only, pending

Environment/preconditions: `mike@inference-host`, Intel LLVM2026 disposable image `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`, six CPU build jobs, read-only oneDNN3.14 install. No accepted source/binary mutation; source/five-hash checks before/after. Original `nemotron-bolt-probe`, ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, API18098, sole original; hardware16x380/1000 caps, boost0, GPU275W unchanged. No persistent launcher changes.

Execute, in this order at the result root: `bash build-candidate.sh`, `python3 -B test-trace.py`, `python3 -B run-trace.py`, `python3 -B analyze-trace.py`. Use explicit local and remote Bash across Fish login:

```sh
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '\''cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-early-block-outputs && bash build-candidate.sh && python3 -B test-trace.py && python3 -B run-trace.py && python3 -B analyze-trace.py'\''"'
```

The lead must schedule long jobs through its existing background runner, not foreground waiting. No retry for agreement; retain failures.

Each arm: native385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + view6 + routersix, startup checks, fresh public API20/20, warmup stream then ordinary stream then separate diagnostic `/v1/chat/completions`. Frozen55-token prompt; forced128, temperature0, seed42, cachefalse, ignoreEOS. S0 max1/min2; C0 max1/min0; n_rs_seq1/K2, capacity2/reservation unchanged, Q8 KV/F32 recurrent, context65536, batch/ubatch1024, parallel1/graphs0. Callbacks and verbose tracing off. Diagnostic actual IDs/full131072-F32 raw vectors and exactly eight complete boundary files. Require all128 BOTH-parent parity before any boundary interpretation. Abort on missing/duplicate/late/foreign/partial captures or nonfinite data.

Retain create argv, environment/hardware and protected-source/five-binary hashes, raw build/native/API/request/response/log/alignment reports, `run-r1/analysis.json`, remote `capture/` and `early-block-outputs/`. Raw tensors/weights/logits, full source archives and executables never enter Git. Local metadata can be losslessly gzip-compressed; original remote JSON stays unchanged. Cleanup always disarms in finally, stops candidate containers, restores SAME original identity/Config/HostConfig/all mounts/five hashes/protected sources/hardware, health/READY and sole original. The analyzer independently rechecks restoration.

## Worker verification boundary

Only CPU/syntax/parser/source-spy/fuzz0 tests run by this worker, exclusively on inference-host in this result root's disposable `worker-scratch/`. Local reads/edits/Git are allowed; no local executable Python/AST/tests/patch/source generation. No GPU, API, service, Docker build or hardware operation. Targeted tests must cover real saved parent S0/C0 FIFO/prefill regressions, exact parent mapping/parity gates, source[0] rather than node output, placement before fusion/after skips, same queue/lifetime, no arithmetic diff, malformed metadata/overflow/foreign buffers/names/views, arming/cleanup, exact eight captures and token0-only comparison. CPU spies supplement, never substitute for the pending real build/E2E and BOTH128 invariance.

## Worker observed results — preparation complete, live E2E pending

All **79 distinct CPU tests passed**, in eight bounded groups (10/10/10/10/10/10/10/9). Environment: `inference-host`, Linux `7.2.0-1-cachyos x86_64`, Python `3.14.7`, GCC `16.2.1 20260810`. Group logs are `worker-checks/cpu-group{0..7}-r1.log`. This includes real saved router-single2 S0/C0 all-frame FIFO/position70/prefill regressions and full128 parent-parity wiring; ONLY the eight new boundary records/files in those fixtures are synthetic. No new live observer invariance has been measured.

Source checks apply/reverse the patch with `patch --batch --forward --fuzz=0 -p1` / `patch --reverse --batch --fuzz=0 -p1` in disposable scratch. The patch is **130 added lines, zero removed lines, one source file**: five includes, 123 helper lines, and the two-line hook insertion. Removing that one hook restores the complete parent graph-compute body; existing compute-forward/router/IQ2/matmul/reorder functions remain byte-identical. Hook is at parent graph-loop position immediately after the compute-flag skip and immediately before `ggml_sycl_fuse`, not after a producer's compute-forward. `source->src` chains are never followed; the authoritative copied pointer is `node->src[0]->data`. Source tests verify the residual naming and common logger installation chain (`common_init` → `llama_log_set` → `ggml_log_set` → `common_log_default_callback` → `common_log_add`).

`test-observer.cpp` is a CPU spy template, NOT alternate production logic. The test inserts the actual patched helper and compiles it with `g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined`, against parent ggml public/backend structs. All26 modes passed, including both widths, 28 metadata/bounds/view/ownership exclusions, exact names, distinct consumer-output sentinels, same queue, complete-source file bytes, no recapture over128 later iterations, unarmed/prefill exclusions, marker/file link/stale errors, ordering/foreign-backend/queue rejection, and submit/async exceptions that complete into still-live host storage. The inherited actual router preflight spy also passed85 exclusions. Sanitizers were clean. Parser tests reject missing/reordered/duplicate/out-of-frame/foreign captures, incorrect sizes, nonfinite floats, traversal and links; comparison tests intentionally differ C0 column1 while requiring token0 equality, then mutate each of eight token0 blocks and check first byte/float/earliest block. A last-vector128 mismatch in EITHER arm prevents interpretation.

The worker ran all executable checks/source generation on the remote host, in:

`/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-early-block-outputs/worker-scratch/assets`

Exact bounded-group invocation (executed separately with `G=0` through `G=7`; no concurrent test jobs):

```sh
bash <<'LOCAL'
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host 'bash -s' <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-early-block-outputs/worker-scratch/assets
G=0
python3 -B - "$G" <<'PY' 2>&1 | tee "../checks/cpu-group${G}-r1.log"
import runpy, sys, unittest
m=runpy.run_path('test-trace.py'); names=unittest.defaultTestLoader.getTestCaseNames(m['Trace'])
group=int(sys.argv[1]); chosen=names[group*10:(group+1)*10]
print(f'CPU group {group}: tests {group*10+1}..{min((group+1)*10,len(names))} of {len(names)}',flush=True)
assert chosen
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(m['Trace'](name) for name in chosen))
sys.exit(not result.wasSuccessful())
PY
REMOTE
LOCAL
```

Other exact remote invocations, same cwd/SSH boundary:

- `python3 -B test-trace.py Trace.test_syntax_and_patch_scope Trace.test_block_source_guards_queue_lifetime_logger_and_complete_boundary Trace.test_compiled_actual_block_observer_cpu_spy`: initial `cpu-source-r1.log.gz` failed on two TEST-fixture issues (external generic-fusion function incorrectly searched as a local definition; checking an already-closed FD after an ifstream reused its integer). Corrected test-only assertions; `cpu-source-r2.log` passes all3. No production hook change was needed.
- `python3 -B test-trace.py`: initial all-at-once attempt hit the local60s foreground limit during the attribution test, after19 passed; incomplete `cpu-all-r1.log.gz` retained. One diagnostic process listing confirmed no surviving test/tee process. This is NOT a green full-suite invocation; all79 subsequently passed as the bounded groups above. These two failed/incomplete logs are lossless `gzip -n` copies to preserve their exact trailing whitespace; remote originals remain unchanged.
- `python3 -B test-trace.py Trace.test_block_attribution_same_first_frame_order_backend_queue_and_all_eight`: passed1, `cpu-attribution-r1.log`.
- `bash -n build-candidate.sh` and AST parsing of all Python assets run inside the remote syntax test. Fuzz0 apply/reverse and source comparisons run inside the remote source tests. Initial `git diff --cached --check` flagged four empty patch-context spaces and raw failed-log whitespace. Only patch serialization was normalized remotely (empty context markers to bare empty lines), then `python3 -B test-trace.py Trace.test_syntax_and_patch_scope Trace.test_observer_preserves_router_control_and_native_source_exactly` passed2 in `cpu-patch-format-r1.log`. Final patch SHA256: `e5768d7b9e57730e4cf05a6b21fb9337d1303657c6479aff510b2d36e01c235f`; applied source unchanged.
- Read-only `runpy.run_path('run-trace.py')['binaries'](PARENT)` equaled all five `PARENT_EXPECTED` hashes; SHA256 of parent SYCL source/native test matched the build preconditions. `source-environment.log` records values, exact asymmetric mapping, prepared asset hashes (before patch serialization normalization), environment, and fresh official queue-document excerpt. Candidate source, `run-r1` and continuation do not exist. Early read-only shell-inspection quoting failures are retained in `preparation-failures.log`.

Only the declared result subtree is committed. No raw tensors, weights, logits, executables or full source archives are included. The worker did not create the candidate, run Docker/build/native GPU/API jobs, touch protected runtime sources or hardware, or change the accepted service. **Pending with lead:** integration, real IntelLLVM2026 build, native/startup/API20 gates, real frozen S0/C0 captures, BOTH128 empirical invariance, eight-block comparison, and live restoration checks. Any missing boundary or invariance failure rejects the experiment; no timing or promotion.
