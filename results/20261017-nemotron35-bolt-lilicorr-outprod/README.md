# Isolated SYCL OUT_PROD layout fix for Bolt LiLiCorr

Status: **OUT_PROD layout fix passes native checks and actual-model execution; the LiLiCorr experiment stops at output fidelity.** No timing or promotion. User approved this isolated fix after the assertion recorded in `../20261017-nemotron35-bolt-lilicorr` (commit `1bebffa4`). Original accepted Bolt is restored and independently verified.

## Baseline and intentional difference

Immutable accepted source: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`. Preserve original target GGUF, accepted binaries, launchers, hardware and unrelated dirty files. Candidate source will be a separate reflink/copy at `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-outprod-r1`.

The only proposed runtime change is to make SYCL OUT_PROD accept representable row-contiguous F32 views using their actual GEMM leading dimension, and make advertised support agree with execution's layout requirements. Keep contiguous Q1_0 dequantization and contiguous destination requirements. No assertion bypass, graph rewrite, target re-quantization, runtime upgrade or global kernel tuning. Add native numerical/support-contract cases, including the exact failure shape and nonzero QKV view offset.

Actual failed operands were F32 src0 ne=[128,56,8,1]/nb=[4,12288,512,688128], transposed F32 src1 ne=[56,56,8,1]/nb=[224,4,12544,100352], contiguous dst ne=[128,56,8,1]. Correct non-transposed A leading dimension is3072, not128.

## Fresh external research

- https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/gemm.html — column-major nontrans A uses lda as the element stride between columns; lda must be at least m. This permits the observed interleaved head view without a materialization copy.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/outprod.cpp — original contiguous assertion and hard-coded lda.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ggml-sycl.cpp — original support predicate advertises F32 input regardless of layout.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-cuda/out-prod.cu — analogous backend leading-dimension handling.
- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/tests/test-backend-ops.cpp — existing CPU-reference OUT_PROD and broadcasting tests.

## Defined test process

All builds, tests, analysis and GPU/API workloads run on `mike@inference-host`, explicit Bash and SSH `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`. Remote artifacts: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-outprod`. Use the existing image `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`, IntelLLVM2026.0.0, six build jobs and existing oneDNN. Tools/workloads stay outside the Pi environment.

1. Check patches apply exactly to accepted source and do not touch unrelated paths. Build only the isolated candidate; retain exact commands, patches, logs, hashes and all failures.
2. Stop the original accepted container only under a failure-safe controller. Run focused native CPU-reference tests for exact strided geometry, nonzero view offset, contiguous F32, both src1 orientations, irregular dimensions, batch broadcasting and Q1_0. Check unsupported layouts are rejected, not silently counted as numerical passes. Existing native385 must also pass. No retry-until-lucky.
3. Debug actual-model activation: official pinned LiLiCorr artifact already verified at `../20261017-nemotron35-bolt-lilicorr/draft/` remotely; do not download or alter it. Use depth4/n_min0/p_min0, greedy draft sampling, no draft-backend sampling, target greedy seed42, q8 target KV, F32 recurrent state, context65536, batch/ubatch1024, single sequence, ESIMD1/FUSION1/graphs0. Capture target/draft defaults and snapshot allocation. Compare frozen55/8270-token,128-output,uncached responses against fresh accepted non-speculative reference. Require actual accepts/rejects, post-rejection exact rendered-output identity and no backend failure. Label this a small development fidelity screen, not exhaustive rollback or distribution proof.
4. If activation passes, fresh **non-debug** candidate and baseline each pass native385 and API gate-v2 **20/20 before timing**, plus deterministic fidelity and exact8238-token retrieval `BOLT-7319-CEDAR`. Stop before timing on unexplained divergence or any failure.
5. Only then: unchanged API probe, inherited warmups54/8269 and three measured55/8270-token prompts,128 forced outputs,cache_n=0. Record proposed/accepted drafts, depth, acceptance lengths and target/draft time split where actually exposed; report end-to-end and decode medians. No synthetic or relaxed acceptance. Preserve every sample. Development results only: no transferable GPU-speed, occupancy, saturation, peak-memory, maximum-context or production claims.
6. Nested finally stops all owned experiments and restarts the same original container. Verify ID, complete config/mounts, accepted hashes, unchanged boot/boost0/16 CPU caps3801000/275W, health and READY. Retain stopped experimental containers, raw remote logs and lossless `gzip -n` local evidence. Commit/push only scoped artifacts/source patches.

The Lab preview/patch sockets were confirmed unavailable in the preceding experiment; do not retry or repair them. This declaration is the textual side-effect preview. No accepted source or deployment is being edited.


## Implementation and pre-activation checks

Worker commit `e6207d5b` was integrated as `d1f17be7`. Its runtime patch adds a shared OUT_PROD capability predicate, accepts aligned row-contiguous F32 leading dimensions, preserves contiguous/non-broadcast Q1_0 dequantization, and passes actual `lda` to oneMKL. Lead corrected test permutation axes to `(2,0,1,3)`, aligned Q1 width to its128-element block, kept Q1 non-broadcast, and added irregular strided/broadcast cases. The original worker patch remains recoverable in Git history.

`build-candidate.sh` passed with IntelLLVM2026.0.0/oneMKL2026.0 and six jobs. Source inventory comparison outside `build-sycl/` confirms exactly four differences: `ggml-sycl.cpp`, `outprod.cpp`, `outprod.hpp`, and `tests/test-backend-ops.cpp`. Accepted binary hashes remained unchanged. Candidate server hash is unchanged; candidate SYCL library is `0b2dbb91c0f5f6acaa2d53e109fc4e44d434d1e6ca2aaab5da41c4c9df0d5cd5`, candidate native test executable is `f203b60539b207ed236b60bba035beeb4ebecca1d0c3585e6a666784ade136b8`.

Read-only source review passed the exact lda3072/ldb56/offset8192/plane-stride512 contract and test fail-closed behavior. Review is advisory; actual native/API gates remain necessary. The new negative-layout fixtures are deliberately tested with `-b SYCL0`; they are not a promise of CPU-backend or all-backend suite support (CPU execution of the unsupported stride0 fixture would assert). The candidate is a development patch, not an upstream-ready cross-backend test submission.

`native-outprod.sh` declares, per fusion0 and fusion1: F32 `80/80` (77 numerical cases plus three explicitly expected layout rejections), Q1_0 `18/18` numerical cases, and a separate three-case support-mode rejection check. Exactly three expected-rejection lines are required; all other skips fail. Focused qkv_view tests use deterministic data and maximum absolute error1e-4 versus CPU; existing cases retain their NMSE threshold. The numerical harness also rejects NaNs and mismatched infinities. All these counts and outcomes passed in the actual candidate launch (190 numerical checks plus nine explicit support-contract checks, including the six expected rejections in test mode).

Process failures retained in session history: worker's initial remote patch-check launch hit inherited Fish syntax and was corrected to explicit Bash; the lead independently reran patch dry-run and `git apply --check` during the successful build. Initial cherry-pick was blocked by a colliding untracked README; both documents were preserved, the worker notes moved to `implementation-notes.md`, and the commit was integrated without touching unrelated files. Build warnings about missing Git metadata, absent HTTPS/OpenSSL support and an existing unused `gate` parameter are preserved in `build-r1.log`; runtime dependency/version precheck passed.


## Observed result

- Fresh accepted reference and isolated candidate each passed native385. Reference also passed fresh API gate-v2 **20/20** and completed the frozen short and long reference requests.
- Candidate passed all OUT_PROD numerical/support-contract checks above, then completed its first actual-model short55/128 request with cache_n=0 and finish reason `length`. The debug request window contains **82 completed SYCL OUT_PROD calls** and no assertion/SYCL exception. This reproduces and clears the original layout crash without materializing the view or changing weights.
- Drafting operated: **41 verification events,163 proposed tokens,86 accepted tokens (52.7607%),26 rejection events**. Accepted-count histogram0:9,1:8,2:8,3:2,4:14;40 depth4 proposals and one final depth3 proposal. Mean accepted draft tokens per verification2.09756; including the target/bonus token, mean progress3.09756, consistent with the server's rounded3.10. API counters and logs reconcile exactly. These are descriptive counters for **one debug request that failed fidelity**, not a throughput or quality claim.
- Exact rendered-output comparison **failed**:195 matching characters, then reference continues ` quickly and accurately...` after `restored`, while candidate continues `. If the backup fails...`. Both outputs have128 generated tokens. The candidate long request, fresh non-debug candidate20/20, further fidelity requests and all timing measurements were skipped. Instrumented timing fields are retained but must not be used as a performance comparison.

### Fidelity attribution remains unresolved

Saved historical requests have exactly equal JSON payloads (temperature0,seed42,no cache). However, the current accepted reference differs from both the preceding LiLiCorr reference and the frozen decode-profile short sample after346 characters (`Core Strategy` versus `Foundation`). Those two older references agree with each other. Hence even unchanged accepted binaries do not establish cross-launch rendered-output identity here. The candidate diverges earlier (195), but these records do **not** isolate the cause to OUT_PROD, speculative verification, recurrent rollback, batching numerics or warmup/state history. No altered-seed or relaxed-acceptance explanation is assumed.

Initial saved-data analyzer incorrectly asserted historical output identity and failed. Its actual source is preserved as `analyze-and-verify-r1.py`, with `verification-r1.log.gz`. The corrected analyzer records historical identity/prefix lengths instead of assuming them; it does **not** change or relax the experiment's failed fidelity gate. No inference request was repeated to obtain a favorable result.

**Next proposed experiment (not run):** declare matched reference repeatability controls, compare accepted versus patched runtime with drafting disabled, then localize the first draft-on token/logit divergence and rejection/rollback history. Use a fixed repetition plan, not retry-until-equal. This is separate from the approved narrow OUT_PROD fix; timing remains disabled until correctness/fidelity and fresh non-debug gates pass. The source fix is retained as a development candidate, not promoted.

## Reproduction and restoration

Actual commands on inference-host, from the remote artifact root with `set -euo pipefail`:

```bash
bash build-candidate.sh 2>&1 | tee build-r1.log
python3 -B test-controller.py 2>&1 | tee controller-static-r1.log
python3 -B run-outprod-activation.py 2>&1 | tee activation-r1.log
# The first analysis failed its historical-identity assumption; retained unchanged.
python3 -B analyze-and-verify-r1.py 2>&1 | tee verification-r1.log
# Corrected saved-data analysis; no repeated inference.
python3 -B test-controller.py 2>&1 | tee controller-static-r2.log
python3 -B analyze-and-verify.py 2>&1 | tee verification-r2.log
bash compress-evidence.sh
```

The failed analyzer originally ran under the top-level name `analyze-and-verify.py`; it was copied to its `-r1` filename before correction. Build/run paths and container names refuse reuse. Full native/server argv are retained in `activation-r1/{reference,activation}/create-argv.json`; the actual controller and native script are copied into `activation-r1/`.

At `2026-10-17T10:21:50Z`, independent analysis/restoration verification passed: same original container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, complete config and mount-field identity, all three accepted binary hashes, unchanged candidate hashes, boot/boost0/16 CPU caps3801000/275W, health OK and saved restoration READY. Both owned experiment containers are stopped; only the original accepted Bolt is running. `verification.json` is successful verification of a **failed-fidelity experiment**, not an activation-gate pass. `activation-r1/summary.json` remains failed with timing disabled.

Raw originals remain remotely. `compress-evidence.sh` creates deterministic lossless gzip copies and compares their decompressed bytes to originals; `compressed.sha256` verifies retrieval. No target/draft weights or binaries are committed. The original target, accepted source/binaries, launchers and hardware remain unchanged. All autonomous agents have finished; the writing worktree was removed after clean integration, with its branch retained.
