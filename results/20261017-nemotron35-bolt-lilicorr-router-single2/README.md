# Bolt exact-two-column F32 router singleton control

## Predeclaration — development diagnostic, quality-sensitive

This is ONE computational control, not a timing run, fidelity claim, accepted-runtime change, or promotion. Preparation base: repository `576bcf44519631fdf0b2400beec9581c17f1904f`. The completed MoE operand observer found equal router weights/input at output 2, position 55, but 213 unequal F32 output entries (max absolute difference 3.337860107421875e-6). It was an observer, not this candidate's computational parent.

- Computational parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-single2-r1`.
- Parent raw reference: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-iq2-single2/pair-r2/run-r1` (the nested `pair-r2` is essential).
- Candidate (lead creates): `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1`.
- Remote result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2`; prefix `bolt-lilicorr-router-single2-r1-`.
- Worker-only preparation and CPU tests: that root's `worker-scratch/`. Shared source, model and references are read-only. No GPU, model/API invocation, Docker build, service change, or hardware change during preparation.

Fresh external research supplied by the lead: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html . Column-major USM GEMM computes `C = alpha op(A) op(B) + beta C`; non-transposed B requires `ldb >= k`, C requires `ldc >= m`. Output columns are mathematically independent and use the supplied queue (with optional event dependencies). **There is no bitwise cross-shape guarantee.** This supports testing the hypothesis, not asserting a fix.

## Bounded implementation

Only `ggml/src/ggml-sycl/ggml-sycl.cpp` and `tests/test-backend-ops.cpp` differ from the computational parent. Eligible operations are ordinary single-device, non-view, tightly packed F32 `MUL_MAT` with a leaf weight ending in `.ffn_gate_inp.weight`, exact weight shape 4096 x 240 and exactly two activation/output columns. Reject wrong names/types/shapes, hints, higher dimensions, views/offsets/permutations, split or foreign buffers, reordered storage, invalid allocation ranges and overlapping output/input memory.

Before either enqueue: validate original extents/strides/allocation ranges, copy full metadata for BOTH columns, offset activation by `col * 16384` bytes and destination by `col * 960` bytes, set only their width to one (and the copied destination's activation edge), and validate both complete column descriptors. Original strides, precision/op parameters, names, buffer/extra and graph tensors are not mutated. The dispatcher then invokes the existing singleton dispatcher twice on the same context; there is no fallible post-column-0 fallback, new GEMV arithmetic, queue, precision forcing, or per-token treatment of arbitrary matrices. The eligible F32 path consumes stack metadata synchronously and submits only device pointers/scalars to MKL; it does not retain stack descriptors in a host task. F32 `extra == nullptr` is supported by null-safe checks.

Existing IQ2, latent/attention Q8, head, NVFP4, deterministic oneDNN and legacy SSM selectors remain unchanged. No MoE operand observer is imported. n1, n>=3, non-router and unsupported layouts retain their original dispatch. `MUL_MAT_SYCL` + `LIBRARY_MKL_F32` describe ONE outer/inner operation; the n2 candidate must show TWO such pairs, each with true width one and original width-two strides. Existing records do not expose pointers: distinct column offsets are additionally tested from the actual source helper and native numerical fixture, not inferred from names.

## Defined lead E2E (not executed by the worker)

Prerequisites: preserved reference captures and inherited width1/native/API harness dependencies under `/home/mike/b70-evals`; original `nemotron-bolt-probe` ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, API port 18098. CPU caps 16 x 3801000 kHz, boost 0, GPU 275 W stay unchanged. All five accepted/parent artifacts and protected source hashes must pass before build/run and after restoration. Parent hashes:

- `libggml-sycl.so.0`: `959874f59e890957c1a4ec1488be1c322bdc7eeba675b7b2c9e0cdaa2ead66db`
- `test-backend-ops`: `c5592fd927626a468c1464cf4f42ee2de5894a0d55f6b1ded023809f5ffd8db9`
- `libllama-server-impl.so`: `43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede`
- `libllama-common.so`: `934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812`
- `llama-server`: `22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d`

The lead stages these six assets at the remote result root and uses its existing background runner for the build and full journey. Every remote command must have explicit local AND remote Bash because the remote login shell is Fish:

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -s"' <<'REMOTE'
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2
# LEAD ONLY, in this order; retain stdout/stderr and nonzero exits.
bash build-candidate.sh
python3 -B test-trace.py
python3 -B run-trace.py
python3 -B analyze-trace.py
REMOTE
```

Build: six jobs, existing disposable `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04` image, IntelLLVM 2026, read-only oneDNN 3.14 mount. Reuse parent configuration/cache; only SYCL and native-test artifacts may change. The inherited disposable-container bootstrap installs ninja/curl development dependencies (not a claim of zero container package changes). Accepted source/archive/launchers and weights are immutable. Full native/model checks require this build; CPU source tests are not a substitute.

Each fresh arm: native 385 + OUT_PROD 190 numerical/9 support + IQ2 six + latent seven + attention thirteen + view six, plus dedicated router six numerical/actual-route cases; startup gates; fresh API20/20; warmup; ordinary streaming; separate diagnostic real emitted IDs and raw 131072-F32 vectors. Existing counts, tolerances, state/canary gates and exact-one `amax=1` / `amax=1.0...0` parsing stay intact. New router cases are widths 1/2/3, non-router width2, offset view and padded view exclusions; CPU-reference numerical checks use deliberately distinct input columns.

S0: max1/min2 suppressed. C0: max1/min0 active. Both n_rs_seq1/K2, output capacity2 and identical prefill reservation; frozen 55-token prompt, forced128, temperature0, seed42, cachefalse, ignoreEOS, tensor callbacks and oneDNN verbose off. Inherited target/drafter configuration and launch environment are retained. Actual target router metadata must pass on every post-prompt target frame, not just text equality; nested drafter/backend routes remain separate. Existing first-frame IQ2/SSM/head/latent/attention gates stay intact.

S0 must preserve ALL128 parent raw vectors, text, IDs, batches, positions and rows. C0 parent comparison is report-only on common prior history; do NOT apply the MoE observer's both-arm equality gate. Cross-arm raw comparisons end at and include the first differing output. Save failures, do not rerun for agreement. Raw logits/tensors remain remote; no weights, raw vectors, full source archive, or generated executable enters Git.

Retain existing-style `build-attempt-r1/`, pre/post-build checks, five-binary/source manifests, `run-r1/` controller/declaration, create argv, native/startup/diagnostic/server logs, requests/responses, raw captures, per-arm alignments, summary/analysis and restoration evidence. Cleanup restores the SAME original container ID, Config, HostConfig, all mounts, all five binaries, source hashes, unchanged hardware, health/READY and sole running container. Failed candidate containers/results are retained as evidence. No timing or promotion regardless of outcome; publication checklist is intentionally incomplete (no standard performance/quality suite).

## Worker verification

**Preparation close gate passed:** 66 remote CPU tests; final log `worker-scratch/cpu-tests-r3.log` (54.901 s), environment/asset hashes `worker-scratch/worker-environment.txt`. Host: inference-host, Linux 7.2.0-1-cachyos x86_64, Python 3.14.7, GCC 16.2.1. Exact command, through the SSH/Bash form above:

```bash
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2/worker-scratch/assets
python3 -B test-trace.py > ../cpu-tests-r3.log 2>&1
```

The suite includes Python AST/duplicate-method checks, `bash -n build-candidate.sh`, fuzz-zero patch dry-run/application against scratch copies, exact unchanged-source/old-fixture comparisons, native parser/count/type/shape/stride mutations, saved-real-capture analyzer wiring, all-frame router activation rejection, nested drafter separation, both-arm comparison semantics, partial-launch regressions and all five pinned parent binary hashes. New numerical test translation unit passes host `g++ -std=c++17 -fsyntax-only` with parent `ggml/include`, `include`, `common`, `tests` includes. The actual new C++ preflight and dispatcher prefix also compile against real ggml public structs with `g++ -std=c++17 -O1 -Wall -Wextra -Werror -fsanitize=undefined`; a CPU spy checks **85 exclusions**, both distinct column pointers/results, full metadata preservation, canaries, null extras and failure of second-column preflight before any singleton call. Buffer ownership is checked before interpreting backend `extra`. These are source/CPU contract tests, **not** MKL/SYCL numerical evidence.

Retained failed attempts: `worker-scratch/preparation-attempt-r1.txt` records local shell quoting rejection before SSH (no local workload); `cpu-tests-r1.log` retains seven test-fixture/parent-adaptation failures and two extractor errors. Corrected tests passed in `cpu-tests-r2.log`; the final buffer-check ordering improvement and foreign-extra exclusions passed in r3. `patch-dry-run-r1.log` retains the initial clean source application check. No model reruns occurred.

**Not run:** IntelLLVM/SYCL build, actual SYCL native numerical/route gates, GPU/model/API journey, restoration exercise, fidelity comparison, timing or promotion. These require the lead's build and full public-boundary journey above. Remaining risk: supported real-model descriptors must satisfy the intentionally narrow non-view preflight; otherwise actual-route gates fail rather than accepting text coincidence. Even successful singleton activation may not improve fidelity. Accepted runtime, protected trees, weights, services and hardware were not modified.

## Parser correction and one-use continuation

The saved S0 failure was a parser attribution failure, not missing route output: asynchronous `BOLT_VERIFY` batch markers can arrive after the direct `BOLT_ROUTE` records for the same target call. `route_frames()` now parses the two logger streams independently, anchors the target backend once from the unique block-0 SSM records, and matches the complete direct target-operation stream to the independent FIFO batch stream by exact order and retained-stride-derived width. It does not assume cross-logger positional containment, equate a backend pointer with a llama context, zip-truncate, or silently drop leftovers. It requires one complete target route group per `BOLT_VERIFY` frame, in order, with exact cardinality and no unknown leading widths. Other-backend routes remain separate.

Prefill widths 5/46/4 are explicitly accounted for. The real `analyze_arm` path now requires the exact frozen batches and positions `5@[0..4]`, `46@[5..50]`, and `4@[51..54]` before alignment or route attribution. Each post-prompt frame requires the complete target operation sequence and strict route/type/shape/stride/reorder/precision checks; n1 requires one router pair and n2 requires two singleton pairs with original n2 strides. The n46 inner router operation remains DNNL F32. Original width is derived from retained strides rather than operation names.

Regression coverage includes the real saved position-70 six-record logger shift, synthetic shifts across verification boundaries, extra target-prefix/head and unknown-leading-VERIFY rejection, changed/reordered/missing prefills through `analyze_arm`, dropped/extra/bad-width operations, reordered nonidentical-width groups, malformed/duplicate records, nested drafter backend separation, wrong n/stride/type/context, and partial target groups. The continuation regression verifies exact original failure/traceback, restored state, S0-only phase, absent C0, frozen payload/create argv, five-binary/source gates, S0 full-128 parent equality, corrected 127-frame router attribution, constant-only controller rewriting, and C0 report-only semantics without an operand-observer or both-arm parity gate.

Observed on `mike@inference-host` in the separate parser scratch `/home/mike/b70-evals/parser-scratch-router-log-order` (no continuation execution):

- `python3 -B -m py_compile analyze-trace.py test-trace.py continue-c0.py`: pass.
- `python3 -B -m unittest -v test-trace.py`: **70 tests passed**; new retained log at `test-trace-router-log-order-followup-final.log`.
- Corrected saved-S0 analysis: 130 verification frames, widths `5,46,4,1...`, 15 target routes at position 70, 127 post-prompt router batches, 128 samples, parent comparison through ordinal 128 with no difference, and S0 parent invariant true.
- `continue-c0.py` was CPU-tested against saved S0 only. Its continuation `main()` was not run. No C0, native/model, fidelity, timing, or promotion result is claimed; timing and promotion remain false.

The parser/continuation implementation used the official Python contracts for ordered JSON decoding, argument-list subprocess execution, and unittest assertions: https://docs.python.org/3/library/json.html, https://docs.python.org/3/library/subprocess.html, and https://docs.python.org/3/library/unittest.html.

## Lead result — router control is not a fidelity fix

Executed on `mike@inference-host`, through explicit local/remote Bash with the SSH flags above:

1. `bash build-candidate.sh` — real IntelLLVM/SYCL build, dependency checks, protected-source and five-binary checks passed; only SYCL and native-test binaries changed. `python3 -B test-trace.py` then passed66 tests (`lead-cpu-r1.log.gz`). Unified-diff blank context lines were retained verbatim; the initial Git whitespace check flagged those patch-format lines, not changed C++ whitespace. Fuzz-zero application and compilation passed.
2. `python3 -B run-trace.py` — initial S0 passed native/API/request gates, but all-frame route attribution failed at position70. The failed `run-r1/summary.json`, controller log and full saved capture are preserved. C0 had not started. Source/recovery review is recorded in `review-notes.md`.
3. After parser-only correction, `python3 -B test-trace.py` passed **70 tests** (`lead-cpu-parser-r2.log.gz`). `python3 -B continue-c0.py` revalidated SAVED S0, ran only missing C0 once under `continuation-c0/run-r1/C0`, and completed strict comparison/restoration (`continuation-c0.log.gz`, `continuation-comparison.json`). No rebuild or S0 inference rerun occurred.

Both arms passed the inherited native385, OUT_PROD190/9, IQ2six, latent7, attention13, attention-view6 and the new router6 numerical/actual-route cases, followed by fresh API20/20. Both ordinary and diagnostic responses agree within their arm. S0 preserves all128 IQ2-parent raw vectors, text/IDs/batch inputs/positions/rows. The corrected router gate covers all127 S0 singleton batches and **all67 C0 two-token batches**, each C0 router operation represented by two real singleton outer/inner pairs with retained original-two-column strides. C0 accepted60 of67 proposed draft tokens.

**Fidelity remains unresolved.** The first S0/C0 raw-vector difference is still output2/position55, before any rollback. Cross-arm raw comparison stops at and includes the first emitted-token difference, now ordinal41: S0 selects3367 and C0 selects1032, each its own raw argmax. The IQ2 parent differed at40. This one-token shift is not evidence of improved quality or performance. C0 also differs from its own IQ2 parent in raw scores from2 and emitted IDs from40; that comparison ends at40. No comparison beyond equal prior history is used for attribution.

Candidate SYCL SHA256: `b2b3b7950c471bb5658f8474eb1866157fd3311a183e07d03a665842a8cad3d9`; native test: `5f43858b1cbd1083ced719a8700f93c30a0e68e94f307541f0bee9e43f5e8389`. Wrapper, implementation and common libraries remain identical to the computational parent. Full manifests are retained.

The continuation checked restoration against BOTH runs. A separate read-only `check_restoration` invocation rechecked the live original container identity/config/HostConfig/all mounts, all five binaries, protected sources, hardware, retained READY responses, live health and sole-running-original condition; see `post-run-verification.log.gz`. The original accepted archive/container remains unchanged. An auxiliary metadata-print command initially used the nonexistent sample key `token` and exited with KeyError; it was corrected to `selected` and rerun on saved JSON only, not inference.

Raw logits remain remotely under `run-r1/S0/capture` and `continuation-c0/run-r1/C0/capture`. S0's validated alignment is `reanalysis-S0.json`; its original failed cell is not rewritten. Local Git includes scoped metadata and lossless gzip logs (mtime0), plus worker/parser CPU logs under `worker-checks/` and `parser-checks/`; no full scratch source, weights or raw logits. Timing and promotion remain prohibited.

Next bounded step: reuse the established router/latent-down operand observer on this new computational parent, with BOTH-arm all128 observer parity required. Directly test whether the actual router outputs now match; dispatch activation alone is not proof. Do not change further arithmetic until that observation is available.
