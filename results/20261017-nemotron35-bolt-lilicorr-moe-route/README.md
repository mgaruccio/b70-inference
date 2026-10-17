# Bolt first-MoE-block host route observation

Observational follow-up to2ccb554e. The head-MMVQ and NVFP4-width2 controls leave a score difference from token2 before rollback and output difference at116. The last control changed C0's actual NVFP4 route but left all128 raw vectors in each arm identical to its parent. Do not repeat that control or infer that first-block SSM routing explains the remaining difference.

## Fresh research and scope

Fetched pinned official source https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/fused-gemm.cpp into the remote root. Its `fg_visit_type` includes IQ2_XXS (line963). Actual routes still require observation, not inference from source eligibility.

The existing stdlib `prepare-draft.py:header` inspected the actual target GGUF read-only. `first-blocks-header.json` preserves names/shapes/types: block1 expert up/down and shared-expert up/down are IQ2_XXS (type16), latent down/up are Q8_0 (type8), router gate input is F32. Block0/2 SSM projections are NVFP4 (type40); enum values verified against pinned ggml.h. Earlier NVFP4 MoE exclusion audits do not describe the actual IQ2_XXS block.

Clone `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-nvfp4-width2-r1` into `...-lilicorr-moe-route-r1`. Change ONLY the existing host-side `bolt_route_selected_weight` predicate in ggml/src/ggml-sycl/ggml-sycl.cpp: additionally select names beginning `blk.1.`. Keep block0/output selection, all route call sites, kernels, dispatch predicates, synchronization, precision, capture schema and flags unchanged. No tensor readbacks or callbacks. Keep inherited head-MMVQ, width2 NVFP4 initialization and legacy-only SSM controls unchanged. This is not a computational intervention.

## Predeclared end-to-end process

All execution/analysis/tests/build/inference on mike@inference-host through explicit local AND remote Bash; SSH BatchMode=yes,ConnectTimeout=8,ControlMaster=no,ControlPath=none,ForwardAgent=no. Remote root `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-moe-route`. Local editing/git only. Lab preview/show_patch unavailable; do not retry/repair. No packages on host, persistent launchers, accepted-source/weight changes, or hardware changes.

CPU-only build with existing Intel2026/oneDNN3.14 and six jobs; fuzz0 patch and source scope exactly one host filter/file. Protect accepted and parent five artifacts. Only candidate SYCL library may differ; wrapper/server implementation/common/backend tests match parent. Reserve4GiB for captures.

Same fresh S0(max1/min2 suppressed) then C0(max1/min0 active), n_rs_seq1/K2, block8,p_min0,greedy draft,no draft backend sampling,ngl999. Same model/drafter, frozen55-token prompt/128 outputs, warmup, ordinary stream and separate nonstream full-score diagnostic at http://127.0.0.1:18098. Context65536,batch/ubatch1024,parallel1,q8KV,F32state,graphs0,FUSION/ESIMD/OPT/VMM/DNN1,BOLT_TRACE_TENSORS0,ONEDNN_VERBOSE0. Hardware boost0/sixteen3801000kHz CPU caps/275W. Capture enabled only by existing BOLT_VERIFY_CAPTURE_DIR.

Each arm retains native385 plus190 numerical/9 support checks, zero custom-prefill dispatch, API20/20, startup/finish/count/cache checks, ordinary/diagnostic consistency, strict128-row framing/path/size/batch/position/emission/acceptance/removal checks and API/log count agreement. Keep known actual head and SSM route/type/reorder/1-or2-column gates. Partition target backend via block0 SSM anchors; preserve nested drafter/other-backend routes separately, never compare backend pointers to llama context pointers.

Require actual target block1 expert up/down and latent down/up route records in selected S0 singleton batches55/56 and C0 two-row batch55/56, with canonical iq2_xxs/q8_0 types respectively. Record shared-expert/router routes where existing call sites emit them. Do not require a guessed MoE route or fabricate missing fusion evidence; missing required evidence fails closed. Preserve dimensions/strides/IDs/precision hints/reorder/result fields as recorded.

**Both arms must preserve all128 raw vectors byte-for-byte against their own NVFP4-width2 parent**, along with emitted IDs, prompt IDs, ordinary/diagnostic text, actual input batches and selected row/position metadata. Restore the full BOTH-arm observational gate; the previous computational control gated only S0. Any observer difference blocks attribution, with failures retained and no retry-until-match. Compare S0/C0 only through identical prior output history including the first differing output.

Commands: `bash build-candidate.sh`, `python3 -B test-trace.py`, `python3 -B run-trace.py`, `python3 -B analyze-trace.py`. Stop experimental arms before final log capture. Nested-finally restore the SAME original nemotron-bolt-probe; independently verify ID/config/HostConfig/all mounts/five hashes/hardware/health/READY. Leave experimental containers stopped. Store raw arrays remotely only, metadata and deterministic compressed logs in scoped Git evidence. No timing or promotion. Use the observed routes to choose one subsequent computational control, not multiple simultaneous changes.

## Observed result

Both arms passed the full observational gate: all128 raw vectors, IDs/text, input batches and selected row/position metadata match their respective NVFP4-width2 parents. Native385 +190 numerical/9 support, API20/20 per arm, within-arm text, trace alignment and independent original-container restoration passed. Candidate SYCL SHA256 `7e365cb794637ad9b20fe9b7472932fcfafa822b2cb5c1577d82b2f41d51e2f6`; other four artifacts match parent. First score difference remains2 before rollback, first output mismatch116; C0 accepts58/69. No timing or promotion.

Actual target routes at S0 positions55 and56 versus C0 batch55/56:
- IQ2_XXS expert up/down: S0 `MOE_SINGLE_MMVQ_PLAIN_Q8_1`, C0 successful `MOE_GROUPED_GEMM_RESULT` with result1; reorder0 in both. Up activation dimensions are [1024,1,1] vs [1024,1,2]; down [2048,9,1] vs [2048,9,2]. The nine-axis is expert selection, not nine target tokens.
- Q8_0 latent down/up: S0 `DMMV_ESIMD`, C0 `MMVQ_REORDERED`, both installed reorder1.
- IQ2_XXS shared expert up/down: `MMVQ_PLAIN` in both, with1/2 activation columns.
- F32 router: `MUL_MAT_SYCL` then `LIBRARY_MKL_F32` in both. Equal route names do not prove equal intermediate bytes.
- First-block NVFP4 SSM and Q8_0 output retain the previously aligned reordered MMVQ routes. Nested other-backend records remain separate.

Lead verification strengthened the full analyzer negative test to start from matching real parent captures in BOTH arms, then change only batch metadata. Initial strengthened test lacked its temporary capture directory and failed before the intended check (`controller-tests-r2.log`); fixed the fixture only, retained failure, and31/31 remote tests passed (`controller-tests-r3.log`). Production invariants were not weakened; no inference rerun occurred.

Source follow-up confirms the existing grouped-dequant failure path uses per-expert `ggml_sycl_mul_mat` on gathered rows. The target header contains80 expert-up/down tensors, all IQ2_XXS. `opt_for_reorder_id` excludes IQ2_XXS. Next isolate the two-token IQ2_XXS grouped-GEMM selection, leaving latent projections and other paths unchanged; require unchanged S0 plus actual C0 fallback evidence rather than assume convergence. A two-token guard can also affect two-token prefill, so do not label it decode-only.
