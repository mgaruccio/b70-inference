# Bolt contiguous checkpoint-window output observer

## Scope and status

This is one bounded, development-only, observation-only observer adaptation for Bolt LiLiCorr. Fidelity is required before timing or promotion, and the accepted runtime remains unchanged. It captures complete post-residual F32 block outputs at physical checkpoints **7 through 15 inclusive** to locate the first unequal complete boundary in **8 through 15**. It does not identify an internal operation cause and makes no claim about unobserved later blocks, whole-model fidelity, performance, or promotion.

The computational parent is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1` (not an observer checkout). The lead-owned disposable candidate is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-block-window-r1`; the result root is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-window`; the container prefix is `bolt-lilicorr-block-window-r1-`; and the patch is `block-window.patch`.

Physical IDs and ordered slots are distinct. Source names are `nemotron_h_block_out-<physical-id>`, files are `block-<physical-id>.f32.bin`, and the `block` log field is the physical ID. The ordered set is `(7,8,9,10,11,12,13,14,15)`, with checkpoint 7 retained as the known-equal anchor. Each S0 arm emits 9 x 16 KiB = **144 KiB** of capture payload; each C0 arm emits 9 x 32 KiB = **288 KiB**. Only token 0 (16,384 bytes) is compared across arms after all BOTH128 parent gates pass; C0 column 1 is never compared.

## Source and runtime boundary

Only `ggml/src/ggml-sycl/ggml-sycl.cpp` is patched, and only `libggml-sycl.so.0` may differ from the computational parent. The hook remains after view/non-compute skips and before `ggml_sycl_fuse` at the existing `RMS_NORM` consumer. It snapshots exact `src[0]`, accepts only the materialized F32 `ADD` source (including MoE ADD+ADD fusion), and uses the existing `ctx.stream()` in-order queue with `memcpy(...).wait_and_throw()` plus lifetime-safe error waiting. Ownership, device, contiguous layout, bounds, non-view, marker, exclusive-file, and same-backend/same-queue guards remain fail-closed. No tensor callbacks, arithmetic/fusion/router changes, new knobs, framework, or parser refactor are introduced.

The private container capture mount and environment are both `/block-window`; the worker verifies the C++ directory literal, patch, run mount, environment, and spy fixture agree. The private remote scratch is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-window/worker-scratch`. Raw vectors, logits, weights, full source trees, and executables stay outside Git.

## Fidelity gates and lead-owned E2E process

The inherited direct-BOLT_ROUTE independent FIFO parser, exact target route groups and cardinality/width/stride checks, frozen prefill widths 5/46/4, native385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + attention-view6 + router6 gates, asymmetric S0/C0 mapping, BOTH128 immediate-parent raw-byte/input parity, and restoration controls remain unchanged. BOLT_VERIFY and BOLT_BLOCK_OUTPUT share the same common-logger FIFO. Direct BOLT_ROUTE uses a separate stream; its order is not compared against common-logger emission order. The first diagnostic target is position 55, token 0, ordinal 2, with S0 width 1 and C0 width 2.

The lead's frozen journey is a six-job Intel LLVM 2026 / oneDNN 3.14 read-only disposable build and run on `mike@inference-host`, using the declared native/API20 gates, ordinary warmup/stream, and separate frozen diagnostic request: 128 outputs, temperature 0, seed 42, cache false, ignore EOS, S0 `max=1/min=2`, C0 `max=1/min=0`, `n_rs_seq=1`, K=2, unchanged capacity/reservation, q8 KV with F32 recurrent state, context 65536, batch/ubatch 1024, parallel 1, graphs 0, callbacks verbose off. No agreement retries are permitted; failures remain recorded. The worker does not build the candidate or run GPU/API inference.

The lead's recorded invocation remains fail-fast over explicit local/remote Bash:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc 'cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-window && bash build-candidate.sh && python3 -B test-trace.py && python3 -B run-trace.py && python3 -B analyze-trace.py'"
```

The controller finally restores accepted `nemotron-bolt-probe` container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, endpoint `18098`, original identity/config/HostConfig/all mounts, all five binary hashes, unchanged hardware (`boost0/16x3801000`, GPU275W), live health `READY`, and the sole original process.

Before interpreting the contiguous result, after BOTH128 parity, the lead will compare token 0's 16,384 bytes at physical checkpoints 7 and 15 in each arm against the saved sparse captures at `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-checkpoints/run-r1/{S0,C0}/block-checkpoints/block-{7,15}.f32.bin`. This is a predeclared post-run check only; it does not add reference infrastructure or change the analyzer schema.

## Worker checks and research

All executable source-generation, syntax, patch, and CPU checks run remotely on `mike@inference-host` through explicit local/remote Bash with `ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`. The full CPU suite, source patch apply/reverse, pinned source hashes, and observer spy are worker checks; real disposable build, native/API/GPU E2E, captures, analyzer result, restoration, timing, and promotion remain lead-owned and pending.

Official queue reference: <https://github.khronos.org/SYCL_Reference/iface/queue.html>. It supports the existing in-order queue and event completion/wait approach; it does not establish floating-point invariance, so the BOTH128 byte-parity gate remains mandatory.

## Non-goals

No accepted-runtime change, timing comparison, promotion, arithmetic/fusion/router modification, full-fidelity claim, or conclusion about blocks after this observed window is made by this preparation slice.

## Lead-observed result — first unequal complete boundary is block8

On `mike@inference-host`, explicit local Bash piped a fail-fast command sequence through the declared SSH flags to remote `bash -s`: `bash build-candidate.sh`, `python3 -B test-trace.py 2>&1 | tee lead-cpu-r1.log`, `python3 -B run-trace.py 2>&1 | tee controller-r1.log`, then `python3 -B analyze-trace.py 2>&1 | tee analysis-r1.log`, from this remote result root. Build/source/dependency/five-binary gates passed. **All79 lead CPU tests passed in136.043s.** Both arms completed without continuation or inference rerun.

Both arms passed native385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + view6 + router6, fresh API20/20, warmup, ordinary stream and separate diagnostic. All nine complete-boundary captures were attributed to position55/token0. **BOTH arms preserve all128 immediate-parent raw vectors byte-for-byte, plus prompt/IDs/text/batch inputs/selected positions/rows.**

The predeclared endpoint check also passed: for each arm, token0 at physical7 and15 exactly matches its saved sparse-checkpoint counterpart (four comparisons,16,384 bytes each). The check first revalidated both runs' both-arm128 gates, attribution and restoration, rejected symlinks/wrong file lengths, then compared token0 bytes and recorded their SHA256 values in `endpoint-comparison-r1.json`. No model request was repeated for this post-run analysis.

Checkpoint7 remains equal across S0/C0. **Checkpoint8 is the first unequal complete boundary**, starting at byte4/F32 index1. Blocks9,10,11 first differ at byte4;12,13 at byte16;14,15 at byte8. There is no unobserved boundary between equal7 and unequal8. This localizes the current difference to the block8 computation or its inputs/state beyond the equal incoming residual; it does not identify an internal operation or establish equality of persistent state. Final raw scores still first differ at output2 before rollback; emitted tokens first differ at41.

Candidate SYCL SHA256 `e6b27838a1ae715684913995cb94080cefd16840cfc0b750699ba2c53a58ed7e`; the other four artifacts remain identical to router-single2. Controller and independent analyzer restoration checks passed: original container identity/config/HostConfig/all mounts, five hashes, protected sources, unchanged hardware, health, READY and sole original process. Accepted runtime unchanged; timing and promotion remain blocked.

Canonical remote artifacts are `run-r1/analysis.json`, per-arm `alignment.json`, raw `capture/`, nine `block-window/block-*.f32.bin` per arm, and the endpoint-comparison JSON. Local logs and metadata JSON larger than1MB are losslessly gzipped with mtime0. Initial preparation syntax, negative-name fixture and malformed-hunk failures are preserved separately in `worker-checks/preparation-failures.log.gz`, with their final79-test distinction; no inference was run during those failures. Raw activations/logits/weights stay remote.

Correction after inspecting actual GGUF metadata: block8 is MoE, not SSM; block9 is the next SSM. The initial SSM classification was incorrect, but the complete-boundary byte comparisons are unaffected. Block8 has F32 router [4096,304], Q8_0 latent-down/up [4096,1024]/[1024,4096], IQ2_XXS expert up/down [1024,2304,304]/[2304,1024,304], and IQ2_XXS shared up/down [4096,5376]/[5376,4096]. The current singleton-router guard explicitly requires240 outputs, so it does not cover block8; this is a coverage gap, not yet a demonstrated cause.

Next bounded step: reuse the proven router/latent-down operand observer for block8, with exact304 router sizing and only those two additional route selectors. Capture full weights and actual src1 before each matmul, then dst afterward; preserve the existing computational control unchanged. Require both-arm128 parity and compare full weights plus token0 inputs before attributing any unequal outputs to a path difference. If inputs differ, move upstream instead. The direct operand equality check is the relevant input gate for this probe; no additional complete-boundary capture is needed. Disk check found12,706,996KiB free; maintain the existing4GiB reserve, with no cleanup authorized.
