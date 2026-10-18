# Bolt contiguous checkpoint-window output observer

## Scope and status

This is one bounded, development-only, observation-only observer adaptation for Bolt LiLiCorr. Fidelity is required before timing or promotion, and the accepted runtime remains unchanged. It captures complete post-residual F32 block outputs at physical checkpoints **7 through 15 inclusive** to locate the first unequal complete boundary in **8 through 15**. It does not identify an internal operation cause and makes no claim about unobserved later blocks, whole-model fidelity, performance, or promotion.

The computational parent is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1` (not an observer checkout). The lead-owned disposable candidate is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-block-window-r1`; the result root is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-window`; the container prefix is `bolt-lilicorr-block-window-r1-`; and the patch is `block-window.patch`.

Physical IDs and ordered slots are distinct. Source names are `nemotron_h_block_out-<physical-id>`, files are `block-<physical-id>.f32.bin`, and the `block` log field is the physical ID. The ordered set is `(7,8,9,10,11,12,13,14,15)`, with checkpoint 7 retained as the known-equal anchor. Each S0 arm emits 9 x 16 KiB = **144 KiB** of capture payload; each C0 arm emits 9 x 32 KiB = **288 KiB**. Only token 0 (16,384 bytes) is compared across arms after all BOTH128 parent gates pass; C0 column 1 is never compared.

## Source and runtime boundary

Only `ggml/src/ggml-sycl/ggml-sycl.cpp` is patched, and only `libggml-sycl.so.0` may differ from the computational parent. The hook remains after view/non-compute skips and before `ggml_sycl_fuse` at the existing `RMS_NORM` consumer. It snapshots exact `src[0]`, accepts only the materialized F32 `ADD` source (including MoE ADD+ADD fusion), and uses the existing `ctx.stream()` in-order queue with `memcpy(...).wait_and_throw()` plus lifetime-safe error waiting. Ownership, device, contiguous layout, bounds, non-view, marker, exclusive-file, and same-backend/same-queue guards remain fail-closed. No tensor callbacks, arithmetic/fusion/router changes, new knobs, framework, or parser refactor are introduced.

The private container capture mount and environment are both `/block-window`; the worker verifies the C++ directory literal, patch, run mount, environment, and spy fixture agree. The private remote scratch is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-window/worker-scratch`. Raw vectors, logits, weights, full source trees, and executables stay outside Git.

## Fidelity gates and lead-owned E2E process

The inherited direct-BOLT_ROUTE independent FIFO parser, exact target route groups and cardinality/width/stride checks, frozen prefill widths 5/46/4, native385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + attention-view6 + router6 gates, asymmetric S0/C0 mapping, BOTH128 immediate-parent raw-byte/input parity, and restoration controls remain unchanged. Common BOLT_VERIFY and BLOCK_OUTPUT records use their own FIFO streams; cross-stream order is never compared. The first diagnostic target is position 55, token 0, ordinal 2, with S0 width 1 and C0 width 2.

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
