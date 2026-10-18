# Bolt sparse later-checkpoint output observer

## Scope and status

This is one bounded, development-only, observation-only probe for Bolt LiLiCorr. Fidelity is required before timing or promotion; the accepted runtime is unchanged. The probe captures complete post-residual F32 block outputs at the **observed physical checkpoints** `[7, 15, 31, 47, 63, 79, 86]` and compares only token 0 at position 55 after both arms pass the full 128-vector immediate-parent gate. Run tier is a development diagnostic, baseline is the router-single2 computational parent, and the only intentional difference is this default-off SYCL observer/output capture; no performance comparison is run.

The computational parent is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1` (not an earlier observer). The lead creates the disposable candidate at `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-block-checkpoints-r1`. The result root is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-checkpoints`; disposable containers use prefix `bolt-lilicorr-block-checkpoints-r1-`.

Physical IDs and ordered slots are intentionally separate. Source names are `nemotron_h_block_out-<physical-id>`, files are `block-<physical-id>.f32.bin`, and the `block` log field is the physical ID. The first checkpoint repeats the known-equal checkpoint-7 anchor. The analyzer reports every observed checkpoint, `first_differing_checkpoint`, `previous_equal_checkpoint`, and a bounded interval; it never calls an unobserved coarse point equal or labels one as an observed divergence.

Block 87 is intentionally omitted. The `nemotron-h.cpp` final-input path may interpose `GET_ROWS` before the final norm; checkpoint 86 is consumed by norm 87 safely before that last-layer cropping path. No claim extends through block 87, final norm/head, or the model's 88 blocks.

## Source and runtime boundary

Only `ggml/src/ggml-sycl/ggml-sycl.cpp` is patched, and only `libggml-sycl.so.0` may differ from the router-single2 parent. The observer is before SYCL graph fusion at the existing `RMS_NORM` consumer, snapshots `src[0]`, requires the exact complete `ADD`/F32/4096 layout and owning SYCL buffer, copies on the existing in-order `ctx.stream()`, waits for completion, and writes a fresh exclusive file. It is default-off, armed only by the private `/block-checkpoints/armed` marker, ordered once per diagnostic process, and rejects aliases, views, foreign buffers, bad dimensions, mixed backend/queue, stale files, marker links/FIFOs, late errors, and lifetime violations. Backend copies such as `SYCL0#...` are not accepted.

The inherited corrected direct-route FIFO parser, all-frame cardinality checks, frozen prefill checks, native gates, asymmetric S0/C0 mapping, BOTH128 parent parity, restoration identity checks, and no-timing/no-promotion controls remain unchanged. S0 uses `n=1` and C0 uses `n=2`; prefill widths 5/46/4 are not captured. The first post-prompt diagnostic target is ordinal 2, position 55, row 0, with the same backend and queue for all seven captures.

## Defined end-to-end process (lead-owned, pending here)

On `mike@inference-host`, with the original Bolt container healthy and sole-running, the lead must run the declared disposable six-job Intel LLVM 2026 / oneDNN 3.14 read-only process:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc 'cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block-checkpoints && bash build-candidate.sh && python3 -B test-trace.py && python3 -B run-trace.py && python3 -B analyze-trace.py'"
```

The real journey is native385 plus OUT_PROD190/support9, IQ2six, latent7, attention13, attention-view6, and router6 for each arm; fresh API20/20; ordinary warmup and stream; then a separate frozen diagnostic request. It uses prompt position 55, forced 128 outputs, temperature 0, seed 42, cache false, ignore EOS, S0 `max=1/min=2`, C0 `max=1/min=0`, `n_rs_seq=1`, K=2, unchanged capacity/reservation, q8 KV with F32 recurrent state, context 65536, batch/ubatch 1024, parallel 1, graphs 0, and callback verbose off. The run must retain exact command/environment, source and five-binary hashes, request/response/log/alignment evidence, remote `capture/`, and private `block-checkpoints/` files, while keeping raw tensors/logits/weights/full archives/executables out of Git. Restoration must prove the original container identity/config/HostConfig/mounts, hashes, hardware, health, READY smoke, and sole original process. No timing or promotion is permitted.

## Worker checks completed

All executable source-generation and CPU-only checks are run on `mike@inference-host` under the corresponding remote `worker-scratch/assets` directory. The targeted suite compiles the actual patched observer with the remote parent headers, exercises source-copy and token-0/full-width behavior, physical-ID/slot ordering, exact norm consumers, same-queue enforcement, ownership/layout/stride/bounds/view guards, wrong names, stale/exclusive files, marker security, late errors, and host lifetime. It also checks Python AST syntax, shell syntax, patch scope/reversibility, inherited parser identity, and the candidate source boundary. Real Docker/build/GPU/API execution remains lead-owned and pending.

## External research

Fresh official reference: <https://github.khronos.org/SYCL_Reference/iface/queue.html>. Its queue documentation supports the approach of using the existing in-order queue and waiting for the same-queue `memcpy` event before reading host bytes; it does not establish floating-point invariance, so the BOTH128 gate and byte comparisons remain mandatory.

## Non-goals

This result is not a timing comparison, promotion, arithmetic/fusion/router change, full-fidelity claim, or proof about unobserved intermediate blocks. Historical early-checkpoint assets are reused only as a harness precedent; no historical raw activations are copied here.

## Lead preflight corrections

The worker's intermediate capture-directory mismatch was fixed before integration: patch, environment and mount now agree on `/block-checkpoints`. Lead review additionally made checkpoint7 equality a mandatory rejection gate, not merely a reported discrepancy, and added its negative regression plus previous-equal-checkpoint assertions. These are parser/test-only changes; the seven-checkpoint runtime patch is unchanged. Full remote CPU and real E2E validation remain pending at this declaration.

The lead invokes all commands through explicit local Bash and remote `bash -s` over the recorded SSH flags, using fail-fast chaining: build, complete CPU suite, paired controller, analyzer. Any nonzero exit stops the remaining steps. No inference retry or continuation is pre-authorized by a successful targeted worker test.
