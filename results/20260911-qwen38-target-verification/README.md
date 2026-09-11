# Fixed-depth target-verification internals

Development-only follow-up to `20260911-qwen38-step-profile-64k`. MTP4 remains the performance baseline; DSpark and DFlash are not abandoned. No production launcher/model/power/service changes are authorized.

## Process and sources

Reuse the prior campaign's real HTTP driver, pinned images and archived patches. MTP stays at K4, original capacity 212992 and batching8192. Profile one deterministic 65536-input/128-output request, C1, greedy seed42, prefix cache off, thinking off. Native profiler starts after first nonempty output, delays three later decode steps, then records five steps. The entire 128-token response must validate. Before/after host checks require unchanged launcher SHA,275W, stoppedGlimmer and no running containers; cleanup removes only the owned cell.

`target-annotations.py` labels target `nn.Module` forward ranges only while inside the target root forward. This excludes shared embedding calls from the drafter. It uses the existing disposable `qwen38_step_timing_patch.py` worker-import shim, mounted with the expected module alias; it does not run the previous graph-event overlay. Hooks attach only during a native profile window and are removed at stop. Eager mode and shape recording deliberately add overhead; these are **not throughput measurements**.

Fresh official sources:
- <https://docs.pytorch.org/docs/2.13/profiler.html>: XPU activities; record_shapes/with_stack add overhead, and retained tensor references can alter optimizations.
- <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>: `record_function` labels, CPU+XPU tracing, warmup/scheduled short captures, Chrome trace export.

The pinned vLLM layerwise NVTX helper imports `torch.cuda.nvtx`; this campaign instead uses documented PyTorch user annotations. No profiler installation or runtime upgrade.

## Executed commands

Stage these campaign `.py` files plus canonical `scripts/experiments/qwen38_step_timing_patch.py` into the same-named remote campaign directory, then on inference-host:

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-target-verification
python3 -u "$R/run-target-internals.py" --cell mtp4 --out "$R/mtp4-eager-01"
```

The smoke script runs inside the pinned MTP serving image, not Pi's environment. Its first CPU-only profiler invocation without `/dev/dri` failed: `zeInit returned2013265921`, `PTI_ERROR_INTERNAL`, `Fail to enable Kineto Profiler on XPU ... error code200`. Exposing the device to initialize PTI fixed the smoke; tensors/operations remained CPU-only. `smoke-annotations-02.log` confirms output equality, three expected target annotations, exclusion of a shared-module call outside the target root, and hook cleanup.

## Observed attribution

`mtp4-eager-01/` passed HTTP usage65536/128, finite metrics and host checks. Exactly five native `execute_context_0(0)_generation_1(5)` ranges and five target-root ranges were captured. All kernel events use one device queue. The compressed raw trace stays on inference-host at:

`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-target-verification/mtp4-eager-01/profile/rank0.1789162588794532488.pt.trace.json.gz`

`attribution.json` stores its SHA256. Reproduce with `python3 summarize-trace.py <trace.json.gz>`. Each kernel is counted once, matched through its CPU External id to the innermost target module range. CPU-range timing is not added to device time.275/20080 kernel events lack a CPU-op match, totaling0.639ms over the complete five-step trace; unmatched/outside-target work remains explicit.

Target-forward eager kernel sums, divided by five steps:
- Linear/GEMM kernels:24.564ms. Gate/up projection11.299ms; down projection5.279ms.
- Full attention `_vllm_fa2_C::varlen_fwd`:14.047ms (main split-KV decode13.465ms, reduction0.582ms).
- GDN core `_xpu_C::gdn_attention`:1.284ms.

Full attention receives Q[5,24,256], FP8 paged K/V[176,1664,4,256], query cumulative lengths[6], used lengths[5], block table[5,128]. These observed shapes motivate examining multi-token verification dispatch and KV reuse, not guessing at more acceptance/depth changes. Aggregate subsystem totals include their projections/norms; do not add them to the linear or core-operation totals above.

## Native attention route qualification: rejected

Fresh upstream sources, checked against installed `vllm_xpu_kernels` 0.1.12.3 in the pinned image:
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/vllm_xpu_kernels/flash_attn_interface.py>: small uniform causal queries expand into independent single-token decode sequences, preserving per-query causal lengths. Default `VLLM_XPU_SPEC_DECODE_MAX_QLEN=16` selects this Split-K route; lowering it to 4 routes Q5 to chunk-prefill.
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/flash_attn/flash_api.cpp> and <https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/flash_attn/heuristics.h>: native dispatch and split heuristics.
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/cmake/flash_attn/configs.cmake>: compiled head-size/policy constraints; the actual native operator must qualify, not merely accept a Python argument.

Installed source also shows that `_spec_decode_varlen_fwd` passes `num_splits=None`; the vLLM `_xpu_ops` facade does not forward `num_splits_kv`. It is not a usable tuning control through this serving path. No precision-preserving small-M INT4 GEMM tuning switch was identified.

`check-attention-paths.py` uses synthetic Q[5,24,256] and FP8 paged KV[40,1664,4,256], 65541 live KV tokens, scalar descale views, and the native facade. It forbids reference fallback, checks finite output and numerical agreement, then graph-captures each route and times 12 replays after three warm replays.

Run on inference-host (no other container running):

```bash
test -z "$(docker ps -q)"
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-target-verification
IMAGE=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
mkdir "$R/attention-paths-02"
set -o pipefail
docker run --pull=never --rm --name b70-attention-path-check \
  --device /dev/dri --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -v "$R:/experiment:ro" -v "$R/attention-paths-02:/output" \
  --entrypoint /opt/venv/bin/python "$IMAGE" \
  -P /experiment/check-attention-paths.py 2>&1 | tee "$R/attention-paths-02/driver.log"
```

Use a new output directory on rerun. The disposable container removes itself on exit. Raw JSON/logs are retained in `attention-paths-01/` and `attention-paths-02/`. The first fixture failed both routes because descales were allocated as four scalars; the native API requires a view of one scalar. Changing the fixture to `ones(()).expand(1,4)` matched the serving representation. Both native routes then passed without fallback:

- Current expanded Split-K decode: median **1.192161 ms**.
- Chunk-prefill: median **13.3503645 ms**, **11.1985× slower**.
- Agreement: maximum absolute error **1.52588e-5**, RMSE **2.75896e-6**, cosine **0.99999994**; `assert_close(rtol=0.02, atol=0.0001)` passed.

This is a development-only synthetic operator preflight, not a serving A/B or a throughput claim. It uses 40 allocated pages rather than the model's 176, with matching live 64K geometry. Current-stream event intervals can include host enqueue gaps. The alternate selector is rejected, so it was not promoted to a full serving trial.

## Decision and next boundary

Keep the MTP4 baseline and all production settings unchanged. DSpark and DFlash remain candidates; this campaign directly measured MTP4 target verification, not fresh DSpark/DFlash performance. No speedup or standard-compliant benchmark result is claimed.

The next proposed kernel experiment is to process verification queries/GQA heads sharing KV together **while retaining Split-K parallelism**. Existing decode provides parallelism but repeats KV reads; existing chunk-prefill offers reuse but loses too much parallelism for this small query count. Linear/GEMM remains the largest measured kernel category. No custom kernel or persistent alternate serving path was implemented here.

Any subsequent optimization must retain MTP4 depth, target/draft quantization, acceptance rules and original context capacity. It must pass real API correctness gates and a fresh one-warmup/six-measurement graph-mode A/B before a speedup claim. A kernel prototype is a separate implementation step, not an inferred extension of this profiling campaign.

Native Lab previews were attempted but unavailable (`connect ENOENT .../lab.sock`); no Lab-managed configuration or run was changed.
