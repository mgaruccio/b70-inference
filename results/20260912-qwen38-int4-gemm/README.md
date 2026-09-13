# Native small-batch INT4 GEMM investigation

Development-only follow-up after rejecting grouped-query Split-K. Investigate existing oneDNN dispatch before any custom kernel. Production remains unchanged: pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`, MTP4, GPTQ symmetric INT4 G128, FP16 compute, FP8 KV, capacity 212992/batch 8192, B70 at 275 W.

## Source findings and fresh research

- <https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h>: native W4A16 uses oneDNN matmul with group scales/zero points and FP16 fpmath. Primitive caching keys include M/N/K and leading dimensions. The packed weight layout is significant, not interchangeable by changing strides alone.
- <https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/tests/test_int4_gemm_onednn.py>: canonical packed-K-contiguous weight transform, symmetric scalar zero point 8, group size 128, `rtol=atol=0.01` numerical comparison.
- <https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/tests/register_ops.py>: explicitly import `vllm_xpu_kernels._xpu_C` to register GEMM. The package initializer alone registers attention, not this operator.
- <https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html>: `ONEDNN_VERBOSE=profile,filter=matmul` reports implementations, descriptors and cache/creation/execute information. Verbose and queue profiling add non-negligible overhead; these timings cannot be used as final throughput results.
- Installed image source `vllm/model_executor/kernels/linear/mixed_precision/xpu.py:105-121` confirms the actual call: reshaped FP16 activations, `w_q.t()`, optional bias, scales, zero points, group size, group indices. No change to activation quantization or model weights is tested.

Initial web index searches returned no matches; direct official source/doc fetches succeeded. One SSH banner timeout occurred before source inspection; the single retry succeeded. No GPU experiment was started by that failed connection.

## Predeclared process

1. On idle `inference-host`, check launcher SHA `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, power `275000000`, no running containers, stopped Glimmer.
2. Native operator test with synthetic operands at the measured M5 shapes: gate/up K5120,N34816; down K17408,N5120. Identical packed weights/scales for native M5, padded M8/M16, and five M1 calls. Include padding/copies/concatenation in captured graphs. Compare all outputs against native M5 and 32 evenly spaced columns against independent CPU unpack/dequantize/FP32 accumulation, using the upstream 0.01 tolerances. Verify mutated-input graph replay, then 12 rotated/interleaved samples per route after three warm calls/replays. Capture verbose dispatch separately; **disable verbosity for timing**. No serving-performance claim from synthetic operator timings.
3. Real application boundary: reuse the annotated eager profile driver for a streamed 65536-input/128-output HTTP completion, greedy seed42, prefix cache/thinking off. Profile starts only after first nonempty output, delays three decode steps, and captures five; validate full output usage, parse/finite metrics and completion. Enable oneDNN matmul diagnostics solely to identify actual serving dispatch. Retain raw request/SSE, launcher/environment, server log and trace reference. This is profiling, not a new throughput baseline.
4. Only consider serving A/B if a native route passes and looks promising. No automatic deployment. Remove owned containers and recheck unchanged host contract. Retain failures and negative findings; no generic tuning infrastructure or package upgrades.

ML runtime stays in the existing immutable container, never Pi. Stage the two scripts into the same-named remote campaign directory; the profile wrapper also reuses `target-annotations.py` from `20260911-qwen38-target-verification` and canonical `scripts/experiments/qwen38_step_timing_patch.py` staged alongside it. Previous driver directories must remain present.

## Commands

On inference-host:

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-int4-gemm
IMAGE=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
# Run serially on the idle GPU. /output is the campaign directory.
docker run --pull=never --rm --name b70-native-gemm --device /dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE -e ZE_AFFINITY_MASK=0 \
  -e ONEDNN_VERBOSE=0 -v "$R:/experiment:ro" -v "$R:/output" \
  --entrypoint /opt/venv/bin/python "$IMAGE" \
  -P /experiment/probe-native-gemm.py --out /output/operator-03
# Separate process: same command with ONEDNN_VERBOSE=profile,filter=matmul,
# --describe and --out /output/dispatch-02. Never compare its elapsed times.
python3 -u "$R/profile-native-gemm.py" --out "$R/serving-profile-01"
```

Use new output directories on retry. Preserve console logs alongside result directories. Native Lab preview was attempted but unavailable (`connect ENOENT .../lab.sock`); no Lab config is touched.

## Results

- `operator-01` and `dispatch-01` failed before the first GEMM because the probe imported only the package initializer. The retained exception is `_OpNamespace '_xpu_C' ... no attribute 'int4_gemm_w4a16'`. Adding the explicit native extension import follows upstream tests; no runtime/package change was made.
- `operator-02` passed all eight route/shape numerical comparisons and mutated-input graph checks. Pad8 and five M1 calls matched native5 exactly on tested operands; pad16 did not (up to 0.0078125 initially, 0.015625 after input mutation), although it met the upstream numerical tolerance. Single-replay timings were noisy, especially for the down projection; retain them but use the batched follow-up rather than treating a tiny difference as a win.
- `dispatch-02` passed. All tested shapes use oneDNN `jit:gemm:any` with `wei:u4 ... blocked:ba`, FP16 fpmath, G128 scales and scalar zero point. This identifies the implementation family; it does not prove identical internal tiling across M values.
- `serving-profile-01` passed the real 65536/128-token HTTP request, profiler controls and finite/parse checks; host unchanged. Its log reports oneDNN **3.12.0**, commit `80afa71049cd69a3df32adcccb623b12cd7baa22`, SYCL/Level Zero on B70, and the same expected `jit:gemm:any` W4A16 implementation for actual model operations. Verbose M5 gate/up and down medians are approximately 0.198 and 0.100 ms, but verbosity/profiling overhead makes them diagnostic only—not throughput evidence.
- `operator-03` passed repeated qualification and **12 rotated/interleaved batches of 16 graph replays per route**, normalizing each event interval by 16 to amortize event/host overhead. Raw batch durations and normalized samples are both retained. No correctness tolerance was changed.

Raw native trace remains on inference-host:
`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-int4-gemm/serving-profile-01/profile/rank0.1789185296697807176.pt.trace.json.gz`

Size: 5715819 bytes. SHA256: `74170554eb482904068c0cd0d6856921f3f96ca405a50a6e1a1dd1454623c64f`. Local server/request/metadata artifacts are retained alongside this reference.

Trace inspection confirms exactly **five target-root ranges** and **five `execute_context_0(0)_generation_1(5)` ranges**, with no prefill execute-context range in the captured window.

### Batched operator findings

Median normalized milliseconds per route (gate/up; down), with inclusive IQR in parentheses:

- Native M5: **0.186590 (0.000683); 0.111078 (0.000658)**.
- Pad8: **0.190184 (0.000568); 0.108644 (0.000629)**. Gate/up is 1.93% slower, down 2.19% faster. The small down-only difference is about **2.43 microseconds** per isolated call, not an established serving speedup.
- Pad16: **0.195775 (0.001056); 0.117041 (0.000838)**, roughly 5% slower on both, with the observed rounding differences noted above.
- Five M1 calls: **0.832972 (0.000820); 0.436997 (0.001305)**, approximately **4.46× / 3.93×** native latency.

Adding the two shape medians as a rough operator-pair comparison gives 0.297668 ms native versus 0.298828 ms pad8: no shared dispatch improvement. This sum is not a measured model-step latency. The probe uses repeated synthetic/hot weights; it is not the real model's full weight stream. The minor down-only result was not promoted to a new serving path or A/B campaign.

**Decision:** keep native M5. No useful general padding/decomposition change was found for the two measured dominant GEMMs. This does not prove the oneDNN kernel optimal or rule out native-kernel work; it rules out these inexpensive dispatch alternatives as a compelling next optimization. No custom GEMM, activation quantization, weight repacking, deployment or DSpark/DFlash change was made.

Final checks found the launcher SHA and 275 W cap unchanged, Glimmer exited, no running containers and no render-device holders. All failed probes, successful qualification, verbose dispatch, and real-request artifacts are retained.

## 2026-09-13 follow-up: installed native policy audit

**No supported runtime candidate found; no new GPU benchmark or serving A/B.**
After the bounded attention split-count trial, inspect the native small-M
GEMM policy before repeating any experiment. Baseline remains best MTP4;
no padding/M1 decomposition rerun, custom kernel, precision change or package
upgrade is part of this audit. The earlier eager gate/up sum11.3ms identifies
where time is spent, not achievable savings or graph-mode serving latency.

Fresh exact-revision sources (lead checked the implementation passages):

- [oneDNN commit80afa71049cd69a3df32adcccb623b12cd7baa22](https://github.com/uxlfoundation/oneDNN/commit/80afa71049cd69a3df32adcccb623b12cd7baa22)
  exists upstream and corresponds to v3.12 release notes, matching the revision
  observed by the earlier real serving profile. No newly inferred runtime version.
- [jit.hpp](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit.hpp):
  `gen_t::pd_t` names itself `jit:gemm:any`, derives compute mode from attributes,
  and calls `select_kernel` with device/shape/layout information, then tries
  internal catalog entries. The Python GPTQ operator has no policy argument.
- [jit/gen_kernel.cpp](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/gen_kernel.cpp):
  `gen_nocopy_desc_t::select_kernel` selects catalog entries. Important nuance:
  there **is** a `GEMM_KERNEL` strategy override in `gen_desc_t::finalize`, but
  only inside `#ifdef DNNL_DEV_MODE`. Its own warning says it overrides problem
  datatypes and can cause inaccuracies for incompatible precision/layouts.
  It is not a supported production runtime tiling control. `ALLOW_IACC` applies
  only when both operands are integer and grouped scales do not require floating
  DPAS; it is not applicable to this FP16×INT4 G128 path.
- [intel/utils.hpp](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/utils.hpp):
  `gpu_utils::dev_getenv` returns defaults without reading the environment when
  `DNNL_DEV_MODE` is absent. This also bounds the generator-DSL controls.
- [jit_xe_hp_systolic.cpp](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit_xe_hp_systolic.cpp):
  `pd_t::init` accepts matched FP16/BF16 operands or int8 pairs, not mixed W4A16.
  Its attractive small-M unroll policies therefore are not a usable substitute.
- [native W4A16 wrapper](https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h):
  explicitly sets FP16 fpmath for FP16 input. Changing global default fpmath
  is not an exposed tiling control and does not override this explicit choice.

### Installed evidence and checks

`native-policy-audit-20260913.log` retains a fresh CPU-only inspection of the
same immutable MTP image on inference-host. Container used `--pull=never --rm
--name b70-gemm-policy-inspect --entrypoint /bin/bash` and **no GPU devices**.
Inside `/opt/venv/lib/python3.12/site-packages`, checks were:

```bash
F=vllm_xpu_kernels/_xpu_C.abi3.so
sha256sum "$F" vllm/model_executor/kernels/linear/mixed_precision/xpu.py
head -4 vllm_xpu_kernels-0.1.12.3.dist-info/METADATA
sed -n '103,122p' vllm/model_executor/kernels/linear/mixed_precision/xpu.py
for S in GEMM_KERNEL ALLOW_IACC enable_generator_dsl generator_dsl_specialize DEFAULT_FPMATH_MODE; do
  if strings "$F" | grep -Fx "$S"; then echo "present: $S"; else echo "absent: $S"; fi
done
nm -C "$F" | grep -E 'int4_gemm_w4a16|gen_t::pd_t::init|xe_hp_systolic_t::pd_t::init|gen_desc_t::finalize|enable_generator_dsl'
```

Installed native extension SHA256 remained
`71c21e5231908cfa67f45b389de7cef6e96e564c769198a675725197234d07fc`.
Wrapper SHA256 was
`de94f0fc2813c5f86e44369490809a7c1aab44cac8e20e5a64a5a1896ba7d9ba`.
All four development-control literals were absent; `DEFAULT_FPMATH_MODE` was
present. Native W4A16/gen/systolic symbols were present. Literal absence alone
is not a general ABI/build proof; combined with the exact-revision compile
and datatype guards, no reachable supported alternative was established.
No fake tuning env var was passed to the server and no speedup is claimed.

Inspection exit0; final host check at2026-09-13T00:15:39-04:00 showed no running
containers or reported render-device users, unchanged launcher SHA and275W.
Several initial guessed upstream source paths returned404; the valid paths
above were subsequently retrieved. Native Lab preview was again unavailable.

**Decision:** stop this configuration-only route. A different GEMM tile policy
would require a separately scoped source/kernel/build experiment. No application
code was modified, so no new numerical/graph/HTTP test was applicable; the prior
real-request dispatch evidence remains the serving-path evidence, not a new
A/B. Do not represent this audit as proving the current kernel optimal.
