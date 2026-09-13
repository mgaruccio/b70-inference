# Native GEMM catalog source experiment

Development-only, explicitly authorized source/build follow-up to the native
policy audit. No promotion or production launcher/power change. No generic tuner,
custom GEMM, padding/decomposition repeat, precision or model changes.

## Predeclared bounds

Installed baseline: best MTP4, immutable image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
native kernels0.1.12.3, Torch2.13.0+xpu. FP16 GPTQ u4 symmetric G128, scalar
zero point8, FP8 targetKV, INT4 S+M1 draft, GDN mixed-split-v5, K4,
context212992/batch8192/C1, graphs[1,2,4,8], prefix off.

Build only on inference-host under
`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-gemm-catalog/`.
Use cached compiler image
`intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`
(Intel DPC++2026.0.0) and existing pinned serving runtime. Copy compiler files
into this scoped directory and mount read-only in disposable build containers;
no compiler/ML install into the host or interactive Pi. No device access during
CPU compilation. Preserve source revisions, patches, commands, logs and failures.
Bound the initial build to90 minutes; do not loop/rebuild indefinitely.

Build oneDNN commit`80afa71049cd69a3df32adcccb623b12cd7baa22` and reuse the
native W4A16 wrapper sources. Public native source tagv0.1.12 is not a proven
byte-identical source for the installed0.1.12.3 wheel, so a rebuilt-default
control is mandatory. Keep original native extension loaded/untouched; any
isolated test binding must use a separate operator namespace and hidden oneDNN
symbols. Do not replace other attention/GDN operators.

Policy change is limited to selecting automatic or at most two alternative
eligible catalog entries for the gate/up geometry M5/K5120/N34816. Preserve
original datatype/layout/group-scale metadata and all finalization, post-op,
determinism and kernel-creation checks. Do not use unrestricted `GEMM_KERNEL`
strings or enable broad development-mode precision overrides. Log chosen
catalog identity; fail on missing/invalid requested entries rather than silently
claiming an override. Primitive caches require a fresh process for each policy.

## Real test process and decision gates

1. Idle host: no running containers or GPU compute users, launcher SHA
   `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`,
   power275000000 microwatts. Builds may run CPU-only; GPU probes run serially.
2. Exercise the actual native W4A16 public operator through the existing
   `20260912-qwen38-int4-gemm/probe-native-gemm.py` fixture conventions, not a
   Python matmul substitute. Gate/up M5/K5120/N34816; control down
   M5/K17408/N5120. Identical packed weights/scales; upstream fixed
   `rtol=atol=0.01`, independent FP32 unpack/dequant reference, finite outputs.
   Include mutated-input captured-graph replay. Compare rebuilt automatic with
   installed native before testing alternatives. Retain failures and all samples.
3. Capture verbose catalog/implementation diagnostics separately from timing.
   Timed processes have verbose off. Warm compilation and graph capture first;
   measure12 alternating batches ×16 graph replays against the installed
   baseline in each candidate process. Report medians/IQR, full samples and
   exact candidate identity. Rebuilt automatic parity is not assumed.
4. Require at least5% lower gate/up median with no greater than5% down-control
   regression and passing numerical/graph gates before serving integration.
   This is a development screen, not a statistical throughput claim.
5. Only an operator winner gets a disposable serving adapter and real HTTP
   fixed-MTP4 A/B using existing step-profile/best-three lifecycle. Preserve
   depth/precision/capacity/acceptance/seed and prefix policy. Run existing
   canary/finite-boundary/functional gates, then cold512/8192/32768/65536-input,
   128-output greedy seed42 streaming requests, one warmup plus six measured
   per point. Validate SSE termination/token usage, finite metrics, text/token
   divergence and speculative counters. Preserve raw requests/responses, exact
   argv, source hashes and logs. No promotion or full-quality claim.
6. If no buildable/valid/useful alternative exists, report that result and stop;
   no forced serving trial. Remove owned containers, verify host invariants,
   retain bounded evidence and commit/push completed results.

## Fresh primary sources

- https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit.hpp
  `gen_t::pd_t::init` selects catalog entries and retains finalization, post-op,
  determinism and kernel creation checks. Use these checks, not a custom kernel.
- https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/gen_kernel.cpp
  Automatic catalog selection; unrestricted development override can alter
  datatypes and is not appropriate for a precision-preserving experiment.
- https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/CMakeLists.txt
  Pins matching oneDNN commit and Torch2.13; independent extension build toggles.
- https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/xpu/onednn/onednn_matmul.cpp
  Native W4A16 function uses original output layout/device guard, optional group
  indexing and oneDNN wrapper. Reuse rather than rewrite numerical computation.

Native Lab previews unavailable (`connect ENOENT .../lab.sock`); scoped CLI only.
