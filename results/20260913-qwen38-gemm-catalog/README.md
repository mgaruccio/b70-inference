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

## Executed build and failure record

All following commands were invoked locally with explicit Bash; scripts perform
remote operations. Snapshots in each attempt preserve the exact version used.

```bash
bash results/20260913-qwen38-gemm-catalog/prepare.sh
bash results/20260913-qwen38-gemm-catalog/build.sh build-01
# After diagnosing each failure, reverse only its recorded patch on the remote
# oneDNN checkout with git apply --reverse --check, then git apply --reverse;
# git diff --exit-code confirms pristine source for guarded reapplication.
bash results/20260913-qwen38-gemm-catalog/build.sh build-02
bash results/20260913-qwen38-gemm-catalog/build.sh build-03
bash results/20260913-qwen38-gemm-catalog/run-probe.sh operator-auto-01 -1 timing
bash results/20260913-qwen38-gemm-catalog/run-probe.sh operator-index1-01 1 timing
bash results/20260913-qwen38-gemm-catalog/run-probe.sh operator-index2-01 2 timing
python3 results/20260913-qwen38-gemm-catalog/analyze.py
```

- Preparation succeeded. Native tagv0.1.12 resolved to
  `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b`; oneDNN resolved to the pinned
  `80afa71049cd69a3df32adcccb623b12cd7baa22`. The native checkout remains
  unmodified. The installed0.1.12.3 wheel's exact source commit is still unknown.
- `build-01` failed in upstream `gpu_sdpa_list.cpp:35`: the reduced primitive
  list `MATMUL;REORDER` produced an invalid registration-map constructor.
  No unrelated SDPA source fix was added. Retry uses the standard `ALL`
  primitive set, still Xe2-only, with graph component disabled.
- `build-02` compiled and installed static oneDNN, then failed in Torch's
  binding CMake discovery because `SYCL_ROOT` was unset. The copied compiler
  lives at its versioned path, not the default `compiler/latest` path expected
  by the installed `FindSYCLToolkit.cmake`. Set `SYCL_ROOT` explicitly.
- `build-03` succeeded with the same catalog patch; both previous logs and
  exit codes remain. Effective CMake cache: Release, static/PIC, CPU runtime
  NONE, GPU SYCL, Xe2, ALL primitives, development mode OFF, graph OFF.
  CPU build containers had no GPU devices,12 CPU limit,24GB RAM/no added swap.
  Compiler was2026.0.0. Benign CMake warnings about SYCL package naming and
  optional Kineto library discovery were retained; linking succeeded.

The final `build.sh` incorporates these two build-configuration corrections;
it is not byte-identical to the failed-attempt snapshots. The patch generator
emits a diff and refuses changed source/repeated application; the driver checks
and applies it explicitly. Final oneDNN diff is exactly67 insertions in
`src/gpu/intel/gemm/jit.hpp`, with every original line/check preserved.

## Actual native-path qualification

The shared library registers only `b70_gemm_catalog::int4_gemm_w4a16`, using the
original upstream seven-argument native implementation, not a Python reference
or custom numerical kernel. Original `_xpu_C` stays loaded and supplies each
process's installed control. Dynamic-symbol checks found no exported oneDNN
symbols. All three probes loaded the same library SHA256:

`f69aed9aec33459b4484095ce4e03899856ce892f6451d719dcb5f42a0e38f3a`.

The gate/up creation log confirms internal M34816/N5/K5120, batch1, G128,
13 returned catalog entries, and a successful selected identity. Automatic
selected index0. Only indices1 and2 were forced, in separate processes. Their
successful selected identities match the requested entries; no fallback was
used to masquerade as a forced candidate. Full strings are in logs/analysis.
The unmodified down shape does not emit the gate/up selector log.

Rebuilt automatic was **bit-identical** to installed output for both initial
and mutated-input tests on both shapes. Each alternative passed the same fixed
`rtol=atol=0.01` gate, but changed some gate/up FP16 results: maximum initial
absolute differences0.00390625 (index1) and0.0078125 (index2); both reached
0.0078125 after input mutation. Down remained bit-identical in all processes.
Independent CPU FP32 accumulation checked32 evenly spaced unpacked/dequantized
columns; all output elements were checked against installed output. This is
synthetic operator qualification, not full-model output equivalence.

All six shape/process combinations completed numerical and captured-graph
checks.12 alternating event batches ×16 replays ×2 routes ×2 shapes ×3 processes
produced **144 timing batches /2304 timed graph replays**, all retained. No
outliers were removed (including the first automatic-process installed gate/up
batch). `ONEDNN_VERBOSE=0` throughout timing. Creation-only experiment logs
identify catalogs without per-execution verbose profiling; no separate verbose
run was needed because the explicit selector log established identity.

## Result: both tested alternatives lose

Median gate/up graph-replay milliseconds (inclusive IQR in parentheses):

| Process | Installed control | Rebuilt route | Change vs paired installed |
|---|---:|---:|---:|
|Automatic/index0|0.187210 (0.000732)|0.186654 (0.000769)|−0.30%|
|Forced index1|0.187015 (0.000562)|0.194956 (0.001128)|+4.25% slower|
|Forced index2|0.187202 (0.001167)|0.239170 (0.001113)|+27.76% slower|

Compared directly with rebuilt automatic, index1/index2 were4.45%/28.14%
slower. Normalizing each process's rebuilt/installed ratio by the automatic
process's ratio gives4.56%/28.14% slower. This distinguishes a strategy result
from a rebuilt-versus-installed difference; it is still a development screen,
not a confidence interval or serving throughput estimate. Down-control medians
and all per-route IQRs are in `analysis.json`.

**Keep the installed native GEMM.** Neither alternative approaches the5%
improvement gate, so no serving adapter or HTTP A/B was created or launched.
No model, precision, draft depth, acceptance, launcher or power change occurred.
Only two of the13 returned entries were tested; this does not prove global
optimality or rule out other kernel changes. Hot synthetic weights and single
projection graph replays do not reproduce the full model's weight stream.

## Artifacts and final verification

`analyze.py` validates build exits/configuration, library/source identity,
selected catalog identities, all numerical/graph success records and all timing
samples, then recomputes the no-win decision. No full standards/quality suite,
capacity qualification or serving-throughput claim is made.

`preparation-01/`, `build-01/`–`build-03/`, and the three `operator-*/` folders
retain exact scripts, logs, patches, CMake configuration, raw JSON and symbol
checks. Build sources and compiler remain scoped on inference-host for
reproduction (compiler2.9GB; oneDNN source99MB, build155MB, install79MB).
The binary remains there at `binding-build/libb70_gemm_catalog.so`,24293864
bytes, with SHA256 above; it is not installed into production or committed.

`final-host.log` at2026-09-13T01:06:33-04:00 confirms no running containers or
reported render-device users, unchanged launcher SHA and275W. Owned build/probe
containers were removed. Advisor output repeatedly lacked visible completed
tool results; actual logs/exit codes and direct source inspection govern the
reported outcomes, not those visibility errors.

Final read-only review passed with no blocking findings. Numerical success for
forced entries is the executed probe's assertion result, not a reconstruction
from stored output tensors. The forced-entry rejection branch was source-reviewed
but not exercised because both candidates created successfully. Final static
checks passed for12 retained Python files,4 JSON files and10 shell scripts;
`analyze.py` exited0 and reproduced144 batches/2304 replays with no winner.
The unrestricted staged whitespace check reports trailing whitespace/EOF blanks
in raw CMake logs/cache and unified-diff context lines. These artifacts remain
unchanged; the staged check excluding only those generated artifacts passes.

## Follow-up: all remaining entries rejected

After the separate MTP4 draft-attribution campaign, the user authorized continued
evidence-led optimization. This follow-up extends the same isolated native
selector to indices3–12; entries1/2 were not rerun. All original numerical
operations and oneDNN validation remain unchanged. Fresh source check:
<https://raw.githubusercontent.com/uxlfoundation/oneDNN/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit.hpp>.

Before `build-04`, the previous library and patch were preserved remotely in
`pre-remaining-backup/`; its library SHA256 is the original `f69aed9a...` above.
The exact build-03 patch was reverse-checked/reversed and `git diff --exit-code`
confirmed pristine pinned oneDNN source before applying the expanded selector.
Build-04 exited0 with the same compiler/runtime/configuration. Current scoped
library SHA256 is
`17b2350f5885607dba229ac87c6cd5ba723173485f37e2d75cb938e547d6cd64`.

Executed local commands (scripts execute all compiler/GPU work on inference-host):
```bash
bash results/20260913-qwen38-gemm-catalog/build.sh build-04
bash results/20260913-qwen38-gemm-catalog/run-probe.sh operator-auto-02 -1 timing
for index in {3..12}; do
  bash results/20260913-qwen38-gemm-catalog/run-probe.sh "operator-index${index}-01" "$index" timing
done
python3 results/20260913-qwen38-gemm-catalog/analyze-remaining.py
```

Rebuilt automatic again passed initial and mutated-input bit identity against
installed on both shapes. Its gate/up median was0.186606ms versus0.187576ms
installed. All ten forced candidates passed unchanged `rtol=atol=0.01`, sampled
FP32 reference and captured-graph mutation assertions. All selected identities
matched the requested entries; down remained exact. No failed cells or excluded
samples.11 processes ×2 shapes ×2 routes ×12 batches ×16 replays =8448 timed
replays in528 batches, including the fresh automatic qualification.

Gate/up changes versus each process's installed control:
- index3: +42.95%; index4: +53.74%; index5: +77.23%;
- index6: +133.74%; index7: +172.91%; index8: +204.61%;
- index9: +126.33%; index10: +120.36%; index11: +116.84%; index12: +117.74%.

**No qualifying candidate and no serving A/B.** Together with the original
trial this screens all13 returned entries for this exact gate/up descriptor,
not all possible kernels/shapes or global optimality. `analysis-remaining.json`
retains medians, installed-normalized comparisons and native identities; raw
operator files retain every timing batch. Original `analysis.json` remains
reproducible. `final-remaining-host.log` confirms no running containers or
reported render-device users, unchanged launcher SHA and275W. Production was
not modified. Python AST checks, shell syntax and both analyzers passed.
