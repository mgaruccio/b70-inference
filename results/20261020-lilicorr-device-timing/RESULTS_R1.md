# R1: SDK build passed; GPU-event diagnostic rejected

**No drafter speedup, device-time result, W3 public qualification or promotion.**
The requested dramatically faster drafter remains unachieved.

## Preparation and build

- All 32 preparation CPU/source checks passed; core and lifecycle reviews passed.
- Frozen oneAPI 2026.0.0 SDK rebuilt exactly **204 objects**: SYCL 105, llama 44,
  common 55. Only `libggml-sycl.so.0`, `libllama-common.so` and `libllama.so.0.7.0`
  changed. Complete 3,806-file candidate source and six-artifact checks passed.
- Raw SDK commands/output and original/candidate checks: `build-attempt-r1/`.
  The original exported source/binaries were preserved. No GPU was mounted for build.

## Actual journey failure

The existing `A1/W3a/W3b/A2` journey was invoked with
`python3 -B run-profile.py --lead-run-profile` on `inference-host`, package
`/home/mike/b70-evals/20261020-lilicorr-device-timing`.

Only **A1** completed (25 public checks and six measured outputs). Its unfused and
fused router native phases both passed 82/82. W3a exited **1**, not OOM, before
model readiness or any W3 public request. Its first failing phase was
`BOLT_ROUTER_NATIVE_FUSION0`: **77/82**. Four `inplace_sigmoid` cases at widths
304/368/432/496 and the 368-expert, 1,024-token case failed. Sentinel `-FLT_MAX`
values and incorrect indices show substantive corruption, not small rounding drift.
No fusion1 or subsequent singleton qualification was reached in that W3 startup.

Keep `profile-r1/summary.json` **failed**, unchanged. Raw failure:
`profile-r1/VSPF-W3a/restored-final.stdout.log`; terminal state:
`preserved-container.json`. Invocation logs are retained alongside the run.
Automatic restoration passed. The distinct
`python3 -B run-profile.py --lead-after-profile-restore` process verified original
identity/source/health/sole GPU ownership (`analysis-live-restoration.json`, passed)
before analysis correctly rejected the incomplete capture.

## Native-only discriminator

`python3 -B native-router-controls.py --lead-run-native-controls` reused the same
exclusive, signal-aware lifecycle and unchanged 82-case BOLT_ROUTER command.
It ran fresh processes with profiling unset/1 and fusion0/1 against both sources.
No trial model or public inference requests were made.

- Qualified source: all four combinations passed every strict native check.
- Diagnostic source: profiling **off** passed both fusion settings; profiling
  **on** failed both. Raw eight logs and individual exit/check results are in
  `native-router-controls-r1/{VSPF-A1,VSPF-W3a}/native-matrix.json` and adjacent logs.

This isolates the profiling-enabled diagnostic path, not a proven compiler or
GPU-driver root cause. It is not acceptable to qualify only the profile-off mode
and then serve with the failing mode.

The matrix's initial restoration report failed only its stopped-container
bookkeeping check: Docker reported `OomKillDisable=false` before start and `null`
after exit. Original identity/source/health/sole ownership checks passed. A new,
separate process repeated all original/qualified checks and exact stopped-container
checks, normalizing only that known false/null native-container field. It passed:
`native-router-controls-r1/independent-original-restoration-oom-normalized.json`.
The earlier failed summaries and verification files remain unchanged.

## Bounds

No usable selected-kernel timing was established. Width-46 library device time
remains unavailable. Diagnostic/oneDNN verbose overhead is not clean speed evidence;
the frozen oneDNN 3.14 installation also disables its experimental profiling API.
No native gate, math, precision, geometry, full-eight/lattice/KV behavior or
persistent launcher was changed to make this pass. Public trials remain blocked
for this profiling-enabled source until the native failure is resolved or this
optional device-event probe is abandoned in favor of a separately qualified mode.

## Decision after bounded official-source research

Abandon this optional GPU-event probe on the frozen stack; continue with a separate
host-boundary variant that restores the qualified queue initializer byte-for-byte.
The [Intel 2026 notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html)
document an out-of-order limitation, not this in-order profiling corruption.
The [2025 notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)
mention `UR_L0_USE_DRIVER_INORDER_LISTS=0` for profiling **overhead**, not correctness.
Neither is evidence for waiving 82/82 or applying a driver/SDK/queue workaround.
