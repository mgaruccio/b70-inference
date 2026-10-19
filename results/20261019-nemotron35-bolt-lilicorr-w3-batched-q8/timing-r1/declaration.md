# W3 Q8 ESIMD batched submission — candidate preparation

**Development tier; not GPU-validated, qualified, timed, or promoted.** This is one
kernel-launch candidate, not a benchmark framework. No remote writes/builds,
GPU/API/service actions, runtime metadata edits, or launcher changes were made.

- Exact baseline: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`.
- Separate proposed candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-batched-q8-r1`.
- Lead-only remote package: `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-w3-batched-q8`.
- Intentional difference: only `batched-q8.patch`, affecting `dmmv.cpp`, `dmmv.hpp`,
  and `ggml-sycl.cpp` under `ggml/src/ggml-sycl/`. Exact pins: `source-hashes.json`.
  Model, requests, arithmetic acceptance, phase/W7 eligibility and IQ2 are unchanged.

## Mapping and fidelity argument

The unchanged `ggml_sycl_latent_dmmv2_eligible` gate still rejects unmarked widths
3..8, other widths, split/off-device widened inputs/outputs, aliasing, bad strides,
wrong precision and all other r2 exclusions. Only its already-eligible **3..8-column
Q8** branch uses the new helper. n=1, n=2 and every fallback keep their old calls.

One `submit` launches global `((nrows+1)/2 * WG, nvectors)`, local `(WG, 1)`.
`get_group(0)` remains row-pair, `get_local_id(0)` remains tid; `get_group(1)` selects
column. The wrapper offsets F32 input by `col * ne10` and output by
`col * (dst_on_device ? ne0 : row_diff)`, exactly like the removed wide loop.
Each workgroup has its own `WG*2` local floats. No cross-column communication or
reduction is introduced; r2's alias guard already forbids write/read overlap.

The original body is templated on item type (1D or 2D), otherwise byte-identical:
STRIPE=8, block traversal/tails, row pairs and odd-row handling, FMA expressions,
`reduce<float>` and the final ascending-WG sum are unchanged. The original 1D
launcher and selector are untouched. The new selector repeats its exact
64/32/16-block thresholds for WG=8/4/2, else WG=1. No waits, queue-property changes,
F16 activation conversion, MMVQ substitution or Q4_K crossover. Existing
`DMMV_ESIMD` tracing remains unchanged and honest. Compiler/device equivalence is
still a **native byte-check obligation**, not proved by source identity.

## Research and test process declared before implementation

Fresh primary research was supplied by the lead on 2026-10-19:

- https://github.com/ggml-org/llama.cpp/pull/21845 — precedent for replacing
  per-column launches with one multi-column launch, not arithmetic permission.
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/mmvq.cpp —
  independent column reductions are a useful mapping; do **not** replace ESIMD.
- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html —
  host submission overhead and device execution are distinct; fewer submissions
  is a hypothesis, not a measured speedup. Q4_K's 6272 threshold is inapplicable.

The prior W3 slowdown motivates continued optimization. Cumulative draft-callback
walltime is not isolated GPU kernel cost and does not establish causality. Target
verification batching must also be measured; the n=5 unmarked prefill fallback
is intentional, not a missed verification route.

## Local checks (passed; CPU only)

The exact remote r2 files plus `UPSTREAM_COMMIT` were retrieved read-only with
`ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlMaster=no -o ControlPath=none
-o ForwardAgent=no mike@inference-host 'bash -s'`, running only `cd`/`tar -cf -`.
The local source subset is ignored under `worker-scratch/parent`; none is vendored.
`cpu-checks.log` retains commands, environment and raw successful results.

```bash
ROOT="$PWD/results/20261019-nemotron35-bolt-lilicorr-w3-batched-q8"
P="$ROOT/worker-scratch/parent"; C="$ROOT/worker-scratch/applied"
GIT_CEILING_DIRECTORIES="$ROOT/worker-scratch" git -C "$P" apply --check --verbose "$ROOT/batched-q8.patch"
patch -d "$P" --fuzz=0 --dry-run --batch --forward -p1 < "$ROOT/batched-q8.patch"
# C was a fresh copy of P, patched with the same command without --dry-run.
GIT_CEILING_DIRECTORIES="$ROOT/worker-scratch" git -C "$C" apply --reverse --check --verbose "$ROOT/batched-q8.patch"
python3 -B "$ROOT/check-source.py" --parent "$P" --candidate "$C"
python3 -B "$ROOT/test-contract.py" --parent "$P" --candidate "$C"
bash -n "$ROOT/build-candidate.sh"
```

Observed: forward/reverse Git checks and zero-fuzz patch application pass against
the retrieved r2 bytes; resulting files match all candidate hashes. Source gate
rejects an altered patch, wrong parent and unpatched candidate. Python files and
all three build-script Python heredocs parse. GCC 16.2.1 / Python 3.14.7, desktop
Linux x86_64: **85 dispatch cases each with/without ESIMD; 840 enabled-path launch
mapping cases**, with UBSan. Actual extracted dispatch/launch code is compiled
against recording stubs; GPU arithmetic is **not** emulated. Tests check WG
boundaries, odd rows, nonzero bases, pitches, widths, guards and submission count;
source comparison checks every unchanged arithmetic/guard/IQ2/tracing byte.

Initial test-only failures (retained in `cpu-development-failures.log`): reused
fixture omitted the preceding `opt_for_reorder` state; disabled-ESIMD stub lacked
`GGML_UNUSED`. Both were fixed in the CPU harness, not by relaxing source guards.
Initial nested `git apply` checks skipped paths; the log retains those non-results
and the corrected checks using `GIT_CEILING_DIRECTORIES` (all three files checked).

## Lead continuation — defined real E2E, NOT executed here

1. Run `bash build-candidate.sh` **on inference-host only**, through the lead's
   managed long-job mechanism. It clones r2 once to the separate candidate, uses
   frozen Intel image `sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`,
   `/src` matching the existing CMake cache, and CPU-only build/dependency checks.
   Bootstrap packages stay inside the disposable container. Require the 4 GiB
   reserve plus 1 GiB evidence allowance. Inherited source/proof and six-artifact
   protection checks run before/after; no old manifests are rewritten. Actual
   six binary hashes and changed names are saved **before** asserting the expected
   sole change `libggml-sycl.so.0`; unexpected relinks require inspection, not
   copying the prior three-binary expectation. Preserve failed attempts.
2. With original healthy/sole GPU owner recorded first, use the same SDK/B70,
   models and flags. Mount this immutable candidate as `/src:ro`; run the existing
   `verify-singletons/native.py --print-native-command` output with its existing
   mounts. Require all **116 existing Q8/IQ2 native checks**, unchanged per-case
   byte/tolerance gates, inherited F32 router checks and separate N9 controls.
   No native fixture or acceptance change is part of this patch.
3. Through the existing public controller, qualify all 25 tasks A/W3, preserving
   frozen A plus a fresh baseline, zero cached prompt and exact outputs. Only then
   run the existing `singletons-w3-dev-timing/controller.py` ABBA machinery:
   short55/long8270, greedy128, one warmup + three measurements per cold cell
   (six per arm/point). The lead owns the candidate-binding adapter; none is added
   here. Follow `BENCHMARKING_STANDARDS.md`; remain development-tier until its
   publication checklist is complete.
4. Retain native/stdout/stderr, HTTP/SSE, timings, telemetry and errors in fresh
   run directories under this package. Stop all candidate containers; restore
   original READY/health/identity/source proof before independent raw analysis.
   Never promote or rewrite persistent inference launchers. No speed claim until
   this real journey passes and actual measurements justify it.

Outstanding: actual frozen-SDK compilation of the 2D ESIMD instantiation and its
barrier/local-accessor lowering, real native bit equivalence, public qualification,
and performance. Local checks cannot discharge these. This commit completes only
candidate preparation; the lead continues build/fidelity/performance iterations.

## Candidate-bound execution

`run-performance.py` reuses the completed ABBA controller and independent raw
analyzer from `20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing`.
Only private loaded namespaces receive fresh output/container names and the
explicit r2-parent-to-candidate `/src:ro` bind. Qualification namespaces and
saved observations are never rebound or rewritten. Actual Docker records are
checked against the explicitly derived expected mount; all other configuration,
image, PID, native, public, telemetry, and restoration checks remain required.
The completed baseline runner's six source hashes are checked against its saved
r3 pins. New driver/package bytes and actual candidate source/binary proofs are
checked before/after execution. Candidate proof requires the successful build
and exactly the declared three-file/backend-library-only change.

Lead commands, on `inference-host` in this package (after a successful build):

```bash
set -euo pipefail
python3 -B test-performance.py -v 2>&1 | tee performance-cpu.log
python3 -B run-performance.py --lead-run-timing 2>&1 | tee performance-capture-r1.log
python3 -B run-performance.py --lead-after-timing-restore 2>&1 | tee performance-analysis-r1.log
```

Output is `timing-r1/`; no existing attempt is reused. The binding fixtures check
actual inherited function-global wiring, exact name/source-bind deltas, strict
synthetic Docker comparisons, and no mutation of observations. Desktop result:
five passed, one host-only test skipped. An initial direct import of the frozen
qualification gate on the desktop correctly failed because its actual absolute
prerequisite paths are remote; the host-only check is explicit, with no fake
hostname/path or weakened source assertion. All six must pass on the host before
capture. This is still preparation, not a native/public/performance result.
