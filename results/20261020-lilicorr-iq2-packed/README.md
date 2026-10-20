# IQ2_XXS n4 packed integer dot — CPU/source slice

**Development-only; not promoted. CPU/source close gate passed, SDK/device/public qualification pending. No speed claim.** The dramatic net-speedup goal remains unmet. The batch4 baseline's completed [RESULTS_R1.md](../20261020-lilicorr-iq2-batch4/RESULTS_R1.md), not its earlier README qualification blocker, remains authoritative.

## Exact bounded change

`iq2-packed.patch` is incremental **after** the immutable `../20261020-lilicorr-iq2-batch4/iq2-batch4.patch`. Apply both, in that order, to a disposable copy of the qualified source. `source-pins.json` records the inputs and expected batch4/candidate `mmvq.cpp` hashes. The codebook pin is `0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c`.

Only `ggml/src/ggml-sycl/mmvq.cpp` changes: add private `vec_dot_iq2_xxs_q8_1_moe_n4`, then change the batch4 helper's single vec-dot template argument. The existing selector still requires `n == 4`. The original adapter, native per-output K loop and XOR reduction, quantizer, selector, strides, local `(1, MMV_Y=1, WARP16)` geometry, and n1/n2/n3/negative/fallback paths stay byte-identical to batch4. No four-output sharing, extra accumulators, axis reordering, precision/FP-mode changes, queues/waits/graphs, flags or APIs. Full8-position / 7×8 / max3 / noncausal rejected-KV behavior is untouched.

The adjacent IQ2_XS **packed-sign idiom**, not its arithmetic/scales, is reused: two existing `dpct::dp4a` calls per eight values. XXS keeps its actual `ksigns_iq2xs[aux32 & 127]` lookup, four seven-bit advances, and identical floating expression:

```cpp
const float d = (float)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
return d * sumi;
```

## Arithmetic and access proof

- All 256 pinned grid entries (2,048 bytes) contain only **8, 25, 43**, never zero. For a negated byte `m`, `(m ^ 255) + 1 = 256 - m` is 213..248, never 256: no carry can cross a byte. Positive bytes remain unchanged. Exhaustive extracted-source checks cover every grid × 128 actual sign codes × eight bytes, plus all 256 possible sign-mask bytes.
- Both packed DPCT operands are **signed `int`**, so its helper sign-extends both byte vectors, including Q8 **−128**. Every product has absolute value at most `43*128 = 5504`; every prefix/32-product sum is bounded by **176128 < INT32_MAX** (also `< 2^24`). Integer grouping and the conversion to float are exact. The float expression and native accumulation/reduction order are unchanged.
- Original IQ2_XS uses `*((const int *)q8 + offset)`; the existing aligned access helper explicitly requires four-byte alignment. This candidate instead uses fixed-size `memcpy` into signed integers, avoiding an aliasing cast or new alignment requirement. IQ2 blocks retain their original half/uint16 accesses; codebook reads use the declared uint64 array. CPU layout checks confirm IQ2 size/alignment 66/2 and Q8 36/4, with Q8 payload at offset 4. Native fixtures also exercise IQ2 blocks at two-mod-four addresses.
- Frozen `dpct/helper.hpp` implements `dp4a` by vector byte extraction and four scalar product additions. This is **not** proof of a hardware DP4A instruction, efficient lowering, register usage, or speedup.

## CPU process and observed results

Predeclared tier: development CPU/source verification, **not a performance comparison**. Baseline: immutable qualified batch4. Intentional difference: only the private n4 integer dot. Preconditions: pinned read-only source snapshots (defaults in metadata), existing Python and host C++ compiler; no installs/SDK/container/GPU/service access.

Executed from this linked worktree:

```bash
set -o pipefail
python3 -B results/20261020-lilicorr-iq2-packed/test-packed.py 2>&1 \
  | tee results/20261020-lilicorr-iq2-packed/cpu-checks.txt
```

Observed: both CPU invocations exited **0**, all assertions passed, **no UBSan findings**. Raw commands, environment and output: `cpu-checks.txt` (Linux x86_64, Python 3.14.7, GCC 16.2.1). The initial Git whitespace gate flagged a blank unified-diff context line; trimmed only patch context and reran the same command with `tee -a`, passing again. Compiler: C++17/O2, warnings-as-errors, strict FP, contraction off, UBSan with no recovery; CPU-only unknown `#pragma unroll` warning suppressed. These are test flags, not changes to the frozen SDK flags.

`test-packed.py` checks exact forward/reverse/re-forward composition and proves removal of only the new helper plus reversal of only its callee restores every seed byte. It compiles the **actual extracted** original/new C++ functions, original adapter, sign-mask function, DPCT accumulator alias/body, block declarations/tables, and unchanged native K/XOR function. Integer-return clones change only the FP tail. A small CPU shim models DPCT byte extraction and subgroup snapshots; `_Float16` models binary16 scales, with layout/edge checks. No SYCL build is attempted.

Observed coverage:

- **262,144** packed/scalar signed-weight byte comparisons; no zero/out-of-range codebook bytes.
- **1,684,480** integer-sum and F32-bit comparisons: every grid/sign pair at iqs0..7; all 63,488 finite binary16 encodings in **each** scale operand (not their full Cartesian product); 22×22 signed-zero/subnormal/normal/max-finite scale edges ×16 scale nibbles ×8 iqs ×6 Q8 patterns; additional deterministic random cases and every Q8 value −128..127.
- Extracted native K=**1024/2048** (the frozen native fixture widths): **768** rows, each comparing all 16 lane partials and final XOR-reduced output, across four token slices, shared/per-expert Q8 and nontrivial expert IDs. CPU subgroup snapshots do not assert a GPU execution order.

Generated source/binary/seed copies are confined to ignored `.scratch/` and removed by the test. Raw output stays here. Inputs may be relocated with `--source-root` / `--codebook`; pins remain mandatory. No previous case or managed guidance was modified.

## Fresh external research supplied by lead/researcher

- [Intel execution model](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/execution-model-overview.html): no workgroup-order assumption; keep launch geometry unchanged.
- [Intel register spills](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-register-spills.html): reject four-way staging/sharing and its spill risk for this smaller experiment.
- [Intel compiler 2026 FP optimizations](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/floating-point-optimizations.html): no new FP mode or reassociation; preserve source expression and frozen build flags.
- [DPCT namespace reference](https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2025-2/dpct-namespace-reference.html): helpers are source-level interfaces; the frozen helper implementation is the arithmetic authority, not a hardware-instruction guarantee.

## Lead-owned real-system gate — not executed by this worker

Use the frozen SDK and immutable clean batch4 seed with only the SYCL artifact changed, reusing the existing 105-object SDK/module driver and clean lifecycle. First require the unchanged native Q8 **82/54** and IQ2 **34/16** case/byte assertions per arm, ready and restored, including the existing truthful-parent handling and negative checks.

Then run the real HTTP **A1/W3a/W3b/A2** journey: 25 public checks per cell (**100 total**) plus **24 measured** responses; short55/long8270, temperature0, seed42, ignoreEOS, max128, `cache_prompt=false`, one warmup plus three measured per point/cell. Serving diagnostics OFF; exact output and counters (short129/84, long133/82), preserved drafter semantics. Retain exact commands, raw logs, source/binary evidence. Independently and separately verify original identity/source/health/sole-GPU ownership restoration. Cleanup stops candidates and restores the original; no launcher/service changes or promotion.

CPU results do not replace that public boundary. SDK compatibility/lowering, device bit fidelity, register pressure, actual clean end-to-end speed, and restoration remain **unverified for this candidate**. Even a passing lead run is development-tier, not standard-publishable benchmarking.
