# IQ2_XXS n4 two-accumulator experiment

**Development-only; not promoted and not a speed claim.** This is a tightly bounded source/CPU experiment after the passed `../20261020-lilicorr-iq2-batch4/iq2-batch4.patch`. It is not layered on `../20261020-lilicorr-iq2-packed/iq2-packed.patch`.

## Required end-to-end process (declared before implementation)

The real acceptance boundary is the existing drafter service on `inference-host` with the Arc Pro B70, not this CPU check. The lead must run this process before treating the candidate as useful:

1. Preconditions: use the exact frozen oneAPI 2026 cached image and existing qualified configuration, a fresh private SDK copy, the unchanged original service as the restore source, and the immutable batch4 seed. Never mutate the original or seed. Record source, binary, generated-graph, device, pool, process, and physical kernel metadata before the candidate.
2. Rebuild the full 105-object module with the existing SDK build path. Check all exports/generated graph files and the protected six-artifact boundary; only the SYCL library may differ. Retain the build log, hashes, and source proof.
3. First run the unchanged native qualifier and representative CPU cases. Require the current native metadata and exact cases: Q8 **82 cases / 54 byte-exact** and IQ2 **34 cases / 16 byte-exact** per arm. Require the candidate's physical DLL/kernel-handle mapping, not just a route label.
4. Run the existing **unprofiled** native operator ABBA×3 against the passed IQ2 batch4, all 24 rows. Run LD/UR diagnostics in a separate invocation, outside timing, and retain actual DLL/kernel-handle mapping. A profiling queue or a native-check failure is a **NO-GO**.
5. Treat cache/random-ID coverage as limited: IDs 0..8 and 240 are not a serving replay, and the actual pool model may differ. Do not call this a whole actual serving-workload result. Only if the unprofiled native operator has a clear gain, run the unchanged public journey: A1/W3a/W3b/A2, 100 public jobs, 8 warmups, and 24 measured responses; short 55 and long 8270 tokens, temperature 0, seed 42, `cache_prompt=false`, `ignore_eos`, `max_tokens=128`, full 8-position / 7×8 lattice, max3, and rejected-KV behavior. Require the existing short **129/84** and long **133/82** drafter counters and all quality/fidelity checks.
6. Restore on every exit path. Independently over SSH and Python verify original identity, source, health, sole-GPU ownership, and stopped-experiment proof. The existing exact native `id=false`/`null` OOM comparator remains allowed. Preserve raw logs/metadata and stop the candidate; do not change launchers, queues, flags, drivers, or services.

Lead-owned commands, after staging and validating those inputs, are the existing package commands:

```sh
cd /home/mike/b70-evals/20261020-lilicorr-iq2-acc2
python3 -B build-candidate.py
python3 -B source-proof.py --built
python3 -B run-performance.py --lead-run-timing
python3 -B run-performance.py --lead-after-timing-restore
```

This worker did **not** execute the SDK, B70, service, public, or performance stages. They remain unverified; CPU exactness is not a substitute for that public boundary.

## Exact bounded source change

`iq2-acc2.patch` changes only `ggml/src/ggml-sycl/mmvq.cpp` after batch4:

- add private `vec_dot_iq2_xxs_q8_1_moe_n4_acc2` using the already-reviewed packed signed-byte helper math;
- replace only the batch4 helper's vec-dot template argument with the new helper;
- keep the four signed `dpct::dp4a` calls, grid/sign loads, four `aux32 >>= 7` advances, `memcpy` Q8 loads, and the existing byte-identical floating tail;
- use only two non-aliased scalar integer accumulators (`l < 2` → `sumi0`, otherwise `sumi1`), then `const int sumi = sumi0 + sumi1` immediately before the existing `d` expression.

The patch is cumulative from batch4, with exact forward/reverse/re-forward checks. All other source files and all non-batch4 IQ2 widths/selectors are byte-identical. Full8/7×8/max3/rejected-KV semantics, native exact dtype/arithmetic/precision, SG16 geometry, grid, local geometry, strides, eligibility, queues, flags, SDK, and driver remain unchanged. `QK_K == 256` is the only candidate branch.

`source-pins.json` pins every qualified SYCL source file, the codebook, both patch hashes, the batch4 seed hash, and the candidate `mmvq.cpp` hash. The candidate hash is `ebdfaec9026f51e03d0b3a048f53cd76f375307f80d51a573971eb7786367559`. The one space-only blank unified-diff context line in `iq2-acc2.patch` is intentional patch syntax and must not be whitespace-cleaned.

## Arithmetic and CPU close gate

The extracted-source test uses the actual original scalar helper, actual candidate helper body, actual frozen DPCT accumulator alias and `dp4a` implementation, actual codebook/tables, unchanged native K/XOR function, and a small CPU subgroup snapshot shim. The candidate body is extracted from the patch result; no parallel hand-coded fast path is used. Mocked CPU `dp4a` is arithmetic evidence only, not hardware-DP4A or lowering evidence.

Pinned bounds are exact: every grid byte is `{8, 25, 43}` and every Q8 byte is signed `[-128, 127]`; each product is at most `43*128`, each two-way partial has 16 products and is at most **88,064**, and the final 32-product sum is at most **176,128**, below `INT32_MAX` and `2^24`. The coefficient is the floating scale outside the integer sum; no looser coefficient bound is used. Signed `int` DP4A operands and `memcpy` loads cover Q8 `-128` and avoid alias/alignment assumptions.

Run from this worktree:

```sh
set -o pipefail
python3 -B results/20261020-lilicorr-iq2-acc2/test-candidate.py 2>&1 \
  | tee results/20261020-lilicorr-iq2-acc2/cpu-checks.txt
```

Observed on Linux x86_64 with Python 3.14.7 and GCC 16.2.1: Git and GNU `patch --fuzz=0` forward/reverse/re-forward composition passed; only `mmvq.cpp` changed; all source pins and native K/XOR source assertions passed; UBSan C++17/O2 compilation passed; all assertions passed. Coverage included 262,144 grid/sign byte comparisons, every Q8 value `-128..127`, finite binary16 scale encodings in each operand, signed-zero/subnormal/normal/max-finite scale edges, deterministic random scale/input cases, and extracted native K=1024/2048 subgroup/XOR cases (768 rows, 16 lanes, four token slices, shared/per-expert Q8). The final output is retained verbatim in `cpu-checks.txt`.

## Fresh primary research recorded

- [Intel IGC DP4A instruction](https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DP4A.md): describes four-byte integer dot accumulation; retain the existing signed overload and do not infer hardware performance from source alone.
- [SPIR-V `OpSDot`](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpSDot): defines signed dot semantics. The selected old SPIR-V path does not directly use `OpSDot`; this experiment makes no contrary claim.
- [Intel register and performance guidance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/registers-and-performance.html): dependency reduction can trade against register pressure. Two accumulators are only a bounded dependency hypothesis and may spill; no ILP or hardware gain is claimed from source or ISA text.

The experiment remains development-tier until the lead-owned SDK/device/public process above is independently complete. No speedup, dramatic or otherwise, is claimed here.
