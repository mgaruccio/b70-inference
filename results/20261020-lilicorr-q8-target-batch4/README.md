# Target Q8_0 n4 submission candidate — not GPU-qualified

**Development/source-only result. No SYCL build, native byte qualification, GPU execution, or serving measurement was performed.** The goal remains a dramatically faster exact-fidelity serving path versus NoDraft; fewer submissions and CPU checks do not establish that goal. No promotion is authorized by this package. The publication checklist in [BENCHMARKING_STANDARDS.md](../../BENCHMARKING_STANDARDS.md) is incomplete; these are not standard-compliant performance results.

## Baselines, scope and pins

- Source baseline: the supplied frozen `/tmp/lilicorr-verify-r2-source-inspect-20261020` export (177 SYCL files plus `UPSTREAM_COMMIT`), **then** the existing [IQ2 batch4 patch](../20261020-lilicorr-iq2-batch4/iq2-batch4.patch). This is the passed `...-lilicorr-iq2-batch4-r1` seed, **not the rejected packed R2 candidate**, and not bare upstream. Immutable upstream: `b6334e37e6ef286fc5091a41b6561b2113d915fb`.
- Serving baseline for the lead's future run: original NoDraft A1/A2, unchanged. Candidate W3a/W3b retains full8-position drafter / 7x8 / max3 and the passed IQ2 batch4 seed, adding only this target-weight submission change.
- `q8-target-batch4.patch` is incremental against that seed: **114 insertions, 4 deletions**, only `ggml/src/ggml-sycl/{ggml-sycl.cpp,dmmv.cpp,dmmv.hpp}`. It does not modify IQ2 arithmetic/quantization or already-batched Q8 drafter ncols8 code. This is Q8_0 **weights with F32 activations**, on the existing `no_quantize_q8_1` ESIMD DMMV route.
- `source-pins.json` records upstream/original/seed identities, both patch hashes, and 12 relevant file SHA-256 pins for each source stage. Tests compare all 178 files after application/round trips, permitting only the three listed differences. The supplied export is read-only; temporary copies stay under this package and are removed.
- Only these five package files are committed. Existing qualifiers, runners, launchers, source trees and configuration are untouched. W7 and rejected noncausal-KV dependency/cleanup changes remain excluded, as do NVFP4 optimization, DFlash configuration detours, SDK/driver/math/graph/queue workarounds, and new runtime knobs or scratch buffers.

## Implementation and contract

The unchanged `ggml_sycl_latent_dmmv2_eligible` still gates the normal MUL_MAT target path by names, types, layouts, buffers, disjointness, private scope/F32 marks, and ESIMD/reorder settings. Only `src1_ncols == 4` attempts the new entrypoint. Every other width, especially n1 NoDraft, and every preflight rejection keeps the **byte-for-byte original per-column loop** or original ineligible route.

The only native helper edits are an optional `row_pair_override = -1` argument and selection of that row index. Existing calls keep the default group index. For the new launch:

- global groups = `ceil(row_diff/2) * 4`; `token = group % 4`, `row_pair = group / 4`;
- each group uses the existing `y + token*ne10` and `dst + token*(dst_on_device ? ne0 : row_diff)` bases; caller-provided row-low offsets are not applied again;
- WG remains 8/4/2/1 at native nblk-row thresholds 64/32/16/else; local range, `WG*2` float SLM, STRIPE8 loops, SIMD32 float accumulators, native MAC helper, local barrier and increasing-p float reduction are unchanged;
- groups are independent: no inter-column synchronization, quantization, half conversion, wider accumulator, reassociation, new queue or event, or scratch allocation.

The private entrypoint checks null/alignment, positive/int-bounded dimensions and row spans, exact input stride, nonoverlapping output-column stride, byte/product/global-range bounds, address wrap, and input/output span disjointness **before** enqueue. Tensor allocation coverage and scope/layout/type validation remain the unchanged caller's responsibility. A false return means no submission occurred; submission exceptions propagate through the original error macros, never into partial-launch fallback. `launched_wg` is set only after submission. Without ESIMD, the entrypoint is a false-returning stub and the new device launcher is excluded.

The existing `BOLT_ROUTE ... DMMV_ESIMD` family record is unchanged; it does not count launches. After successful n4 submission, and only when existing default-off `bolt_route_enabled()` is true, one host line records:

```text
BOLT_Q8_VERIFY_BATCH4 weight=<name> dst_name=<case> n=4 row_low=<low> row_high=<high> rowpairs=<ceil(rows/2)> wg=<1|2|4|8> physical_launches=1
```

This is submission metadata, not a claim that asynchronous device execution or byte qualification passed. No singleton records are synthesized. The lead must require this record for exact eligible Q8 n4 cases while preserving all frozen numeric/negative/canary/byte checks. Existing qualifiers were intentionally not edited here.

## Research gate supplied before implementation

Fresh primary research was completed by the researcher and supplied in the assignment before code changes:

- [Intel DPC++ 2026.0 ESIMD extension](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/explicit-simd-sycl-extension.html): ESIMD uses explicit vectors and subgroup size 1; preserve native local geometry/vector operations rather than treating work-items as conventional SIMD lanes.
- [Intel LLVM ESIMD specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/sycl_ext_intel_esimd.md): explicit IDs, memory accesses and SLM underpin this mapping. Each independent group uses the same local IDs, local storage and helper arithmetic.
- [Intel thread mapping/occupancy guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/thread-mapping-and-gpu-occupancy.html): group execution order is arbitrary and barriers are local. Interleaving columns of a weight row pair **may** improve locality; neither scheduling order nor cache residency is assumed or measured here.

No new adapter/queue flags or SDK changes follow from this research.

## Checks actually executed

From the dedicated linked worktree:

```sh
python3 -B results/20261020-lilicorr-q8-target-batch4/test-candidate.py
git apply --stat results/20261020-lilicorr-q8-target-batch4/q8-target-batch4.patch
```

**8 tests passed.** `cpu-checks.txt` retains exact commands, versions/environment, raw output, the initial test-only false-positive failure and corrected rerun, plus parser and close-gate results. The initial substring check for `simd<` also matched the unchanged helper name ending `_esimd<WG>`; a word-boundary assertion fixed the test without changing candidate source.

Coverage:

- GNU patch `--batch --forward --fuzz=0 --verbose`: both seed derivation and candidate application, no fuzz or offsets.
- Git `apply --whitespace=error`: forward check/apply, reverse check/apply, re-forward check/apply. All 178 file hashes match GNU's candidate or the passed seed as appropriate, covering asymmetric hunk regressions.
- Full-file restoration after removing only documented additions/two row-index edits proves untouched native arithmetic, original WG selector/default call, eligibility, non4 fallback, precision, diagnostics family and IQ2 seed code.
- 532 odd/even-row geometries across every WG; row-pair/token bijection, local IDs/SLM slots, unique output writes.
- 400 offset cases: device/split-style destination strides, row-low prefixes, odd rows, stripe/tail coverage, reordered-weight spans, gaps and canaries. Split/padded cases exercise address formulas only; unchanged outer eligibility still rejects split/padded tensors.
- Extracted **actual host preflight/selector** with an inert CPU launch spy: 147 boundary cases plus propagated submission failure; the no-ESIMD stub passes the same 147 inputs with zero calls. GCC C++17 `-Wall -Wextra -Werror` and UBSan are used only for this small host fixture, **not** for the SYCL translation units or device arithmetic.
- ast-grep C++ parser comparison: the same 29 pre-existing macro/preprocessor ERROR snippets in seed and candidate (dmmv.cpp 12, ggml-sycl.cpp 17, header 0), no added errors. This is not a compiler pass.

Tests require Python 3, GNU patch, Git, and a host C++17 compiler with UBSan. `Q8_TARGET_SOURCE_ROOT` can select another read-only exact export; pins must still match. `CXX` can select the host compiler. Temporary `.cpu-*` trees are cleaned on completion. Raw patch context deliberately retains original whitespace; added source lines are checked with `git apply --whitespace=error` and the test's whitespace assertion.

## Defined real E2E process — lead only, NOT executed here

**Tier:** development exact-fidelity/net-serving comparison, not a community-comparable publication. **Journey:** the real target/drafter service receives the frozen HTTP/SSE public requests, first after strict native qualification and then with diagnostics off for measurement. No CPU model or stub substitutes for this boundary.

1. Preconditions: exclusive clean B70 ownership; original service/source/binary identity and health captured; no persistent launcher changes. Keep model, driver, SDK, math flags, queue/graph behavior, prompts, decoding and full8/7x8/max3 unchanged. Derive the candidate from the passed IQ2 batch4 source, verify this incremental patch and source hashes, and retain commands/output/failures.
2. Use the cached frozen SDK image `intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`. Reuse the [existing generated module closure](../20261020-lilicorr-iq2-batch4/build-module.py), rebuilding **all 105 SYCL objects** (59 top-level + 46 templates), not just the two changed translation units. Of the six protected public artifacts, only `libggml-sycl.so.0` may change. Do not regenerate the graph or change compiler/link flags.
3. Extend only the native qualifier's physical-metadata requirement for eligible Q8 n4 cases. Before serving, require strict **Q8 82 cases / 54 exact** and **IQ2 34 cases / 16 exact**, all existing byte comparisons, negative cases, padding, aliases, canaries and frozen checks unchanged. Require matching case names, row span, row-pair count, native WG and `physical_launches=1` from the new record, without weakening singleton controls or other widths.
4. Reuse the [exclusive clean ABBA controller](../20261020-lilicorr-iq2-batch4/run-performance.py), preserving original NoDraft A1/A2 and using this rebuilt candidate only for W3a/W3b. **The untouched seed controller does not test this candidate.** The lead must first retarget its existing candidate source/binary inputs and qualifier to these exact pins; this package intentionally contains no parallel runner. Then the established entrypoints are:

   ```sh
   # CANDIDATE_CONTROLLER is the lead-retargeted existing controller, not the seed runner.
   python3 -B "$CANDIDATE_CONTROLLER" --lead-run-timing
   # A separate process after capture, including failed captures:
   python3 -B "$CANDIDATE_CONTROLLER" --lead-after-timing-restore
   ```

5. Execute all **100 public jobs** (25 per cell), existing 8 warmups and **24 measured samples**. Frozen inputs are short 55-token and long 8270-token prompts, temperature 0, seed 42, `cache_prompt=false`, ignore EOS, output 128. Require exact public outputs and W3 draft/accepted counters **129/84 short, 133/82 long**. Time with capture/route diagnostics off. Retain original requests, responses/SSE exchanges, native output, build commands and outputs, six-artifact comparison, process/source identities, failures, and timing/analysis under the lead's candidate run directory.
6. Existing cleanup restores the original service even on failure. The separate post-capture invocation must independently repeat original identity, source, health and sole-GPU-ownership checks. Preserve its raw results, not only the in-controller cleanup status.

**Unresolved:** frozen full-module build/lowering, exact device bytes, new native metadata qualification, public fidelity, artifact isolation, resource behavior and actual net speed. No promotion unless the unchanged real journey demonstrates a dramatic net win over NoDraft with exact fidelity. No GPU/SSH/Docker, delegated work, or remote/environment changes occurred in this slice.
