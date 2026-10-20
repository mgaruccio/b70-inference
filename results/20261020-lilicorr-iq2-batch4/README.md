# Private IQ2_XXS batch-4 source candidate — NOT qualified

**Development/source-only artifact. Unbuilt, unapplied to the inference tree, no GPU or performance result.** No remote access, services, installs, descendants, model/launcher changes, or running-binary changes were made. The goal remains exact fidelity and a net B70 win versus NoDraft, not merely fewer submissions. W7 and all previously excluded approaches remain excluded.

## Base and patch

- Exact inspected source: `/tmp/lilicorr-verify-r2-source-inspect-20261020`, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`, with the lead-verified r2 modifications. **Not** bare upstream, r1, or local system source. Origin supplied by lead: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2` (no `.git`).
- `source-pins.sha256`: exact SHA-256 pins for the three patch bases, upstream marker, and six relevant unchanged dependency/configuration files. The lead verified the complete 177-file inspection copy separately; this slice checks these ten pins, not a new full-source attestation.
- `iq2-batch4.patch`: **74 insertions, zero deletions, three files**: `ggml-sycl.cpp` +39, `mmvq.cpp` +25, `mmvq.hpp` +10, all under `ggml/src/ggml-sycl/`.
- Repository base: `b57fd0c0e0dd182879ea4b37ac97381052606e22`; private branch `pi/lilicorr-iq2-batch4`. Full source copies remain ignored under `.scratch/`; no source tree is committed.

The existing `ggml_sycl_iq2_single2_eligible` and **all** singleton-view preflights still execute before the new helper. The new branch further requires 2..4 tokens, fully contiguous flattened activations, and parent input/output disjointness (including n=2, where the old route does not require it). Original scope/precision marks for n=3..4 are unchanged. All other dtypes, reordered/noncontiguous/invalid layouts, n=1, and wider inputs retain the old route. Any new pre-enqueue rejection also falls through to the unchanged per-token loop.

Scratch is capped at `min(1 MiB, existing buffer maximum / 2)` before multiplication/allocation, leaving room for pool look-ahead/alignment. Native n=4 up/down shapes need 4,608 / 82,944 bytes respectively. A null allocation returns before submission; allocator exceptions remain fatal under existing pool behavior, not a post-enqueue fallback. Exhausted live pool/VRAM is not made recoverable by this patch.

One unchanged plain `quantize_row_q8_1_sycl<quantize_q8_1>` submission covers `ne11 * ntokens` rows, then one GEMV submission on the same existing in-order queue. The latter adds only grid dimension 0 for tokens, offsets Q8/IDs/dst pointers, and calls the **same** `mul_mat_vec_q_moe` IQ2 specialization. Each token reads its own IDs, preserving order and duplicates. IDs `nb[1]` and dst `nb[2]` are explicit byte strides; padded dst is still rejected by existing host eligibility. Pool release follows both enqueues with the original queue/lifetime convention. No wait, queue, graph, reorder, quantizer, vec-dot, K-loop, XOR-reduction, or floating-point flag changes.

**Corrected frozen geometry:** the lead confirmed `GGML_SYCL_WARP_SIZE=16`, not 32. This patch retains symbolic `WARP_SIZE`, `block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE)`, and `[[sycl::reqd_sub_group_size(WARP_SIZE)]]` exactly. Equal source/local geometry is **not** proof of bitwise output identity or identical compiler lowering.

## Truthful route; qualification blocker

An eligible parent emits **one** `MOE_IQ2_BATCH4_MMVQ_PLAIN_Q8_1` record for the physical batched GEMV, not N counterfeit singleton records. Existing `bolt_route_record` supplies original parent metadata. With K=input width, M=output width, E=total experts, U=selected experts, R=1 or U, N=tokens, S=parent IDs `nb[1]`:

- weight `ne=(K,M,E,1)`; its original strides remain unchanged.
- src1 `ne=(K,R,N,1)`, `nb=(4,4K,4KR,4KRN)`.
- dst `ne=(M,U,N,1)`, `nb=(4,4M,4MU,4MUN)`.
- IDs `ne=(U,N,1,1)`, `nb=(4,S,NS,NS)`; S may include padding.
- `reorder=0`, `expert=result=-1`; original names, op/op_id, precision tags, device/context, and auxiliary fields are retained. In particular `ne2` is E, while `src1_ne2=dst_ne2=ids_ne1=N`.

`results/20261019-nemotron35-bolt-lilicorr-verify-singletons/native.py` is **unchanged**. It currently requires `[MOE_SINGLE_MMVQ_PLAIN_Q8_1] * N` and singleton dimensions for eligible wide IQ2 cases. The CPU test confirms its rejection of a synthetic n=4 parent record at that exact route assertion. Lead-side qualification must explicitly distinguish the new eligible 2..4-token parent route from singleton references and all old fallback cases. Do not globally allow this route or relax numeric byte equality, annotation/layout/alias guards, canaries, or negative-route expectations.

## Checks actually executed

From the dedicated worktree:

```sh
python3 -B results/20261020-lilicorr-iq2-batch4/test-candidate.py
git apply --stat results/20261020-lilicorr-iq2-batch4/iq2-batch4.patch
```

**8 tests passed.** Exact command/environment/output and parser diagnostic observations: `cpu-checks.txt` (Linux x86_64, Python 3.14.7, Git 2.56.0). Tests apply/check the patch with `git apply --whitespace=error` only in a temporary ignored `.scratch/` subtree, removed on exit. `IQ2_SOURCE_ROOT` may select another read-only exact base; pins must still match.

Coverage: ten source pins; full-file equality after removing only the additions; existing fallback/arithmetic preservation; preflight/enqueue ordering; exact original subgroup geometry/IQ2 specialization; 144 offset configurations with independent/duplicate IDs, ID padding, synthetic dst padding, and canaries; 24 flattened-quantizer address/padding configurations using 16 lanes; scratch-cap integer boundaries; truthful route and expected old-parser rejection. These are source checks and CPU address models, **not execution of C++ guards or quantization/GEMV numerics**. The source parser reported the same 23 pre-existing ERROR nodes in baseline and candidate, none in new code; the complete observed list is retained. This is not a clean compiler check.

## Lead-only qualification, still mandatory

The research/E2E gates were supplied by the lead before this slice. Primary sources: [SYCL 2020](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html), [group algorithms](https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html), Intel [subgroups/SIMD](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/sub-groups-and-simd-vectorization.html) and [kernel launch](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html). Applied conclusions: preserve the actual subgroup geometry and arithmetic; group barriers cannot synchronize workgroups; quantization/GEMV stay separate on the existing in-order queue. Concurrent columns may improve scheduling/cache locality; no specific speedup is claimed.

Only after frozen capture and actual route evidence justify proceeding, the lead may build with frozen SDK `intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0` (source SDK before nounset, no host installs). Lead correction requires all 105 module objects rebuilt because object lineage is unproven, even though two TUs directly change. Verify exact source patch and six-artifact comparison: only `libggml-sycl` may differ unless header dependencies precisely justify another delta.

Then run the existing native IQ2/Q8 wide-versus-single-column **byte** comparisons, including all negative guards, strides, padding, aliases and canaries, with honest candidate-route qualification; then unchanged 100 public cases and 8 warm/24 measured A1/W3a/W3b/A2 public-SSE trials (short 55, long 8270; temperature 0, seed 42, cache false, max 128, ignore EOS). Exact outputs and W3 short 129/84, long 133/82 remain mandatory. Preserve independent original identity/source/health and sole-GPU restoration, retain failures/raw evidence, time with diagnostics off, and require a net win versus NoDraft before promotion. Native correctness, real negative cases, public E2E, artifact deltas, memory headroom, and speed are **all unresolved here**; no tolerances or waivers were introduced.
