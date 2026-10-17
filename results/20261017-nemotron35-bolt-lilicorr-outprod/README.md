# SYCL OUT_PROD layout fix (r1)

Source baseline: accepted immutable tree `b6334e37e6ef286fc5091a41b6561b2113d915fb` at `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`.

## Patches

- `outprod-layout-r1.patch`: shared SYCL capability predicate and runtime `lda` handling. F32 permits contiguous rows with aligned padded column/plane strides; Q1_0 remains contiguous/dequantized and resets `lda=ne00`; dst remains contiguous; src1 checks match the transposed/nontransposed GEMM contract.
- `outprod-tests-r1.patch`: exact LiLiCorr QKV-view CPU-reference coverage, transposed/nontransposed src1, irregular tails and repeat/broadcast dimensions, contiguous F32/Q1_0 regressions, and explicit expected-unsupported stride/column cases.

The focused QKV test uses backing `[3*m*bs0,k,bs1]`, view `[m,k,bs0,bs1]`, `nb1=backing.nb1`, `nb2=m*4`, `nb3=backing.nb2`, and offset `2*m*bs0*4`.

## Checks

Remote patch check (in a temporary copy of the accepted source):

```text
patch --dry-run --forward -p1 < outprod-layout-r1.patch
patch --dry-run --forward -p1 < outprod-tests-r1.patch
PATCH_DRY_RUN_OK
```

Native/build/GPU/API/runtime tests were not run by this worker; the lead owns those gates and the later timing process. No accepted source or persistent launcher was modified.

Design reference: Intel oneMKL GEMM column-major leading-dimension contract (<https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/gemm.html>) and the pinned CUDA OUT_PROD implementation.
