# Raw-Q8 device embedding candidate — rejected

The isolated R2 SDK build passed: all 44 module objects (including 11 Unity
units) were compiled using the existing generated commands. Only
`libllama.so.0.7.0` changed; all other pinned artifacts, GPU kernels, protected
source, model files and launchers stayed unchanged.

## Actual GPU run

`python3 -B run-performance.py --mode candidate --lead-run-timing` completed the
unchanged exclusive ABBA journey. Both W3 startup gates confirmed a SYCL0
raw-Q8 table: 570,425,344 tensor/buffer bytes (544 MiB). Automatic restoration
passed. A **new independent process** ran
`--mode candidate --lead-after-timing-restore`, verified original identity,
source, health and sole ownership, then rejected analysis.

All six measured **short** requests changed from the required **129 proposed /
84 accepted** to **126 / 85**. Long counters remained 133/82. Analysis records
24 rows, six raw-timing acceptance errors, `status=failed`, and
`independently_restored=true`. Startup activation and a changed acceptance count
are not sufficient to pass the experiment. The gate was not weakened or rerun
with relaxed expectations. **No candidate speed comparison or promotion is
claimed.**

## Concrete precision mismatch

The qualified backend has `GGML_SYCL_F16=ON`. In
`ggml/src/ggml-sycl/common.hpp:216–218`, `dfloat`/`dfloat2` are half/half2.
`dequantize.hpp:771–788` dequantizes Q8 into that half2 and multiplies there;
`getrows.cpp` converts those results to the float output. In contrast, CPU
`ggml-cpu/ops.cpp:5288` calls `ggml_get_type_traits(type)->to_float`, whose
`ggml-quants.c:553–566` Q8 reference multiplies float32 scale × int8.

Thus moving raw Q8 storage changed embedding-value rounding despite unchanged
weight bytes. This is a concrete mismatch consistent with the live failure,
not a claim that startup/mocked tests had proved numerical equivalence.
[Khronos documents `sycl::half` as IEEE 16-bit storage](https://github.khronos.org/SYCL_Reference/iface/scalar-types.html).

The next isolated variant will materialize the **same native CPU-reference F32
values once**, upload a bounded chunk at a time, and use the existing GPU F32
GET_ROWS copy. That avoids changing any GPU backend/kernel/global F16 setting
or the target table. Expected device memory becomes about 2 GiB; capacity,
exact acceptance and end-to-end timing remain pending. Failed Q8 data, source,
patch and pins remain unchanged.

## Raw evidence

`candidate-r1/` retains the failed analysis, every public/native/timing request,
activation and independent restoration. `build-attempt-r2/` retains the actual
successful SDK object/link/source/binary proof. All 2,530 transferred files
matched remote SHA256. Git log retention uses 98 verified lossless copies
(917,924,749 raw bytes → 54,188,038 gzip bytes); raw originals remain locally
and remotely. Decompress for original filenames. No failed observation was
rewritten as a pass.
