# Experimental Bolt NVFP4 → F32 prefill workgroup packing — R2

**Patch artifact only. No worker GPU/build result or speed claim.** Tier: development.

R2 is the bounded correction to the R1 NVFP4 packing experiment. It keeps the same
256-work-item, eight-block workgroup and the same per-lane scale conversion, nibble
mapping, multiplication, and F32 destination cast. The packed kernel is now also
instantiated for the reordered `[all qs][all d]` layout used by the actual Bolt
prefill path. Its reordered reads use the offsets and arithmetic of the existing
reordered original kernel. This slice does not change precision, GEMM, decode, KV,
weights, launchers, or settings.

R1 artifacts remain unchanged:

- `nvfp4-prefill-pack.patch`
- `test-nvfp4-sycl.cpp`
- `README.md`
- `nvfp4-native-test-definitions-fix.patch`

## Exact base and contract

Apply `nvfp4-prefill-pack-r2.patch` to a disposable copy of the accepted SSM
archive:

```text
mike@inference-host:/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
upstream pin: b6334e37e6ef286fc5091a41b6561b2113d915fb
```

The patch is cumulative against that archive. It adds the R2 native target
`test-nvfp4-sycl-r2`; it does not require applying the R1 artifact first. The
backend directory compile-definition lookup from R1 is retained, including
`GGML_SYCL_F16` and `GGML_SYCL_WARP_SIZE`, without importing the private oneDNN
property.

The dispatcher selects the packed kernel only when all of these hold:

- F32 destination;
- fusion enabled;
- at least 8 NVFP4 blocks (512 elements); and
- the queue reports `max_work_group_size >= 256`.

Unlike R1, `reordered` is not an exclusion. `dequantize_block_nvfp4_packed<dst_t,
reordered>` uses the original AoS `block_nvfp4` fields for `reordered=false`, and
for `reordered=true` uses:

```text
q: x[i * (QK_NVFP4 / 2) + sub * (QK_NVFP4_SUB / 2) + j]
d: x[ne / 2 + i * (QK_NVFP4 / QK_NVFP4_SUB) + sub]
```

The ordinary 32-lane kernel remains the fallback for fusion-off, short, half,
and unsupported-device calls. Debug mode emits a host-side marker that identifies
both layout and destination:

```text
[SYCL][NVFP4] path=packed|original reordered=0|1 dst=f32|f16 k=<elements> blocks=<blocks>
```

## Native exactness test

`tests/test-nvfp4-sycl-r2.cpp` is built as `test-nvfp4-sycl-r2` for a static SYCL
build. It calls the production `ggml_get_to_fp32_sycl` dispatch.

The unchanged raw fixture has 11 block counts:

```text
1, 7, 8, 9, 15, 17, 63, 64, 65, 127, 513
```

A layout loop runs every fixture in both original and reordered storage. For each
layout it runs fusion-off (original) and fusion-on, then compares the complete F32
payload bitwise. The nine eligible counts must select packed fusion-on; the 1- and
7-block cases exercise the existing short original fallback. The 64-block fixture
still covers all 256 raw scale bytes and all 16 NVFP4 nibbles. Every run checks
input bytes, both canary regions, tails, and the full output payload.

The finite-scale 17-block extras retain the CPU reference and half fallback. The
reordered finite-scale case now exercises the actual packed path: fusion-off
reordered output is retained as a bitwise reference, fusion-on reordered output is
compared against it and against the CPU reference. The raw layout loop continues
to cover the disabled-fusion control.

On the intended B70 device (`max_work_group_size >= 256`), the R2 test should
produce 22 raw case lines and, with debug enabled, these marker counts:

```text
path=packed reordered=0 dst=f32 : 10
path=packed reordered=1 dst=f32 : 10
path=original reordered=0 dst=f32: 13
path=original reordered=1 dst=f32: 14
path=original reordered=0 dst=f16: 1
```

Thus the aggregate is 20 packed and 28 original markers, followed by:

```text
NVFP4 native exactness PASS
```

Lead-owned build/run handoff (oneAPI `icx`/`icpx`, SYCL runtime, Intel GPU; do
not run this worker artifact in the Pi checkout):

```bash
source /opt/intel/oneapi/setvars.sh --force
cd "$D"  # disposable accepted archive with this patch applied
cmake -S . -B build-sycl -DLLAMA_BUILD_TESTS=ON \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_VULKAN=OFF \
  -DGGML_BACKEND_DL=OFF
cmake --build build-sycl --config Release -j6 --target test-nvfp4-sycl-r2

export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_GRAPH=0
export GGML_SYCL_ENABLE_VMM=1 GGML_SYCL_DEBUG=1
export LD_LIBRARY_PATH="$D/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}"
./build-sycl/bin/test-nvfp4-sycl-r2 2>&1 | tee "$R/native/nvfp4-exactness-r2.log"
test "$(grep -cF 'path=packed reordered=0 dst=f32' "$R/native/nvfp4-exactness-r2.log")" -eq 10
test "$(grep -cF 'path=packed reordered=1 dst=f32' "$R/native/nvfp4-exactness-r2.log")" -eq 10
test "$(grep -cF 'path=original reordered=0 dst=f32' "$R/native/nvfp4-exactness-r2.log")" -eq 13
test "$(grep -cF 'path=original reordered=1 dst=f32' "$R/native/nvfp4-exactness-r2.log")" -eq 14
test "$(grep -cF 'path=original reordered=0 dst=f16' "$R/native/nvfp4-exactness-r2.log")" -eq 1
```

Keep compiler/runtime output and failures. Run the existing ordinary 385-case
native gate with its unchanged thresholds as well; this standalone target is not a
broad CTest gate. A device below the work-group limit can only exercise the
original fallback and is not an R2 activation result.

## Lead-owned end-to-end sequence

After the native gate, verify the requested public path in order:

1. Run the debug actual-model gate with the unchanged original weights, q8 KV,
   F32 recurrent state, context 65536, batch/ubatch 1024, ESIMD1, FUSION1, and
   graphs0. Use the existing 8270-token/output-1 probe and require markers proving
   `path=packed reordered=1 dst=f32` on the actual Bolt branch; retain the raw
   output and dispatch log.
2. Restore debug-off and run the fresh non-debug gate-v2, requiring 20/20 before
   timing. Preserve the exact environment and hardware/runtime settings.
3. Run the unchanged short-55 and long-8270 prefill workloads: one warm-up and
   three measured repetitions, 128 forced output tokens, and raw prefill/TTFT/
   decode records.
4. Run exact `BOLT-7319-CEDAR` retrieval for 8238 tokens and retain the response
   and verification artifacts.

No timing result is attributable to R2 until native correctness, actual reordered
activation, 20/20 quality, and retrieval all pass. This remains development-tier
work, not a publication-standard benchmark.

## Research and worker checks

Fresh official references affecting this slice:

- [Pinned `convert.cpp`](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/convert.cpp)
  and [`dequantize.hpp`](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/dequantize.hpp):
  preserve the 32-lane mapping and the reordered byte offsets used by R2.
- [Intel unitrace documentation](https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md):
  confirms `--start-paused`, `--resume`, and `--stop` for the separate diagnostic
  window; profiling is not correctness evidence or a speed claim.

Worker-side checks were limited to the artifact path and did not build or launch a
GPU:

- read-only SSH copied the accepted archive using explicit BatchMode,
  ConnectTimeout, no ControlMaster/ControlPath, and no agent forwarding;
- `nvfp4-prefill-pack-r2.patch` applied and reverse-checked cleanly against the
  accepted archive;
- `git diff --no-index --check` on the applied tree was clean; and
- R1 artifacts and the accepted remote source were not modified.

The lead owns all oneAPI builds, native execution, actual-model debug activation,
API gates, timing, retrieval, and artifact retention.
