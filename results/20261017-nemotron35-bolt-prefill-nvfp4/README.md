# Experimental Bolt NVFP4 → F32 prefill workgroup packing

**Patch artifact only. No worker GPU/build result or speed claim.** Tier: development.
This slice changes only the NVFP4 conversion dispatch used by prefill; it does not
change decode, GEMM arithmetic, settings, launchers, weights, or a user tuning knob.

## Exact base and contract

Apply `nvfp4-prefill-pack.patch` to a disposable copy of:

```text
mike@inference-host:/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1
upstream archive: b6334e37e6ef286fc5091a41b6561b2113d915fb
```

The archive has no `.git` directory. The relevant source bytes were fetched with
read-only SSH and matched locally:

```text
ggml/src/ggml-sycl/convert.cpp    d4a54d6a08f4cd823fa1bf900600538fdb0423a0d1ac4dbd9364579d84e89d5e
ggml/src/ggml-sycl/dequantize.hpp 68265ed704802b6c027359ed03bc008d723b822766794dcfb1b407e0ecc14a63
tests/CMakeLists.txt              d1e60fec1ea04cfce88095c7c462c177a60f7b0ce4c0ed94f35f5b9b5e48eed4
```

The original 32-lane conversion remains intact as
`dequantize_row_nvfp4_sycl_original`. The candidate is selected only when all of
these hold:

- F32 destination and non-reordered NVFP4;
- `g_ggml_sycl_enable_fusion` is enabled;
- at least 8 NVFP4 blocks (512 elements); and
- the queue device reports `max_work_group_size >= 256`.

The packed launch uses 256 lanes, eight consecutive 64-element blocks per
workgroup, and `int64_t` block/output indexing. A partial final workgroup returns
before reading or writing out-of-range blocks. It keeps the original per-lane
scale lookup, nibble mapping, multiplication, and destination cast. Half,
reordered, short, fusion-off, and unsupported-device calls use the original
kernel. Debug mode emits one host-side marker per conversion call:

```text
[SYCL][NVFP4] path=packed|original k=<elements> blocks=<blocks>
```

## Native exactness test

`tests/test-nvfp4-sycl.cpp` is built as `test-nvfp4-sycl` only for a static SYCL
build. It calls the production `ggml_get_to_fp32_sycl` dispatch: fusion-off is
the same-device original reference and fusion-on is the candidate path. It
checks full-array raw F32 bits, unchanged input bytes, destination guards and
tails, and exercises block counts `1, 7, 8, 9, 15, 17, 63, 64, 65, 127, 513`.
The 64-block case asserts coverage of all 256 raw scale bytes and all 16
nibbles, including the zero/special raw scale encodings. It also performs finite
CPU-reference comparisons and explicitly checks fusion-off, half, and reordered
fallbacks.

Lead-owned build/run handoff (oneAPI `icx`/`icpx`, SYCL runtime, Intel GPU; do
not run this in the Pi worker):

```bash
source /opt/intel/oneapi/setvars.sh --force
cd "$D"  # disposable exact archive with this patch applied
cmake -S . -B build-sycl -DLLAMA_BUILD_TESTS=ON \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_VULKAN=OFF \
  -DGGML_BACKEND_DL=OFF
cmake --build build-sycl --config Release -j6 --target test-nvfp4-sycl

export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_GRAPH=0
export GGML_SYCL_ENABLE_VMM=1 GGML_SYCL_DEBUG=1
export LD_LIBRARY_PATH="$D/build-sycl/bin:/opt/onednn/lib:${LD_LIBRARY_PATH:-}"
./build-sycl/bin/test-nvfp4-sycl 2>&1 | tee "$R/native/nvfp4-exactness.log"
test "$(grep -cF '[SYCL][NVFP4] path=packed' "$R/native/nvfp4-exactness.log")" -eq 10
test "$(grep -cF '[SYCL][NVFP4] path=original' "$R/native/nvfp4-exactness.log")" -eq 15
```

On the intended device (`max_work_group_size >= 256`), the log should contain
10 packed markers, 15 original markers, every listed case, fallback status, and
`NVFP4 native exactness PASS`. Keep compiler/runtime output and failures. Run the
existing ordinary native tests with their unchanged thresholds; this standalone
test is not registered as a broad CTest gate.

## Lead-owned end-to-end sequence

After native exactness, follow the existing public-boundary process in order:

1. Existing 385-case native gate with debug enabled, then real Nemotron activation
   and marker verification; no decode/GEMM/settings changes.
2. Fresh non-debug 20/20 gate-v2 before timing, with original weights, q8 KV,
   F32 recurrent state, context 65536, batch/ubatch 1024, ESIMD1, FUSION1,
   graphs0, and unchanged hardware/runtime settings.
3. Existing short-55 and long-8270 prefill runs: one warm-up plus three measured
   runs, 128 forced output tokens, unchanged prompts/outputs, retaining raw
   prefill/TTFT/decode data.
4. Exact 8238-token retrieval check (`BOLT-7319-CEDAR`) and artifact retention.

This is not a publication-standard benchmark; no timing result is attributable
to this patch until those gates pass. The current comparison context is the
previously measured native/API baseline (fresh native 385, API 20/20, exact 8238);
the worker did not rerun it.

## Research and worker checks

Fresh official references used for this slice:

- [Intel unitrace README](https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md): confirms the supported unitrace workflow and pause/resume collection; profiling is not a correctness or speed claim.
- [Pinned llama.cpp `convert.cpp`](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/convert.cpp) and [`dequantize.hpp`](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/dequantize.hpp): establish the pinned 32-lane dispatch and original NVFP4 lane mapping.

Static checks performed without building or launching a GPU:

- `git apply --check` and apply succeeded on a temporary copy of the fetched
  baseline; `git diff --check` was clean.
- Remote/local SHA-256 values above matched.
- No remote source, container, launcher, model, or persistent host configuration
  was modified. Native build/test execution is intentionally left to the lead.
