# EXPERIMENTAL Bolt NVFP4 → BF16 prefill

**Incremental patch artifact only. Native SYCL compilation, GPU tests and real
API tests NOT RUN. No new correctness PASS, speedup or promotion is claimed.**
Development tier; quality-sensitive activation rounding. Not a standard-compliant
benchmark publication. Original models, production launcher, library and driver
remain unchanged. No inference-host writes, builds or workloads were performed.

User authority: “another gear shift time. lets get nemotron 3.5 bolt running on
the b70 and see how fast we can get it”; “no need to stop here, continue working
and making performance improvments to bolt”.

## Base and bounded implementation

Apply `nvfp4-bf16.patch` **after** llama.cpp
`b6334e37e6ef286fc5091a41b6561b2113d915fb` plus
[`../20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch`](../20261016-nemotron35-bolt-nvfp4-reorder/nvfp4-reorder.patch).
It is NOT a replacement for that patch or a patch against bare upstream.

The comparison baseline remains the winning original-weight NVFP4 R1 runtime,
not the rejected Q8/Q6 copies. Recorded development baseline: 39.45–39.47 tok/s
short decode, 38.13–38.16 after 8K, approximately 955 tok/s cold 8K prefill; see
[`../20261016-nemotron35-bolt-sycl/README.md`](../20261016-nemotron35-bolt-sycl/README.md)
for the existing results and predeclared BF16 acceptance process.

The incremental patch touches four upstream files:

- `ggml/src/ggml-sycl/ggml-sycl.cpp`: share R1's unchanged weight eligibility
  predicate, then add a library-GEMM branch for NVFP4 SSM weights only.
- `ggml/src/ggml-sycl/convert.cpp`: instantiate R1's existing dequantization
  templates with BF16 output, selecting original AoS or reordered SoA metadata.
- `ggml/src/ggml-sycl/gemm.hpp`: BF16 operands explicitly use strict fpmath;
  leave accumulation at oneDNN's default strict F32. The legacy F16 permission
  remains unchanged for non-BF16 operands.
- `tests/test-backend-ops.cpp`: separate candidate selectors; no changes to the
  accepted R1 test class, its 56 registrations or its `5e-4` NMSE threshold.

New dispatch requires **all** of: optimized oneDNN and OPT enabled; named
`.ssm_in.weight`/`.ssm_out.weight` NVFP4 weights satisfying R1's owned/contiguous,
2D, aligned, regular-SYCL-buffer and integer-size limits; dense MUL_MAT;
**BF16 accumulator request**; source precision requests undefined or permitting
BF16 (raw requests, not the global relaxation override); dynamic precision not
forced F32; F32 activations and F32 destination; contiguous non-view 2D
activations/destination in regular SYCL buffers; the whole weight row range and
whole activation column range; **more than 8 columns**; integer-safe GEMM stride
products; the context device's same in-order queue.

The new branch dequantizes weights and converts activations to BF16, then calls
existing `row_gemm` BF16/BF16→F32. No persistent cache or new buffer ownership:
conversions, oneDNN execution and subsequent pool reuse share the existing
in-order queue. No synchronization is removed. A oneDNN `unimplemented` exception
falls through to the existing path; other errors are not suppressed. A debug
marker is an attempted dispatch, not proof that the primitive executed.

Default/F32 requests, ≤8-column NVFP4 decode, non-SSM, MoE, views, split/multi-device
calls and unsupported shapes retain their existing dispatch. The pre-existing
BF16-source branch is retained. In particular, pinned Mamba2 `ssm_out` passes a
reshaped activation into `build_lora_mm` (`src/models/mamba-base.cpp`): if that
activation remains a view at the SYCL boundary, this prototype leaves it on its
fallback. Do not infer that every SSM projection will change. Require actual
model `ssm_in` BF16 dispatch before accepting this as more than an unexercised
path; broadening view support is not part of this artifact.

## Source/research findings

Fresh official research was provided in the baseline README. During this slice,
HTTPS reads rechecked [oneDNN strict accumulation](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_attributes_accumulation_mode.html)
(default F32 for floating primitives) and [SYCL BF16 conversion](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_bfloat16.asciidoc)
(round-to-nearest-even). [MatMul datatypes](https://uxlfoundation.github.io/oneDNN/v3.14/dev_guide_matmul.html)
support BF16/BF16→F32. These establish feasibility, not execution or speed.

**Correction to the handoff:** pinned `common.hpp:34–39` DOES conditionally define
`GGML_SYCL_HAS_BF16`, when `__INTEL_LLVM_COMPILER` and the BF16 header are present.
Read-only SSH confirms the host has that code, `CMAKE_CXX_COMPILER=icpx`, empty
`CMAKE_CXX_FLAGS`, and `GGML_SYCL_F16=ON`. An absent `-D` in build flags therefore
does not prove BF16 code was compiled out. The existing BF16-source branch does
not match NVFP4 weights regardless. We preserve it, rather than replacing it on
the mistaken assumption it is dormant.

Five reconstructed R1 files matched read-only host SHA256 results exactly:

```text
57996971b0e058104a268b497a2814c775d0a6dcbc99b1f905c44e93e59d59ef  ggml/src/ggml-sycl/common.hpp
d4a54d6a08f4cd823fa1bf900600538fdb0423a0d1ac4dbd9364579d84e89d5e  ggml/src/ggml-sycl/convert.cpp
218162efabb3f4de58560e219716c2ff5c5f8c058c3b782701c27cbaf8040997  ggml/src/ggml-sycl/gemm.hpp
ca9931603e7cecbc536f2819be8e6a79a1d431af2f3f5a380af5fd50c819ff3a  ggml/src/ggml-sycl/ggml-sycl.cpp
8d130c47cedcef41ae53cde7e92f1fbc889ef2ac1a95ef4e27c9a16e27515dc0  tests/test-backend-ops.cpp
```

SSH used `BatchMode=yes ConnectTimeout=8 ControlMaster=no ControlPath=none
ForwardAgent=no` and `bash -lc`; source root was
`mike@inference-host:/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-r1`.
Only `sha256sum`, `sed`, and `grep` ran there.

Keep installed oneDNN **3.14.0**, commit
`3f27bec21fdce6b550f233bf6e0afd086ee8d543`, library SHA256
`114f9c84cde4283123bbce3306db9b4a80c29cf3512afcfd871c85461ddfab11`.
The recorded R1 actual-model descriptor was F32/F32/F32 with `attr-fpmath:f16`:
not strict-F32 arithmetic, XMX proof or a bottleneck profile.

## Local checks actually executed

Environment: linked worktree `pi/bolt-nvfp4-bf16`; Linux `7.2.9-2-cachyos`, x86_64;
Python `3.14.7`; Git `2.56.0`; GCC `16.2.1 20260810`. Public pinned files were
fetched from `https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/`
into ignored `.pi/bf16-scratch/{base,r1,candidate}`; R1 was applied to the last
two trees. Initial reconstruction also requested nonexistent
`ggml-sycl-backend.hpp` (HTTP 404); it is unnecessary and was not used.
No model/runtime dependencies were installed.

Commands from this worktree root:

```sh
python3 -B results/20261016-nemotron35-bolt-bf16/test_patch.py \
  .pi/bf16-scratch/r1 .pi/bf16-scratch/candidate
c++ -std=c++17 -fsyntax-only -I.pi/bf16-scratch/candidate/ggml/include \
  .pi/bf16-scratch/candidate/tests/test-backend-ops.cpp
git apply --check --whitespace=error --directory=.pi/bf16-scratch/verify \
  results/20261016-nemotron35-bolt-bf16/nvfp4-bf16.patch
git apply --whitespace=error --directory=.pi/bf16-scratch/verify \
  results/20261016-nemotron35-bolt-bf16/nvfp4-bf16.patch
git apply --reverse --check --directory=.pi/bf16-scratch/verify \
  results/20261016-nemotron35-bolt-bf16/nvfp4-bf16.patch
git apply --reverse --directory=.pi/bf16-scratch/verify \
  results/20261016-nemotron35-bolt-bf16/nvfp4-bf16.patch
```

Observed: **4 local tests pass**, output `R1=56 lifetime=20 reference=4`;
C++ test-source syntax check exits 0 without diagnostics; strict apply/reverse
checks exit 0. Byte comparisons across 16 reconstructed files confirm applied
tree equals candidate and reversed tree equals R1. The optional script checks
unchanged R1 source/registrations, executes the actual registration loops with
stub cases, checks all ten RNE vector expectations, all low-seven-bit NVFP4
scale/nibble combinations' BF16 representability, and the one-hot test layout in
AoS/SoA. These are source/scalar checks, **not native execution**, compiler/API
validation of the SYCL changes, or substitutes for the following acceptance.

## Lead-only build and native acceptance — NOT RUN

Use a separate source/build derived from R1. In that source directory:

```sh
git apply --check --whitespace=error /absolute/path/nvfp4-bf16.patch
git apply --whitespace=error /absolute/path/nvfp4-bf16.patch
```

Use the same oneAPI 2026.0 development image, pinned library mount and baseline
SYCL CMake settings. Explicitly enable the existing definition **only in the
candidate build**: with the observed empty baseline CXX flags,
`cmake -S "$PROTO_SRC" -B "$PROTO_BUILD" -DLLAMA_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS:STRING=-DGGML_SYCL_HAS_BF16`.
This is redundant if header auto-detection already enabled it; it is not a new
CMake option. Do not change F16/global math settings or replace nonempty baseline
flags. Build `llama-server` and `test-backend-ops`, retaining exact configuration,
commands, diagnostics, loaded-library identity and binary hashes. No build may
run concurrently with GPU measurements. Compilation remains an open gate.

Run on the exclusive B70 in the candidate runtime environment (`SYCL0`, or its
reported equivalent), retaining stdout/stderr/exit status:

```sh
# Each invocation must execute and pass exactly 56 cases, unchanged 5e-4 NMSE.
GGML_SYCL_ENABLE_OPT=1 "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_reorder=1'
GGML_SYCL_ENABLE_OPT=0 "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_reorder=1'

# Exactly 20 cases each, all three outputs checked; unchanged 5e-4 NMSE.
GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_DEBUG=1 ONEDNN_VERBOSE=1 \
  "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_bf16_lifetime=1'
GGML_SYCL_ENABLE_OPT=0 "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_bf16_lifetime=1'

# Exactly 4 cases: two dense references (5e-4), two exact RNE cases (zero tolerance).
GGML_SYCL_ENABLE_OPT=1 GGML_SYCL_DEBUG=1 ONEDNN_VERBOSE=1 \
  "$PROTO_BUILD/bin/test-backend-ops" test \
  -b SYCL0 -o MUL_MAT -p 'nvfp4_bf16_reference=1'
```

**No skipped/unsupported case counts as a pass.** The new selectors do not match
the old selector. They are candidate-path tests, not general BF16 requirements
on arbitrary backends. OPT=0 is a fallback control for the lifetime suite, not
an acceptable run of the path-sensitive reference suite.

The 20 lifetime cases are: 10 at `(m,k)=(37,64/192)`, `n=1/8/9/17/128`; four at
real `(m,k)=(18560,4096)/(4096,8192)`, `n=9/128`; and six small `n=9/128` controls
(F32 accumulator, non-SSM name, F32 activation-precision request). Each uses one
weight allocation and dependent prefill→decode→prefill operations. In OPT=1,
ten cases should execute BF16 AoS prefill then BF16 SoA prefill after R1 decode;
four are small-column decode controls, six are excluded-path controls.

The four reference cases use `(m,n,k)=(37,9,192)`, with original weights or a
preceding decode on the same allocation. Two use one-hot NVFP4 rows to expose
actual F32→BF16 conversion output exactly, including positive/negative
halfway-even/odd, neighboring values and exponent carry. F32 bypass cannot pass
these vectors. The other two use the R1 raw-scale/nibble data, CPU dequantization
and explicit BF16 rounding of both operands followed by scalar **F32 FMA**;
GPU output must pass both this reference and the ordinary CPU comparison.

Dispatch evidence must accompany counts: `/nvfp4_bf16` markers with `layout=AoS`
and `layout=SoA` at the expected transitions, followed by actual GPU oneDNN
BF16/BF16/F32 MatMul records. Require no F16/relaxed fpmath or accumulation for
those calls (strict defaults may be omitted from verbose output). No such
marker/primitive should appear for F32/non-SSM/source-F32 or ≤8-column controls.
A marker alone, unsupported-primitive fallback, or numerical PASS alone does
not establish successful BF16 execution or native XMX use.

## Lead-only actual-model process — NOT RUN

Follow the already-declared BF16 process in the baseline README, not a parallel
benchmark. Same original verified GGUF, B70, 65,536 context, batch/microbatch
1024, ESIMD=1, q8 KV, F32 recurrent state, template, sampling and cache policy.
Intentional differences are this patch and the explicit candidate-only compile
definition; diagnostic verbosity is never part of a performance comparison.
After compilation/dependency checks and the existing ownership preflight:

```sh
python3 -B run-cell.py --name nvfp4-bf16-diag \
  --source /home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-bf16 \
  --context 65536 --batch 1024 --ubatch 1024 --esimd 1 \
  --dnnl-verbose --quality --long-bench
```

This is a **proposed lead-only command**, not an executed command. Through the
real server require health/models, smoke, **20/20 API cases**, and exact 8K
retrieval. `server.log` must show actual model SSM-in prefill (e.g.
`1x18560x4096:1x4096x1024`) with BF16 source/weights, F32 destination and strict
math/default strict F32 accumulation. If the model never reaches the new
branch, reject it as unexercised; synthetic native tests do not substitute.

Only then run matched **uninstrumented** short 55/128 and long 8270/128 requests,
one warmup plus three repetitions, and independently repeat an apparent gain.
Record actual tokens/cache usage, TTFT, server timings and client decode versus
the two R1 launches. Retain patch/build/library/binary hashes, native counts,
logs, frozen controllers/create argv, raw API responses and pre/post GPU state
in the lead's named cell directories under
`/home/mike/b70-evals/20261016-nemotron35-bolt-sycl/` (mirror accepted artifacts
into the repository using the existing process). Every success or failure must
remove only the known Bolt container and restore the unchanged Qwen launcher,
checking its observed 7+1 usage and finite logprobs. Any numerical/API failure
rejects the candidate; no gain leaves R1 as winner. No promotion is authorized.
