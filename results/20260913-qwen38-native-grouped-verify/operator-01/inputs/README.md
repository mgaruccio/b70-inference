# Native grouped verification prototype

**Development only. Not compiled/GPU-qualified by the worker; no speed or serving
claim.** No production launcher, model weights, precision, acceptance policy,
capacity (212992), batch budget (8192), or C1/MTP4K4 configuration changes.

## Bounded implementation

- `patch-native.py` emits a patch of only `paged_decode.hpp` and
  `collective/chunk_prefill_mainloop.hpp` at native
  `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b`. Both added template switches default
  to false; prefill and ordinary decode retain their original behavior.
- `binding.cpp` instantiates **q16_h256_p64, FP16Q/FP8e4m3fnKV, 32 splits,
  PackedVerify=true** only. Private `b70_grouped_verify::forward` registration,
  hidden ELF symbols. No oneDNN build or `_vllm_fa2_C` replacement.
- `grouped_verify.py` is a pre-helper-expansion seam, not a serving adapter install.
  It packs `q[t,h,d]` to `[0,(h//6)*30+t*6+h%6,d]`, shape `[1,120,256]`,
  with four KV heads. Native scheduling tiles rows 0..15 and 16..29; rows >=30
  are masked and native bounded stores discard them. There are two independent
  KV traversals per KV head instead of five; actual speed remains unmeasured.
- Native global identity-tile `partition_C` supplies the row including tile offset
  16. The new mask is `key < max(device_used - 4 + global_row//6, 1)`.
  FP8-to-FP16 conversion, scale folding, FP16 DPAS, online softmax, epilogue and
  split reduction are unchanged. Native initializes maxima to lowest finite,
  sums to zero and skips zero-sum splits; no new empty-split reduction algorithm.
- Device `clamp_min(used,1)` drives both scheduling and masking each replay;
  for used<=1 every valid row's causal limit is 1. Device `[0,1]` packed Q
  metadata replaces helper expansion; the single parent `[1,128]` page table
  is never expanded. No host scalar is derived from device tensor contents.
- Only HND KV `[176,1664,4,256]`, strides `[1664*4*256,256,1664*256,1]`,
  positive pitched Q, contiguous output, scalar-broadcast descales and full
  causal attention are eligible. Runtime tensor strides and descale pointers
  reach native unchanged. Unsupported metadata may use the supplied original
  helper; eligible exceptions propagate, never silently fall back.

As with native, valid page IDs and `0 <= used <= max_seqlen_k <= 212992` are
caller preconditions, not a new malformed-input API. Dummy `cu=[0,0], used=0`
with valid page zero follows the installed uniform helper: first-key attention,
not zero output. Negative/out-of-range pages are never submitted to native.

## Fresh source research (2026-09-13)

Primary sources consulted; the implementation follows these exact revisions:

- [Native mainloop](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp): ordinary decode has causal masking disabled; existing local-mask coordinate partition gives global Q/K indices.
- [Kernel layouts](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp) and [scheduler](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/attn/xe_2/collective/chunk_prefill_scheduler.hpp): GQA heads already occupy the Q-row dimension; grid tiles `ceil(head_group/16)`. Device lengths are read as `cumulative_length[batch]`, despite the field name.
- [Native config](https://github.com/vllm-project/vllm-xpu-kernels/blob/1796aa8bc8db4ac68d9cd19636cef88f3af81d2b/csrc/xpu/attn/xe_2/paged_decode.hpp): standalone policy instantiation reuses the native launcher. q8 already groups six heads and is NOT this prototype.
- [SYCL-TLA commit API](https://api.github.com/repos/intel/sycl-tla/commits/cd763790ad2f74d7294435ecf77682bac0062c3a) confirmed the exact **40-character** native CMake pin `cd763790ad2f74d7294435ecf77682bac0062c3a`.
- [PyTorch 2.13 XPU graphs](https://docs.pytorch.org/docs/2.13/generated/torch.xpu.graph.html): capture uses a side stream by default; retain graph/input/output owners and mutate device contents, not host-captured scalars.
- Local baseline contract: `../20260913-qwen38-native-split-count/operator-01/installed-spec-helper.py`, particularly its per-row `clamp_(min=1)`.

## Lead-owned build and real operator qualification

**All compiler/ML/GPU commands run on inference-host only**, using immutable
image `f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
Torch 2.13.0+xpu, existing read-only Intel compiler 2026.0 and native Git objects
in sibling `20260913-qwen38-gemm-catalog`. No full source clone or upstream CMake
build: archive only native attention headers and extract SYCL-TLA headers/FA2.
AOT targets B70 `bmg-g31-a0`, matching native Xe2 GRF/SYCL flags.

Stage from the integrated checkout (only this new remote experiment root):

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-native-grouped-verify
L=results/20260913-qwen38-native-grouped-verify
ssh inference-host "mkdir -p '$R'"
scp "$L"/{build.sh,run-probe.sh,CMakeLists.txt,binding.cpp,patch-native.py,grouped_verify.py,probe.py,README.md} "inference-host:$R/"
scp results/20260912-qwen38-grouped-split-k/check-grouped-split-k.py "inference-host:$R/"
# Run as lead-owned background jobs, not foreground multi-minute shell work:
ssh inference-host "bash '$R/build.sh' build-01"
# Only after lead confirms idle GPU/275 W and unchanged best-production launcher:
ssh inference-host "env B70_IDLE_CONFIRMED=1 bash '$R/run-probe.sh' build-01 operator-01"
```

Build is bounded to 90 minutes, 12 CPUs/24 GiB, with no GPU device mounted.
The source/compiler mounts are read-only; missing native Git objects fail instead
of mutating the shared source. Every build cell is new; failures, exact commands,
exit codes, patch, CMake files and binary remain there. Probe is bounded to 45
minutes and one isolated GPU container, with no other containers allowed.
Cleanup removes **only its named experiment-owned container**, never other jobs.
Both scripts reject use outside the fixed remote directory/host.

Predeclared journey and expected results:

1. Control = original `flash_attn_varlen_func` -> installed speculative helper
   -> `_vllm_fa2_C`, with split count fixed32 (auto equivalence also checked).
   Candidate uses the **same public entry** with only pre-expansion interception
   and calls the distinct compiled namespace. Reference is independent CPU FP32;
   only reference/Case/comparison functions are reused from the old split-K probe,
   never its Triton algorithm.
2. Seed42, exact production HND cache allocation and random permuted parent pages.
   Test lengths 0..6, 63/64/65, 1663/1664/1665/1668, 1984/1985,
   2048/2049/2053, 8197/32773/65541, two pitched-Q layouts, caller/allocated
   contiguous output, unit and nonunit scalar-broadcast descales. NaN-fill outputs
   before comparisons. Mutation of the final K/V must leave earlier query rows
   bitwise invariant and visibly affect the last-row FP32 control.
3. Capture real public-route graphs, then mutate Q, used lengths and page-table
   **contents at the same addresses** through long/short/page-boundary/dummy cases.
   Compare each replay to fresh native and FP32. Dummy cu/used/page-zero is included.
   Verify unsupported-shape fallback and propagation of an injected eligible failure.
4. Fixed candidate/native `rtol=.02, atol=1e-4`. Retain equally strict FP32
   diagnostics even if native fails them. The predeclared supplemental allowance
   `max(1e-4,eps_fp16*max|descaledV|)` is separately labeled, not a tolerance tuned
   after seeing errors. Candidate/native and both supplemental FP32 checks must pass.
5. After three eager warmups and three graph warmups, time 12 alternating A/B
   batches of 16 replays at 8K/32K/64K (used=context+5). Include Q pack, output
   unpack and device metadata operations. Keep all 72 raw batch timings, medians,
   IQRs and actual dispatch traces. Require >=5% reduction at 32K/64K and <=5%
   regression at 8K. Exit0 = qualified operator; exit2 = correct but no operator
   win; exit1/other = failure. None of these means serving gain.

Retain `operator-NN/{driver.log,exit-code.txt,result.json,collect-env.txt,image.json,
dispatch-*.json,installed-spec-helper.py,inputs/}` plus its corresponding build
cell. Host checks record/require launcher SHA
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4` and 275 W before/after.
Do not install a serving adapter unless the operator gate wins. Later real HTTP
canaries/functional-repeatability tests and interleaved serving confirmation are
lead-owned, outside this slice. This is not a standard-publishable benchmark.

## Worker checks / remaining risk

`python3 results/20260913-qwen38-native-grouped-verify/check-local.py --upstream`
runs stdlib-only symbolic pack/unpack, two-tile/short-length mask, metadata and
fail-closed-dispatch fixtures, Python AST, `bash -n`, pinned-source patch/drift
checks. No Torch/compiler/GPU is loaded. Binding source also has zero ReadSeek
parse errors. These checks passed; they do **not** validate SYCL compilation,
CuTe fragment semantics on hardware, numerical parity, graph capture, or speed.
The lead must execute the above real operator process before qualification.
