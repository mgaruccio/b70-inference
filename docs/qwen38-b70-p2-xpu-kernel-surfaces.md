# P2: B70 XPU-kernel surfaces for GPT-pro items 5, 6, and 11

Research snapshot: 2026-08-23. This is a source map, not a kernel change or a
performance result. No kernel was compiled and no B70 server was started.

## Pins and provenance

The image/handoff facts are:

- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- vLLM: `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`
  (`0.27.2rc1.dev77+gac7509e2b`).
- Installed package: `vllm-xpu-kernels 0.1.12.3`; the running image exposes
  compiled libraries (including `libgdn_attn_kernels_xe_2.so`), not headers or
  a source checkout. The runtime GDN Python module is site-packages
  `vllm/v1/attention/backends/gdn_attn.py`.

The statements below labeled **image** are facts supplied by the handoff or
runtime pin. Statements labeled **upstream** are facts from a public checkout
and must not be treated as proof of what the image loaded.

## 1. Checkout map

### Closest public source

There is no public Git tag/ref named `v0.1.12.3`. The public refs checked were:

| Ref | SHA | Relation |
| --- | --- | --- |
| `v0.1.12` / `v0.1.12.1` | `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b` | previous tagged source |
| `release/v0.1.12.2` | `e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88` | nearest public release branch; `v0.1.12-2-ge8b12ae` by `git describe` |
| `v0.1.13` | `07d44bcb4f7d856c235f36774c1ee94cde3e8b76` | later tagged source; not an image-parity checkout |
| `main` at research time | `4543b580fecca68a7dd54ddaf6e444dc5f11a6a4` | later upstream drift |

Use the `release/v0.1.12.2` tree at `e8b12a...` as the closest audit
checkout, and record that it is **nearest, not proven exact provenance** for the
0.1.12.3 wheel. The PyPI 0.1.12.3 upload is dated 2026-08-10, but its wheel
metadata contains no source SHA/build manifest. Do not silently substitute
current `main`: it contains the post-split GDN interface and later dependency
changes.

A later worker can fetch without putting a large tree in this worktree:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/vllm-project/vllm-xpu-kernels.git /tmp/vllm-xpu-kernels
cd /tmp/vllm-xpu-kernels
git fetch origin e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88
git checkout --detach e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88
```

The e8b12a CMake source records these dependency pins:

- oneDNN repository `https://github.com/uxlfoundation/oneDNN.git`, commit
  `80afa71049cd69a3df32adcccb623b12cd7baa22` (`rls-v3.12`).
- SYCL-TLA repository `https://github.com/intel/sycl-tla.git`, revision
  `cd763790ad2f74d7294435ecf77682bac0062c3a`.
- Xe2/GDN compilation is controlled by `VLLM_XPU_ENABLE_XE2` and
  `GDN_KERNELS_ENABLED`; the GDN subdirectory produces
  `gdn_attn_kernels_xe_2`, which is linked into `_xpu_C`.

### Build prerequisites (do not run here)

The e8b12a build metadata and `AGENTS.md` require, at a high level:

- Python 3.9--3.12 for CMake discovery (`pyproject.toml` permits `<3.14`),
  PyTorch `2.13.0+xpu`, CMake >=3.26, Ninja, and substantial build memory
  (the project estimates roughly 8 GB per compile process).
- Intel oneAPI 2026.0/DPC++ (`icx`/`icpx`); source
  `/opt/intel/oneapi/setvars.sh` before native configuration.
- A working XPU/Level Zero driver and BMG target. e8b12a's default
  `DPCPP_SYCL_TARGET` is `intel_gpu_bmg_g21`, while CMake lists both
  `intel_gpu_bmg_g21` and `intel_gpu_bmg_g31`; a B70 build must explicitly
  verify the G31/AOT target instead of inheriting the default blindly.
- oneDNN available to `find_package(oneDNN)`, and SYCL-TLA/CUTLASS fetched or
  supplied via `VLLM_CUTLASS_SRC_DIR`. `MAX_JOBS`,
  `VLLM_XPU_XE2_AOT_DEVICES`, `VLLM_XPU_ENABLE_XE2`, and the kernel-family
  toggles are the relevant controls.

The source's normal shape is `uv`/virtualenv plus an editable install
(`uv pip install --no-build-isolation -e .`); the exact image-compatible torch,
oneAPI, driver, and AOT settings still need to be established on the B70 box.

## 2. Item 5: small-M GPTQ-G128 W4A16

### Call path and where M enters

The pinned vLLM call is
`vllm/model_executor/kernels/linear/mixed_precision/xpu.py:XPUwNa16LinearKernel.apply_weights`:

1. `x.reshape(-1, x.shape[-1])` flattens all token/batch dimensions into `M`.
2. `torch.ops._xpu_C.int4_gemm_w4a16(...)` receives the flattened activation,
   transposed packed weight, scale, zero point, `group_size`, and optional
   `g_idx`.
3. In the closest kernel checkout,
   `csrc/xpu/onednn/onednn_matmul.cpp:int4_gemm_w4a16` optionally applies
   `index_select` for `g_idx`, creates the output, and calls
   `oneDNN::dnnl_matmul_w4a16_int4`.
4. `csrc/xpu/onednn/int4_gemm_w4a16.h:dnnl_matmul_w4a16_int4` computes:
   `m = product(mat1.sizes()[:-1])`, `k = mat1.sizes()[-1]`, and
   `n = result.sizes().back()`. For the pinned flattened call, `m` is the
   number of verify/draft rows. The packed B shape is `[k/8, n]`; scales are
   `[k/group_size, n]`, with G128 supplied as `group_size=128`.
5. The wrapper passes `m,n,k,lda,ldb,ldc` into
   `matmul_primitive_create_and_cache`. The cache key in
   `csrc/xpu/onednn/onednn_ext.h:matmul_primitive_cache_t::get` includes the
   dimensions and strides, so M can select/cache a distinct oneDNN primitive.

There is no separate GPTQ `gemv` launcher in `vllm-xpu-kernels` at this pin.
`csrc/xpu/grouped_gemm/xe_2` is an MoE grouped-GEMM path, not the dense GPTQ
linear path. Any GEMV-like choice is inside the linked oneDNN primitive
selection and must be audited there.

### Actual Xe2 launch surface

For the pinned oneDNN commit, the relevant source is outside the XPU-kernels
repository:

- `src/gpu/intel/gemm/jit/pd.cpp:init_GEMMProblem` maps oneDNN's `desc()->m(),
  n(), k()` and quantization group fields into the GEMMStone problem.
- `src/gpu/intel/gemm/jit.cpp:gen_t::launch_nocopy` forms the dispatch. It starts
  with `gws[0] = div_up(m, unroll[LoopM])`, `gws[1] = div_up(n, unroll[LoopN])`,
  and a K dimension based on `kParallel()`/`wgK`; `lws` comes from the selected
  `wg[LoopM]`, `wg[LoopN]`, and `wgK`. C-interleave, fixed-workgroup/power-of-two
  rounding, fused-EU padding, K padding, subgroup scaling, and batch scaling
  can enlarge the final range. The selected `unroll`, `wg`, and kernel strategy
  are runtime/architecture dependent.

Thus source inspection establishes where M is consumed, but not the B70
selected tile or final workgroup count. The worker should not infer a
67--80% idle result from the MLXFast hypothesis.

### Counters that would establish idle workgroups

Run the same GPTQ-G128 W4A16 shape at `M=1..9` (especially 2, 3, 4, 5), holding
K, N, dtype, scales, bias, strides, and graph mode fixed. Retain oneDNN verbose
primitive records and a GPU kernel trace. The smoking-gun record needs all of:

- final dispatch `gws/lws` and kernel name/variant; submitted workgroups and
  valid output-row/output-column tiles. Compute the invalid-row fraction after
  all oneDNN rounding, not from `M` alone;
- active versus predicated/empty workgroups or SIMD lanes, EU/thread occupancy,
  and execution-mask activity;
- DPAS/XMX active/busy cycles (or the BMG driver's equivalent), EU active and
  stall cycles, and GRF/register-spill indicators;
- LSC/global load-store, L3, and DRAM bytes/transactions, plus kernel elapsed
  time and achieved throughput.

A credible item-5 result is a common verify width with a large submitted-to-
valid tile gap, low active SIMD/XMX utilization or high predication, and a
repeatable latency drop versus the generic plan. Exact metric names vary by
BMG driver/profiler; save the raw report and tool version. A specialized plan
must still pass the target-token/parity gate before any serving claim.

## 3. Item 6: graph-padding and speculative metadata

### Python path in the pinned vLLM source

`vllm/v1/attention/backends/gdn_attn.py` contains the graph metadata contract:

- `GDNAttentionMetadata` carries `spec_query_start_loc`,
  `non_spec_query_start_loc`, `spec_state_indices_tensor`,
  `spec_token_indx`, `num_accepted_tokens`, and `num_actual_tokens`.
- `GDNAttentionMetadataBuilder.__init__` preallocates graph-sized state/index/
  query/accepted-token buffers.
- `build` separates spec rows, compacts `spec_token_indx`, and creates active
  query-start arrays. It also reclassifies non-spec one-token decodes as
  prefills when spec rows exist; this is a semantic workaround, not a kernel
  graph-padding fix.
- In the full-graph path (`build`, around the `use_full_cuda_graph` block),
  state indices, masks, query-start locations, and accepted-token counts are
  copied into buffers sized by `batch_size = m.num_reqs`; padded request rows
  are filled with NULL/false/repeated terminal values. `m.num_actual_tokens`
  remains the active token count. The data tensors can therefore have a
  captured/padded leading dimension while metadata has a captured request
  dimension.

### Compiled GDN contract and known change

In the closest e8b12a checkout:

- `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:causal_conv1d` and
  `:gated_delta_rule` already accept padded **data** leading dimensions and
  narrow `z`, projected qkvz/ba, and `core_attn_out` to
  `[0:num_actual_tokens)`. This behavior came from the padded-leading-dim work
  (`ae5ba48517e68c713d83351d560799514cf4c2b3`, issue #320).
- The same interface still requires exact speculative metadata sizes:
  `spec_query_start_loc.size(0) == num_spec_decodes + 1`,
  `spec_state_indices_tensor.size(0) == num_spec_decodes`, and
  `num_accepted_tokens.size(0) == num_spec_decodes`. It also requires exact
  contiguity/dtype contracts.
- `tests/gdn_attn/test_gdn_attn_padded.py` proves only the data-leading-dim
  contract for non-spec decode and checks that padded output tails remain
  untouched. It is not a graph-padded speculative-metadata regression test.

The relevant upstream change is still open:

- [vllm-xpu-kernels PR #391](https://github.com/vllm-project/vllm-xpu-kernels/pull/391),
  head `d9dd065e91698d97a0de02aadae044a9ea3369ba`, fixes
  [issue #389](https://github.com/vllm-project/vllm-xpu-kernels/issues/389).
  Its `csrc/xpu/gdn_attn/gdn_attn_interface.cpp` hunk changes the three
  speculative checks to `>= num_spec...` and narrows query-start,
  state-index, and accepted-token tensors to active rows before launching.
  It also bounds the padded `update_states` tail in
  `causal_conv1d.hpp` and `xe_2/chunk_causal_conv1d_xe2.hpp`. The PR is based
  on the older fused interface and is stale relative to the post-split
  `e8b12a`/current-main interface; it is a specification of the needed
  behavior, not an image-ready patch.

### What the B70 overlay covers and leaves behind

`patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py`
is the static-buffer half of closed/unmerged [vLLM PR
#43955](https://github.com/vllm-project/vllm/pull/43955), against the exact
vLLM/image pins. It currently covers:

- a reusable `spec_token_arange` and capacity failure instead of allocating an
  arange each pure-spec build;
- an empty reusable non-spec index without a redundant zero-length copy;
- slicing compact spec rows rather than CPU-mask indexing; and
- reuse of the static arange during graph metadata copying.

It intentionally does **not** change `gpu_model_runner.py`, the runner's
accepted-token lifecycle, the `_xpu_C.gdn_attention`/split-op ABI, any C++
metadata validation or active-prefix narrowing, graph capture shape policy,
or a compiled kernel. The runner hunk of PR #43955 is therefore still absent.
The remaining item-6 question is whether to port the active-prefix contract
from PR #391 to the split interface and then measure graph capacity versus
active rows; do not call the current overlay a compiled-GDN graph-padding fix.

## 4. Item 11: speculative GDN state and native Xe2 candidate

### What the closest source actually launches

At e8b12a, the split interface in
`csrc/xpu/gdn_attn/gdn_attn_interface.cpp` sends spec decode to the generic
SYCL headers, while Xe2 is used for the non-spec chunk/prefill path:

- `csrc/xpu/gdn_attn/causal_conv1d.hpp:causal_conv1d_spec_kernel` has
  `get_nd_range(num_spec_decodes, qkvz_elems)`: global groups
  `(num_spec_decodes, ceil(qkvz_elems/1024))`, local `(1,256)`. Each group
  processes a feature slice and walks all `num_spec_tokens` for one request.
- The conv state is `[cache_batch_size, width-1, conv_elems]`. For a spec row,
  `num_accepted_tokens[batch] - 1` selects the initial cache column (clamped
  to zero); `cache_indices[batch, t_local]` selects the per-token writeback
  slot. Every token checkpoints the trailing `width-1` state, so the next
  round can seed from the accepted prefix rather than the rejected suffix.
- `token_indx` maps compact local spec tokens to global projected qkvz/ba and z
  positions. q/k/v/b/a outputs are compact, row-major local buffers; z is
  written through the global token index. This is a deliberate compact-versus-
  global split, not an interleaved q/k/v output buffer.
- `csrc/xpu/gdn_attn/gated_delta_rule.hpp:gated_delta_rule_spec_kernel` has
  global `(num_spec_decodes, num_v_heads, ceil(head_v_dim/32))`, local
  `(1,1,256)`. It reads the SSM state at the accepted-prefix cache slot,
  iterates all speculative tokens, writes `core_attn_out` through
  `token_indx`, and writes each token's state to its corresponding cache
  column.
- The SSM state is a flat per-cache-slot tensor with row stride
  `ssm_state.stride(0)` and logical shape
  `[cache_batch_size, num_v_heads, head_v_dim, head_k_dim]`. The state-index
  tensors are contiguous row-major metadata; the explicit row stride is used
  for two-dimensional speculative cache indices. No separate interleaved
  state layout is visible in this checkout.

The interface rejects a single call mixing spec and non-spec tokens
(`causal_conv1d`, around its initial validation). The Python v5 mixed-batch
split is consequently a correctness crutch; it is not a native mixed-class
state machine.

### Native Xe2 speculative GDN is upstream drift, not the pinned artifact

The open [vllm-xpu-kernels PR #477](https://github.com/vllm-project/vllm-xpu-kernels/pull/477),
head `b7bbaf8605909d04621f1022cdb158924a106397`, is the concrete native-Xe2
spec-decode candidate:

- New files: `csrc/xpu/gdn_attn/xe_2/gated_delta_rule_decode_xe2.{h,cpp,hpp}`;
  `gdn_attn_interface.cpp` dispatches `gated_delta_rule_decode_xe2_spec` when
  Xe2 is enabled.
- `gated_delta_rule_decode_kernel_xe2`/the spec variant use local
  `(1,1,256)`, subgroup 32, and global `(batch_or_spec_decodes, num_v_heads, 1)`;
  each workgroup loops over the V dimension instead of using the generic third
  bucket dimension.
- The spec variant keeps the same semantics: initial state is
  `cache_indices[request, num_accepted_tokens[request]-1]`; each speculative
  token writes a new SSM state at `cache_indices[request,t_local]`; and
  `token_indx` redirects output to the global active buffer. The candidate
  requires contiguous 2-D cache indices and exact row/count sizes, so it also
  needs an explicit graph-padding policy before it can be the item-6 answer.

The wheel/image cannot be assumed to contain PR #477: the closest public source
has no `gated_delta_rule_decode_xe2.*` files, and the handoff identifies only
compiled libraries. Open related context is [PR
#476](https://github.com/vllm-project/vllm-xpu-kernels/pull/476) (conv1d,
claims a spec latency reduction), [issue
#510](https://github.com/vllm-project/vllm-xpu-kernels/issues/510) (mixed
spec/non-spec causal-conv rejection), and [PR
#394](https://github.com/vllm-project/vllm-xpu-kernels/pull/394) (older native
recurrent decode-in-mix-batch proposal). None is a proof that the pinned image
has a native Xe2 spec kernel. No open item-5-specific small-M W4A16/GPTQ PR was
found; current W4A16 search hits are historical/closed, while open W8A16 and
MoE GEMM PRs are different paths.

## 5. File/symbol table

| Path / symbol | Item | Why it matters | Status / unknown |
| --- | ---: | --- | --- |
| vLLM pinned `vllm/model_executor/kernels/linear/mixed_precision/xpu.py:XPUwNa16LinearKernel.apply_weights` | 5 | Flattens token rows to M and calls the W4A16 op | **Confirmed** pinned Python call; target model's actual M distribution still needs a trace |
| `csrc/xpu/onednn/onednn_matmul.cpp:int4_gemm_w4a16` | 5 | Optional `g_idx`, output shape, and dispatch into oneDNN | **Confirmed** e8b12a source |
| `csrc/xpu/onednn/int4_gemm_w4a16.h:dnnl_matmul_w4a16_int4` | 5 | Computes M/K/N, strides, G128 scale mask, primitive-cache key | **Confirmed**; no custom GEMV here |
| `csrc/xpu/onednn/onednn_ext.h:matmul_primitive_cache_t::get` | 5 | Builds oneDNN descriptors keyed by M/N/K and strides | **Confirmed**; selected kernel/tile is runtime unknown |
| oneDNN `src/gpu/intel/gemm/jit/pd.cpp:init_GEMMProblem` | 5 | Converts oneDNN descriptor/quantization into GEMMStone problem | **Confirmed** at `80afa710...`; external dependency |
| oneDNN `src/gpu/intel/gemm/jit.cpp:gen_t::launch_nocopy` | 5 | Forms/routs final `gws/lws`, including M tiling and padding | **Confirmed** source surface; B70 final geometry unknown |
| vLLM pinned `gdn_attn.py:GDNAttentionMetadataBuilder.build` | 6 | Creates active and graph-sized spec metadata | **Confirmed** pinned source; runner interaction remains unmodified |
| `patch_gdn_metadata.py:patch_text` / `PURE_PATCHED` / `COPY_PATCHED` | 6 | Reuses static metadata buffers and compact arange | **Confirmed** overlay only; not C++ narrowing |
| `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:causal_conv1d` and `gated_delta_rule` | 6 | Narrows padded data tensors but validates spec metadata exactly | **Confirmed** e8b12a; graph-padded metadata still unknown/unfixed |
| `tests/gdn_attn/test_gdn_attn_padded.py:test_gdn_attention_accepts_padded_leading_dim` | 6 | Regression for padded data rows and untouched tails | **Confirmed**; does not cover padded spec metadata |
| PR #391 `gdn_attn_interface.cpp` active-prefix hunk | 6 | Proposed `>=` checks and active metadata slices | **Open**, head `d9dd065e...`; stale pre-split candidate |
| `csrc/xpu/gdn_attn/causal_conv1d.hpp:causal_conv1d_spec_kernel` | 11 | Conv state seed, token indirection, per-token conv rollback | **Confirmed** generic SYCL at e8b12a; not native Xe2 |
| `csrc/xpu/gdn_attn/gated_delta_rule.hpp:gated_delta_rule_spec_kernel` | 11 | SSM recurrence, accepted-prefix seed, per-token state writeback | **Confirmed** generic SYCL at e8b12a; not native Xe2 |
| `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_xe2.*` | 11 | Native Xe2 chunk/prefill GDN path | **Confirmed** compiled/source surface; spec dispatch is separate |
| PR #477 `xe_2/gated_delta_rule_decode_xe2.{h,cpp,hpp}` | 11 | Native Xe2 decode/spec kernel candidate and geometry | **Open**, head `b7bbaf8...`; absent from nearest checkout |
| issue #510 `causal_conv1d` mixed-class rejection | 11 | Documents why Python v5 split exists under concurrency | **Open** issue; no durable pinned-image fix |

## 6. What still requires a real B70 checkout/build

A later worker cannot establish these from the compiled-only image or this map:

1. The exact source SHA used to build the image's 0.1.12.3 wheel, unless the
   image/wheel build manifest is recovered.
2. The oneDNN primitive actually selected for the B70 GPTQ layer, its final
   `unroll/wg` tile, GEMV-vs-GEMM choice, and final dispatch dimensions at
   `M=2..5`.
3. Hardware counters proving idle workgroups, XMX starvation, GRF spills, or
   memory-bound behavior; this needs the B70 driver/profiler and repeated
   `M=1..9` runs.
4. Whether PR #391's active-prefix contract can be rebased onto the split
   interface without corrupting graph replay, and whether cap 32/64/128 shape
   changes preserve the 131k ceiling.
5. Whether PR #477's native Xe2 spec kernel builds against the image's exact
   Torch/oneAPI/SYCL-TLA ABI, matches generic state numerics, and improves
   GDN at the actual acceptance widths.
6. Mixed spec/non-spec concurrency, rollback under partial accept, and
   graph-capture correctness on the real service. These are not proven by the
   non-spec padded test or by the Python metadata overlay.

## Primary sources and research record

- [vllm-xpu-kernels repository](https://github.com/vllm-project/vllm-xpu-kernels)
  and [nearest checkout at `e8b12aef...`](https://github.com/vllm-project/vllm-xpu-kernels/tree/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88).
- [e8b12a CMake/build metadata](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/CMakeLists.txt),
  [W4A16 wrapper](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/onednn/int4_gemm_w4a16.h),
  [GDN interface](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/gdn_attn/gdn_attn_interface.cpp),
  [conv spec kernel](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/gdn_attn/causal_conv1d.hpp),
  [GDR spec kernel](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/gdn_attn/gated_delta_rule.hpp),
  and [padded-data test](https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/tests/gdn_attn/test_gdn_attn_padded.py).
- [Pinned vLLM `gdn_attn.py`](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/attention/backends/gdn_attn.py)
  and [pinned XPU W4A16 caller](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/kernels/linear/mixed_precision/xpu.py).
- [vLLM PR #43955](https://github.com/vllm-project/vllm/pull/43955), closed and
  unmerged; the static metadata hunk is the basis of the local overlay and the
  runner hunk is intentionally absent.
- [Graph-padded XPU PR #391](https://github.com/vllm-project/vllm-xpu-kernels/pull/391),
  [issue #389](https://github.com/vllm-project/vllm-xpu-kernels/issues/389),
  [native Xe2 GDR PR #477](https://github.com/vllm-project/vllm-xpu-kernels/pull/477),
  [conv PR #476](https://github.com/vllm-project/vllm-xpu-kernels/pull/476),
  and [mixed-batch issue #510](https://github.com/vllm-project/vllm-xpu-kernels/issues/510).
- [oneDNN pinned commit](https://github.com/uxlfoundation/oneDNN/tree/80afa71049cd69a3df32adcccb623b12cd7baa22),
  especially [`gemm/jit.cpp`](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit.cpp)
  and [`gemm/jit/pd.cpp`](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/pd.cpp).
- [PyPI 0.1.12.3 metadata](https://pypi.org/project/vllm-xpu-kernels/0.1.12.3/)
  (package version/date, but no source SHA) and [oneDNN verbose-mode
  documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)
  (the source of the recommended primitive trace).
