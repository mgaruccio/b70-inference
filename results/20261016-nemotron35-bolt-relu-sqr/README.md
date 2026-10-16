# Experimental Bolt F32 RELU → SQR fusion

**Experimental, not accepted:** the corrected fusion builds and passes all154 native
GPU cases (112 legacy +21 fusion-on +21 fusion-off), but the model API gate fails
19/20 on `drawer` → `rewrad`. Timing and retrieval were blocked. The unchanged R1
baseline also showed this failure on one launch and passed on the next; this is
not evidence that fusion caused it. No speed improvement is claimed.

This bounded F32 SYCL patch leaves weights, precision policy, production launchers,
and host configuration unchanged. Full evidence: [execution record](../20261016-nemotron35-bolt-sycl/execution.md#bounded-kernel-fusion-experiment).
Actual-model debug traces confirm **80 fewer kernel launches per decode evaluation**:
all80 RELU→SQR pairs use the fused path, including every observed routed width.
This is dispatch verification, not a throughput measurement. Qwen was restored.

Target source: `llama.cpp-bolt-b6334e37-nvfp4-r1`, source revision
`b6334e37e6ef286fc5091a41b6561b2113d915fb` plus the accepted NVFP4 R1 patch.
Apply `relu-sqr.patch` to a copied candidate source; do not apply it to the
production checkout.

## Contract and implementation

The Nemotron-H Bolt graph uses `LLM_FFN_RELU_SQR` in routed and shared FFN
paths. The patch recognizes only the exact adjacent graph chain
`UNARY(RELU) -> SQR` and computes the existing unary semantics directly
as `op_sqr(op_relu(x))` in one F32 pointwise kernel.

Fusion requires all of the following:

- the existing `GGML_SYCL_ENABLE_FUSION` switch is enabled;
- both graph nodes are the exact RELU/SQR pair and the source, intermediate,
  and destination tensors are F32;
- equal shape, contiguous source/intermediate/destination, non-empty output, and
  fewer than `2^31` elements;
- `ggml_can_fuse()` accepts the chain, preserving single-use, view, and output
  protection for the intermediate RELU.

Every other type, layout, shape, consumer, output request, or graph form falls
back to the ordinary two-unary dispatch. The scheduler consumes exactly one
following node after a successful fusion. With existing SYCL debug enabled, the
native path emits the existing-style marker containing
`fused RELU_SQR`; no new global control or profiling is added.

## Files

- `relu-sqr.patch` — source and focused native-test patch.
- `test_patch.py` — stdlib-only source/scalar checks; it does not compile or
  execute SYCL.

## Source-level check

From this result directory, against an unmodified R1 tree and a candidate tree
with the patch applied:

```sh
python3 -B test_patch.py /absolute/path/to/r1 /absolute/path/to/candidate
```

## Native test selector and expected count

The patch keeps the unique `RELU_SQR` operation label and adds the selector
prefix `relu_sqr=1`. The native `test-backend-ops` selector must execute exactly
**21 cases per invocation** with no skipped or unsupported cases:

- 4 existing F16/F32 controls;
- 4 existing Bolt shape controls: routed width 2,048 and shared width 5,376 at
  decode and 1,024-column prefill/multi-column shapes;
- 8 actual routed FFN-up expert shapes from the original GGUF: widths 1,536,
  2,048, 2,304, and 2,560, each with `n_expert_used=9` at `{width,9,1,1}`
  and `{width,9,31,1}`;
- a 257×3 non-workgroup-multiple case;
- contiguous rejection controls for a strided input, an extra RELU consumer,
  and an explicitly requested RELU output;
- a fused pair followed by a consumer, verifying that the scheduler does not
  overskip the subsequent node.

After the lead builds the copied candidate on the inference host, run both the
candidate path and the existing fallback control:

```sh
# 21/21 candidate cases; retain output and require the fused marker.
GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_DEBUG=1 \
  "$PROTO_BUILD/bin/test-backend-ops" test -b SYCL0 -o RELU_SQR -p 'relu_sqr=1'

# 21/21 ordinary two-launch cases; same CPU reference and strict comparison.
GGML_SYCL_ENABLE_FUSION=0 \
  "$PROTO_BUILD/bin/test-backend-ops" test -b SYCL0 -o RELU_SQR -p 'relu_sqr=1'
```

The test harness compares the complete graph with its CPU reference through
`err()`/`max_nmse_err()`. `test_relu_sqr::max_nmse_err()` returns `0.0` for F32,
so every finite F32 element must match exactly; only the retained F16 fallback
controls use the inherited framework tolerance. The reuse case additionally
checks the RELU intermediate and side-consumer result; the output case checks
that an explicitly requested intermediate remains materialized. Do not relax
the F32 tolerance to hide a mismatch.

## Lead-only acceptance and benchmark gate

The lead owns the copied-source build and native execution. Before any timing,
run the 21-case selector with fusion on and off, the existing accepted NVFP4
native tests, and the fixed 20-case API guards. Only after all guards pass may
the lead run the predeclared short `55 -> 128` and long `8270 -> 128` requests
with one warm-up and three measured repetitions, plus exact 8K retrieval and a
matched repeat of any apparent gain. Restore Qwen and verify the canary and
invariants afterward. No speed claim is made by this artifact.

This remains a development result, not a publication-standard performance
comparison. Follow `BENCHMARKING_STANDARDS.md` for any later baseline/candidate
measurement; no timing, environment publication, or persistent launcher change
is included here.

Fresh research informing the bounded approach (not a direct port):

- [B70 inference cookbook PR 41](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/pull/41) — similar exact-shape fusion coverage.
- [SYCL event reference](https://github.khronos.org/SYCL_Reference/iface/event.html) — profiling/event operations may block; this patch enables no profiling.
