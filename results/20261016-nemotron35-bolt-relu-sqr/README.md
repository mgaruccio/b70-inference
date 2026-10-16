# Experimental Bolt F32 RELU → SQR fusion

Patch artifact only. This is one bounded, precision-preserving SYCL increment
against accepted Bolt NVFP4 R1. It does not change MoE scheduling, SSMs, graphs,
precision policy, launchers, models, or persistent inference configuration. The
worker performed no compiler, GPU, profiler, or inference-host workload.

Target source: `llama.cpp-bolt-b6334e37-nvfp4-r1`, source revision
`b6334e37e6ef286fc5091a41b6561b2113d915fb` plus the accepted NVFP4 R1 patch.
Apply `relu-sqr.patch` to a copied candidate source; do not apply it to the
production checkout.

## Contract and implementation

The Nemotron-H Bolt graph uses `LLM_FFN_RELU_SQR` in routed and shared FFN
paths. The patch recognizes only the exact adjacent graph chain
`UNARY(RELU) -> UNARY(SQR)` and computes the existing unary semantics directly
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
**13 cases per invocation** with no skipped or unsupported cases:

- 4 existing F16/F32 controls;
- 4 Bolt shapes: routed width 2,048 and shared width 5,376 at decode and
  1,024-column prefill/multi-column shapes;
- a 257×3 non-workgroup-multiple case;
- contiguous rejection controls for a strided input, an extra RELU consumer,
  and an explicitly requested RELU output;
- a fused pair followed by a consumer, verifying that the scheduler does not
  overskip the subsequent node.

After the lead builds the copied candidate on the inference host, run both the
candidate path and the existing fallback control:

```sh
# 13/13 candidate cases; retain output and require the fused marker.
GGML_SYCL_ENABLE_FUSION=1 GGML_SYCL_DEBUG=1 \
  "$PROTO_BUILD/bin/test-backend-ops" test -b SYCL0 -o RELU_SQR -p 'relu_sqr=1'

# 13/13 ordinary two-launch cases; same CPU reference and strict comparison.
GGML_SYCL_ENABLE_FUSION=0 \
  "$PROTO_BUILD/bin/test-backend-ops" test -b SYCL0 -o RELU_SQR -p 'relu_sqr=1'
```

The test harness compares the complete graph with its CPU reference. The reuse
case additionally checks the RELU intermediate and side-consumer result; the
output case checks that an explicitly requested intermediate remains materialized.
The F32 pointwise contract is finite and should be exact; do not relax the
native tolerance to hide a mismatch.

## Lead-only acceptance and benchmark gate

The lead owns the copied-source build and native execution. Before any timing,
run the 13-case selector with fusion on and off, the existing accepted NVFP4
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
