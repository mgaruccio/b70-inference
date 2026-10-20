# Existing native operator comparison — development only

## Question and control

Does the rejected exact packed n4 implementation improve either representative
IQ2 operation over the passed batch4 implementation? Static ISA instruction counts
are not timing, and the eight-instruction serial dp4a chain is only a hypothesis.
No accumulator-split patch exists. No inference/source/library/launcher change is
made by this comparison. This is not a serving speedup or a publishable benchmark.

A is the completed `...-lilicorr-iq2-batch4-r1` source/build; B is rejected
`...-lilicorr-iq2-packed-r2`. The existing packed source proof checks both complete
exports/builds against their completed builds and the immutable original export;
only `libggml-sycl.so.0` differs between A and B among the protected artifacts.
The frozen native test executable is identical. Each process executes its own
variant's absolute executable and uses that variant's library directory. The
separate loader diagnostic must confirm the actual loaded SYCL library path.

## Environment and inputs

Use `inference-host` B70, the pinned cached oneAPI image and existing qualified
container configuration, with exclusive GPU ownership. No SDK/driver/install,
math flag, graph/queue/adapter workaround, or event profiling is allowed.

`native-perf-cases.txt` uses the frozen executable's existing `--test-file` format:

- up: IQ2_XXS weights `[1024,2048,240,1]`, F32 input `[1024,1,4,1]`;
- down: IQ2_XXS weights `[2048,1024,240,1]`, F32 input `[2048,9,4,1]`;
- selected IDs `[9,4,1,1]`, strides `[4,960,3840,3840]`;
- native F32 accumulation tag 10 and existing private scope word `0x4e535631`;
- contiguous native outputs `[M,9,4,1]`.

ABI values come from pinned `ggml.h` SHA256
`475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3`;
`test-backend-ops.cpp` SHA256 is
`3a2167a0b4d9e26feea23ef299e11fe6468baf19be6e0304ebde57ea240e1c6d`.

**Input limitation:** the unchanged generic initializer uses shuffled IDs 0..8,
not arbitrary IDs across all 240 experts. Weights/activations are independently
randomized per process. This is a cache-hot same-shape operator experiment, not a
replay of serving inputs. Do not infer total serving latency or a dramatic gain.

## Executable journey and expected results

1. Run CPU fixture/argv/lifecycle-adapter checks locally; sync only the new files.
2. Invoke `python3 -B run-native-perf.py --lead-run-native-perf` remotely. Existing
   qualified preflight verifies immutable original identity/source/health/owner,
   all source/build proofs, capacity, names and code hashes before disruption.
3. The inherited exclusive capture stops the original service and runs one native
   container with the two source exports mounted read-only. For **each** variant,
   rerun unchanged Q8 82/54-byte and IQ2 34/16-byte gates through the strict existing
   qualifier. Keep every log, raw capture and checker output, including failures.
4. Run unchanged `test-backend-ops test -b SYCL0 -o MUL_MAT_ID --test-file
   /iq2-batch4/native-perf-cases.txt -j 1` for each variant with diagnostic-only
   `LD_DEBUG=libs SYCL_UR_TRACE=2 GGML_SYCL_DEBUG=1`. Require the actual variant DLL,
   UR launch-call trace, no profiling queue, two CPU comparisons and no skips.
   Retain full UR diagnostics for handle review; do not assume a tracer prints a
   host C++ identifier. The unchanged named IQ2 gates independently validate the
   native n4 route records and exact bytes before the representative test.
5. Disable loader/UR/debug/capture diagnostics. Run the unchanged `perf` command
   on the same two-case file in **ABBA x3** order (six processes/variant). Its
   existing warmup and >=1-second steady graph measurements must produce exactly
   24 supported finite positive result rows. Preserve all samples and failures.
   Inspect medians/dispersion and both directions, without selecting best runs.
6. Inherited `capture` restores the original on success or failure. In a **new
   process**, invoke `python3 -B run-native-perf.py --lead-verify-restored` even if
   capture failed. Require original identity/source/health and sole observed GPU
   ownership, and stopped experimental containers. Do not remove failed outputs.

Artifacts: `native-perf-r1/` under the remote packed-R2 package, synchronized to
this result directory. Retain the exact argv, declaration, metadata, full raw
native/diagnostic/perf outputs and independent restoration evidence. No completed
comparison will be silently overwritten or rerun into this output directory.

## Timing scope and follow-up

The frozen `eval_perf` clocks synchronous `ggml_backend_graph_compute`, duplicates
the output operation and reports mean wall microseconds per graph operation. This
includes Q8_1 quantization, native dispatch, math and amortized synchronization;
it is **not a device event or pure GEMV-kernel time**. Diagnostic tracing is absent
from timed runs. A follow-up candidate must demonstrate a gain over passed
batch4 and native exactness before the existing full real serving comparison against
NoDraft (100 public jobs, 24 measured samples, full 8-position drafter/7x8/max3,
exact counters and independent restoration). This microbenchmark cannot replace it.

## Fresh primary research affecting the process

- https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-register-spills.html
  — shorter live ranges/removing dependencies can help scheduling, but extra
  temporaries can spill; validate performance instead of inferring it from ISA.
- https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/EnvironmentVariables.md
  — `SYCL_UR_TRACE=2` traces UR calls. Use only for separate host diagnostics, not
  timings; do not change queue properties or force an adapter. Frozen runtime
  behavior and captured logs remain authoritative over rolling documentation.
