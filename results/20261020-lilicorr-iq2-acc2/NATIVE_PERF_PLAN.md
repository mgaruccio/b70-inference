# Exact acc2 native operator screen — development only

Question: does the two-accumulator integer helper beat the **passed IQ2 batch4**
implementation? The rejected packed variant is not the baseline. This is not a
serving benchmark, promotion or proof that any source dependency was the cause.

## Environment, preconditions and test data

`inference-host` B70, existing cached frozen oneAPI2026 image and qualified
configuration, exclusive GPU ownership. Fresh acc2 build must pass whole source,
seed/original, graph,105-object closure and six-artifact checks. Both variants'
absolute immutable executables/library paths are mounted read-only. Only their
SYCL library differs; native test executable is identical. No installation,
profiling queue, SDK/driver/adapter/flag/math/geometry/launcher change.

The unchanged sibling native test-file is copied byte-for-byte:
IQ2_XXS up `[K1024,M2048,240,1]` / F32 `[1024,1,4,1]`, down
`[K2048,M1024,240,1]` / F32 `[2048,9,4,1]`; selected IDs `[9,4,1,1]`
with 960-byte expert stride; accumulation10 and existing private marker0x4e535631.
The ABI/header and executable-source hashes are asserted in the runner.

**Limitation:** generic initialization shuffles IDs0..8, not arbitrary IDs over
all240; random weights/inputs differ per process. These fixed representative K/M
shapes and cache-hot data do not reproduce the whole serving input distribution
or establish aggregate target cost. Do not call this a full-model replay.

## Executable real journey

1. Run `python3 -B test-native-perf.py -v` locally and remotely before capture.
2. In this declared package root, invoke
   `python3 -B run-native-perf.py --lead-run-native-perf`. The inherited preflight
   validates original identity/source/health/owner, both complete source/build
   proofs, adapter hashes, capacity, unique names and protected config before
   stopping the original. Output `native-perf-r1` is fresh and never overwritten.
3. The exclusive W3 native container runs unchanged native Q8 82/54-byte and
   IQ2 34/16-byte gates for **each** library via the strict qualifier. Then its
   ordinary `test-backend-ops test -b SYCL0 -o MUL_MAT_ID --test-file
   /iq2-batch4/native-perf-cases.txt -j 1` performs both CPU comparisons with
   separate LD_DEBUG/SYCL_UR_TRACE/GGML_SYCL_DEBUG diagnostics. Require actual
   variant DLL and real UR launches, no skips and no profiling flag. Preserve
   full names/handles; confirm their binding rather than inferring from counts.
4. Diagnostics/capture OFF, run the same immutable `perf` CLI in ABBA x3 order:
   base,acc2,acc2,base repeated three times. Built-in warmup and >=1-second steady
   graph runs must yield exactly24 finite positive rows (six processes/variant,
   two operations each). Retain all rows/failures; no best-run or outlier removal.
5. Inherited finally restores original on all paths. In a **new SSH/Python
   process**, run `python3 -B run-native-perf.py --lead-verify-restored` regardless
   of capture status. Require exact original/source/health/sole GPU owner and
   stopped experiments. Only during stopped-cell comparison, reuse the existing
   exact-native-Id false/null OomKillDisable spelling comparator; captured data
   and original-container checks are untouched. Every other field remains strict.
6. Retain declaration, argv, all raw native/diagnostic/perf outputs, SHA inputs,
   container/ownership data and independent restoration under `native-perf-r1/`.
   Sync without deleting failures. No persistent promotion or source cleanup.

The native tool measures synchronous whole operator-graph wall microseconds,
including Q8_1 quantization/dispatch/math/amortized synchronization, **not pure
GPU kernel/event time**. Only a clear native gain merits the unchanged full
public A1/W3a/W3b/A2 comparison (100 jobs,24 measured, exact fidelity/counters,
full8/7x8/max3 and independent restoration) defined in `README.md`.

Fresh primary research and exact signedness/overflow/order arguments are in
`README.md`. Extra integer liveness may spill at the observed128GRF/SIMD16;
source ILP and instruction counts alone do not establish a win.
