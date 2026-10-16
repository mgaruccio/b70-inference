# Bolt staged SSM prefill — patch-only candidate

**Development tier; not a performance result or a promoted configuration. Device build, native execution, real-model activation, API gates, retrieval and timings: NOT RUN by this worker.** Only the two-file patch and this handoff are delivered. No remote sources, running servers, launchers, weights, build settings or runners were changed.

The requested direction is prefill, not decode. The lead-reported current router+MoE baseline passed 20/20 gate-v2 and exact retrieval; uncached 8270-token prefill median was **8583.184 ms**, TTFT **8.6699 s**. Its warmed, output-1 unitrace capture attributed **1.497786 s / 17.4413%** of **8.587583 s summed device activity** to 440 SSM scans. That attribution is motivation, **not** a prediction of wall-time savings. Staging adds a launch and scratch traffic; a gain is not guaranteed, especially for short batches.

## Exact baseline and scope

Apply `ssm-prefill.patch` to a disposable copy of the **current archive**, not upstream alone:

```text
mike@inference-host:/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-router-r1
upstream: b6334e37e6ef286fc5091a41b6561b2113d915fb
```

Source SHA-256, matched between read-only SSH output and the worker's scratch copy:

```text
ec8401059c853491c69c4ac9d8f78b1da03d887861a1826d94a1e64de8d0319b  ggml/src/ggml-sycl/ssm_scan.cpp
3127310a7ffea17c480d3bdf9653f5173a0018818bc3b6104cc1a2a69625ff9a  tests/test-backend-ops.cpp
```

Only those two files change in the patch. No header or backend-dispatch API changes. The original scan kernel and its launch helper remain byte-identical; the existing operation entry gains an eligibility check before its original launch.

Actual Bolt dimensions were verified in the retained `../20261016-nemotron35-bolt-sycl/ssm-cache-r1-dispatch/server.log.gz`: state `[128,64,128,1]` (also cache capacity 4), x `[64,128,nt,1]`, dt `[128,nt,1]`, A `[1,128]`, B/C `[128,8,nt,1]`. The actual shape is **128 state coordinates, head dimension 64, 128 heads, 8 groups**. This log establishes the model shape, not activation of this new candidate.

## Implementation and numerical contract

1. On the existing in-order queue, compute one F32 softplus and one F32 decay per `(sequence, token, head)`:
   - `dt <= 20.0f ? sycl::log1p(sycl::exp(dt)) : dt`
   - `sycl::exp(dt_soft_plus * A)`
2. The staged scan loads these values but retains the original token loop, `x * dt_soft_plus`, `(state * dA) + (B * x_dt)`, state/C accumulation, subgroup reduction, output store, snapshot indexing and final store. The recurrence/reduction/snapshot/final-store block, beginning at `x_dt`, is byte-identical to the original.
3. No x/B transform, reassociation, prefix/tree scan, reduced precision, compiler flag or decode-kernel change.

**Required acceptance is same-backend bitwise equality**, not merely numerical proximity. Identical source expressions are not proof: different kernel contexts, transcendental lowering, contraction and subnormal handling can still produce different bits. Any native mismatch blocks the candidate; retain the failure, do not weaken the test or substitute an approximate kernel. The separate CPU numerical comparison keeps the default **1e-7 NMSE** limit.

Eligibility is deliberately bounded:

- Existing `GGML_SYCL_ENABLE_FUSION=1`; `2 <= nt <= 1024`; `1 <= K <= 4`; F32 Mamba-2, `d_state=128`, `head_dim=64`.
- `(heads,groups,sequences)` is `(128,8,1)`, `(16,2,2)`, or `(4,1|4,1|2)`: actual Bolt plus native controls.
- State, dt, A and ids contiguous; state strides additionally canonical, including its last dimension. This matters for snapshot output even when the input has only one cache row.
- x/B/C have contiguous channel/group planes, with ordinary or padded token/sequence strides. Axis-permuted layouts fall back. Strides and input extents fit the original signed-int indexing, with float-aligned pitches.
- Output storage must not overlap any input. Read-only input/input sharing is allowed; ids select input cache rows only. Output sequence order remains batch order, not ids order. No host readback of ids is introduced; ids must satisfy the existing valid-index contract.
- The queue must be in order and the existing VMM pool must be enabled and supported. This additional conservative restriction avoids the legacy pool's full-cache path, which can immediately free USM. No pool implementation is changed.

One `ggml_sycl_pool_alloc<float>` holds both tables through both submissions. The VMM pool only rewinds its cursor on release; backing stays mapped, and subsequent reuse on the same in-order queue follows the consumer. No asynchronous host buffer or new synchronization is used. Maximum requested scratch for Bolt is **1 MiB** (`2 * 128 * 1024 * sizeof(float)`); VMM commit granularity can be larger. Graphs remain off in the required validation configuration; graph replay is not validated here.

Decode (`nt=1`), fusion-off, VMM-off, unlisted shapes/layouts and larger nt/K retain the original path. Existing `scope_op_debug_print` emits `call ggml_sycl_ssm_scan_prefill:` only on staged dispatch; it is debug-only and must not be enabled for timing.

## Native regression contract — expected, not yet observed

Selector: `test-backend-ops test -b SYCL0 -o BOLT_SSM_PREFILL`.

**34 cases per run**, **23 staged candidates / 11 fallback candidates** with fusion on and VMM available. Every case also executes one **original-kernel K=5 reference**, which must never stage. With fusion off there are **0** staged dispatches; with fusion on **23**. There are 68 scan operations per full run, including references.

Coverage:

- Actual Bolt and small 4-head controls at **1, 2, 32, 78, 128, 256, 512, 1024 tokens** (16 cases). 78 is the remainder of 8270 with 1024-token batches.
- Bolt-size K=4 strides; groups=heads; reversed sequence ids; K=3/K=4 snapshots and unwritten slots; padded token/sequence strides and nonzero view offsets; permuted-axis fallback; gather from row 3 of a four-row Bolt state; duplicate ids with independent output sequences; threshold/edge inputs (10 cases).
- State size 256, head dimension 32, unlisted heads/groups/sequences, 1025 tokens, K=5, and decode with K=4 (8 fallback cases).
- Edge dt includes negative underflow inputs, signed zero, the smallest positive F32 subnormal, values immediately below/at/above 20, and 80; A includes negative zero and weak/strong decay. No NaN/Inf inputs or undefined aliases.

The reference changes **only K**, a runtime snapshot-store guard in the unchanged kernel. It computes the same y and first K state slots; fusion-off execution independently checks this equivalence before fusion-on testing. There is no forced CPU fallback, environment mutation inside a process, artificial operation flag, or mutable input/output fixture alias.

The graph concatenates the **entire candidate output** with the corresponding original-kernel reference prefix. That prefix excludes only reference-only extra slots, never candidate output. A length assertion rejects incomplete comparisons. The test reads every F32 word and compares raw bits, including signed zeros, final state, every candidate snapshot and initialized unwritten slots. It also independently checks unwritten-slot canaries. No arithmetic delta or scalar reduction can hide a mismatch.

Each case emits two `[BOLT_SSM_PREFILL]` full-array count records (`under_test` and `CPU`); failures emit `[BOLT_SSM_PREFILL_MISMATCH]` with the first index/region and raw words after counting all mismatches. Per full run, **each backend** compares **18,883,328 y words + 19,939,328 state words**. CPU NMSE is checked separately for y and each complete state slot, in both halves, at the unchanged 1e-7 limit. Expected records: **34 per backend**, all mismatch counts zero. The largest case has a 72 MiB joined output; buffers do not multiply all token/head/K combinations.

### Lead-owned executable sequence

Run only in the existing isolated inference-host oneAPI 2026.0 / oneDNN environment, with the GPU dedicated to Bolt. `D` is a disposable copy of the exact router archive, `P` this patch, and `R` the lead's raw-results directory outside Pi (for example `/home/mike/b70-evals/20261016-nemotron35-bolt-prefill-ssm`). Preserve original source and server state. These commands are a handoff, **not worker execution evidence**:

```bash
cd "$D"
git apply --verbose --check --whitespace=error "$P"
git apply --verbose --whitespace=error "$P"
# Require real "Checking patch"/"Applied patch" output, never "Skipped patch".
cmake -S . -B build-sycl -DLLAMA_BUILD_TESTS=ON
cmake --build build-sycl --config Release -j6 --target llama-server test-backend-ops

# In the existing GPU runtime environment, after the lead's dependency precheck:
set -euo pipefail
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_ENABLE_ESIMD=1 GGML_SYCL_ENABLE_GRAPH=0
export GGML_SYCL_ENABLE_VMM=1 GGML_SYCL_DEBUG=1
mkdir -p "$R/native"
for fusion in 0 1; do
    GGML_SYCL_ENABLE_FUSION="$fusion" ./build-sycl/bin/test-backend-ops \
        test -b SYCL0 -o BOLT_SSM_PREFILL 2>&1 | tee "$R/native/fusion-$fusion.log"
    test "$(grep -cF '[BOLT_SSM_PREFILL] backend=under_test ' "$R/native/fusion-$fusion.log")" -eq 34
    test "$(grep -cF '[BOLT_SSM_PREFILL] backend=CPU ' "$R/native/fusion-$fusion.log")" -eq 34
    ! grep -F '[BOLT_SSM_PREFILL_MISMATCH]' "$R/native/fusion-$fusion.log"
done
test "$(grep -cF '[SYCL][OP] call ggml_sycl_ssm_scan_prefill:' "$R/native/fusion-0.log" || true)" -eq 0
test "$(grep -cF '[SYCL][OP] call ggml_sycl_ssm_scan_prefill:' "$R/native/fusion-1.log" || true)" -eq 23

# One eligible-shape control must instead fall back when the VMM pool is disabled.
GGML_SYCL_ENABLE_VMM=0 GGML_SYCL_ENABLE_FUSION=1 ./build-sycl/bin/test-backend-ops \
    test -b SYCL0 -o BOLT_SSM_PREFILL \
    -p 'n_seq_tokens=2,n_head=128,n_group=8,n_seqs=1,K=1,' \
    2>&1 | tee "$R/native/vmm-off.log"
test "$(grep -cF '[BOLT_SSM_PREFILL] backend=under_test ' "$R/native/vmm-off.log")" -eq 1
test "$(grep -cF '[SYCL][OP] call ggml_sycl_ssm_scan_prefill:' "$R/native/vmm-off.log" || true)" -eq 0
```

Require **34/34 pass in both full runs**, plus 1/1 VMM-off, no unsupported selected cases, no numerical failures, the word/record totals above and exact activation counts. Keep all compiler/native output, including failures. Follow with the lead's existing relevant legacy native gates; do not replace them with these cases.

Then the lead executes the already-defined public-boundary journey, in order:

1. Debug real-model activation using the original Bolt weights: verify the staged marker on eligible prefill batches (including 1024/78), never decode. Retain source shapes/strides and activation counts.
2. Fresh **nondebug 20/20 API gate-v2**, before any timed warm-up. Keep q8 KV, F32 recurrent state, c65536, b/ub1024, single sequence, ESIMD1, FUSION1, graphs0, hardware and all other baseline settings unchanged.
3. With the existing serving runner and exact unchanged prompts, baseline versus candidate: short **55** and long **8270** uncached prompt tokens, **128** output tokens, greedy/EOS-ignore policy unchanged; **one warm-up + three measured runs** each. Record raw prefill, TTFT, decode and full outputs, not only medians.
4. Existing exact **8238-token retrieval** input must return the expected sentinel. Preserve request/response/event artifacts, token/cache counts, commands, environment, failures and summaries in `R`; no replacement proxy or synthetic-only gate.
5. Clean up only candidate test processes/artifacts as the lead directs. Do not promote a persistent launcher, change weights, restore another model, or share the GPU with another workload.

This reduced development process is **not** the standard publication suite in `BENCHMARKING_STANDARDS.md`; its publication checklist is incomplete. The worker did not add or edit a runner.

## Worker verification actually performed

The linked worktree was verified before writing. Source was copied only to ignored `.pi/ssm-prefill/base/`, using read-only remote `bash --noprofile --norc -c` and:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host
```

Remote commands were `cd` plus `tar -cf -` for the two touched files and required headers/pool/CPU/backend reference sources, bounded `grep` reads, and `sha256sum` for the two patched files. No build, model, GPU or container commands were run locally or remotely.

Observed static checks (all substantive checks passed):

```bash
# From the worktree root; P is results/.../ssm-prefill.patch's absolute path.
git apply --verbose --check --whitespace=error --directory=.pi/ssm-prefill/base "$P"
git apply --verbose --reverse --check --directory=.pi/ssm-prefill/candidate "$P"
# Copy just the two originals into .pi/ssm-prefill/applied/, then:
git apply --verbose --whitespace=error --directory=.pi/ssm-prefill/applied "$P"
cmp .pi/ssm-prefill/candidate/ggml/src/ggml-sycl/ssm_scan.cpp .pi/ssm-prefill/applied/ggml/src/ggml-sycl/ssm_scan.cpp
cmp .pi/ssm-prefill/candidate/tests/test-backend-ops.cpp .pi/ssm-prefill/applied/tests/test-backend-ops.cpp
git apply --verbose --reverse --check --directory=.pi/ssm-prefill/applied "$P"
/usr/bin/c++ -std=c++17 -fsyntax-only -I.pi/ssm-prefill/base/ggml/include .pi/ssm-prefill/candidate/tests/test-backend-ops.cpp
```

- Applicability and whitespace: both files explicitly checked, applied cleanly, matched candidate byte-for-byte, reverse-check passed. Initial checks from inside ignored scratch printed `Skipped patch`; those no-op checks were rejected and rerun from the root with explicit `--directory` as above.
- `git diff --cached --check` initially flagged four space-only unified-diff context lines. Their context prefixes were normalized to empty lines (accepted by `git apply`); final artifact whitespace and applicability checks passed without changing patched source.
- Full native-test translation-unit syntax: exit 0, GCC **16.2.1 20260810**. No executable or object was built or run.
- Source comparison assertions: original scan/helper unchanged; staged recurrence/reduction/snapshots/final stores byte-identical. Registration enumeration: 34 cases, 23/11 eligibility, 34 forced fallback references; padded-view bounds and aggregate word counts checked statically.
- ast-grep C++ `ERROR` query on patched `ssm_scan.cpp`: `[]` (exit 1 means no matches). This is a parse check, **not SYCL type checking or device compilation**.
- Native/SYCL/serving/performance results remain **NOT RUN**. No numerical identity or speed claim is made.

## External sources and conclusions

Fresh official references consulted during this slice:

- https://github.khronos.org/SYCL_Reference/iface/queue.html — in-order queues order submissions on the same queue; staging and consumption share it, and other queue types fall back.
- https://github.khronos.org/SYCL_Reference/iface/usm_allocations.html — USM deallocation is not a substitute for waiting for users; use retained pool backing, not short-lived raw USM. The source audit motivates the VMM-only guard.
- https://github.khronos.org/SYCL_Reference/iface/math-functions.html — SYCL's OpenCL math precision guarantees do not establish bitwise equivalence between differently compiled kernels. Keep the same functions/expressions and require native exactness.

The lead/researcher's fresh sources carried into this slice:

- https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ssm_scan.cpp — preserve the sequential recurrence; patch the exact router archive, not an assumed pristine upstream tree.
- https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md — profile attribution motivates testing; it does not prove speedup.
- https://github.com/ggml-org/llama.cpp/pull/23142 — MoE counting-sort work is already present in the baseline; no blind backport or adjacent MoE change.
