# Bolt R1 + MoE: sigmoid/bias/full-sort fusion

**Fixture-v2 source-only development candidate. The initial remote SYCL build passed, but fusion-off native testing failed 78/82; this test-only revision needs fresh remote validation. No fusion-on/model test, speed claim or promotion.** Baseline is accepted R1 plus the corrected `../20261016-nemotron35-bolt-moe-reduce/moe-reduce.patch` (validated 40.1887/38.8496 short/long decode tok/s), not an SSM/post-fusion stack. The patch changes only `ggml/src/ggml-sycl/ggml-sycl.cpp` and `tests/test-backend-ops.cpp`.

## Contract

- Fuse consecutive executable SIGMOID → ADD → descending ARGSORT, only contiguous F32 `[experts,tokens,1,1]`, experts `{240,304,368,432,496}`, tokens `1..1024`, and F32 bias `[experts,1,1,1]`. Emit the original raw probabilities, biased values and **full I32 indices**.
- Admit only one optional intervening metadata node: the model's raw-probability reshape `[1,experts,tokens,1]` before ADD. `build_moe_ffn` visits this external weight view before the selection branch. Never skip an executable node; no graph/controller changes.
- Match the original local/shared-memory sort's device limit, workgroup and tasks-per-thread calculation (decline non-divisible workgroups). Compute sigmoid once per coordinate with the original `1.f/(1.f+op_exp(-x))`, then a workgroup **global-and-local** barrier publishes the outputs before calling the **unchanged** bitonic sort. Strict comparisons, ties and non-power-of-two padding are unchanged. No top-k substitution, global multipass changes, approximation, `restrict`, or math flags.
- Allow exact-coordinate logits→probabilities or logits→biased reuse. Also allow full-range indices→dead computed logits reuse after the global barrier, but reject input/output-marked logits, external consumers and old-logit views. Views of the new index output remain valid.
- Reject shifted/cross-row output/logit overlaps, probabilities/biased overlap, indices/probabilities overlap, and any bias/output overlap. **Indices/biased overlap remains excluded**, even when biased is dead: the unchanged sort's last barrier fences only local memory. Do not add that alias without a different, separately approved implementation.
- `ggml_can_fuse_subgraph_ext` lists all three materialized outputs, allowing their external consumers/output flags while still requiring COMPUTE. Existing `GGML_SYCL_ENABLE_FUSION` controls dispatch; the sole new debug marker is `ggml_sycl_op_router_fused`.

## Primary sources and conclusions

Pinned upstream is `b6334e37e6ef286fc5091a41b6561b2113d915fb`; accepted R1 source is the supplied local snapshot, with corrected MoE applied first. Lead/researcher fresh research preceded implementation. The worker additionally fetched the pinned allocator and graph builder on 2026-10-16.

- [SYCL sort and launcher](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ggml-sycl.cpp): strict, non-stable bitonic ordering; local barriers do not publish the new global writes.
- [Sigmoid](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.cpp) and [op_exp](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/element_wise.hpp): preserve F32 formula and `sycl::exp`.
- [Fusion semantics](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml.c): listed outputs bypass visibility checks, not COMPUTE checks.
- [Allocator](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-alloc.c): UNARY can reuse a dead same-layout parent; rejecting all aliases would exclude ordinary allocations.
- [Graph builder](https://raw.githubusercontent.com/ggml-org/llama.cpp/b6334e37e6ef286fc5091a41b6561b2113d915fb/src/llama-graph.cpp): unbiased probabilities feed weights separately; their reshape is metadata, not an intervening launch.

## Fixed native gate (lead, remote B70 only)

**82 cases per mode, 164 total; expected router starts off/on = 0/67.**

- 40 eligible: all five widths × tokens `{1,2,17,1024}` × ordinary/tied data. Tied data includes signed zero, extreme finite logits, saturation and exact bias-induced ties; every width exercises padding.
- 25 eligible: each width at 17 tokens with sigmoid-inplace, exact logits/biased reuse, exact dead-logits/indices reuse, real gallocr reuse, and the model's raw-probability reshape. The five gallocr cases assert actual logits/probability pointer equality in both graphs, not simulated reuse; they also include the model reshape and an external weight-view consumer.
- Two eligible external probability/biased-view consumers.
- 15 fallbacks: width 239; 1025 tokens; extra dimension; strided logits/bias; row-varying bias; ascending sort; intervening executable copy; inplace ADD; repeated bias aliasing the first/second probability row; wrong unary; index reuse of output-marked, externally consumed, or viewed logits.

Index-alias cases mark only the I32 tensor OUTPUT: `ggml_set_output` recursively marks `view_src` owners and would otherwise make positive dead-logit reuse fall back. Both graphs assert that the logit owner is OUTPUT only for `indices_logits_output`, so consumer/view cases exercise their intended guards. Production visibility checks and the 82/67 gate are unchanged.

Each case uses independent deterministic inputs and an unfused same-backend control, separated by an executable output-marked CONT. Complete probability/biased arrays must match bitwise; complete original I32 tensors and both index arrays must match exactly. CPU F32 NMSE remains **1e-7**. CPU index checking follows existing `test_top_k(ties)` convention: full permutations and identical ranked values, not an unspecified CPU tie order; same-backend tie indices are never relaxed. Sentinels and extra consumers remain checked. No NaN cases or tolerance changes. Shifted output aliases and indices/biased aliases are rejected but not executed as native graphs: their unfused parallel kernels lack a safe reference contract.

Fixture-v2 changes only four negative fixtures: bias-alias logits cycle by row through `{-FLT_MAX,0,FLT_MAX}`, giving exact `{0,0.5,1}` probabilities and half-integer biased keys. First/second-row bias still aliases the inplace sigmoid output, and its pre-sigmoid values differ from the required post-sigmoid bias. Consumer/view fixtures use zero logits and integer bias for distinct ranked keys; an executable CONT reads the overwritten logit storage without arithmetic on I32-as-F32 subnormals. Every copied byte is checked against the independent same-backend control and the corresponding full/prefix I32 output, and CPU/device copy bytes must match exactly (stricter than NMSE). The existing raw/biased/index pair checks, CPU F32 1e-7 gate and 82/67 counts are unchanged; two additional zero-mismatch copy markers are required per mode.

On a disposable isolated **R1+MoE** source/build, with absolute `SRC`, `BUILD`, `PATCH`, `RUN`:

```bash
set -euo pipefail
patch --dry-run --batch -p1 -d "$SRC" < "$PATCH"
patch --batch -p1 -d "$SRC" < "$PATCH"
cmake --build "$BUILD" --target test-backend-ops llama-server --parallel 2
mkdir -p "$RUN"
for f in 0 1; do
    env GGML_SYCL_ENABLE_GRAPH=0 GGML_SYCL_ENABLE_FUSION="$f" GGML_SYCL_DEBUG=1 \
        "$BUILD/bin/test-backend-ops" test -b SYCL0 -o '^BOLT_ROUTER$' -j 1 \
        > "$RUN/router-native-$f.log" 2>&1
    grep -F '82/82 tests passed' "$RUN/router-native-$f.log"
    test "$(grep -cF 'BOLT_ROUTER floats: mismatches=0 ' "$RUN/router-native-$f.log" || true)" -eq 82
    test "$(grep -cF 'BOLT_ROUTER indices: mismatches=0 ' "$RUN/router-native-$f.log" || true)" -eq 82
    test "$(grep -cF 'BOLT_ROUTER allocator: logits=probs exact (2 graphs)' "$RUN/router-native-$f.log" || true)" -eq 5
    test "$(grep -cF 'BOLT_ROUTER consumer bytes: mismatches=0 ' "$RUN/router-native-$f.log" || true)" -eq 2
done
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_router_fused:' "$RUN/router-native-0.log" || true)" -eq 0
test "$(grep -cF '[SYCL][OP] call ggml_sycl_op_router_fused:' "$RUN/router-native-1.log" || true)" -eq 67
```

Zero, unsupported or skipped cases cannot satisfy these fixed counts. Also run the lead's unchanged legacy 112 and MoE 40 off/on cases: **316 native cases total**. Retain raw logs, including failures; no identical retry of a failed candidate.

## Retained initial failure and fixture-v2 status

The lead's initial `router-r1-dispatch/server.log` under `../20261016-nemotron35-bolt-sycl/` is retained unchanged. Observed: fusion-off passed 78/82; both bias-alias cases had zero pair mismatches before the custom CPU rank oracle returned infinity; consumer/view cases had SCALE NMSE=1 on both sides despite zero routing pair mismatches. Fusion-on and model testing were not attempted.

Source inspection confirms the old rank oracle compares CPU indices against SYCL-computed score keys, while ordinary sigmoid/add scores need only satisfy 1e-7 CPU NMSE; complementary logits can create near-tie keys without identical rounding across backends. It also confirms SCALE operated on index bit patterns interpreted as subnormal F32. Near-tie ordering and FTZ differences are **hypotheses**, not tensor-level diagnoses from this log. The revision removes those fixture assumptions without relaxing an oracle or changing production code. Use a fresh isolated R1+MoE source/build and a new `router-r1-fixture-v2` run directory; never overwrite/retry the failed run in place. Remote results for this revision remain pending.

## Predefined public-boundary gate (lead-owned; not implemented here)

Use the isolated oneAPI 2026.0 B70 build. After native gates, debug smoke must show all **40 decode layers** using this marker (the supplied baseline has 840 sigmoid starts = 40×21 decode evaluations). Do not time debug runs. Run fresh non-debug fixed **20/20 HTTP API checks before timing**. Only if green, run matched 55/8270-input, 128-forced-output requests, one warm-up plus three repetitions, and exact 8238-token retrieval `BOLT-7319-CEDAR`. Compare against R1+MoE, keeping original weights, q8 KV, F32 recurrent state, context 65536, batch/ubatch 1024, ESIMD=1, FUSION=1, graphs=0. Retain exact server/request commands, environment, complete API outputs and raw timings in the lead's normal run logs. Failed native/API candidates stay untimed. Keep the fastest validated Bolt resident; clean up only disposable experiment scratch. This is a development experiment, not a standard-compliant publication comparison.

## Worker close gate

Observed locally: GCC `16.2.1 20260810`; the following all passed against the supplied R1 snapshot **after applying the corrected MoE patch**:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
    -I .pi/router-fusion/candidate/ggml/include \
    .pi/router-fusion/candidate/tests/test-backend-ops.cpp
patch --dry-run --batch -p1 -d .pi/router-fusion/base \
    < results/20261016-nemotron35-bolt-router-fusion/router-fusion.patch
patch --dry-run --batch -R -p1 -d .pi/router-fusion/candidate \
    < results/20261016-nemotron35-bolt-router-fusion/router-fusion.patch
git apply --check --whitespace=error --directory=.pi/router-fusion/base \
    results/20261016-nemotron35-bolt-router-fusion/router-fusion.patch
git apply --check --reverse --directory=.pi/router-fusion/candidate \
    results/20261016-nemotron35-bolt-router-fusion/router-fusion.patch
```

Added source lines are whitespace-clean; the original sort kernels and launcher are byte-identical. Scratch is ignored under `.pi/router-fusion/`. No SSH, build, model or GPU workload ran locally; no oneAPI compiler was on PATH, so device syntax checking was not run. **Remote build, exact native numerics/markers, model dispatch, API and performance remain pending.** Only this README and `router-fusion.patch` are deliverables.
