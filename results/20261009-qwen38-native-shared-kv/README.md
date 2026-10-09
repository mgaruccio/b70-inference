# Native shared-KV MTP4 revalidation — development only


## Outcome

**Observed 64K decode improvement; not production-qualified.** Fresh current-boost
A/B/B/A pooled medians are54.8002 →58.3290 tok/s (**+6.44%**,12 measurements per
arm across six repeated prompt clusters). The earlier same-build HumanEval+
regression flag remains unresolved; production is unchanged.

| Cell | Arm | Decode median (inclusive IQR), tok/s | Total latency median, s |
|---|---|---:|---:|
| A1 | Native auto |54.1711 (2.1882)|56.2262|
| B1 | Shared KV |57.6998 (3.8729)|56.0766|
| B2 | Shared KV |58.3348 (2.4393)|56.0782|
| A2 | Native auto |54.8446 (2.9880)|56.2052|

Both candidate-cell medians exceed both controls. Paired median decode gains
are+5.23%/+5.23% against A1 and+3.94%/+5.07% against A2. Native paired drift is
+0.075%; candidate paired drift+0.020%. These are descriptive results from one
A/B/B/A block, not cross-day robustness or a statistical quality-equivalence test.

Pooled total request latency changes56.2106 →56.0782s (**−0.24%**); TTFT remains
about53.88s. The decode improvement must not be advertised as a6.44% end-to-end
latency reduction. Pooled draft acceptance changes50.732% →51.329%; native
proposed2048/accepted1039 across512 draft steps, candidate2032/1043 across508.
Actual emitted tokens per step are3.0000 →3.0236. Changing output/acceptance
trajectories can contribute to serving gains; fixed-input operator latency is
reported separately below.

All24 measured requests and four warmups validated. Every cell passed3 canaries,
131 finite-boundary checks and8 functional checks. Both candidates proved
eligible interleaved-cache dispatch plus FULL graph capture/run, with no
unsupported-Q5 logs. KV capacity is226397 tokens in every cell; configured
context stays212992, while this fresh serving test reaches65536 prompt tokens.
All current host guards, source checks and owned-container cleanups passed.

Short output-ID identity is19/19 between A1/B1/B2,18/19 against A2; A1:A2 also
matches18/19 (`parity-code-2` differs). Long measured text identity is2/6 or3/6
for candidate/control pairs,2/6 for native A1:A2 and4/6 for candidate B1:B2.
Baseline nondeterminism does not establish candidate quality equivalence.
`comparison.json` retains every paired value, archive hash, contract,
acceptance count and output hash; `SHA256SUMS` covers all seven archives.

User requested continued native GEMM/attention optimization after the split-count
and row-padding screens. Discovery found an already implemented native candidate
on branch `pi/grouped-quality`, commit `470d02e2`, rather than a need to rewrite
attention. Reuse its unchanged build-06 and adapters; do not rerun the known-slow
Triton grouped prototype or promote a production launcher.

## Source and mechanism

Original campaign: `results/20260913-qwen38-native-grouped-verify/` at commit
`470d02e2`. Original native source pin `1796aa8bc8db4ac68d9cd19636cef88f3af81d2b`,
SYCL-TLA `cd763790ad2f74d7294435ecf77682bac0062c3a`, DPC++2026.0.0.
Build-06 library SHA256:
`e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec`.
Public-source parity with installed kernels0.1.12.3 is not proven; compare against
the actually installed native operator, not just a rebuilt control.

The candidate packs Q[5,24,256] into [1,120,256], grouping five positions and six
GQA heads per KV head. A dedicated `PackedVerify` native mask uses global packed
row/6 to retain `max(L-4+t,1)` causal bounds. This is not unchanged decode with
extra heads. It retains native FP8 scale handling, split32 and q8 DPAS tiles;
pack/unpack costs are included. q8 spans the30 packed rows in four tiles, so do
not imply that the implementation reduces five KV scans to one.

Fresh primary-source research consulted:
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp>
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp>
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/vllm_xpu_kernels/flash_attn_interface.py>
- <https://github.com/vllm-project/vllm-xpu-kernels/blob/e8b12aefae6b9df9b712799eef0ec0cd9ce7ac88/docs/group_split_kv_design.md>

Generic upstream warnings about unchanged decode lacking multirow causal masks
are addressed by this explicit patch. The actual installed helper and prior
build/test evidence support literal block size1664 and real interleaved KV
strides; a nominal16/32/64 policy list is not evidence that1664 is unsupported.
No full-prefill rerouting or new kernel design is needed for this test.

## Predeclared process

Fixed host: same boot `c93aea98-bb22-4cf6-879f-42d2980cb726`, boost0,
all16 CPU caps3801000 kHz, GPU275W, no competing containers. Image digest
`f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`, vLLM
ac7509e2b, torch2.13.0+xpu, kernels0.1.12.3. Model/target FP16 GPTQ-G128,
FP8 KV, native MTP4, context212992, batch8192, C1, graph sizes[1,2,4,8],
prefix caching/thinking off. Production launcher remains unchanged.

1. Verify current guards and candidate/source hashes. Reuse existing425KiB
   library; no compiler install/full rebuild (host has only10GB free).
2. Run unchanged build-06 `probe.py`: independent CPU FP32 paged causal reference,
   native comparison, tiny/page-boundary/8K/32K/64K lengths, pitched Q, actual
   interleaved KV layout, caller output, nonunit descales. Historical HND-layout
   qualification remains prior evidence, not a fresh test. Also test
   future-key isolation, same-address mutable graph metadata and fallback/error
   checks. Its existing rtol.02/atol1e-4 native comparison and separately labeled
   FP16-reference allowance are fixed before execution; this is not an exact
   output/quality-equivalence claim. Retain every error and raw timing.
3. If the operator qualification passes and a repeated benefit remains, execute
   fresh A1/B1/B2/A2 at64K through the existing real HTTP serving lifecycle. Each
   cell runs its existing canary/finite-boundary/functional suite, then one
   warmup and six measured65536-input/128-output requests, greedy seed42,
   ignored EOS and identical independently nonced prompt sets across cells.
4. Require candidate eligible-dispatch evidence and FULL graph capture/run, not
   a silent fallback. Preserve raw SSE, requests, outputs, metrics, acceptance
   deltas, all timings, source/launch/runtime/configuration and failure records.
5. Compare paired performance, acceptance and output variability against both
   controls. Retain configured and runtime-reported capacity. This is a
   **quality-sensitive development experiment**: baseline nondeterminism does
   not establish candidate quality equivalence. No standard/publishable or
   production qualification without the benchmarking standard's broader tests.
6. Serial GPU work only. Existing lifecycle cleanup plus current pre/post
   CPU/boot/power/launcher/kernel guards. No deployment or persistent inference
   change; the already installed boost-off service remains untouched.

Remote campaign root:
`/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-native-shared-kv`.
Exact per-cell commands and frozen dependencies will be retained with results.
Native Lab preview unavailable (`ENOENT`); authorized ordinary CLI used.


## Fresh operator result

`operator-01` exited0: all35 checks passed,46 eligible eager/capture calls,
no new guarded kernel errors, owned-container cleanup and postconditions passed.
The native operator timing control explicitly uses split32; the probe separately
checks automatic-vs32 outputs. Serving controls use unchanged automatic splitting.
Each timing is12 alternating-order rounds of16 graph replays, including Q packing,
device metadata preparation and output unpacking:

- 8197 keys: native0.206522ms, candidate0.174310ms; latency −15.60%.
- 32773 keys: native0.543304ms, candidate0.452324ms; latency −16.75%.
- 65541 keys: native0.931615ms, candidate0.767044ms; latency −17.67%.

This passes the existing operator gate; it is not a serving-throughput result
or a bitwise/quality-equivalence result. Raw per-round timings, comparison errors,
FP32 diagnostics, dispatch traces and current guards are in `operator-01.tar.gz`.

The eager candidate/native max absolute difference is zero except the1985/2049
boundary cases (and future-key mutation), where it is0.000030517578125. Strict
FP32 diagnostics fail at `eager/3`, `/4`, `/6` and `allocated_output` for **both**
native and candidate; their separately predeclared FP16-reference allowances
pass. These retained diagnostics must not be described as exact FP32 agreement.

Recovered source/build inputs, binary, source patch, build metadata, original
native source archive, serving/diagnostic dependencies and content hashes are in
`reused-candidate.tar.gz`. The30MB SYCL-TLA source archive is not duplicated;
its pinned commit and archive SHA256 are recorded inside the bundle. Archived
historical README prose is not current qualification (early q16/HND text is stale).
Native source licensing is in the bundle; the matching SYCL-TLA BSD-3-Clause
notice is retained alongside it as `SYCL-TLA-LICENSE.txt`.

Archive SHA256:
- `operator-01.tar.gz`: `5dd114b94d1163495f0785e2286be8d6563ac516873f60311f8f4edd88c8bb54`
- `reused-candidate.tar.gz`: `f72301b2e55166dc213f2e18501e79f17e0501f8330ec36bc1e33ec0de75a8f5`


## Existing quality flag — unchanged by a speed result

Quality commit `bac718884c7af20ae943e3c12534e911e07b2876` tested this exact
build-06 library and f01 image in `results/20260913-qwen38-grouped-quality/`.
HumanEval+ passed140/164 target-only,139/164 native MTP4,136/164 candidate.
`HumanEval/1`, `/19` and `/130` regressed, with no offsetting passes. The reported
paired normal95% interval for candidate−native was−3.89 to+0.23 percentage points;
one sample per task/arm and no prespecified noninferiority margin do **not** prove
quality neutrality or isolate the kernel from runtime nondeterminism.

Other paired deltas: IFEval−0.18pp, GSM8K+0.08pp, MBPP+0.26pp. Those do not cancel
the HumanEval+ flag. The prior verdict was **do not promote on throughput alone**.
This current-boost performance remeasurement is permitted as development work,
not a clearance of that flag. Its source README and analysis/comparison JSON are
preserved in `prior-quality.tar.gz` with the original commit attribution.


## Commands and verification

Executed on `inference-host` through SSH with BatchMode,12-second connect timeout,
30-second keepalives and6 missed-keepalive tolerance. The operator precedes the
serial A/B/B/A cells; use fresh output directory names when rerunning:

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-native-shared-kv
python3 -B "$R/run-cell.py" --arm operator --out "$R/operator-01"
python3 -B "$R/run-cell.py" --arm baseline --out "$R/a1"
python3 -B "$R/run-cell.py" --arm candidate --out "$R/b1"
python3 -B "$R/run-cell.py" --arm candidate --out "$R/b2"
python3 -B "$R/run-cell.py" --arm baseline --out "$R/a2"
```

The underlying exact Docker/serving arguments, image/runtime captures, requests,
metrics, source hashes, cleanup and guard records are inside each archive.
Operator runtime is bounded to45minutes; serving startup to1800seconds and the
long-context benchmark to3600seconds, using the existing lifecycle cleanup.

Offline reproduction from this result directory (stdlib only):

```bash
sha256sum -c SHA256SUMS
python3 -B compare-cells.py a1.tar.gz b1.tar.gz b2.tar.gz a2.tar.gz > comparison.json
```

Supplementary local checks: wrapper CLI/AST, analyzer AST, synthetic delta and
inclusive-IQR calculations, rejection of host/contract/capacity/prompt drift,
and loading/self-comparing the actual A1 archive. These supplement, not replace,
the real operator and public HTTP executions above. No unrelated repository
changes, package installations or production launcher changes are included.
