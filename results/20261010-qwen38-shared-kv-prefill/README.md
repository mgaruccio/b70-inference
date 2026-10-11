# Shared-KV cold-prefill screen — 2026-10-10

Development-tier experiment, not standard-publishable and not a production promotion.
Starting point is the existing build-06 shared-KV performance leader, **not native
MTP4**. Its unresolved quality caveat remains (earlier HumanEval+ 136/164 vs
139/164 native). No weights, quantization or acceptance policy are changed.


## Decision

**Keep the 8192/.95 shared-KV recipe.** The 16K budget failed the context-memory
check; the memory-bounded 12K bundle passed but did not improve TTFT (roughly
0–0.3% slower). No configuration was promoted.

The isolated 32K profile points to **large-M GEMM first, prefill attention
second**: two GEMM kernels account for 55.94% of GPU work and three full-attention
kernels for 32.87%. Initially recovered from truncated SSH output, these totals
are now reproduced locally from the checksum-verified complete raw trace.
Future kernel work should retain the shared-KV decode dispatch, model precision
and context. This experiment establishes a bottleneck, not a kernel speedup.
Continuation implemented and tested fused INT4 Triton and bounded transient
FP16-tile GEMMs. Both passed numerical checks but were slower on the tested
large-M projections; neither entered serving. **Next target: prefill-only
attention**, with shared-KV decode preserved. Full results and failures are below.

## Research and hypothesis

Fresh official/upstream sources consulted on 2026-10-10:

- <https://docs.vllm.ai/en/latest/configuration/optimization/>: larger
  `max-num-batched-tokens` can improve TTFT/prefill utilization. First screen:
  8192 baseline vs 16384; keep chunking enabled and isolate C1 requests.
- <https://github.com/vllm-project/vllm/issues/53928>: reported XPU Qwen GPTQ/MTP
  failure with mixed speculative/non-speculative chunked-prefill work. No
  concurrent decode+prefill testing or chunking removal in this screen.
- <https://docs.vllm.ai/en/stable/getting_started/quickstart/> and
  <https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/platforms/xpu.py>:
  Flash/Triton are XPU attention options, but compatibility with the shared-KV
  seam must be checked; this is a possible later experiment, not a measured gain.
- <https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/model_executor/kernels/linear/mixed_precision/xpu.py>:
  GPTQ uses the native XPU W4A16 GEMM path, not CUDA Marlin. A kernel change needs
  prefill-specific profiling; prior small-M/decode profiles cannot justify it.
- <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>: prefix
  reuse can reduce prefill, but is a distinct workload from cold-prefill speed.

## Fixed environment and real end-to-end process

- `inference-host`, one 32 GB B70; CPU boost off / 3801000 kHz caps.
- Pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- Qwen3.8-27B GPTQ symmetric G128, FP16 target, FP8 KV, MTP4 S+M1,
  mixed-GDN-v5, shared-KV build-06, XPU graphs, thinking on.
- Context 212992, utilization .95, C1, prefix cache on / mamba align.
- Temporary 230 W cap in both arms; original 275 W restored after each arm.
- Persistent launcher untouched. Experimental listener restricted to loopback.

`run-prefill.py` reuses the prior private shared-KV launcher, established host
checks, exact-token prompt generator and real streaming chat benchmark. Only
the server token budget differs. The client `--budget` is metadata, **not** the
server setting. All inference, dependencies and benchmark execution stay on the
remote inference host, outside Pi's environment.

1. Verify idle GPU/container state, boot/CPU/power, model-build and launcher
   hashes, and kernel health. Launch a disposable server and wait for `/health`.
2. Check the shared-KV eligible-dispatch/library log. Reject any unsupported
   five-row shape rather than counting native fallback as a candidate result.
3. Generate six independently nonced prompts per exact rendered input length:
   8192, 32768, 65536. Reuse the same prompt dataset across fresh-server arms.
4. Exercise `/v1/chat/completions` with greedy sampling, ignored EOS, one output
   token. Generic plus shape warmups are discarded; five measurements per point.
5. Also run the existing 512-input/128-output cell as a decode-path check, and
   require runtime FULL-graph execution evidence from the established adapter.
6. Require exact endpoint token counts and zero prefix-cache hit deltas. Preserve
   SSE/output text, all samples/failures, metric snapshots, commands and logs.
7. Compare median/IQR TTFT and server `request_prefill_time_seconds` deltas when
   available and attributable to exactly one request (count delta = 1).
   Input tokens/TTFT is explicitly a client-side proxy, not pure GPU kernel time.
   Server prefill time is the runtime's request-stage metric, not a GPU profile.
8. Stop only the owned container, restore original power and idle state, recheck
   persistent launcher and kernel health. Preserve failed cells without retries
   disguised as successful measurements.

This preserves the leader's cache-on recipe with measured cold misses; it does
not claim the standard's cache-disabled/full-suite protocol. This reduced C1
screen cannot establish concurrent-serving safety, maximum-context performance,
quality neutrality, or a production-ready improvement.

## Commands and artifacts

On `inference-host`:

```sh
cd /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261010-qwen38-shared-kv-prefill
python3 -u run-prefill.py --budget 8192
python3 -u run-prefill.py --budget 16384
```

Each `batch-*` directory retains `launcher.sh`, `controller.py`, `commands.json`,
`container.json`, `server.log`, `dispatch.json`, `summary.json`, per-cell
`results.json`/SSE, `status.json`, host/kernel checks and cleanup evidence.
The prompts are stored once at the campaign root on the inference host.

## Observed baseline and initial 16K failure

The baseline controller completed with exit 0. Its streamed summaries report:

| Cold input tokens | Median client TTFT (s) | Input tokens / TTFT (tok/s) |
|---:|---:|---:|
| 8192 | 4.690 | 1746.6 |
| 32768 | 23.475 | 1395.9 |
| 65536 | 59.409 | 1103.1 |

Five measured requests per point, one output token, zero reported cache-hit
deltas. The 512→128 decode-path cell reported 115.40 tok/s median. These are
client timing/proxy results, **not isolated GPU prefill throughput**. The
controller's successful exit includes its execution and restoration checks;
the underlying server-side metrics and raw artifacts still need retrieval.

The 16384-budget controller exited before health. Subsequent recovery established
that this was the runtime's **KV-capacity admission check**, not a measured
slowdown: 7.88 GiB required for 212992 tokens versus 7.52 GiB available (estimated
maximum context 201344). No GPU kernel errors were reported. Remote readback
confirmed no running containers and restoration to 275 W.

Artifact retrieval was initially blocked by intermittent SSH transport failures.
A per-command `ProxyCommand=tailscale nc %h %p`, compression and standard
Curve25519/Ed25519/AES-GCM options allowed subsequent diagnostic reads; no network
or SSH configuration was changed. The earlier jump-host attempt was policy-denied
and was not pursued. Initial stdout is retained in `baseline-stdout.txt` and
`candidate-stdout.txt`; full server evidence is being recovered separately.

### Memory-bounded 12K candidate

Rather than reduce context, test a **configuration bundle**:

- batch budget 8192 → 12288;
- utilization .95 → .965 (reduced memory headroom);
- explicitly retain the leader's 152 KV blocks with
  `--num-gpu-blocks-override 152`.

The pinned runtime's `kv_cache_utils.py:2162-2182` supports this override but can
allocate beyond profiling if misused. The controller therefore requires its
logged pre-override profiled block count to be **at least 152**, in addition to
unchanged shared-KV shapes and runtime execution. The original .95 baseline
reported 8.23 GiB available KV, 2.86 GiB peak activation, 0.15 GiB graph memory,
and 29.41/30.30 GiB initially free; the failed 16K budget left only 7.52 GiB KV.
These observations motivate a smaller increase, not forcing the failed 16K
allocation or claiming memory headroom is free.

Run the same cold prompts, five measurements, server metrics, decode-path check
and restoration process defined above:

```sh
python3 -u run-prefill.py --budget 12288 --tuned-memory
```

The `batch-12288-util0965-blocks152` cell completed with exit 0, preserved
shared-KV, and recorded no new kernel errors. Independent readback confirmed
an idle host and 275 W restored.

| Input | 8K-budget median TTFT | 12K-budget median TTFT | Input/TTFT change |
|---:|---:|---:|---:|
| 8192 | 4.6903 s | 4.6914 s | −0.023% |
| 32768 | 23.4747 s | 23.5510 s | −0.324% |
| 65536 | 59.4095 s | 59.5910 s | −0.305% |

Five measurements per point, same prompts, one output token, no cache hits.
Short decode was effectively unchanged (115.40 → 115.58 tok/s; same aggregate
548 proposed / 509 accepted MTP tokens). **Reject the 12K bundle:** no useful
prefill gain and less memory headroom. These tiny sequential-run differences
are not a claim of statistically established regression. No quality-neutrality
or production-promotion claim. Cache reuse remains a different workload.


### Other options screened against the actual stack

- Upstream [default chunk-prefill policies](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/attn/kernel_configs/chunk_prefill_default.conf)
  include `256,true,true,false,false,false`, the relevant causal paged shape
  for this model's **256-wide** full-attention heads. Generic Qwen 128-head
  coverage is not evidence for this model. The installed wheel is 0.1.12.3,
  not byte-identical to the v0.1.12 source tag; inspect actual dispatch/fallback
  logs before proposing a rebuild. No missing-policy evidence yet.
- The full `TRITON_ATTN` backend changes decode too, bypassing the shared-KV
  FlashAttention seam. It is not a drop-in optimization of the requested leader.
  A prefill-only dispatcher would require a separate implementation and FP8-KV
  correctness checks, not merely a backend environment variable.
- [XPU W4A16 implementation](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h)
  uses its own primitive cache (capacity 512), so
  `ONEDNN_PRIMITIVE_CACHE_CAPACITY` is not a demonstrated tuning lever here.
  Large-M GEMM work requires a prefill-specific profile; verbose/profiling modes
  must stay out of latency comparisons.
- `VLLM_XPU_SPEC_DECODE_MAX_QLEN=0/1` disables a small-query speculative fast
  path. It targets decode, not these long cold prefills, and changes the path
  the user explicitly asked to retain; do not pursue it as a prefill fix.


### Diagnostic prefill profile

Following the [unitrace session-control documentation](https://raw.githubusercontent.com/intel/pti-gpu/master/tools/unitrace/README.md),
reuse the already installed tracer, without rebuilding or installing anything:

```sh
python3 -u run-prefill.py --budget 8192 --profile-prefill
```

`profile-32k` retains the unchanged 8192/.95 shared-KV baseline. Start unitrace
paused; warm up/verify 512→128 shared-KV decode; warm the 32K prefill shape with
tracing paused. Resume only around a second, unique 32768→1 streamed HTTP
request, require zero prefix-cache hits, then pause and shut down the owned
server to flush traces. Save the request/SSE, metric snapshots, launcher,
control commands, trace and host-restoration checks. Interpret kernel duration
shares as diagnostic work accounting, not wall-time speedups; profiled TTFT
must not enter the uninstrumented comparison above.


Observed: profile controller exit 0, exact 32768→1, zero cache hits, profiled
TTFT 23.43755 s. Runtime/host checks passed, no new guarded kernel errors, and
independent readback confirmed no running containers and 275 W. The worker
flushed `profile-32k/unitrace/python3.269.json` (11,217,159 bytes); empty tracer
files for helper processes are not performance evidence.

`analyze-prefill.py` counts only `gpu_op` / `ph=X` events. The remote analysis
computed 10,158 GPU events, summed GPU work **23.29101 s**, interval union
**23.29080 s**, and span **23.36965 s**. Leading work:

- `gemm_kernel[SIMD16 {128; 1; 1} {64; 2; 1}]`: 42.54%, 1170 calls.
- `gemm_kernel[SIMD16 {32; 1; 1} {128; 4; 1}]`: 13.40%, 396 calls.
- Three `XeFMHAFwdKernel` variants: 21.87% + 5.59% + 5.41%.
- GDN's leading kernels are much smaller individually (about 1.78%, 1.77%,
  0.94%, 0.46%); a decode-only GDN fusion is not the first prefill lever.

The nearly full GPU coverage is consistent with kernel work, rather than large
scheduler/launch gaps, dominating this 32K request; it does not rule out better
tiling or utilization. Kernel names establish primitive families, **not exact
projection or layer ownership**. Before implementing a large-M GEMM change, attribute the
actual M/N/K/group-size shapes and compare the current oneDNN selection with a
bounded alternative; do not change quantization or cache dequantized weights
at the expense of context. A prefill-only attention experiment is the second
option, with native/shared-KV dispatch preserved for small verification queries
and explicit FP8-KV numerical checks. Neither is claimed implemented or faster.

### Evidence availability and verification limits

The full remote bundle is:

```text
/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261010-qwen38-shared-kv-prefill/raw-results.tar.gz
size: 2434891 bytes
sha256: 24bc3edc82c35f446d8bb2cdbe1b28e9731d49a16337f143015741ef86778976
```

It contains both successful timing arms, the failed 16K arm, profile trace,
raw requests/metrics and prompts. On continuation, a bounded SCP retrieval
succeeded. `raw-results.tar.gz` is now retained locally and its exact size and
SHA256 above were verified before reading the archive. Earlier transfer failures
remain in `profile-analysis-partial.txt`; they are not model/runtime failures.

Local complete-trace analysis in `profile-analysis-complete.json` reproduces all
10,158 events and the reported totals, with 202 kernel-name groups. The two
leading GEMMs account for 55.94% of summed work, not an achievable speedup.
The trace records launch IDs but not GEMM M/N/K or projection ownership.

`server-prefill-metrics.json` retains all five per-request histogram sum deltas
per timing point, each with count delta exactly one and zero cache hits. Median
server prefill seconds (baseline → 12K bundle): 8K **4.670556 → 4.670946**,
32K **23.409536 → 23.456149**, 64K **59.281412 → 59.452943**. These independently
confirm no batching benefit. Server request timing is not isolated GPU throughput.

Local stdout files retain the completed run summaries; `profile-host-check.txt`
retains the trace-file listing and cleanup readback. Neither max-context requests
nor quality suites were rerun; configured context is preserved, not freshly
capacity-qualified. This is not a standard-publishable result.

Local checks passed: controller Python syntax/help/invalid-combination guards;
analyzer GPU-only filtering, aggregation, interval overlap, percentage math,
and rejection of empty/negative/zero-duration traces. Real HTTP execution and
cleanup passed for baseline, 12K candidate and the isolated profile. The
persistent launcher remains unchanged.


## Continuation: prefill GEMM attribution and bounded prototype

Still development-tier; the fixed baseline remains build-06 shared-KV,
8192/.95, FP16/GPTQ symmetric G128, FP8 KV, C1, 212992 context at 230 W.
Fresh primary research: [oneDNN verbose controls](https://github.com/uxlfoundation/oneDNN/blob/master/doc/performance/verbose.md),
[the pinned W4A16 wrapper](https://github.com/vllm-project/vllm-xpu-kernels/blob/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h),
and [the pinned oneDNN decompression implementation](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit/include/gemmstone/problem.hpp).
The current op already uses grouped FP16 weight decompression. There is no
established runtime large-M selector. A possible alternative is a fused Triton
large-M kernel with register-tiled dequantization and no persistent dense weights;
its numerical behavior and performance must be measured, not assumed.

Installed-package inspection is in `installed-runtime-inspection.txt`: torch
2.13.0+xpu reports oneDNN 3.12.0 / `80afa710...`, kernels 0.1.12.3. The command
failed its final operator lookup because importing `vllm_xpu_kernels` alone does
not register `_xpu_C`; the platform explicitly imports
`vllm_xpu_kernels._xpu_C`. Host source checkout versions do not prove wheel identity.
No package upgrades or global git/SSH configuration changes were made.

The first `--gemm-diagnostic` attempt used `ONEDNN_VERBOSE=profile_exec`.
It failed **before health**, during graph capture: `RuntimeError: wait cannot be
called for a queue which is recording to a command graph.` This is a diagnostic
mode incompatibility, not a model slowdown. `gemm-diagnostic-stdout.txt` and
`gemm-diagnostic-failure.txt` retain the failure and cleanup readback: restored
275 W, no running container, unchanged launcher hash, no new guarded kernel errors.
The failed directory is preserved remotely as `gemm-diagnostic-32k`.

The bounded retry uses **creation-only** logging (`ONEDNN_VERBOSE=profile_create`)
in a separate `gemm-create-diagnostic-32k` directory, without disabling graphs:

```sh
python3 -u run-prefill-gemm.py --budget 8192 --gemm-diagnostic
```

Its predefined real-system test process reuses the existing idle/CPU/boot/hash/
kernel guards, disposable loopback server and power cleanup. Run the existing
512→128 HTTP decode check (shared-KV eligible dispatch and runtime FULL graphs),
then one unique 32768→1 shape warmup and a second unique 32768→1 diagnostic
request through `/v1/chat/completions`. Require exact token counts and zero cache
hits, retain SSE/metrics/server log/commands/status and verify cleanup.
Creation logs provide primitive shapes/selection, not measured per-kernel work
shares or uninstrumented latency. This diagnostic does not establish a speedup.


Creation-only retry **passed**, exit 0. Exact 32768→1, zero cache hits, runtime
shared-KV/FULL-graph checks and host cleanup passed. Diagnostic TTFT 23.404948 s
is excluded from speed comparisons. Independent readback showed no running
containers and 275 W. `gemm-create-diagnostic-stdout.txt` and
`gemm-create-host-check.txt` retain the observations. Both failed and successful
raw runs are in `gemm-diagnostic-results.tar.gz` (SHA256
`2758532e148183d7ee5d6e9f5e5d9bc5c874ac90ff0aaaf6e6fca1192af96365`,
verified locally). Actual GPU oneDNN reports 3.12.0 / `80afa710...`, SYCL/Level
Zero B70, driver 1.15.39122, and `jit:gemm:any` with grouped FP16 scales, G128,
u4 weights, scalar s8 zero point and FP16 math.

`gemm-shapes.json` records all 49 quantized primitive creation rows. Distinct
large projection `(K,N)` pairs are `(5120,16384)`, `(6144,5120)`,
`(5120,34816)`, `(17408,5120)`, `(5120,14336)`, `(10240,5120)`.
Large created M values include **8192, 6656, 3328 and 2816**; a candidate tested
only at M=8192 would not cover this run. Creation rows are not executed-call
counts, and do not map the two unitrace GEMM names to exact layers.


### Fused INT4 prototype screen

`prefill_int4.py` implements three bounded static tiles with register-only INT4
unpacking, FP16 dequantized operands/output and FP32 accumulation. No dense
weight cache is retained. Its serving-shaped dispatcher has the installed ABI
`x, qweight, bias, scales, qzeros, group_size, g_idx`, and preserves the native
op for M≤32. It is **not wired into vLLM serving** and is not ready for promotion.
The direct microbenchmark deliberately tests even M32 through the candidate;
that is not evidence the serving decode path changed.

`bench-prefill-int4.py` reads real safetensors one projection at a time, retains
actual keys/layouts, and uses the installed symmetric/desc_act=False preparation
(scalar zero point 8, g_idx=None, packed weight transpose). Stored packed qzeros
and stored g_idx do not imply asymmetric or act-order execution in this recipe.
It compares M8192/M6656/M32, three tile variants, two discarded warmups, and five
interleaved synchronized full-call timings, including Python/validation/launch
cost. FP32-reference columns, full-output error/finite/allclose/cosine metrics and
allocated/peak memory are retained. These unmerged on-disk projections are an
initial screen, not all the fused runtime shapes above or serving throughput.

Predeclared supplemental process: guard idle/CPU/boot/hashes/kernel state; run
`python3 -u run-gemm-microbench.py` in a disposable no-network container with the
same pinned image/model/230 W; preserve report, logs, exact command and source;
stop only owned container ID, restore 275 W and verify no new kernel errors or
launcher changes. A viable kernel would still require uninstrumented HTTP cold
prefill A/B at 8K/32K/64K, exact/cache-hit checks and shared-KV/FULL-graph decode
checks, plus output checks. No direct-op speedup may be called a serving gain.

First screen `gemm-screen-01` executed the native baseline on four real
projections (12 M/shape cases), but **every candidate failed compilation**:
Triton 3.7.2 rejects plain Python `PACKING` inside a JIT kernel. No candidate
output or numerical/performance result was observed. The raw archive is
`gemm-screen-01.tar.gz`, SHA256
`2d5378fdb30cf21d21bd56b812e7e25f7c3fe2d03aa0fb172fb77cc3a4d445c6`
(verified locally); controller reports benchmark returncode 1, restored 275 W,
no new guarded kernel errors. Review also caught per-call construction of the
nested JIT function, which would defeat a stable warmed launch cache.
Both issues were corrected: module-scoped stable JIT function and explicit
PACKING/GROUP_SIZE constexpr parameters, with no permissive environment override.
The corrected screen uses a fresh `gemm-screen-02` directory.


### Results and decision from the corrected GPU screens

**Reject both approaches; retain the original shared-KV/oneDNN recipe.**

- `gemm-screen-02`: all 12 shape/M cases passed, all three fused tiles compiled
  and produced finite outputs within `atol=rtol=.01`. The best fused tile was
  **6.4–7.6× the native full-call median** across the eight large-M cases.
  `gemm-screen-02-summary.json` retains each arm and its numerical comparison.
  Raw archive SHA256:
  `bbe3034be489a5523c24a0aed1b4d72585b77534491b84e8527d7babef04c67c`.
- `gemm-screen-03`: opt-in `--configs tiled-fp16`, with at most **64 MiB of
  transient FP16 weights** per tile, `torch.mm` and no persistent dense cache.
  All 12 cases passed the same numerical tolerance. K-major dense tiles were
  **6–23% higher in full-call median** across large-M cases, not faster.
  Raw archive SHA256:
  `196ae59dbb8469717c82bcb85411f0d4c6c5489af8b4928ed499a01170be9bf8`.
- `gemm-screen-04`: match the native weight's **N-major physical orientation**,
  passing the FP16 tile's transpose to `torch.mm`; same 64 MiB weight cap and
  interleaved test. All 12 cases passed, but large-M medians were still
  **4–30% higher**. Example M8192 gate projection: native **9.6485 ms** vs tiled
  **10.0556 ms**; M6656 down projection: **7.5473 ms** vs **9.7792 ms**.
  Raw archive SHA256:
  `52909c1991237190be20fcfb21e72511c11fe3e51d1c98410f7476795e01ec03`.

All archive checksums were verified locally. Per-run `*-summary.json`,
`*-stdout.txt` and `*-host-check.txt` preserve the compact findings; archives
include actual commands, source snapshots, full timing samples, numerical
reports, hardware checks and failures. Exact screen command:
`python3 -u run-gemm-microbench.py`; the committed controller is the final
one-shot screen-04 version (it refuses an existing directory), with screen-01/
02/03 controllers retained inside their archives. Each screen used the unchanged
pinned image and 230 W, then passed idle/275 W/launcher/kernel restoration checks.

These are **direct-op development results**, not an end-to-end serving regression
or a statistically established small-delta result. The 64 MiB cap applies only
to the dequantized weight tile, not total workspace. Reported memory peaks
include large full-output numerical-check temporaries; do not use them as
serving activation peaks or context-capacity qualification.

Numerical tolerance is not proof of model-output identity. N-major tiled M8192
outputs were bit-identical in these samples; M6656 and M32 were not (maximum
absolute difference .015625). No fidelity/quality-neutrality claim is made.
Neither rejected prototype was integrated into serving, so candidate HTTP A/B,
new logits/generation checks, max-context and quality suites were not run.
The original baseline HTTP journey and shared-KV/FULL-graph verification passed
in the creation-only diagnostic, with exact 32K tokens and zero cache hits.
There is **no new cold-prefill speedup and no promotion** from this continuation.

Next concrete implementation target is **prefill-only attention**, preserving
the shared-KV small-query/verification dispatch. The recovered full-attention
work share (~32.87%) motivates that investigation; it is not an achievable
wall-time gain. A global Triton attention switch remains inappropriate. This
screen rejects the tested GEMM implementations, not every possible GEMM change.
