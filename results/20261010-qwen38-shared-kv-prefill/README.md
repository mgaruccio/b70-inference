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
kernels for 32.87%. These leading rows were recovered from a truncated SSH
analysis response; the complete raw trace remains on the inference host.
Future kernel work should retain the shared-KV decode dispatch, model precision
and context. This experiment establishes a bottleneck, not a kernel speedup.

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
raw requests/metrics and prompts. Repeated SSH/SFTP bulk retrieval failed; the
bundle is **not** claimed downloaded or independently archive-verified locally.
`profile-analysis-partial.txt` retains the truncated remote analysis plus its
SSH error. Thirteen complete leading rows account for 95.95% of reported GPU
work; the full-kernel-list tail was not recovered. The totals above came from
the complete trace analysis before its output transfer failed.

Local stdout files retain the completed run summaries; `profile-host-check.txt`
retains the trace-file listing and cleanup readback. Server-side per-request
prefill metrics are saved remotely but were not locally reviewed, so the
performance comparison uses client TTFT only. Neither max-context requests nor
quality suites were rerun; configured context is preserved, not freshly
capacity-qualified. This is not a standard-publishable result.

Local checks passed: controller Python syntax/help/invalid-combination guards;
analyzer GPU-only filtering, aggregation, interval overlap, percentage math,
and rejection of empty/negative/zero-duration traces. Real HTTP execution and
cleanup passed for baseline, 12K candidate and the isolated profile. The
persistent launcher remains unchanged.
