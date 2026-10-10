# Shared-KV cold-prefill screen — 2026-10-10

Development-tier experiment, not standard-publishable and not a production promotion.
Starting point is the existing build-06 shared-KV performance leader, **not native
MTP4**. Its unresolved quality caveat remains (earlier HumanEval+ 136/164 vs
139/164 native). No weights, quantization or acceptance policy are changed.

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

## Observed status: baseline complete, candidate blocked

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

The 16384-budget controller exited 1 with `Server exited before health; see
server.log`. No candidate throughput was measured. Do not interpret this as
an OOM, a shape-contract failure, or a slowdown without the actual server log.

Artifact retrieval was blocked by intermittent SSH transport failures. A
bounded verbose check authenticated successfully but then timed out opening
the session; Tailscale ping still answered. A separate jump-host attempt was
policy-denied and was not pursued. No network or SSH configuration was changed.
The stalled artifact-copy task was stopped. The host controller has a `finally`
cleanup path, but the failed candidate's final container/power state has **not**
been independently read back. Inspect it before any further launch.

Available local logs: `baseline-stdout.txt`, `candidate-stdout.txt`. Full evidence
remains under the host campaign directory above. No persistent inference
launcher was changed. No speedup, quality-neutrality or promotion claim is made.

### Next decision, once host access is reliable

1. Retrieve candidate `server.log`, `status.json`, `cleanup.json`; confirm idle
   container state, original 275 W and persistent launcher hash.
2. Retrieve the baseline raw results and inspect per-request server metric count
   deltas before using the prefill-stage timing data.
3. Diagnose the 16K startup failure before another experiment. Do not reduce
   context or relax the shared-KV shape checks silently to get a passing cell.
4. If compatible, finish the larger-budget comparison; otherwise record that
   tradeoff and keep 8192. Flash-vs-Triton and large-M GPTQ GEMM profiling remain
   untested options, not demonstrated improvements. Cache reuse is a separate
   practical latency optimization and must not be counted as cold-prefill speed.
