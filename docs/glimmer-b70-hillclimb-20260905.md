# Glimmer B70: over 300 aggregate tok/s — 2026-09-05

## Retained change

`scripts/start-muse-vllm-concurrent.sh` now uses **DFlash K3 instead of K4**.
Everything else is unchanged: eight active sequences, native 131072 context,
GPTQ target and GPTQ assistant, float16 activations, FP8 KV, XPU graphs,
`DFLASH_KV_MODE=none`, 2048 batch-token budget, memory utilization 0.90,
prefix caching disabled, localhost port 18080. No new kernels, weight changes,
DFlash2 migration, driver changes, or power tuning were needed.

Host: `inference-host`, Arc Pro B70 (`8086:e223`), Linux `7.2.0-1-cachyos`.
Pinned image digest: `f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`;
vLLM `0.27.2rc1.dev77+gac7509e2b.xpu`, `vllm-xpu-kernels==0.1.13.2`.
The GPU was idle initially and was freed after testing. Original C1 and Qwen
launchers were not changed; no service was installed or left running.

## Matched short-input screen

Existing sky prompt, eight simultaneous HTTP SSE clients, temperature 1,
top_p .95, top_k 64, seed 42, 256 output tokens per request. One shape-matched
warmup, then three measured waves per freshly started cell. Context remained
131072 throughout; this is not a larger-client-count or reduced-context result.
Aggregate e2e throughput = total completion tokens / concurrent wave wall time,
including prefill, reasoning, content, and queueing. Not per-stream decode.

| Draft depth | Median aggregate tok/s | Three-wave range | Mean acceptance length* |
|---|---:|---:|---:|
| K4 baseline, reproduced this session | 282.245 | 281.306–282.455 | 2.158 |
| K2 | 296.943 | 296.866–296.949 | 1.989 |
| **K3** | **302.642** | **301.516–302.834** | **2.147** |
| K6 | 273.681 | 272.846–274.342 | 2.282 |

All measured requests had no errors, 256 completion tokens, and visible reasoning
or content. These capped probes are **not completed-answer quality scores**.
*Acceptance length = 1 + accepted draft tokens / draft steps, from before/after
Prometheus counter deltas. K3 preserves almost K4's acceptance length while
proposing/verifying fewer tokens. This explains the tuning direction; it is not
a kernel-level profile. Observed median card power was ~274 W for K3/K4/K6,
~258 W for K2, with 2800 MHz median reported frequency.

## Independent restart and end-to-end validation

The edited repository launcher was copied to the host and used for a fresh K3
start, not a separate implementation. Five measured waves before and after
long-context stress, following one initial eight-client warmup:

- Before stress: **304.539** median aggregate tok/s (303.681–305.260).
- After stress: **303.118** median aggregate tok/s (303.064–303.345).
- All 80 replay requests returned 256 tokens with visible text and no errors.
- Conservative post-stress improvement: **7.4%** over the same-session 282.245
  baseline; **9.0%** over the earlier 278.104 native-context result.
- Existing GSM8K Janet / HumanEval-0 / MT-Bench Hawaii smoke checks: **8/8 before
  and 8/8 after stress**, with natural stops and the existing answer/code/content
  checks. Baseline K4 also passed 8/8. This is not a broad quality-equivalence claim.
- Twelve staggered 2042-token requests, 64–768 output tokens: peak eight active,
  queue observed, replacement requests streamed before the initial eight finished,
  clean drain, zero preemptions; 27.23 seconds.
- **128994 prompt + 282 output tokens**: beginning/middle/end checkpoint retrieval
  correct, natural stop, zero preemptions; 94.75 seconds.
- **Eight × 65532 prompt + 2048 output tokens**: all outputs visible and complete,
  peak eight active, 84.17% KV usage, zero preemptions, clean drain; 416.69 seconds.

The earlier K4 forced-length six-request near-capacity empty-response failure
was **not retested or fixed**. Nor were six simultaneous near-native requests
revalidated for K3. Keep the [prior context caveats](glimmer-b70-concurrency-20260905.md):
131072 is a per-request limit, not eight reserved full-native slots. Long-context
retrieval and answer smoke tests do not prove general reasoning quality.

## Executed test process and evidence

All requests used the real service at `http://127.0.0.1:18080`, via
`/v1/chat/completions`, `/tokenize`, `/v1/models`, and `/metrics`.
Local-state evidence is on `inference-host` under:

`~/b70-evals/muse-glimmer/20260906-hillclimb/`

(The directory name contains 20260906; the actual run date was September 5.)
It contains `baseline.sh`, `cell.sh`, per-cell generated launchers, JSON timings,
metrics before/after, server logs, `winner-launch.sh`, `verify-from-client.sh`,
completed answers, and raw long-context SSE responses. No raw logs are committed.

Executed host commands (R is that evidence directory, OLD is the existing
`~/b70-evals/muse-glimmer/20260905-concurrency` client directory):

```bash
bash "$R/cell.sh" 8 2 8
bash "$R/cell.sh" 8 3 8
bash "$R/cell.sh" 8 6 8
# Independent restart used the exact edited repository launcher:
docker rm -f muse-b70-hillclimb  # stop this experiment's K6 cell first
NAME=muse-b70-hillclimb \
  PATCH="$HOME/inference/launchers/b70/patch-vllm-dflash-gptq-context-kv.py" \
  bash "$R/winner-launch.sh"
bash "$HOME/inference/launchers/b70/wait-vllm-health.sh" 420 18080
# One --reps 1 warmup first; --reps 5 before and after the stress tests:
python3 "$OLD/instrument.py" --base http://127.0.0.1:18080/v1 \
  --model muse-glimmer-gptq --concurrency 8 --reps 5 --max-tokens 256 \
  --label winner-after-stress --out "$R/winner-after-stress.json" --log ''
python3 "$OLD/quality.py" 8 "$R/winner-after-stress-quality.json"
python3 "$R/context-probe.py" queue 2048 12 winner-queue12
python3 "$R/context-probe.py" retrieval 129000 1 winner-native-retrieval
python3 "$R/context-probe.py" capacity 65536 8 winner-c8-64k-resident 2048
# Executed after all checks and saving the server log:
docker rm -f muse-b70-hillclimb
```

`context-probe.py` is the existing client with only its output directory changed.
The saved verification script contains the exact execution order. Cleanup removed
only the experiment's container; `docker ps` was empty afterward.

## Fresh research informing the approach

- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/):
  gains depend on workload, hardware, and sampling; evaluate aggregate throughput
  rather than assuming single-stream latency or draft acceptance predicts it.
- [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/):
  batch-token budget trades throughput against latency. Left unchanged to isolate
  draft depth on this short-input, decode-heavy workload.
- [Dynamic speculation](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/):
  batch-dependent depth is an option for mixed loads; not needed for this bounded win.
- [DFlash2 integration](https://github.com/vllm-project/vllm/pull/52816) and
  [B70 dtype issue](https://github.com/vllm-project/vllm/issues/55250): DFlash2 is a
  distinct runner/model compatibility experiment, with a reported FP16 acceptance
  failure. Not attempted; do not treat upstream NVIDIA results as B70 evidence.
- [Quantized drafter issue](https://github.com/vllm-project/vllm/issues/51581):
  retained and verified the existing packed-QKV fallback instead of bypassing
  quantization scales in fused context-KV projection.
