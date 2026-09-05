# Glimmer B70 concurrency and native context — 2026-09-05

## Retained candidate

`scripts/start-muse-vllm-concurrent.sh`: the established pinned vLLM-XPU image,
`vllm-xpu-kernels==0.1.13.2`, existing GPTQ target and GPTQ assistant, packed-QKV
fallback (`DFLASH_KV_MODE=none`), XPU graphs, **DFlash K4 / max-num-seqs 8**.
Native context is **131072**, FP8 KV, memory utilization 0.90, batch token budget
2048, chunked prefill enabled by vLLM, prefix caching disabled.

This is separate from the original C1/K20 launcher. Qwen is unchanged. The new
launcher binds localhost port 18080, does not remove existing containers, and
requires the GPU to be free. It is not an automatically installed service.

```bash
bash scripts/start-muse-vllm-concurrent.sh
bash scripts/wait-vllm-health.sh 420 18080
curl -fsS http://127.0.0.1:18080/v1/models
# After use, free the shared B70 before starting Qwen:
docker rm -f muse-vllm-xpu-concurrent
```

## Controlled short-input screen (8k configured context)

Same sky prompt, sampling (temperature 1, top_p .95, top_k 64, seed 42), 256 output
tokens, one shape-matched warmup and three measured waves at each client load.
Aggregate throughput is sum of completion tokens divided by wave wall time,
including prefill, queueing, reasoning, and answer tokens. These are capped
performance probes, **not completed-answer or broad quality scores**.

| Server | Load 1 | Load 4 | Load 8 |
|---|---:|---:|---:|
| Original C1/K20 | 48.110 | 48.085 | 48.080 |
| C8/K20 | 52.377 | 93.975 | 152.629 |
| C8/no speculation | 31.846 | 122.725 | 236.485 |
| C8/K4 | 51.964 | 158.345 | 278.255 |

Values are median aggregate e2e tok/s. C8/K4's eight-request wave took 7.36s versus
42.60s for C1/K20: 5.79x aggregate throughput. No request errors in this screen.
The concurrency client now recognizes vLLM's `delta.reasoning` as well as legacy
`delta.reasoning_content`. Its historical client decode ratio is approximate for
multi-token speculative chunks; use aggregate e2e for the comparison above.

Completed-answer checks used existing share-suite prompts (GSM8K Janet, HumanEval/0,
MT-Bench Hawaii), greedy sampling, and 2048 maximum output tokens. C1/K20 passed
3/3; C8/K20, C8/no-spec, and C8/K4 each passed 8/8 mixed requests. Checks are final
$18, the existing two code assertions, and nonempty >=400-character writing with
natural stop. This is a smoke cohort, not proof of general quality equivalence.

Final replay with **131072 configured context** retained **278.104 aggregate
e2e tok/s** median at C8 (277.809–278.473), with zero request errors. All eight
post-stress completed-answer checks passed. Thus the retained context setting
does not trade away the short-input concurrency gain on this tested workload.

## Native context and the misleading "180k" number

**Do not treat six near-native requests as validated:** the forced-length
6×128994 + 2048 test reached 97.46% KV use, but two requests emitted no visible
text despite reporting 2048 output tokens. All HTTP streams ended and the queue
drained with zero preemptions; this is still a failed response-validity gate.
A streaming/non-streaming × normal-EOS/ignore-EOS diagnostic follows below.

Single-request replay of the same 128994-token prompt passed all four combinations:
streaming/non-streaming × normal EOS/`ignore_eos=true`. Both normal-EOS responses
stopped naturally with all three codes; both forced-length responses had visible
content. Raw responses are `diag-stream*-ignore*-raw.txt`; summary is
`native-long-response-matrix.json`. This does **not** resolve the high-pressure
six-request failure or establish whether it is model, parser, speculation, or cache
behavior. It does show that neither streaming nor `ignore_eos` alone reproduces it.

The failed forced-length probe does not mean continuous batching is disabled.
Per-request context limits, active sequence limits, and memory admission are
different controls; reducing max context is not required to enable queueing.

Six **normal-EOS** requests at 128994 prompt tokens subsequently all passed
three-code retrieval, stopped naturally (186–242 output tokens), and drained
in 587.92s with zero preemptions. Peak three running, five waiting, and 45.38% KV
usage. This validates ordinary full-context queue service, **not** six full
contexts resident at once and **not** resolution of the forced-length failure.
Per-request raw SSE is saved as `native-c6-normal-eos-request*-raw.txt`.

The 8k startup reported **180022 tokens / 21.98x concurrency**, with **5.18 GiB**
of KV memory. With the same memory allocation but `max-model-len=131072`, startup
reported **609193 tokens / 4.65x concurrency**. Neither is a universal flat token
pool size. In the pinned `vllm/v1/core/kv_cache_utils.py`,
`get_kv_cache_capacity` returns `int(max_concurrency * max_model_len)`;
`get_max_concurrency_for_kv_cache_config` divides shared blocks by the sum of
per-request blocks across KV groups. The model has 39 sliding-window layers
(window 2048), 13 full-attention layers, and a five-layer sliding-window draft.
Group padding and different draft head geometry also affect allocation.

Public API checks through the saved native-context launcher:

- **130940 prompt + 128 output = 131068 tokens** completed; no preemption.
- **128896 prompt + 284 output** recovered three beginning/middle/end checkpoint
  codes and stopped naturally. This synthetic retrieval probe is not a broad
  long-context reasoning evaluation.
- Eight submitted **65532-token** prompts with 128 outputs all completed, but
  peaked at only three running and 26.5% KV use; this proves queue service, not
  eight simultaneously full contexts.
- Keeping the same eight 65532-token prompts resident with **2048 output tokens
  each** reached **eight running / 84.25% KV use**, zero preemptions, and a clean
  drain in 412.50s. All eight produced the full requested output count. This
  validates >524k prompt tokens concurrently resident, unlike the short-output test.
- Eight submitted **81912-token** prompts with 2048 outputs all completed in
  549.72s. Peak **seven running / 89.86% KV use**, no preemption: the eighth
  request queued instead of eight full contexts being resident together.
- Twelve staggered 2042-token requests, varied 64–768 output lengths: peak eight
  running, waiting rose above zero, replacement requests streamed before the
  original eight all finished, all twelve completed, running/waiting drained,
  and preemption count stayed unchanged. Continuous batching/queueing verified.

## Reproduction and evidence

Host: `inference-host`, single Arc Pro B70, Linux `xe`; experiment container
`muse-b70-concurrency`, localhost port 18080. No power settings changed (observed
275W cap, runtime power control `auto`). Raw local-state evidence is on the host:

`~/b70-evals/muse-glimmer/20260905-concurrency/`

- `cell.sh`, `instrument.py`, `launch-c1.sh`, generated `launch-cell.sh`: screen
  recipe; `c*-load*.json`, metrics before/after, and per-cell server logs.
- `quality.py`, `c*-quality.json`: full completed-answer responses and checks.
- `context-probe.py`, `native-*.json`, `native-*-prompt.txt`: public `/tokenize`
  counts, SSE responses/usage, request timestamps, and 200ms `/metrics` samples.
- `native-start.log`: pinned native-context startup and cache capacity.

Example executed commands on the host (R is the evidence directory):

```bash
bash "$R/cell.sh" 8 4 '1 4 8'
python3 "$R/quality.py" 8 "$R/c8-k4-quality.json"
python3 "$R/context-probe.py" queue 2048 12 native-queue12
python3 "$R/context-probe.py" capacity 130944 1 native-c1-filled
python3 "$R/context-probe.py" retrieval 128900 1 native-c1-retrieval
python3 "$R/context-probe.py" capacity 65536 8 native-c8-64k
python3 "$R/context-probe.py" capacity 65536 8 native-c8-64k-resident 2048
python3 "$R/context-probe.py" capacity 81920 8 native-c8-80k-pressure 2048
# Failed response-validity gate; preserved as a known limit:
python3 "$R/context-probe.py" capacity 129000 6 native-c6-129k-admission 2048
python3 "$R/long-response-matrix.py"
python3 "$R/context-probe.py" retrieval 129000 6 native-c6-normal-eos
python3 "$R/instrument.py" --base http://127.0.0.1:18080/v1 \
  --model muse-glimmer-gptq --concurrency 8 --reps 3 --max-tokens 256 \
  --label native-final-c8 --out "$R/native-final-c8.json" --log ''
python3 "$R/quality.py" 8 "$R/native-final-quality.json"
```

Do not infer support beyond the model's native 131072 context. `max-num-seqs=8`
is an active-request limit, not eight reserved 131k contexts. Longer prompts can
queue and KV pressure may cause recomputation; startup alone is not a load test.

## Fresh primary-source research

- https://docs.vllm.ai/en/stable/features/speculative_decoding/ — speculation is
  a latency optimization; verify throughput under load rather than extrapolating C1.
- https://docs.vllm.ai/en/stable/features/speculative_decoding/dynamic_speculative_decoding/
  — batch-dependent draft depth is possible; this experiment keeps a simpler
  measured static K4, with no untested dynamic-speculation mapping.
- https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/ — full-attention
  and sliding-window groups share blocks; padding and window eviction matter.
- https://huggingface.co/meta-models/Muse-Glimmer-30B — native sequence length.
- https://docs.vllm.ai/en/v0.27.0/configuration/optimization/ — chunked prefill.
- https://docs.vllm.ai/en/stable/usage/metrics/ — running/waiting/cache/preemption
  metrics used for admission and drain verification.
