# Restored Intel 1.5 / EAGLE-4 comparison

**Development-tier engineering results, not a standard-compliant community benchmark.** The initial catastrophic slowdown was an implementation regression in the Intel FP8 port, not evidence that the EAGLE-4 model was intrinsically slow. The historical [report](REPORT.md) and its raw results are retained rather than overwritten.

## Scope and configuration

Only Intel's public release recipe is the baseline; no private baseline or calibration profile was used. The target, image, model revision, eager execution, context limit, batching settings, and launch arguments remain those in [REPORT.md](REPORT.md#fixed-environment-and-explicit-differences) and the adjacent launchers. No runtime, driver, precision, or model upgrade was used to recover performance.

- Published MTP: memory fraction 0.8, four speculative tokens, released engine and legacy runner.
- Matched-memory MTP control: same recipe at 0.9.
- Restored EAGLE: approved memory fraction 0.9, FP16 draft KV, FP8 target KV, 4096-token draft window, V2 runner/CUSTOM attention, static tree with 15 proposals, width 16, top-k 4 and depth 14. Target length remains 40000; prefix caching remains disabled.
- This compares installed configuration bundles, **not a drafter-only causal experiment**. EAGLE still does not fit the requested 40K configuration at 0.8; the historical allocation failure remains relevant.

## What was repaired

1. Restored the B70 draft kernel for the actual Intel FP16 cache stride `(1048576, 10240, 256, 1)`, alongside the previously qualified stride. Unknown layouts remain rejected.
2. Added ordinary native FP8 attention for the actual `uint8 [blocks, 4, 1024, 512]` cache: each slot contains 256 K bytes and 256 V bytes. Native attention receives typed E4M3 views and live FP32 scalar K/V descales, not a numerical conversion of raw bytes.
3. Published the exact runtime KV-zeroing contract from the worker's cache configuration. Native FP8 entry is fail-closed unless the target layer belongs to an authorized, zeroed full-attention group. Draft layers are excluded in both chain and tree modes.
4. Added native tree preparation that reads both committed and newly written rows from the FP8 cache, applies live scales, and packs ancestor-only sequences for compiled FA2. Poisoned live K/V tests ensure new rows are not accidentally taken from unquantized inputs.
5. Adapted the pinned Intel causal ABI and version selector. Qualified the width-16 FP8 workspace through 40000 tokens; FP16 and width-8 paths retain their previous 16384 limit.

The production source tested over HTTP is `ebc1d2c`; final source `24f1c632ad9c641cd3dcd0391c9ed8112e79763f` changes only diagnostic tests. Engine integration remains `6cae86cf1575c283485030b4104f20c6af8b63e0`.

## Native qualification

On the same B70 and pinned Intel image: **99 passed, zero skipped**, 14 deprecation warnings, 220.26 seconds. This includes ordinary FP8 writer/attention/zeroing cases, eight cached native-tree cases through context 39984, draft eager and graph-replay checks with live cache/table mutations, and existing FP8 cache/verifier regressions.

Initial poisoned-padding probes failed and were investigated, not discarded: unused E4M3 NaN V bytes can contaminate the native P·V operation even when attention probabilities are masked. The final diagnostic explicitly checks that hazard, then requires recovery and dense-reference parity after the real `KVBlockZeroer` and writer run. No tolerance relaxation, xfail, or new skip masks that failure. Only the runtime-zeroed path is authorized for ordinary FP8 dispatch.

Native test evidence: [99-case JUnit result](checks/restored-native-99.xml), retained unmodified from `inference-host:/home/mike/b70-evals/20261025-intel15-eagle4/native-final-24f1c63/out/ordinary-fp8-native.xml`; coordinator copy under `/tmp/intel15-eagle4-comparison/native-final-24f1c63/out/`.

## Saved-input public HTTP results

Real `/v1/completions` streaming requests use saved exact input-token fixtures, temperature 0, seed 0, ignored EOS and 512 output tokens. Restored EAGLE has three measured repetitions per context after four excluded canary/warmup requests; previously collected MTP and ordinary controls have ten each. These are **unequal-sample probes**, separate from the CLI comparison below. Every restored measured request completed with 512 output tokens.

| Arm | Input | N | Median TTFT (s) | Median post-first-token decode (s) | Median full request (s) |
|---|---:|---:|---:|---:|---:|
| Ordinary target reference | 8K | 10 | 3.963 | 18.823 | 22.786 |
| Matched MTP 0.9 | 8K | 10 | 4.115 | 12.284 | 16.400 |
| Restored EAGLE 0.9 | 8K | 3 | 4.353 | 19.790 | 24.104 |
| Ordinary target reference | 32K | 10 | 19.808 | 19.881 | 39.689 |
| Matched MTP 0.9 | 32K | 10 | 20.515 | 11.301 | 31.817 |
| Restored EAGLE 0.9 | 32K | 3 | 20.792 | 19.916 | 40.709 |

Column medians are calculated independently and need not sum. A separate one-output-token diagnostic measured warmed TTFT of 4.131 s at 8K and 20.446 s at 32K. This removes the old ~774-second 32K prefill regression. The restored saved-input 32K/512 median is 40.709 s versus approximately 1151 s on the broken port; MTP remains faster than restored EAGLE in this tested recipe.

Per-request EAGLE `(accepted, proposed, rounds)` counts:
- 8K: `(407,1560,104)`, `(410,1545,103)`, `(397,1710,114)`.
- 32K: `(405,1590,106)`, `(406,1575,105)`, `(386,1890,126)`.

These tree proposal acceptance ratios are not directly interchangeable with a four-token MTP chain's acceptance ratio.

## Equal-sample Intel CLI comparison

The restored run uses the same CLI workload settings as the retained MTP baselines: ten measured random prompts, one warmup, 32768 input tokens, 512 output tokens, concurrency one, temperature/seed zero, ignored EOS. All three arms completed **10 requests, zero failures**, totaling 327680 input and 5120 output tokens each. CLI initialization and its one warmup are excluded from measured timings.

| Metric | Published MTP 0.8 | Matched MTP 0.9 | Restored EAGLE 0.9 |
|---|---:|---:|---:|
| Median TTFT (ms) | 20594.60 | 20597.35 | 20798.13 |
| Median TPOT, excluding first token (ms) | 15.81 | 15.59 | 60.57 |
| P90 TPOT (ms) | 20.88 | 20.74 | 173.35 |
| Aggregate output throughput (tokens/s) | 17.358 | 17.415 | 8.467 |
| Request throughput (requests/s) | 0.03390 | 0.03401 | 0.01654 |
| Total-token throughput (tokens/s) | 1128.295 | 1131.991 | 550.373 |
| Measured duration (s) | 294.958 | 293.995 | 604.681 |
| CLI acceptance rate (%) | 83.801 | 83.751 | 10.132 |
| CLI acceptance length | 4.352 | 4.350 | 2.520 |
| Draft rounds | 1176 | 1177 | 2039 |
| Proposed tokens | 4704 | 4708 | 30585 |
| Accepted tokens | 3942 | 3943 | 3099 |

The restored EAGLE bundle has **51.4% lower aggregate output throughput** and **3.886× median TPOT** than the matched-memory MTP control on this synthetic workload. Aggregate output throughput includes prefill; it is not decode-only throughput. Prefill is now similar to MTP. Decode latency varies substantially across EAGLE requests, with lower aggregate acceptance than on the saved-input HTTP fixture. This is a measured remaining gap, not evidence that every EAGLE configuration or real task behaves this way. No further tuning was introduced to improve this comparison.

Raw unmodified results: [MTP 0.8](mtp-published-benchmark.json), [MTP 0.9](mtp-mem90-benchmark.json), [restored EAGLE](eagle-restored-benchmark.json). The [restored CLI log](checks/eagle-restored-benchmark.log) records arguments, warmup and measured sample counts. The original broken-port JSON remains separate. Prometheus snapshots before and after the entire CLI invocation include initialization/warmup and must not be substituted for the CLI's measured-window acceptance values above.

Exact benchmark command, run inside the qualified server container after sourcing the image's oneAPI/CCL environment:

```bash
vllm bench serve --model Qwen3.8-27B \
  --tokenizer /llm/models/Qwen3.8-27B --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port 8000 --backend vllm \
  --request-rate inf --max-concurrency 1 --ignore-eos --trust-remote-code \
  --dataset-name random --num-prompts 10 --num-warmups 1 \
  --random-input-len 32768 --random-output-len 512 --random-range-ratio 0 \
  --temperature 0 --seed 0 --save-result --save-detailed \
  --result-dir /tmp/intel15-bench --result-filename eagle-restored.json \
  --metric-percentiles 50,90,99
```

The executed orchestration script is `/tmp/intel15-eagle4-comparison/run-restored-benchmark.sh`. Full server log, CLI log, raw JSON and before/after metrics are retained in `restored-ebc1d2c/benchmark/` under both run roots.

## Correctness limits

The three semantic chat canaries passed: `19*23 → 437`, a valid Python `add` function, and `{"answer": 42}`. This is a smoke check, not a broad quality evaluation.

Restored EAGLE produced three distinct sequences at each saved-input context. None exactly matched any retained ordinary-reference sequence. First differing token indices against reference repetition zero were `[3,1,1]` at 8K and `[244,1,1]` at 32K. The MTP 32K control matched the ordinary reference in 7/10 repetitions. Ordinary 8K repetitions themselves produced nine distinct sequences; ordinary 32K produced one. Thus this work does **not** establish exact greedy equivalence or quality non-regression, despite native numerical parity and the canary results.

No BetterBench full run, full long-context/concurrency sweep, broad quality suite, interleaved crossover, or matched thermal study was completed. This report is a bounded development comparison, not a standard-compliant or production-readiness claim. Equal CLI sample counts do not remove the other limitations.

## Reproduction and evidence

- Successful serving artifacts: `inference-host:/home/mike/b70-evals/20261025-intel15-eagle4/restored-ebc1d2c/`, copied to `/tmp/intel15-eagle4-comparison/restored-ebc1d2c/` on the coordinator. `server.log`, `http/results.json`, raw SSE files, `prefill-probe/`, and `semantic/` retain actual requests, timing and outputs.
- Public path: start the adjacent EAGLE launcher with staged qualified source and existing engine patch at fraction 0.9; check `/health`; run `restored-prefill-probe.py`, then `tests/intel15_http_probe.py run --repeats 3` on the saved fixture, then the chat canaries. Stop the owned container after collection. Full executed scripts remain in `/tmp/intel15-eagle4-comparison/`.
- Native path: `TRITON_INTERPRET=0`, pinned image, real `/dev/dri`, compiled FA2 required, and the real Intel cache writer/zeroer. The executed remote script is `native-final-24f1c63/native-final.sh` under the same run root; its JUnit result contains all 99 cases.
- Earlier failed parity, ABI, startup and timeout logs remain under the run roots. The final results do not retroactively turn those attempts into passes.
- After the final benchmark, the owned server was stopped. A fresh `docker ps --format '{{.Names}}'` returned no running containers, and `ss -ltn 'sport = :18110'` returned no listener. No inference deployment was promoted or left running.

Public research informing the repair:
- [Intel 1.5 release](https://github.com/intel/llm-scaler/releases/tag/vllm-1.5) and [pinned XPU patch](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/patches/vllm_xpu_kernels.patch): version and ABI constraints.
- [Pinned-era FA wrapper](https://github.com/vllm-project/vllm-xpu-kernels/blob/a692986/vllm_xpu_kernels/flash_attn_interface.py): typed FP8 cache, scalar K/V scales, FP16 Q with `q_descale=None`; generic FA2 rather than the GQA-incompatible ESIMD shortcut.
- [Paged cache contract](https://github.com/vllm-project/vllm-xpu-kernels/blob/a692986/csrc/xpu/attn/paged_kv_utils.h) and [chunk-prefill mainloop](https://github.com/vllm-project/vllm-xpu-kernels/blob/a692986/csrc/xpu/attn/xe_2/collective/chunk_prefill_mainloop.hpp): runtime strides and invalid-padding V hazard.
- [Intel Triton issue 5581](https://github.com/intel/intel-xpu-backend-for-triton/issues/5581): do not silently compile-specialize unqualified cache strides.

EAGLE-4 is source-available, not OSI open source. No model weights or EAGLE source are distributed here, and no EAGLE upstream push is authorized. Implementation commits remain in their source repositories; only this report and bounded evidence are delivered to the user-owned results repository.
