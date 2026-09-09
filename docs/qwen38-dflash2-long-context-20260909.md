# Qwen3.8 DFlash2 48K / 64K context measurements

Status: **all four 48K/64K cells passed**, with measured longer-context numbers and [raw artifacts](../results/20260909-qwen38-dflash2-long-context/). User request: “ok let's do that then this isn't complete without the longer context numbers”.

## Scope and fresh research

This is a development-tier capacity/long-context extension under [BENCHMARKING_STANDARDS.md](../BENCHMARKING_STANDARDS.md), not a repeat of the complete BetterBench/serving/quality suite. The variable under test is the configured DFlash context ceiling. Run INT4 at 49152 then 65536 total tokens, followed by BF16 controls at those same limits. Preserve the fixed 32768-context [earlier validation](../results/20260909-qwen38-dflash2-boundary-fix/) and the separate [MTP4 / drafter benchmark reference](../results/20260909-qwen38-dflash2-rtn-standard/); MTP4 is a different stack bundle, not a matched quantization control.

Fresh primary sources consulted before modification:

- [Official vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/): `max_model_len` counts **prompt plus output**. The GPU-memory utilization budget does not itself guarantee a requested context fits; explicit KV-memory bytes, if set, replace automatic KV sizing. Keep the existing 0.95 utilization and automatic KV allocation unchanged, and measure rather than assuming the earlier 74440-token estimate remains valid at a larger configured ceiling. Current documentation defaults are not substituted for this pinned runtime's actual settings.
- [Pinned DFlash proposer](https://raw.githubusercontent.com/vllm-project/vllm/73029d42441321b631779db3475031f5ec26dd6c/vllm/v1/spec_decode/dflash.py): context and query buffers have distinct runtime allocations. Source inspection does not establish an executable maximum context; real startup and inference are required.

Only add an explicit DFlash context option to the standard runner, preserving its 32768 default and the existing fixed MTP launcher. No kernel, checkpoint, acceptance-policy, graph or memory-budget tuning is bundled into this sweep.

## Defined public-boundary test process

- **Environment:** idle `inference-host`, one Arc Pro B70; pinned image `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`, vLLM `73029d424`, XPU kernels 0.1.14.1. The exact existing target and BF16/partial RTN INT4 draft checkpoints remain read-only.
- **Fixed settings:** GPTQ INT4/G128 target, FP16 compute, FP8 target KV, BF16 draft KV, strict DFlash K7, legacy V1, server C1, max batched tokens 8192, graphs `[1,2,4,8]`, prefix cache off, thinking off, 275 W cap. No production promotion.
- **Order:** INT4 49152; INT4 65536; BF16 49152; BF16 65536. Each is a new disposable cell/output directory. Runs are sequential; no drift-cancelled significance claim.
- **Preconditions:** no active GPU container; Glimmer remains stopped; persistent launcher SHA256 is `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4` and power cap is 275000000 microwatts. Runtime dependencies stay outside the interactive Pi environment.
- **Startup:** invoke the real `qwen38_standard_bench.py --long-context-only --context LIMIT` path. `/health` must succeed and `/v1/models` must report the requested effective limit. Run the existing 3 answer canaries, 131 finite-logprob boundaries and 8 executable functional checks. Record a startup failure as a capacity/runtime failure, never as a successful larger-context test.
- **Test data and steps:** existing deterministic nonce/template framing, exact tokenization via `/tokenize`, streamed `/v1/completions`, temperature 0, seed 42, ignored EOS, 128 forced output tokens. Retain the standard lengths 512/8192/16384/32768/65536/120000/160000 and add the exact supported boundary: **49024 + 128** or **65408 + 128**. Every supported point gets one warmup and six measured trials. Add a one-token-over probe (**49025 + 128** or **65409 + 128**) requiring a context-limit HTTP 400. Other over-limit points remain real HTTP rejection probes, not omitted rows.
- **Expected results:** exact input and 128-output usage; `[DONE]` and `finish_reason=length`; no SSE errors; health after requests. Any unhealthy server stops that cell immediately. Later cells run only after verified cleanup; no automatic lower-context retry or undocumented setting change hides a failure.
- **Evidence:** frozen executed sources, launcher and exact client argv, environment/package/model configs, raw rendered tokens, request/response/SSE, per-request speculative counters, server logs and per-cell exit codes. Summarize median TTFT, post-first-stream decode proxy, end-to-end throughput and inclusive IQR from six valid measurements, excluding warmups. Report startup memory/KV estimates and maximum *successfully tested* prompt separately; do not call startup memory a measured workload peak. No new memory-sampling infrastructure is added.
- **Cleanup:** remove only each owned disposable container; verify launcher hash, power cap and stopped Glimmer. Retain every failed attempt. Commit bounded raw evidence and the final report; do not overwrite previous campaigns.

## Observed results

- Both BF16 and partial RTN INT4 passed at **49152 and 65536 total context**. The largest successful prompt is **65408 input + 128 output**; this is not a 65536-input-plus-output result, a proven maximum, or a 200K claim.
- At 49024 input tokens: BF16 decode proxy **41.08 tok/s (IQR 2.41)**; INT4 **44.17 (3.13)**. Median TTFT is **34.595 / 34.572 seconds** respectively.
- At 65408 input tokens: BF16 decode proxy **42.29 tok/s (IQR 1.93)**; INT4 **42.95 (3.61)**. Median TTFT is **51.398 / 51.375 seconds**. Six measured trials per point, excluding one warmup; these are sequential medians, not statistically significant gains.
- **140 valid forced-length streams**, including 120 measured requests, plus **16 correct context-limit HTTP 400s**, including each one-token-over boundary. All four cells pass the existing 3 canaries / 131 finite boundaries / 8 functional checks. Effective `/v1/models` limits, exact usage, stream termination, health, matching cross-drafter request payloads and executed-source equality were verified.
- **39 focused CPU tests pass**, no skips; changed/new Python syntax checks pass. The whole campaign completed in **58m05s, exit 0**; all four per-cell exits are zero. Cleanup confirms the unchanged persistent launcher, 275 W cap, stopped Glimmer and no remaining containers.
- INT4 still saves **2.00 GiB** of combined model-loading memory. Reported KV-token capacity changes with the configured ceiling: at 64K, BF16 reports **73231** and INT4 **103765**, despite unchanged KV pools of 4.69 / 6.65 GiB. The old 74440 estimate was not a hard maximum; neither is the new estimate. No workload memory peak or capacity beyond 64K was measured.
- [Final report, exact commands, environment, source snapshots, speculative counters and machine-readable analysis](../results/20260909-qwen38-dflash2-long-context/README.md). No production configuration was promoted; the full publication/quality limitations remain explicit.
