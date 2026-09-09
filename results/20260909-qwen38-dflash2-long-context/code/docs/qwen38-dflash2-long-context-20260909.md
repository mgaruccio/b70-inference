# Qwen3.8 DFlash2 48K / 64K context measurements

Status: planned; no larger-context pass or timing is claimed yet. User request: “ok let's do that then this isn't complete without the longer context numbers”.

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

Pending execution. A 65536 total-token configuration, if successful, establishes at most a 65408-token input with this 128-token output test; it is not a 65536-input-token result or a 200K claim.
