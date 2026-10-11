# Xe2 head-256 K64 prefill A/B — 2026-10-11

Disposable measurement only. The golden launcher was not modified.

## Decision

**Do not promote yet.** Under prefix caching disabled, the K64 tile clears the
prefill stop bar: median 98,304-token TTFT improved 6.37% (96.034 s to 89.911 s).
Decode did not regress, and 98,304+32,768 admission succeeded on both arms.
Production still runs with prefix caching on. Disabling it made vLLM force
Mamba cache mode to `none`, so this is not the production cache configuration.

## What changed

Only the Xe2 full-attention head-256 chunk-prefill tile, in a disposable
overlay of `libattn_kernels_xe_2.so`:

- `ShapeQK`: `<_256, _32, _32>` to `<_256, _64, _32>`
- `ShapePV`: `<_256, _32, _32>` to `<_256, _32, _64>`
- `ShapeOut` unchanged: `<_256, _256>`

Built from `vllm-xpu-kernels` `2a6fcfc1` (the 0.1.12.3-era tree) with oneAPI
2026.0 `icpx`, AOT `bmg-g31-a0`, and only the head-256 chunk-prefill config
plus the decode companion `8,256,64`. Compiler log: 256 registers, about 5
spills. Library sha256
`f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019`.

## Protocol

Host `inference-host`, one Arc Pro B70, power cap 275 W. Image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
Model `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` as served by the
golden launcher. Disposable container only, port 8001. Flags matched the
golden launcher except `--no-enable-prefix-caching`. Confirmed in the K64
server log: `enable_prefix_caching=False`, chunked prefill 8192, FlashAttention
block size 64, attention page 1664 (divisible by 64), MTP-4, FP8 KV, util 0.95,
`max-num-seqs 1`. Golden launcher hash remains
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

TTFT is non-streaming time to finish one greedy token (`temperature=0`) on an
exact token-id prompt. Five reps after one unmeasured warmup at each length.
No prefix cache, so there are no cached prompt tokens to report. Prompt token
counts matched the requested lengths.

## Results

Median TTFT, K64 versus stock. Positive is faster.

| Prompt tokens | Stock | K64 | Change |
|---:|---:|---:|---:|
| 2048 | 0.986 s / 2077 tok/s | 0.986 s / 2076 tok/s | -0.04% |
| 8192 | 4.075 s / 2010 tok/s | 4.057 s / 2019 tok/s | +0.45% |
| 32768 | 20.534 s / 1596 tok/s | 19.949 s / 1643 tok/s | +2.85% |
| 98304 | 96.034 s / 1024 tok/s | 89.911 s / 1093 tok/s | +6.37% |

The 98,304-token reps were 95.993–96.101 s stock and 89.741–89.993 s K64.
Whole-request 128-token decode after a 128-token prompt was 69.35 tok/s stock
and 69.04 tok/s K64 (-0.45%, inside noise). Both arms accepted a 98,304-prompt
/ 32,768-generation request with HTTP 200. Greedy one-token outputs matched on
synthetic lengths 5, 68, 69, 70, and 133. That is a parity check, not a
quality certificate.

Raw rows: `results-nocache-stock.json`, `results-nocache-k64.json`.
