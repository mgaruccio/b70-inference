# Glimmer B70: 350 aggregate tok/s attempt — 2026-09-05

350 was the next target after validated C8/K3 **303.1**. This session did **not**
reach 350 and did **not** retain a new launcher change. The C8/K3 cell without
`--async-scheduling` remains the bar.

Prefix caching was excluded on purpose: it would only help repeated-prompt
prefill, not decode.

## Screen

Existing sky prompt, eight simultaneous HTTP SSE clients, temperature 1,
top_p .95, top_k 64, seed 42, 256 output tokens. One shape-matched warmup,
then three measured waves. Context 131072. Aggregate e2e = total completion
tokens / concurrent wave wall time (gated by the slowest of the eight
clients). Host `inference-host`, Arc Pro B70, pinned
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
`vllm-xpu-kernels==0.1.13.2` unless noted. GPU was idle at session start and
empty after cleanup.

| Cell | Median aggregate tok/s | Three-wave range | Notes |
|---|---:|---:|---|
| K3 / bt2048 (reproduced) | **303.141** | 302.672–303.173 | Matches the retained K3 bar |
| K3 / bt8192 | 301.856 | 301.452–302.806 | Official >8192 throughput advice; slower here |
| K3 / bt4096 | 302.556 | 301.538–302.741 | Neutral/slightly worse |
| K3 / bt2048 / `--async-scheduling` (generated cell) | 314.465 then **312.205** cold | 312.380–314.646; 311.837–312.586 | 8/8 quality on the first cell |
| K3 / async + CPU `performance` governor | 314.902 | 312.505–315.103 | Same live cell; within noise |
| K3 / async + `vllm-xpu-kernels==0.1.14.1` | 308.767 | 307.834–309.078 | Booted; slower than 0.1.13.2 |
| **Repo launcher + async** (two independent starts) | **302.022 / 302.005** | 300.979–302.390 | `async_scheduling: True` in logs; **stragglers** |

All capped probes had 8/8 HTTP success, 256 completion tokens, and visible
text. Not completed-answer scores except the 8/8 GSM8K/HumanEval/MT-Bench
smoke on the first async cell and on the first repo-async start.

## Why async was not retained

On generated experiment cells, async packed the eight clients tightly
(elapsed about 6.14–6.55 s) and raised aggregate throughput to 312–314.

The actual `scripts/start-muse-vllm-concurrent.sh` with the same flag
measured **302** twice. Per-request elapsed was about **5.74–6.80 s**: some
streams finished earlier, the wave was gated by a slower one. Client decode
stayed ~43 tok/s. That is worse aggregate than K3 without async (6.24–6.76 s,
303.1).

Because the shipped launcher path did not reproduce 312–314, `--async-scheduling`
was reverted. 350 is not claimed.

## Other closed knobs

- `max-num-batched-tokens` 4096/8192: no gain on this decode-heavy short screen.
- Intel B70 max dynamic clock is 2800 MHz; the card was already there.
  `power1_cap` is 275 W and rejected 290 W.
- CPU `performance` governor: noise on the live async cell.
- DFlash2 stays parked for FP16 XPU (0% acceptance; [issue 55250](https://github.com/vllm-project/vllm/issues/55250)).
- Dynamic speculation does not beat static K3 at fixed C8.
- More clients or shorter context would not be the same comparison.

## Why 350 is still a kernel-class target

Even the unreproduced 314.5 is only +3.7% vs 303. 350 is +15.5%. The 30B GPTQ
target verify remains the dominant cost. Remaining headroom is XPU kernel
efficiency or a working BF16 DFlash2 path.

## Research

- [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
- [SchedulerConfig.async_scheduling](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/)
- [Intel Arc Pro B70 specs](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html)
- [DFlash2 FP16 XPU](https://github.com/vllm-project/vllm/issues/55250)
- [vllm-xpu-kernels 0.1.14](https://github.com/vllm-project/vllm-xpu-kernels/releases)

## Evidence

`~/b70-evals/muse-glimmer/20260906-350tps/` on `inference-host`. No raw logs
committed. Experiment container `muse-b70-350` was removed; `docker ps` empty.
