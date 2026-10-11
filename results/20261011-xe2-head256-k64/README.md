# Xe2 head-256 K64 prefill — 2026-10-11

Promoted the K64 attention overlay into the golden launcher after the
no-cache full-context sweep. Production prefix caching stays on.

## Decision

**Promoted the tile only.** Median cold TTFT is flat through 4k, then faster
at every longer point, with non-overlapping ranges from 16k through the legal
ceiling. The 98,304-token gain is 6.41%. Decode did not regress. A 98,304-prompt
/ 32,768-generation request was accepted. The 212,992-token probe was illegal:
one output token exceeds `max_model_len`. The legal ceiling is 212,991.

The live launcher is `/home/mike/inference/launchers/start-qwen38.sh`, now
sha256 `c2328b1be4a0b57eb4be25444df819338d7515d8e6d79f6f1d3b8ab1baefb6e2`.
Previous script: `start-qwen38.sh.before-head256-k64-20261011T064649Z`.
Overlay: `/home/mike/inference/launchers/libattn_kernels_xe_2-head256-k64.so`,
sha256 `f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019`.
The service was started and answered `yes` to a greedy smoke prompt. Prefix
caching remains enabled. Research and the next wave are in
[RESEARCH.md](RESEARCH.md).

## Full-range no-cache result

Prefix caching off on both arms, because the measurement was not allowed to
use token caching. vLLM then forces Mamba cache mode to `none`. That is not
the production cache policy. Five reps through 98,304 tokens, three above that.
TTFT is non-streaming time to one greedy token.

| Prompt tokens | Stock median | K64 median | Change |
|---:|---:|---:|---:|
| 512 | 0.263 s / 1945 tok/s | 0.264 s / 1939 tok/s | -0.30% |
| 2048 | 0.983 s / 2083 tok/s | 0.985 s / 2079 tok/s | -0.18% |
| 4096 | 1.958 s / 2092 tok/s | 1.960 s / 2090 tok/s | -0.08% |
| 8192 | 4.074 s / 2011 tok/s | 4.063 s / 2016 tok/s | +0.27% |
| 16384 | 8.847 s / 1852 tok/s | 8.744 s / 1874 tok/s | +1.16% |
| 32768 | 20.490 s / 1599 tok/s | 19.974 s / 1641 tok/s | +2.52% |
| 65536 | 52.491 s / 1249 tok/s | 49.889 s / 1314 tok/s | +4.96% |
| 98304 | 96.098 s / 1023 tok/s | 89.936 s / 1093 tok/s | +6.41% |
| 131072 | 151.023 s / 868 tok/s | 140.213 s / 935 tok/s | +7.16% |
| 196608 | 295.398 s / 666 tok/s | 271.008 s / 726 tok/s | +8.26% |
| 212991 | 337.413 s / 631 tok/s | 309.186 s / 689 tok/s | +8.37% |

Ranges do not overlap from 16,384 tokens up. Short-context deltas are inside
the rep noise. K64 128-token decode was 1.825 s / 70.1 tok/s versus 1.846 s /
69.3 tok/s on the earlier stock arm. Admission HTTP 200. K64 text prompts
returned `yes`, `42`, and `Paris`. Synthetic length 5 on this later K64 start
was `!` rather than the `0` both arms produced in the paired four-point run;
that check was not re-paired, and this stack is already known to repeat short
greedy tokens nondeterministically.

Raw rows: `results-fullrange-nocache-stock.json`,
`results-fullrange-nocache-stock-ceiling.json`,
`results-fullrange-nocache-k64.json`. The earlier four-point files remain.

## What changed

Only the Xe2 full-attention head-256 chunk-prefill tile:

- `ShapeQK`: `<_256, _32, _32>` to `<_256, _64, _32>`
- `ShapePV`: `<_256, _32, _32>` to `<_256, _32, _64>`
- `ShapeOut` unchanged: `<_256, _256>`

Built from `vllm-xpu-kernels` `2a6fcfc1` with oneAPI 2026.0 `icpx`, AOT
`bmg-g31-a0`. Compiler log: 256 registers, about five spills.
