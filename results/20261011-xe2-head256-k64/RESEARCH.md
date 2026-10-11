# Xe2 head-256 K64 — research record

This is the decision record for the Qwen3.8-27B prefill wave on the Arc Pro B70.
It is not a production promotion by itself. The measured no-cache four-point
result is in [README.md](README.md). A full-context no-cache sweep is the
promotion gate and is recorded in the same directory when it finishes.

## Target

Xe2 full-attention chunk prefill for head dimension 256, K tile 32 to 64:

```text
ShapeQK  <_256, _32, _32> -> <_256, _64, _32>
ShapePV  <_256, _32, _32> -> <_256, _32, _64>
ShapeOut <_256, _256> unchanged
```

Qwen3.8-27B full attention is 24 Q heads / 4 KV heads, head dim 256, every
fourth layer, no sliding window. The live page measured in this image is
1,664 tokens, which is divisible by 64, so the wider K tile is in scope.
FlashAttention is the backend (`kv cache block size 64`, then the hybrid
aligner raises the attention page to 1,664).

## Why this and not the alternatives

- Published sibling result, not our pin: AnnoyingTechnology,
  [head-256 tuning note](https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docs/xpu-head256-attention-tuning-2026-08-27.md)
  and
  [patch](https://github.com/AnnoyingTechnology/intel-arc-b70-llm-inference/blob/main/docker/patches/vllm-xpu-kernels-xe2-head256-k64.patch).
  At 210 W their full-model 100k TTFT moved 115.19 s to 111.46 s (+3.24%).
  Attention microbench at 100k was +9.57%. Q512/K128 was not built because
  K64 already spills registers on Battlemage.
- More GDN kernel work is not the wave. vllm-xpu-kernels #476+#523 on a B70
  Qwen3.8 serving test was about +0.11% at 131k and slower at fresh 4k.
- Raising `max_num_batched_tokens` to 16384 was slower on the sibling ~100k
  cold A/B. This host's golden budget is already 8192. It was held fixed.
- `lm_head` quant is not a prefill lever here. Prefill logits are selected
  rows, not every prompt token.
- PR #363 reverse odd-head scheduling is a separate arm. It was not stacked.
  Its public numbers are B580 MQA head 128, not this 24/4 head-256 FP8-KV path.
- W4A8, sparse prefill, disaggregated prefill, MTP depth changes, and CUDA or
  Ascend ports were out of scope.

## Build

Pinned image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
vLLM `0.27.2rc1.dev77+gac7509e2b`, kernels `0.1.12.3`. There is no public
`v0.1.12.3` git tag. The wheel uploaded 2026-08-10. `chunk_policy_head256`
is unchanged from tag `v0.1.12.1` through commit `2a6fcfc1` (2026-08-10,
"add gemma3 conf"), which is the last commit before that upload. The two-line
tile patch applied cleanly there.

The runtime image has `ocloc` 26.27.39122.11 and IGC 2.38, but not `icpx`.
The host already had `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`.
The focused library was compiled with that `icpx`, AOT device `bmg-g31-a0`
only, against the runtime image's Torch 2.13.0+xpu. oneDNN was skipped because
the FA2 sources do not reference it. SYCL-TLA was pinned to
`cd763790ad2f74d7294435ecf77682bac0062c3a`. The log reported 256 registers and
about five spills, matching the sibling K64 compile note.

Overlay library sha256
`f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019`.
`_vllm_fa2_C` needs two symbols from `libattn_kernels_xe_2.so`; both are present.

## Measurement contract

User instruction for this wave: do not use token caching. Both arms ran with
`--no-enable-prefix-caching`. vLLM then logged `Mamba cache mode is set to
'none' when prefix caching is disabled`. That is a real confound versus the
golden launcher, which keeps prefix caching and `--mamba-cache-mode align`.
The K64 change itself is only the attention tile.

Other flags matched the golden launcher: context 212992, util 0.95, batch
8192, MTP-4, FP8 KV, `max-num-seqs 1`, power cap 275 W. Golden launcher hash
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4` was not
modified for the measurement.

TTFT is non-streaming time to finish one greedy token on an exact-length
token-id prompt. Five reps. Prefix caching off, so cached-token counts are
absent by design. Prompt token counts matched the requested lengths.

## Four-point no-cache result

| Prompt tokens | Stock median | K64 median | Change |
|---:|---:|---:|---:|
| 2048 | 0.986 s / 2077 tok/s | 0.986 s / 2076 tok/s | -0.04% |
| 8192 | 4.075 s / 2010 tok/s | 4.057 s / 2019 tok/s | +0.45% |
| 32768 | 20.534 s / 1596 tok/s | 19.949 s / 1643 tok/s | +2.85% |
| 98304 | 96.034 s / 1024 tok/s | 89.911 s / 1093 tok/s | +6.37% |

98,304-token reps did not overlap (stock 95.993–96.101 s, K64 89.741–89.993 s).
Short 128-token decode was 69.35 vs 69.04 tok/s. Both arms returned HTTP 200
for a 98,304-prompt / 32,768-generation admission. Synthetic greedy lengths
5, 68, 69, 70, and 133 matched. That is parity, not a quality certificate.

Stop bar used for this wave: kill on admission failure, non-finite output,
decode drop over 5%, or 98k median TTFT gain under 2%. The four-point run
cleared that bar. Promotion still waits on the full-context sweep
(512 through 212992) and text-prompt greedy parity, because a tile that wins
at 98k can still miscompile or regress at the context ceiling.

## Not promoted

The live launcher was left on the stock kernel. A passing full-range sweep
promotes only the K64 library overlay. It does not turn production prefix
caching off, because that forces Mamba cache mode `none` and is a separate
serving-policy change from the tile.
