# Community comparison research — 2026-10-09

This is a bounded source review, not an independent reproduction, exhaustive
leaderboard, or announcement approval. Full local quality qualification was
paused at the user's direction. The community-protocol serving benchmarks were
then completed; see `../20261009-qwen38-community-protocol/README.md`.
No production configuration change or announcement has been made.

## Closest published speed reference

The public [draft-INT4 submission](https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/master/submissions/vllm-qwen38-mtp4-draft-int4.json)
reports **112.65 tok/s** client post-first-token decode at p512/g128, median of
five runs (111.58–117.73), one B70, C1, prefix caching off, 230 W cap (205 W
measured median). Measurement date: 2026-08-18. This is the highest comparable
single-B70 Qwen3.8-27B p512/g128 token-rate report found in the bounded search,
not proof of a universal current best.

The submission explicitly records 131072 configured context, utilization .88,
8192 batched tokens, FP8 KV, `--dtype float16`, GPTQ INT4, MTP4, and both
`B70_DRAFT_LMHEAD_INT4=1` / `B70_DRAFT_MTP_INT4=1`. It uses the same f01e24f6 image,
vLLM 0.27.2rc1.dev77+gac7509e2b and kernels 0.1.12.3 as our campaign. Its notes
also name the mixed-GDN-v5 patch. The matched BF16-draft control was **81.20**,
not the separate older 83.7 result. Acceptance was 94.44% versus 95.86%.

Our existing baseline already uses S+M1, so dismissing 112.65 as irrelevant
because it uses draft INT4 would be incorrect. Conversely, our 64K-input
58.33 tok/s cannot be ranked against its 512-input cell. Our configured context
212992, utilization .95, 275 W cap, CPU controls and patch set must be disclosed;
matching a subset of flags does not establish an apples-to-apples comparison.

The submission explicitly says: **“Speed-only: no token/KL/task-quality parity
vs BF16 draft.”** Approval is acceptance of a self-report, not independent
verification. The subsequent 15-task A/B is not broad model-quality evidence.

## Replay limitations

The [public reference client](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/benchmarks/b70-realworld-context-harness.py)
uses streaming, monotonic timing and post-first-token throughput. The exact
112.65 campaign prompt JSON and raw SSE were not found publicly. A generic
client invocation is not proof of that campaign's exact workload.

[Jonathan Mann's 84.56 report](https://jonathanmann.tech/blog/qwen38-intel-arc-b70/)
uses one B70, BF16-draft MTP4, p512/g128, one warmup and five measurements, with
100% acceptance on a favorable prompt. The exact prompts, complete sampler
settings and raw requests were not found publicly. It is not an exact replay
baseline either. A new comparison can align methodology and publish its own
inputs, but must not pretend to reproduce these unavailable inputs.

## Newer correctness warning and alternative

The [pinned EXL3 route report](https://raw.githubusercontent.com/SergiioB/intel-arc-pro-b70-inference-cookbook/1a76136ea1b3f96bba3b20e42041a6fcc95f56fd/docs/qwen38-27b/EXL3-XPU.md)
reports a September 26/27 arithmetic canary returning 30 in several GPTQ/W4A16
exports while a BF16 TP2 control returned 14. It calls that family “closed for
this model” and recommends EXL3. This is a campaign self-report and a broader
author verdict, not a demonstration about every export. The page does **not**
name our exact SergiioB model revision, pin the failing runtime/export revisions,
or reproduce the complete failing prompts. Do not transfer its EXL3 runtime
pin onto the failing GPTQ runs or claim that our exact build is proven affected.

The page reports EXL3 passing six exact-answer checks, including
`sum(i*i for i in range(4)) == 14` twice and `9*7-3 == 60`, and a correct answer
with a 261920-token prompt at the configured 262144 boundary. Its decode figure
is an **MTP chunk rate that undercounts generated tokens**, explicitly not
comparable to the 112.65 token-rate cell. Broader benchmark quality is not
established by six canaries. Our previous arithmetic canary was 19+23, not the
reported range/squares expression; the tests must not be conflated. Our
"target-only" control remains GPTQ, not a BF16-weight accuracy reference.

The same report describes separate prefix-cache corruption on a W4A16
compressed-tensors route. Our prefix cache is already disabled; that alone does
not address its separate arithmetic warning.

Other sources considered:
- [Achu Mukundan's public long-context campaign](https://achumukundan.dev/data/b70-workhorse/qwen38-20260820-README.md):
  reported 51.02 tok/s at 16K and 36.48 at 130560 input, three samples per cell;
  different workloads, not a short-context ranking.
- [Two-B70 FP8 route](https://github.com/steveseguin/b70-optimization-lab/blob/main/packages/qwen38-27b-fp8-tp2-b70/README.md):
  reported 90.2 tok/s uses two GPUs and a different precision/runtime/MTP setup;
  not a single-card comparison.

**Current conclusion:** a historical speed-only reference exists, but neither
quality-safe community leadership nor announcement readiness is established.
Do not replace the missing public prompts with favorable inputs and call the
result a community-best reproduction. Preserve our full-suite results and
paired performance methodology, and label all remaining comparison gaps.
