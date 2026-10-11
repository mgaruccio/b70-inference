# Qwen B70 — next session

Updated 2026-10-11. The Xe2 head-256 K64 attention overlay is the production default. The September custom-kernel decision below is historical and does not apply to this tile.

> **Current prefill result:** [K64 full-range sweep](../results/20261011-xe2-head256-k64/README.md) and [research record](../results/20261011-xe2-head256-k64/RESEARCH.md), commit `c429527f`. No-cache TTFT is flat through 4k and faster from 16k to 212,991 tokens, up to +8.37%. Production prefix caching stays on.
>
> **Prefill prerequisite resolved; chunk screen completed:** The maintained guard is deployed after the five existing patches. All132 distinct-prefix finite-logprob boundary probes and arithmetic/JSON/Python-AST canaries passed. Both8320 and6656 were slower than8192 at16k/32k; the predeclared kill gate stopped longer tests. Guarded K64 production is running with batch8192 and prefix caching unchanged. [Results, commands and limitations](../results/20261011-qwen38-chunk-alignment/README.md).
>
## Closeout

- Host: `inference-host`. Service `qwen38` is running with the overlay mounted.
- Launcher: `/home/mike/inference/launchers/start-qwen38.sh`, SHA256 `5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d`. Guard deployment backup: `start-qwen38.sh.before-prefill-guard-20261011` (old K64 hash `c2328b1be4a0b57eb4be25444df819338d7515d8e6d79f6f1d3b8ab1baefb6e2`).
- Overlay: `/home/mike/inference/launchers/libattn_kernels_xe_2-head256-k64.so`, SHA256 `f5195ace0074610d266e38322060dc4b71bf39c30dd4b7ab984fa830fa05b019`. Repo source: [`scripts/start-qwen38.sh`](../scripts/start-qwen38.sh).
- Unchanged serving policy: 212992 context, util 0.95, batch 8192, MTP-4, FP8 KV, prefix caching, `mamba-cache-mode align`, C1, 275 W. A 212,992-token prompt plus one output token does not fit; the legal ceiling is 212,991.
- Chunk budgets8320/6656 were screened and rejected; default8192 remains. Unstarted proposals only: `VLLM_XPU_FA_REVERSE_ODD_HEADS` alone, then QKV/attention overlap. No further experiment is running. Do not widen K64, reopen GDN, raise the chunk budget to16384, or try FP8 query MMA.

> **Newer MTP-tuning follow-up:** [Acceptance-aligned validation plan](qwen38-mtp-acceptance-next-session.md) covers the completed million-position experiment, cached checkpoints, and the bounded dev-only diagnostic for the next session. The older speed/kernel results below remain historical context.

## Authoritative current results

- **Matched four-way speed:** [results and protocol](../results/20260914-qwen38-four-way-speed/README.md), implementation/results commit `058f4ad9`. All four columns come from one forward/reverse eight-cell campaign, twelve samples per configuration/context. Do not substitute earlier three-way or custom-only measurements.
- **Full three-arm quality:** [results and limitations](../results/20260913-qwen38-grouped-quality/README.md), commit `bac71888`. Target-only/native/custom each completed2,466 requests, four scored suites,500-prompt divergence and repeated64K token-ID diagnostics.
- **Custom kernel implementation/qualification:** [grouped verification](../results/20260913-qwen38-native-grouped-verify/README.md). Build06 remains experimental and opt-in.

| Context | Native MTP4 | Custom MTP4 | DFlash2 INT4 | DSpark Split-K |
|---|---:|---:|---:|---:|
|512|70.77|69.48|76.29|52.55|
|8K|63.97|63.45|60.76|42.95|
|32K|61.36|63.27|50.28|37.47|
|64K|55.12|57.93|45.13|31.14|

Median post-first streaming decode tok/s, B70/275W/C1. Same workloads; selected bundles retain documented runtime/capacity/batch differences. Custom is **+5.11% versus native at64K** in this matched campaign; the earlier+8.85% was a separate experiment. DFlash has the lowest64K whole-request latency for128 output tokens (55.33s vs custom56.10s) because of faster prefill. See the linked report for IQRs/TTFT and raw evidence.

## Decision / unresolved issue

**Do not promote the custom kernel yet.** HumanEval+ was82.93% versus native84.76% (three fewer passes: HumanEval/1, /19, /130). Other candidate/native quality changes were small: IFEval strict−0.18pp, GSM8K strict+0.08pp, MBPP+ +0.26pp. Native and target-only also differ and repeat nondeterministically; quality neutrality/equivalence is not established. No additional optimization or rerun is running.

If explicitly requested next: repeat/investigate those three code-task transitions across target/native/custom, distinguish formatting/runtime variation from semantic failures, and only then decide on further qualification or promotion. This is follow-up work, not a hidden unfinished implementation task. Full standard-publishable performance coverage (BetterBench20-pass, vLLM bench serve, concurrency/near-limit sweep) remains outside these development campaigns.

## Host and production closeout — 2026-09-14, historical

The launcher hash and stopped-service note in this section were superseded on 2026-10-11. Use the closeout at the top.


- Host: `inference-host`; all ML/compiler/evaluator execution stays there, not in Pi.
- Final check: no running Docker containers or render-device holders; power cap275000000 microwatts. The service is **stopped**, not secretly restarted for closeout.
- Persistent launcher: `/home/mike/inference/launchers/start-qwen38.sh`, SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4` unchanged.
- Default remains native MTP4,212992 context,C1,batch8192,FP8 KV, prefix caching and default thinking enabled; see [golden configuration](qwen38-b70-golden-config.md). Benchmarks deliberately disabled prefix caching/thinking. Experimental settings must not be mistaken for production defaults.
- Remote artifacts: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-qwen38-four-way-speed/` and sibling `20260913-qwen38-grouped-quality/`. Raw results/commands and failure history are committed; reproducible source-download URLs/hashes cover omitted download archives.

Older August handoffs/context measurements remain historical evidence, not the current next-session plan.
