# Qwen B70 — next session

Updated 2026-09-14. Session work is complete; no benchmark or evaluator is left running. Production was not promoted or modified.

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

## Host and production closeout

- Host: `inference-host`; all ML/compiler/evaluator execution stays there, not in Pi.
- Final check: no running Docker containers or render-device holders; power cap275000000 microwatts. The service is **stopped**, not secretly restarted for closeout.
- Persistent launcher: `/home/mike/inference/launchers/start-qwen38.sh`, SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4` unchanged.
- Default remains native MTP4,212992 context,C1,batch8192,FP8 KV, prefix caching and default thinking enabled; see [golden configuration](qwen38-b70-golden-config.md). Benchmarks deliberately disabled prefix caching/thinking. Experimental settings must not be mistaken for production defaults.
- Remote artifacts: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-qwen38-four-way-speed/` and sibling `20260913-qwen38-grouped-quality/`. Raw results/commands and failure history are committed; reproducible source-download URLs/hashes cover omitted download archives.

Older August handoffs/context measurements remain historical evidence, not the current next-session plan.
