# Guarded K64 chunk-budget development screen — 2026-10-11

Status: completed. Maintained prefill guard deployed and verified; both chunk candidates rejected. Guarded production remains batch8192/prefix-on. No chunk-budget promotion.

User-approved scope: resolve the prefill-classification prerequisite, then investigate option 1 (8192 versus 8320/6656). This is a development run, not a standard-publication or quality-neutrality claim.

## Observed result and decision

The real host command completed with exit0. All132 distinct-prefix guard probes passed finite-logprob and exact-token-count assertions, including5,69,133 and49925. Guard, three cold cells and restored production each passed arithmetic/JSON/Python-AST canaries (15 canary responses total).

| Batch budget | 16k median TTFT | Change | 32k median TTFT | Change |
|---:|---:|---:|---:|---:|
|8192|8.7315s|baseline|19.9418s|baseline|
|8320|8.8215s|+1.03% latency|20.2012s|+1.30% latency|
|6656|8.7905s|+0.68% latency|20.0625s|+0.61% latency|

Five measured requests/length/arm, after excluded warmups. Both candidate ranges were entirely slower than baseline at both lengths. Neither cleared the first primary gate, so **baseline-after,98k,212991+1 and128-output decode extension were correctly skipped**. Keep8192; this screen does not establish the page-alignment mechanism or anything about agent-task speed/quality.

Local evidence: [`run.log`](run.log), [`comparison.json`](comparison.json), [`restored-public-probe.json`](restored-public-probe.json) and [`restored-advertised-probe.json`](restored-advertised-probe.json). Comparison values are derived from stdout rounded to four decimal places, not presented as full-precision source JSON. The restored service answered HTTP200 through both the LAN and normal advertised `inference-host:8000` endpoint; each saved five-token completion has a finite logprob and exact5+1 usage. The initial12-second LAN probe timed out; a subsequent180-second-budget probe completed in5.831146s and the advertised-endpoint probe in2.328695s. No cause is inferred from that initial timeout. First per-cell compile/warm requests near61s remain in `run.log` and are not timed samples.

**Artifact limitation:** the runner wrote complete responses, recipes, environment and server logs under `/home/mike/b70-evals/20261011-qwen38-chunk-alignment/run-01`. SSH archive collection/readback timed out twice (150s and120s). The first download was empty/invalid; the repeat remained unverified. The local download was removed rather than committed as verified evidence. Full host-artifact readback is not claimed. This is not a complete publishable benchmark package. Public model inference was independently verified; failed SSH inspection/transfer is not treated as evidence that inference failed. Temporary local SSH transport/control files were removed; no persistent network or SSH configuration changed. No additional benchmark is running.

## Baseline and intentional differences

Single B70 on `inference-host`, pinned XPU image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`, Qwen3.8-27B GPTQ-Int4 with native MTP4 and draft INT4 S+M1, FP8 KV, C1, max context 212992, utilization .95 and K64 attention overlay. Power readback after the user reboot: 275000000 microwatts. No Docker workloads or render-device holders were present at deployment. Around 15 GiB host disk space remained; no weights or compiler installs are needed.

The maintained guard is applied after the five existing patches. Source hash `d26eb73c90a61c9f4f0056c7009cbe8c5a758d22a4db7c3ab8bf3ec28d73e66c` was copied to the launcher's existing cookbook patch path; host readback matched. Guarded launcher hash: `5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d`. Previous launcher was backed up as `start-qwen38.sh.before-prefill-guard-20261011`. The older `baa464…` hash identifies the historical trial artifact, not the new maintained wrapper.

Only batch-token budget differs among the timed cells. All use the same exact token arrays and sampling. Prefix caching is **off** for cold timing and vLLM then forces Mamba cache mode `none`; this is deliberately different from production prefix-on/`align`. K64 remains mounted in every cell. One-token non-streaming elapsed time is the legacy TTFT proxy, not engine-only prefill time. All timed requests additionally request one logprob so finite-output checks are real rather than inferred from HTTP 200.

The effective hybrid attention page measured previously is 1664 tokens, not merely nominal block size 64. Thus 8320 = five pages and 6656 = four pages. Whether changing the scheduler budget reduces boundary work is an unproven hypothesis; this run does not instrument scheduled chunks or establish that mechanism.

## Defined real-system test process

Preconditions: idle host, no running Docker containers, 275 W unchanged, exact K64 library hash, maintained guard installed, guarded production launcher still batch8192/context212992. Runtime and model execution stay on the inference host, outside Pi.

Public boundary: real `/v1/models`, `/v1/completions`, and `/v1/chat/completions` HTTP requests. The runner starts sequential disposable cells from copies of the maintained launcher and retains their exact recipes and logs.

1. Prefix-on/align guard cell: exact-token prompt lengths 1–128,133,197,261,49925, one greedy output token, a distinct first-token prefix for every length (no reused prefixes), finite token logprobs, exact usage counts, and zero reported cache hits. Arithmetic (`42`), JSON (`{"ok":true}`), and Python-AST `add(a,b)` canaries must pass without stripping fences or executing generated code.
2. Cold baseline8192, candidate8320, candidate6656: three boundary probes and the same canaries; excluded 16k/32k warmups; five repetitions per exact length, alternating lengths, identical prompts across arms. No cache hits allowed. Save each response and timing; preserve failures and outliers.
3. Reject a candidate unless both 16k and 32k median TTFT beat baseline. If any initially pass, repeat baseline and require a win against both bracketing baselines before extension. This reduced sequential screen is not a confidence interval or acceptance of quality neutrality.
4. Only qualifying candidates and a fresh8192 control extend to 98k (warmup plus five measurements), three streamed128-output decode checks and one legal212991+1 capacity probe. Report any decode regression over5%; no automatic promotion.
5. Stop owned experiment containers, restore the maintained guarded8192 production launcher with prefix caching enabled, verify model capacity and the three canaries, and retain the restored-service log. Do not restart the historical Glimmer service or alter power, clocks, network configuration, or other workloads.

Executable host command after source upload:

```sh
python3 /tmp/qwen-chunk-guard-20261011/run.py \
  --output /home/mike/b70-evals/20261011-qwen38-chunk-alignment/run-01
```

The runner refuses output reuse. That directory retains source, summary/per-request responses, exact launchers, server logs, models, environment, and any streaming SSE records. Remote-access diagnostics used a temporary Wi-Fi-bound SSH transport with normal authentication/host-key verification; no persistent networking/SSH configuration was changed.

## Supplementary verification completed

```sh
python -m unittest discover -s tests -p test_qwen38_prefill_guard.py -v
python -m unittest discover -s results/20261011-qwen38-chunk-alignment -p test_run.py -v
bash -n scripts/start-qwen38.sh
```

Observed: five guard tests and three campaign gating/restore tests passed; shell syntax and campaign AST syntax passed. The selective review's extension blocker was fixed: no long control arm runs unless at least one candidate beats both baselines. Temporary cells bind localhost only; structured records distinguish effective Mamba mode `none` from production `align`. Four ad-hoc client checks covered finite/missing-NaN rejection, valid canaries, and retaining the invalid-JSON response before failure. These mocked/source checks do not substitute for the real B70 gate.

## Fresh primary research

- https://github.com/vllm-project/vllm/pull/53059 — prefill must be excluded from shape-only uniform-decode classification; preserve intentional forced-decode overrides.
- https://raw.githubusercontent.com/AnnoyingTechnology/intel-arc-b70-llm-inference/main/docker/patches/patch_uniform_decode_prefill.py — validated anchored transformation; the maintained wrapper adds fail-closed replay checking and tests.
- https://docs.vllm.ai/en/latest/configuration/optimization/ — batch-token budget trades prefill/TTFT against scheduling and memory; larger is not automatically faster. Current documentation is not evidence of support in the pinned image.
- Existing deployed K64 evidence and next-wave decision: [research record](../20261011-xe2-head256-k64/RESEARCH.md).
