# Qwen3.8 DFlash2 context-boundary fix artifacts

**PASS: native validation and both BF16 / partial RTN INT4 API arms.** This is a correctness regression campaign, not a new full performance comparison or a production promotion. The [original benchmark package](../20260909-qwen38-dflash2-rtn-standard/) retains the pre-fix measurements and both original crashes.

## Change and scope

The scheduler can shorten the final DFlash K7 verification block to fewer than eight query tokens. The pinned XPU GDN kernel still requires eight rows because its speculative-state table has eight columns. The [overlay](../../scripts/patch-vllm-qwen38-xpu-boundary.py) pads only temporary internal GDN projection/output buffers, retains the full state table and previous accepted count, and publishes only the real output prefix. It does not pad the prompt, alter the scheduler/rejection policy, lower the context limit or change target weights.

This fix is deliberately restricted to **pure C1 speculative calls in vLLM `73029d424` / XPU kernels 0.1.14.1**; it fails closed on source drift. Ordinary full-width calls remain unchanged. Mixed and multi-request partial blocks are not generalized by this overlay.

## Native state validation

Both runs used the actual Arc Pro B70 and installed native XPU kernel in the pinned serving image, with the real target's tensor dimensions. FP16 matches target serving compute; the BF16 test is supplemental dtype coverage, not a separate model-quality evaluation.

| Projection/convolution dtype | Partial cases | Continuations | Result |
|---|---:|---:|---|
| FP16 | 224 | 896 | PASS |
| BF16 | 224 | 896 | PASS |

Coverage: real lengths 1–7, previous accepted counts 1–8, four physical/metadata padding variants, zero/random future suffixes, poisoned physical padding, and continuation after **every** valid accepted prefix. Count 1 includes the target/bonus token and covers zero accepted drafts. Real outputs/z, selectable rolling convolution history windows and SSM checkpoints match native full-width execution exactly (`rtol=0`, `atol=0`). Inputs, unowned state and output tails remain intact. See [FP16 log](native-v2/float16.log), [BF16 log](native-v2/bfloat16.log) and [executed test source](native-v2/check-qwen38-xpu-boundary-state.py).

The first test attempt failed **before API testing** because its oracle treated the convolution cache like per-token SSM checkpoints. In this kernel version convolution instead uses one rolling history line; dummy future rows legitimately differ. The corrected oracle compares all history windows selectable by real accepted prefixes and still executes every continuation. The runtime overlay was unchanged and no numerical tolerance was relaxed. The [failed original log](native-float16.log) and [original test source](code/scripts/check-qwen38-xpu-boundary-state.py) are retained; use `native-v2/` for the corrected test.

## Real public-API regression

| Gate | BF16 DFlash | Partial RTN INT4 DFlash |
|---|---|---|
| 512 / 8192 / 16384 / **32640** input + 128 output | 1 warmup + 6 measured per point: PASS | 1 warmup + 6 measured per point: PASS |
| Nearby 32634–32639 input + 128 output | 6/6 PASS | 6/6 PASS |
| 32768 / 65536 / 120000 / 160000 input, plus 128 output | 4/4 expected context HTTP 400 | 4/4 expected context HTTP 400 |
| Existing canary / finite-boundary / functional gates | 3 / 131 / 8 PASS | 3 / 131 / 8 PASS |

Each arm completed **34 valid forced-length streams**, including warmups and nearby cases, plus four expected HTTP rejections: **68 successful streams and eight correct rejections** in total, in addition to the existing gates. Each original crashing warmup request payload was replayed unchanged; all fixed-arm request payloads also match across arms. Successful streams require exact input/output usage, `[DONE]`, `finish_reason=length`, no SSE errors and healthy server checks. The same graph configuration `[1,2,4,8]` is enabled; graph capture/replay evidence is in both server logs. Neither fixed server log contains the former assertion or a traceback. Strict rejection, prefix cache off and thinking off remain unchanged. This does **not** establish universal greedy-output identity or quality equivalence.

## Environment, commands and reproduction

- Host: `inference-host`, one Intel Arc Pro B70, 275 W cap, isolated disposable `qwen38` container bound to `127.0.0.1:8000`.
- Image: `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`; vLLM `0.28.1rc1.dev278+g73029d424`; XPU kernels `0.1.14.1`; Torch `2.13.0+xpu`.
- Same GPTQ INT4/G128 target with FP16 compute, FP8 target KV, BF16 draft KV, DFlash K7, legacy V1, server C1 and max context **32768**. The RTN checkpoint is only partially INT4; other draft tensors remain BF16.
- Full predeclared journey, primary-source research and constraints: [fix process](../../docs/qwen38-dflash2-boundary-fix-20260909.md).
- Canonical host campaign: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-dflash2-boundary-fix/`.
- Exact native argv: [FP16](native-v2/float16.command.txt), [BF16](native-v2/bfloat16.command.txt). Both run the corrected test in an isolated `--network none` XPU container after applying the overlay.
- Exact API argv: [BF16](bf16-api.command.txt), [INT4](int4-api.command.txt). The [executed API runner](code/scripts/experiments/qwen38_boundary_api.py) uses the existing `DFlashCell`, `/tokenize` and streamed `/v1/completions`, then the unchanged long-context client plus six nearby requests. Cold-client argv: [BF16](bf16-api/long-context-console.command.txt), [INT4](int4-api/long-context-console.command.txt).
- [Initial launch script](run-command.sh) preserves the failed first native gate. [Resume script](resume-api-command.sh) uses the corrected native gate before the two API arms. **Do not overwrite this campaign by replaying its paths:** reproduce with a new campaign/output directory and the same read-only model layout, pinned image and archived sources.

## Artifact index and cleanup

- BF16: [cell summary](bf16-api/summary.json), [cold summary](bf16-api/long-context/summary.json), [all exact-token requests and SSE](bf16-api/long-context/points/), [nearby requests/results](bf16-api/nearby/), [launcher](bf16-api/launcher.sh), [environment](bf16-api/collect_env.txt), [server log](bf16-api/server.log).
- INT4: [cell summary](int4-api/summary.json), [cold summary](int4-api/long-context/summary.json), [all exact-token requests and SSE](int4-api/long-context/points/), [nearby requests/results](int4-api/nearby/), [launcher](int4-api/launcher.sh), [environment](int4-api/collect_env.txt), [server log](int4-api/server.log).
- [Lead CPU test log and exact commands](lead-cpu-tests.txt): all **35 focused tests pass, no skips**, including the stopped-image source check. Syntax checks also passed for all ten requested source/test files. These supplement rather than replace native and API execution.
- `code/` is the frozen initial campaign source; `native-v2/` separately preserves the corrected native oracle. The archived executed overlay, native-v2 test and both serving runners match the delivered source byte-for-byte. Python caches are omitted. Raw failures are not rewritten as successes.
- [Exit codes](cell-exits.tsv): both corrected native runs and both API arms exit 0; the resumed pipeline completed in 23m18s with exit 0. API cell timestamps are recorded in their summaries (2026-09-09 UTC).
- [Final cleanup](cleanup.txt): persistent launcher hash unchanged, cap still 275000000 microwatts, Glimmer stopped, and no disposable containers left running. Both API cell summaries independently record launcher/power cleanup checks.

The persistent launcher must retain SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; `glimmer-tb21-prefix-c8` must remain stopped. No experimental configuration is promoted by this artifact package. The earlier speed ratios remain explicitly **pre-fix**, development-tier results with unresolved quality/publication gates.
