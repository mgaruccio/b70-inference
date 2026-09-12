# DSpark draft-query attribution — development diagnostic

Purpose: identify the cost inside the earlier ~33.45 ms DSpark draft graph before choosing an optimization. This campaign does not alter the model, speculative depth, acceptance rules, or production launcher. Eager summed kernel durations are not graph latency or a throughput comparison.

## Fixed stack and test process

- Host: `inference-host`, Intel B70, 275 W; Glimmer stopped, no competing container/device workload.
- Image: `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4`, vLLM `73029d424`, corrected DSpark layer-norm overlay SHA256 `0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643`.
- Target: Qwen3.8-27B GPTQ INT4 sym G128, FP16 compute, FP8 KV. DSpark BF16 draft/KV, depth 7, greedy draft sampling, standard rejection, adaptive verification disabled. C1, capacity 65664, batch 2048, prefix caching off.
- Intentional diagnostic differences: eager mode, XPU graphs off, native profiler with shapes and temporary draft annotations. `ignore_frontend=false` avoids the known DSpark frontend profiler-control error; no runtime fix.
- Real public journey: reuse `../20260911-qwen38-step-profile-64k/run-step-profile.py`. Send the existing nonce-bearing 65536-token input, temperature 0, seed 42, 128 output tokens, ignore EOS. Start profiling only after the first nonempty streamed SSE; delay three iterations and capture five. Stop after 24 nonempty events; complete the stream.
- Expected: HTTP 200 profile controls, exact input/output token counts, normal length completion, no SSE or metrics errors, trace containing warmed decode rather than prefill. Inspect separate backbone, sequential sampling and context-KV phases. No tensor-to-host reads are added by annotations.
- Evidence: launch argv/config and source hashes, runtime identity, raw SSE/request/metrics, profiler trace and attribution, server log, host before/after. Large raw trace remains on the inference host; retain its path/hash locally.
- Cleanup: driver removes only its owned temporary container and verifies unchanged launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, 275 W power and stopped Glimmer. No production promotion.

Remote campaign directory: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-dspark-draft-profile`.

```sh
# Copy the two campaign scripts to the remote campaign directory.
# Existing sibling driver and worker-import shim are already staged there.
# Shim source: scripts/experiments/qwen38_step_timing_patch.py, staged as
# ../20260911-qwen38-target-verification/qwen38_step_timing_patch.py.
python3 profile-draft.py --out profile-01 > profile-01.console.log 2>&1
python3 summarize-draft.py profile-01/profile/rank0.*.pt.trace.json.gz > profile-01/draft-attribution.json
```

## Fresh primary-source research

- [Pinned DSpark model](https://raw.githubusercontent.com/vllm-project/vllm/73029d424/vllm/model_executor/models/qwen3_dspark.py): parallel noncausal block backbone followed by sequential Markov head. Full-vocabulary logits/Markov work must be measured separately, not assumed negligible.
- [Pinned DFlash model](https://raw.githubusercontent.com/vllm-project/vllm/73029d424/vllm/model_executor/models/qwen3_dflash.py): context KV precomputation and query-block attention are distinct; full-attention layers default noncausal unless overridden. Actual checkpoint has five full-attention layers, 32 query/8 KV heads, head dimension 128, vocabulary 248320, no sliding window. Runtime metadata remains authoritative.
- [Native XPU attention interface](https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/vllm_xpu_kernels/flash_attn_interface.py): small uniform causal queries can expand to native Split-K decode; noncausal queries are excluded. Hypothesis only: noncausal DSpark may use slower chunk-prefill. A future noncausal transform would need the same full parent KV length for every query, not the causal length decrement. No such change is implemented here.

## Local checks

All three scripts parse. A stdlib synthetic trace check passed nested phase/module attribution, CPU external-ID to device mapping, outside-scope exclusion and exclusive phase totals. These checks supplement, not replace, the real API run.

Native Lab preview/patch display was unavailable (`lab.sock` ENOENT); execution follows the user's explicit continuation authorization.

## Observed result

`profile-01` passed: 65536 input / 128 output tokens, length completion, no SSE/metrics errors, both profiler controls HTTP 200. The trace contains exactly five `execute_context_0(0)_generation_1(8)` annotations and five each of draft generation, backbone, sampling and context-KV scopes. There is no prefill in the captured sample.

Exclusive summed device work per generation step:

- **Noncausal attention operator: 21.710 ms** across five draft layers (main chunk-prefill kernel 21.701 ms).
- Remaining backbone: **5.543 ms**, principally BF16 GEMMs. Total backbone: **27.253 ms**.
- Sequential sampling: **6.155 ms**. Base vocabulary projection `[7,5120] × [5120,248320]`: **4.292 ms**; seven Markov projections `[1,256] × [256,248320]`: **1.536 ms**. These are subsets of sampling, not additive parent totals.
- Context-KV preparation: **0.208 ms**, outside generation.
- Backbone plus sampling: **33.409 ms**. This is qualitatively consistent with the earlier ~33.45 ms graph observation but is **not** a paired graph measurement or a throughput result.

Runtime confirms every draft layer is noncausal, no sliding window, 32 query heads / 8 KV heads, dimension 128, BF16 KV. First annotated metadata: query length 7, sequence length 65562. Native attention input shapes are `[7,32,128]` Q and `[227,1664,8,128]` K/V, block table `[1,40]`. The BF16 CUTLASS `XeFMHAFwdKernel` appears 25 times, five layers × five steps; this is the non-Split-K chunk-prefill route.

Attribution details: each device kernel is counted once using its CPU External ID and innermost draft phase/module. Divisor is five generation scopes. 3,089 kernels (53.095 ms total) lack a matching direct CPU ID; an independent runtime-launch correlation audit resolves all of them **outside draft scopes**, with zero unresolved launches. They remain excluded rather than guessed into a draft category. See `profile-01/draft-attribution.json` for exclusive totals, raw operator shapes, kernel names and audit.

Raw trace (retained remotely):

- `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260912-qwen38-dspark-draft-profile/profile-01/profile/rank0.1789188089953386895.pt.trace.json.gz`
- 4,307,960 bytes; SHA256 `8c379d097929ef1f9f09ddbc7831292d665a781622212bcf01de7f71860c0341`.

`profile-01/summary.json`, `host-before.json`, `host-after.json` and `container-cleanup.json` confirm successful request and owned-container removal, unchanged production launcher/power, no running containers and stopped Glimmer. Source overlay hash matches the corrected baseline. Raw request/SSE/metrics, exact launch command, console/server logs and runtime identity are retained alongside the analysis.

Final verification reran the summarizer on the real trace and asserted all five decode-only steps, phase counts, exact API token counts, successful cleanup and zero unlinked draft launches.

**Decision:** attention is the first justified optimization target. Next is a correctness-first native noncausal expanded-query Split-K experiment on this exact BF16 geometry. Every query must retain its full parent KV length; the causal decrement is invalid here. Then require fixed-config real-API correctness and unprofiled graph-mode A/B before any gain or promotion claim. No attention implementation has been changed in this campaign.
