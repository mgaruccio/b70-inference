# Intel 1.5 Qwen3.8-27B / EAGLE-4 engineering comparison

**Regression diagnostic, not completion of the requested comparison. EAGLE previously ran much faster; the Intel FP8 port lost optimized dispatch because of cache-layout and precision-path changes. The measurements below document that regressed port, not the drafter's capability. Restoring and validating the intended fast paths, then rerunning the comparison, remains open. Preserve these observations as failure evidence; do not use them to conclude that EAGLE itself is slower than native MTP.**

## Question and scope

Compare Intel's published release recipe with native MTP against the same target with EAGLE-4, making the minimum compatibility changes. This is an internal engineering evaluation of the installed builds, not a claim about EAGLE-4's best achievable performance.

Only Intel's public recipe was used as the baseline. Source: [Intel release 1.5](https://github.com/intel/llm-scaler/releases/tag/vllm-1.5), [published README](https://github.com/intel/llm-scaler/blob/vllm-0.26.0-b2/vllm/README.md), Intel commit `77c547307e0f0289044c8a610bd9603c19c2c483`.

## Fixed environment and explicit differences

- One Intel Arc Pro B70, PCI ID `0xE223`.
- Intel image digest `sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`; Torch 2.12.0+xpu, Triton 3.7.1, Transformers 5.8.0, vLLM `0.26.1.dev0+g568afb3a1.d20260907.xpu`.
- Official `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 18 shards, 1199 indexed tensors. Qwen3.8 and TP1 are explicit adaptations of Intel's template, not a verbatim reproduction of its example.
- Target: TP1, `sym_int4`, FP16 compute, `fp8_e4m3` KV, FP16 Mamba state, max length 40000, max batched tokens 8192, max sequences 64, requested block size 64, eager execution, prefix caching disabled.
- The runtime expands the target block size to **1024**. Both arms require the same read-only `/dev/dri/by-path` container mount for oneCCL.
- Published baseline: native `qwen3_5_mtp`, four speculative tokens, unmodified released engine, legacy model runner, memory fraction **0.8**.
- Matched memory control: the same native MTP at **0.9**.
- EAGLE: source commit `17d1790`, engine hooks `6cae86c` over Intel-patched base `d0c19eb`, V2 runner, CUSTOM target attention, FP16 draft KV, draft window 4096, static verify budget 15 / width 16 / top-k 4 / depth 14. Memory fraction **0.9** was explicitly approved after the unchanged 0.8 recipe failed to fit. No compiler, driver, image, or target precision upgrade was used.
- Therefore this is **not a drafter-only causal comparison**: runner and attention implementation differ. The matched 0.9 control separates the memory setting, not these other differences.

## Memory qualification

The 0.8 EAGLE attempt loaded both models (21.29 GiB) but failed startup: 2.32 GiB available KV versus 5.25 GiB required. Runtime diagnostics confirmed correct separate cache types and enabled hybrid caching, not accidental full-context draft caching:

| Component | Required allocation |
| --- | ---: |
| 48 FP16 Mamba groups, 16 state blocks each | 1536 MiB |
| 16 FP8 full-attention groups, 40 blocks each | 1280 MiB |
| FP16 draft sliding-window group, 16-token blocks padded to 2 MiB | 2562 MiB |
| Total | 5378 MiB |

The approved 0.9 run reported 5.51 GiB available KV, capacity 41,978 tokens, and 1.05x concurrency at 40K. The 0.8 MTP baseline loaded 18.04 GiB and reported 3.83 GiB available KV and 83,636-token capacity. These capacities are runtime estimates, not multi-request load-test results.

## HTTP probe observations

Host-local clients used identical saved token prompts, greedy temperature 0, seed 0, concurrency 1, ignored EOS, and 512 returned output token IDs. Every request retained raw SSE, usage, timings, and metric deltas. Protocol canaries/warmup were excluded from measured rows. The repeated-input prompts are synthetic workload fixtures, **not a model-quality benchmark**.

| Arm / sample | Input tokens | TTFT (s) | Post-first-token decode (s) | Full request (s) |
| --- | ---: | ---: | ---: | ---: |
| MTP 0.9, median of 10 | 8192 | 4.115 | 12.284 | 16.400 |
| EAGLE 0.9, one smoke request | 8192 | 52.707 | 128.615 | 181.322 |
| MTP 0.9, median of 10 | 32768 | 20.515 | 11.301 | 31.817 |
| EAGLE 0.9, first 32K request | 32768 | 773.938 | 376.753 | 1150.691 |
| EAGLE 0.9, warmed 32K repeat | 32768 | 773.847 | 430.726 | 1204.573 |

The EAGLE warmed repeat confirms that the long-context slowdown is not explained solely by first-use compilation. These rows have unequal sample counts and different generated sequences; they do not establish population-level speed ratios.

## Intel benchmark CLI

Primary performance measurement uses the released `vllm bench serve` CLI: random input length 32768, output length 512, range ratio 0, seed 0, temperature 0, ignored EOS, request rate infinity, concurrency 1. Commands and raw detailed JSON are retained.

Executed inside each arm's container after sourcing `/opt/intel/oneapi/setvars.sh --force` and `/opt/intel/oneapi/ccl/2021.15/env/vars.sh --force`:

```sh
vllm bench serve --model Qwen3.8-27B \
  --tokenizer /llm/models/Qwen3.8-27B --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port 8000 --backend vllm \
  --request-rate inf --max-concurrency 1 --ignore-eos --trust-remote-code \
  --dataset-name random --num-prompts "$N" --num-warmups "$W" \
  --random-input-len 32768 --random-output-len 512 --random-range-ratio 0 \
  --temperature 0 --seed 0 --save-result --save-detailed \
  --result-dir /tmp/intel15-bench --result-filename "$RESULT" \
  --metric-percentiles 50,90,99
```

MTP: `N=10 W=1 RESULT=benchmark.json`; EAGLE: `N=1 W=0 RESULT=benchmark-bounded.json`.

- MTP 0.9: **10 successful / 0 failed**, one explicit warmup; median TTFT 20,597.35 ms, median TPOT 15.59 ms, P90 TPOT 20.74 ms, output throughput 17.42 tokens/s including prefill. Acceptance 83.75%, acceptance length 4.35. The CLI's ITL is chunk-level and must not be substituted for token-normalized TPOT.
- Published MTP 0.8: **10 successful / 0 failed**, one explicit warmup; median TTFT **20,594.60 ms**, median TPOT **15.81 ms**, P90 TPOT **20.88 ms**, output throughput **17.36 tokens/s** including prefill. Acceptance **83.80%**, acceptance length **4.35**. The earlier one-request smoke result is retained separately, not substituted for this final run.
- EAGLE 0.9: **1 successful / 0 failed** on the already-warmed server, zero additional requested warmups. TTFT **773,719.76 ms**, TPOT **3744.00 ms**, full measured run **2686.90 s**, output throughput **0.191 tokens/s** including prefill. Acceptance **0.27%**: 20 accepted / 7365 proposed tokens across 491 rounds (acceptance length 1.04). The CLI also runs an initial test request outside its measured result. This is a bounded single observation, not the originally planned ten-repeat paired protocol.

The CLI cells share workload dimensions and seed, but unequal sample counts do not establish identical random prompt contents. The separate saved-token HTTP probes provide the identical-input comparison. Different warmup counts, unequal repetition counts, runner differences, and dispatch differences prohibit interpreting the CLI numbers as an isolated drafter-model speed ratio.

## Correctness evidence and limits

- Native cache/attention qualification: **52 passed, zero skips**, exact Intel image on B70, including FP8 scales/bytes, accepted-row compaction, and actual Qwen TP1 attention shape.
- Integrated CPU/interpreter checks at `17d1790`: **123 passed, 7 skipped**. These skips are not native passes.
- Engine hook tests in the pinned image: **90 passed** without GPU execution.
- EAGLE and native MTP answered the simple arithmetic, Python-add, and JSON chat canaries correctly. This is narrow semantic evidence, not broad quality qualification.
- Ordinary-target 32K outputs were identical across ten runs. MTP 0.9 matched that sequence in 7/10 runs; the other three first diverged at indices 3 or 244. EAGLE's first 32K output diverged at index 45, and its warmed repeat at index 1 (zero-based). Thus EAGLE has **not demonstrated exact greedy equivalence**; the cause is not established.
- Ordinary-target 8K outputs themselves produced nine distinct sequences in ten identical greedy requests. MTP 0.9 produced ten distinct sequences. EAGLE also differed. Baseline nondeterminism prevents treating every exact-match failure as an EAGLE defect, but does not excuse candidate errors.
- A native extreme-value FP8 fixture exposed cancellation-amplified FP16 attention-coefficient rounding. Independent replay matched all 1408 native output elements exactly. The byte-boundary test was conditioned without altering encoded byte coverage or dense tolerances; the original failure remains retained. This did not establish model-quality equivalence.

## Confirmed dispatch limitations

Read-only review found no concrete prefill-classification bug after the Intel InputBatch compatibility correction. EAGLE tree execution and the V2 runner are active.

1. The specialized B70 draft launch requires a leading cache stride of 1,835,008 elements. Intel's actual padded cache stride is 1,048,576. The guard correctly declines the specialized launch, so generic Triton draft row attention runs.
2. The native FA2 target path is explicitly unavailable for FP8 KV in this EAGLE adapter. Ordinary target attention and FP8 tree verification use generic Triton paths instead.
3. Initial inference did compile kernels, but later 32K requests remained slow without corresponding new compile warnings.

These are implementation/dispatch limitations of this tested port. They prevent attributing its slowdown to the drafter model alone. No unverified kernel optimization was introduced merely to improve the result.

## Artifacts, reproduction, and cleanup

- Remote host run root: `inference-host:/home/mike/b70-evals/20261025-intel15-eagle4/` (not a coordinator-local path).
- Coordinator artifacts: `/tmp/intel15-eagle4-comparison/`.
- Relevant retained run directories: `native-mtp-by-path/`, `ordinary-reference/`, `mtp-mem90-cell1/`, `mtp-published-final/`, `eagle-initial/`, `eagle-cache-diagnostic/`, `eagle-mem90-initial/`, `eagle-mem90-prefillfix/`. Failed-startup logs are copied from the remote host into the same-named coordinator directories; `cache-diagnostic/` is the separate diagnostic-code directory, not its runtime log directory.
- Raw CLI results included beside this report: [published MTP](mtp-published-benchmark.json), [90% MTP control](mtp-mem90-benchmark.json), [90% EAGLE](eagle-mem90-benchmark.json). These are unmodified copies of the CLI outputs, including generated synthetic text and errors. Full server/SSE evidence remains at the retained run roots above.
- `launch-native-mtp.sh MODEL_DIR LABEL [mtp|none] [0.8|0.9]`; `launch-eagle4.sh MODEL_DIR DRAFT_DIR RUN_DIR LABEL [0.8|0.9]`. Defaults preserve 0.8. Candidate staging requires the exact source and engine patch described above; these are not standalone public deployment scripts.
- The original ten-repeat MTP/EAGLE/EAGLE/MTP protocol is **not completed**. After the observed long runtimes and the user's delegation of that decision, this was narrowed to a bounded engineering comparison; retain that limitation rather than presenting a shortened run as the original plan.
- Final host checks verified no running containers, no listener on port 18110, no leftover readiness client, and 74 GiB free disk space. The recovered 5GHz Wi-Fi connection remained active; its persistent preferred `flint-5ghz` profile has autoconnect priority 100, with the original `flint` fallback unchanged. No inference deployment was left running.

EAGLE-4 is source-available, not OSI open source. This report does not distribute model weights or EAGLE source, or establish commercial licensing permission. Implementation commits remain in their source repositories; no EAGLE upstream push is authorized.
