# Shared-KV build: community-protocol development measurements

## Latest: matched 230 W rerun — passed

Both native S+M1 and our shared-KV build completed all four cells at **230 W**.
Five measured requests per cell followed generic and shape warmups. All 40 raw
SSE records validated; identical retained prompts and client were used in both
arms. Both runner exits report success, FULL graph runtime statistics were
recorded in both arms, and the candidate execution-evidence check passed.

| Input / output tokens | Native at 230 W | Shared-KV at 230 W | vs native | Published 230 W reference |
|---|---:|---:|---:|---:|
| 512 / 128 decode | 115.42 tok/s | **115.38 tok/s** | −0.04% | 112.65 tok/s |
| 8192 / 128 decode | 107.16 tok/s | **108.33 tok/s** | +1.09% | 103.63 tok/s |
| 130944 / 128 decode | 61.18 tok/s | **66.03 tok/s** | +7.92% | 62.52 tok/s |
| 8192 / 1 input tokens / TTFT | 1736.45 tok/s | **1736.55 tok/s** | +0.01% | 1696 tok/s |

Shared-KV is numerically +2.42%, +4.54%, +5.61% and +2.39% versus the published
figures. **Only our two local arms are matched-input/configuration comparisons.**
We matched the published power cap, not the full published serving configuration:
our runs retain context 212992, utilization .95, C1 and enabled prefix caching
and thinking. Cache hits were zero. The candidate seam accepts specific KV
shapes (including [152,1664,4,256]) and a [1,128] block table; changing context and
memory settings would require separate compatibility work, not a silent native
fallback. Historical community prompts also remain unavailable.

The only changes versus the earlier local setup are the 230 W cap and
`--cudagraph-metrics` in **both** arms. The vLLM
[CLI documents this flag](https://docs.vllm.ai/en/latest/cli/serve/#--cudagraph-metrics)
as recording graph dispatch modes and frequencies. Each arm retained 12 FULL
runtime-statistics rows, including five-token decode rows. This closes the prior
missing-table check for the **new run**, without changing the earlier failed
record or claiming a per-kernel trace.

Shared-KV decode ranges were 111.19–119.61, 69.93–112.75 and 63.85–73.40 tok/s.
Native ranges were 107.57–119.57, 66.08–111.58 and 52.37–67.99. At 128K,
whole-request medians were 174.529 s native and 174.393 s shared-KV, not a 7.92%
whole-request improvement. Accepted/proposed draft counts were native
508/552, 472/672, 496/616 versus shared-KV 509/548, 477/656, 501/600.
Streamed outputs matched on 13/15 decode pairs and 4/5 prefill pairs. Quality
qualification remains unresolved; sequential five-prompt arms are development
evidence, not a statistically established community ranking.

The original **275 W cap was restored and verified**, both owned containers
stopped, production was unchanged, and no new guarded kernel errors appeared.
The guard was copied privately with only its checked/reported power value and
power-error message changed from 275 W to 230 W; other safeguards were retained.

Executed on `inference-host`:

```sh
python3 -B /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-community-protocol/run-230w.py
```

New artifacts: `benchmark-230w.tar.gz` contains `native-230w-01/`,
`shared-kv-230w-01/` and `power-230w-01/` (private launchers/guard, initial and
restored host checks, and `power.json`). `comparison-230w.json` contains the
revalidated comparison; `run-230w.log` retains the original controller output.
`run-230w.py` is the paired controller. After extracting the archive:

```sh
python3 -B compare.py /tmp/b70-community-230w-verified/native-230w-01 \
  /tmp/b70-community-230w-verified/shared-kv-230w-01 > comparison-230w.json
sha256sum -c SHA256SUMS
```

## Earlier 275 W run — retained below with its original failure

Executed 2026-10-09. The first completed run mistakenly tested the existing native
launcher. The corrected run tests **our experimental shared-KV build**, reusing
that native run's exact measured prompts, public client and serving settings.

**Status:** all 20 candidate HTTP measurements completed and their raw SSE records
validate. The runner nevertheless exited 1 because its final evidence check could
not find a FULL-graph runtime statistics table. This failure is retained, not
waived or relabeled as a fully passing run. These are development measurements,
not the repository's standard-publication package, a community ranking, or a
quality-qualified release.

## Measured results

Five measured requests per cell, after generic and same-shape warmups. All 40
baseline/candidate records have the requested endpoint token counts and zero
reported prefix-cache hits. Numbers below are medians.

| Input / output tokens | Native baseline | Our shared-KV build | Matched-input difference |
|---|---:|---:|---:|
| 512 / 128 decode | 116.00 tok/s | **116.29 tok/s** | +0.25% |
| 8192 / 128 decode | 108.29 tok/s | **109.55 tok/s** | +1.16% |
| 130944 / 128 decode | 62.28 tok/s | **67.22 tok/s** | +7.92% |
| 8192 / 1 input tokens / client TTFT | 1928.02 tok/s | **1930.17 tok/s** | +0.11% |

Decode is `(completion_tokens - 1)/(end - first_generated)`; the one-output-token
cell measures input tokens divided by client TTFT, **not isolated engine prefill**.
At 130944 input tokens, median whole-request time was 160.705 s native versus
160.561 s shared-KV: the decode improvement does not imply 7.92% faster requests.

Shared-KV decode ranges were 112.09–120.59, 67.42–114.09 and 57.52–74.75 tok/s;
native ranges were 107.95–120.22, 68.12–112.60 and 60.20–69.24. The 8K and long
cells vary materially across prompts. These were sequential native-then-candidate
runs, not interleaved repeats; small differences are not established causal gains.

Aggregate accepted/proposed draft tokens:

| Decode cell | Native | Shared-KV |
|---|---:|---:|
| 512 | 508/552 (92.03%) | 509/548 (92.88%) |
| 8192 | 475/664 (71.54%) | 475/676 (70.27%) |
| 130944 | 501/600 (83.50%) | 494/612 (80.72%) |

Streamed reasoning/content matched on 13/15 decode prompt pairs and 4/5 prefill
pairs. Acceptance and output differences mean this is not an isolated fixed-work
kernel speedup. [HumanEval+ qualification remains unresolved](../20261009-qwen38-shared-kv-quality/);
no quality fix, neutrality claim, production promotion or full-quality rerun is
implied by these speed measurements.

## Published reference: context only

The cookbook S+M1 submission reports 112.65 tok/s (512), 103.63 (8192),
62.52 (130944), and 1696 input tok/s per TTFT (8192/1). Our shared-KV observations
are numerically +3.23%, +5.71%, +7.51% and +13.81%, respectively, but **these are
not apples-to-apples community-comparable gains**: historical prompts/raw SSE
are not published, and machines, power and serving settings differ. No
community-best claim follows from this development run.

## Build and serving configuration

Baseline: `/home/mike/inference/launchers/start-qwen38.sh`, SHA256
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

Candidate: a private launcher copy with only the shared-KV read-only mounts,
worker import hook and four environment additions. Production was not edited.
`compare.py` checks the actual container commands, environment and bind mounts.

- Shared-KV library SHA256:
  `e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec`.
  Reused build-06 binary/source: [shared-KV provenance](../20261009-qwen38-native-shared-kv/).
- Added environment: `PYTHONPATH=/experiment`, `B70_STEP_TIMING=1`,
  `B70_GROUPED_SERVING=1`,
  `B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so`.
- One B70; Qwen3.8-27B GPTQ symmetric G128; FP16 target, FP8 KV.
- MTP4, draft INT4 S+M1, mixed-GDN-v5, XPU graphs enabled.
- 212992 configured context, utilization .95, batch-token budget 8192, C1.
- Prefix caching and thinking **enabled** in both arms; measured cache hits zero.
  Client requests temperature 0 and fixed lengths with `--ignore-eos`.
- 275 W cap, CPU boost off, all 16 CPU maximum-frequency caps 3801000 kHz.
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.

The published reference uses 230 W, context 131072, utilization .88 and caching
disabled. Both local arms used the same boot, configuration and checked source
hashes. The owned candidate container stopped successfully; postconditions show
no running containers, unchanged production launcher/host settings and no new
guarded kernel errors.

### Why the runner exited 1

The expected library was installed and an eligible Q=[5,24,256] dispatch was
logged during startup, before graph capture completed. No `unsupported-q5` event
was logged. FULL graph capture completed; all four HTTP benchmark cells finished.
The final reused `_candidate_execution_evidence` check additionally requires a
log table matching `| FULL |`, which was absent. The engine configuration records
`cudagraph_metrics=False`. This run therefore **does not independently prove FULL
graph replay during measured requests**; the eligible dispatch log is startup
evidence, not a per-request execution trace. The overlay's diagnostic probe is
bounded to its first two five-row calls and stops once eligibility is logged;
absence of an unsupported event cannot independently exclude later fallback.

`candidate-execution-evidence.json` retains `full_graph_capture_seen=true` and
`full_graph_run_seen=false`; `exit.json` retains `success=false`. No measurements
were rerun, dropped, or substituted to hide the failure. The original stdout and
traceback are in `shared-kv-run.log`.

## Public client and exact inputs

Cookbook source Git blobs:

- `b70-realworld-context-harness.py`: `a4ca3c28c2d87436f80e93c42cb9f9712f3765e7`
- `b70-generate-exact-prompts.py`: `645c021089c82d671a0c332497cfe2ec503c3f57`

The official generator ran inside the image against `/model`. Six entropy-first
prompts per decode length provide one shape warmup and five measured samples.
A separate 8K prompt set prevents prefill from reusing cached decode prefixes.
Candidate prompt files are byte-identical to the baseline's; all 20 measured
message hashes match in order and are unique across cells.

The original native attempt failed during generic warmup because the client
recognized `delta.reasoning_content`, while the server emits `delta.reasoning`.
Both measured arms use the same one-line parser compatibility addition:
`or delta.get("reasoning")`. Request parameters, timing placement, token counting
and formulas are unchanged. The original failed attempt is also archived.

## Commands, verification and artifacts

Real public-boundary tests on `inference-host`:

```sh
python3 -B /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-community-protocol/run.py
python3 -B /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-community-protocol/run-build.py
```

Each cell uses the archived public client with `--mode context --target <input>
--output <output> --budget 8192 --reps 5 --model qwen38 --root
http://127.0.0.1:8000 --full-output-warmup --ignore-eos`. Output directories and
the private launcher refuse overwrite; these exact commands are retained as
executed, not idempotent rerun commands. Benchmark dependencies stay on the host.

Offline verification, after extracting the two archives:

```sh
python3 -B compare.py /tmp/b70-community-protocol-verified/current-serving-02 \
  /tmp/b70-shared-kv-community-verified/shared-kv-serving-01 > comparison.json
sha256sum -c SHA256SUMS
```

Observed: all 40 measured records validated, including raw SSE completion,
endpoint usage, first-generated timestamp, reconstructed streamed text, timing
formulas, medians, prompt hashes, added-only candidate configuration and cleanup.
This verifies recorded measurements; it does not convert the graph guard or
quality qualification to a pass.

- `comparison.json`: recomputed results, dispersion, acceptance, output matches
  and the preserved failed candidate-run status.
- `summary.json`: original **native baseline** summaries, unchanged.
- `community-benchmark.tar.gz`: native `current-serving/` failed warmup and
  `current-serving-02/` completed baseline, public source and license.
- `shared-kv-benchmark.tar.gz`: `shared-kv-serving-01/`, including exact prompts,
  raw SSE, per-request records, commands, private launcher, container/server
  metadata, build contract, evidence failure, host guards, cleanup and runners.
- `shared-kv-run.log`: original candidate stdout and final traceback.
- `run-build.py`, `run.py`: candidate wrapper and common public-protocol controller.
- `compare.py`, `SHA256SUMS`: reproducible offline verification and integrity checks.

References:
- [Public S+M1 submission](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/submissions/vllm-qwen38-mtp4-draft-int4.json)
- [Published 8K and long-context cells](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27b/QWEN38-VLLM-XPU.md)
- [Public benchmark client](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/benchmarks/b70-realworld-context-harness.py)
