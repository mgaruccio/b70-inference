# B70 + Qwen 3.8 Benchmarking Standards

Status: Working standard  
Scope: Intel Arc Pro B70 inference work, initially Qwen 3.8 27B and speculative-decoding optimizations

## 1. Purpose

This document defines the minimum benchmark and reporting standard for performance work in this repository.

The goals are:

- make results reproducible on another B70;
- make B70 results directly comparable with CUDA, ROCm, and other XPU results;
- distinguish real inference gains from prompt mix, cache effects, thermal variation, or benchmark noise;
- measure the long-context behavior that matters for this deployment, not only short-context peak throughput;
- make speculative-decoding results meaningful by reporting acceptance behavior as well as tokens/sec;
- explicitly measure quality whenever an optimization is not mathematically lossless.

The canonical result is the raw benchmark artifact plus its exact configuration. Social posts, screenshots, and README tables are summaries of that artifact, not the source of truth.

Normative terms **MUST**, **SHOULD**, and **MAY** are used in their usual standards sense.

## 2. Benchmark tiers

### 2.1 Development run

For local iteration only. Any convenient reduced benchmark is acceptable.

Development numbers MUST NOT be presented as community-comparable results.

### 2.2 Standard publishable run

A publishable result MUST include:

1. paired baseline and candidate configurations;
2. BetterBench full run;
3. `vllm bench serve` results;
4. the repository long-context sweep;
5. speculative-decoding metrics when speculation is enabled;
6. environment capture and exact launch arguments;
7. raw machine-readable output.

### 2.3 Quality-sensitive run

Required in addition to the standard run whenever the optimization can change target-model output, including relaxed/cascade acceptance.

It MUST include the quality and output-divergence tests in Section 9.

## 3. Baseline and candidate rules

Every optimization MUST be tested against the current best known baseline on the same machine.

The baseline and candidate MUST use the same:

- GPU and power limit;
- host CPU and memory configuration;
- model revision;
- target quantization;
- KV-cache dtype;
- maximum model length;
- prompt set;
- sampling parameters;
- output length;
- prefix-caching policy;
- software stack, except for the component intentionally changed;
- ambient/thermal operating assumptions as far as practical.

Only the variable under test SHOULD change.

If multiple settings must change together, they MUST be reported as a bundle rather than attributing the gain to one component.

A result SHOULD be expressed both as absolute performance and as delta from baseline.

Example:

```text
Baseline: 81.4 tok/s
Candidate: 103.8 tok/s
Delta: +27.5%
```

## 4. Environment capture

Every published result MUST identify:

### Hardware

- exact GPU model;
- GPU count;
- configured power limit;
- CPU;
- system RAM;
- PCIe generation/link width if relevant;
- any non-default clocks or power settings.

### Model

- Hugging Face/model repository;
- exact model revision or commit when practical;
- target quantization format and quant source;
- speculative drafter/head repository and revision, if separate;
- drafter dtype/quantization;
- KV-cache dtype;
- maximum configured model length.

### Runtime

- vLLM version and preferably commit SHA;
- vLLM XPU kernels version/commit;
- PyTorch/XPU version;
- Intel compute/runtime/driver version;
- Linux kernel and distribution;
- relevant oneAPI/XPU libraries;
- all server launch arguments;
- relevant environment variables.

The repository SHOULD retain a complete `collect_env` output for each published run.

Suggested layout:

```text
results/<run-id>/
  metadata.yaml
  collect_env.txt
  server-command.txt
  betterbench/
  vllm-bench/
  long-context/
  spec-metrics/
  quality/
```

## 5. BetterBench standard

BetterBench is the primary community-facing workload benchmark.

A publishable result MUST:

- use the full benchmark rather than the reduced/quick mode;
- use 20 measured passes per category where supported by the harness;
- save the raw JSON/result files;
- report median throughput;
- report IQR or the benchmark's equivalent dispersion statistic;
- report coefficient of variation when available;
- include decode and prefill results;
- include concurrency behavior rather than only concurrency=1.

For optimization comparisons, paired/interleaved A/B mode SHOULD be used whenever supported.

When paired A/B is available, the report SHOULD include the relative improvement and 95% confidence interval.

Example:

```text
Candidate vs baseline: +18.7%
95% CI: +17.9% to +19.5%
```

## 6. vLLM serving benchmark

`vllm bench serve` is the maintainer-facing benchmark and SHOULD accompany every publishable result.

At minimum, save and report:

- request throughput;
- output-token throughput;
- total-token throughput;
- TTFT;
- inter-token latency / generation latency metrics exposed by the current vLLM release;
- concurrency or request-rate configuration;
- input/output token distributions.

The exact command MUST be preserved with the result.

Where practical, run a small concurrency sweep rather than a single point. A reasonable default is:

```text
concurrency: 1, 2, 4, 8
```

Higher concurrency MAY be added when evaluating serving rather than single-user interactive performance.

## 7. Repository long-context sweep

Short-context throughput is not sufficient for the B70/Qwen 27B target workload. Every publishable optimization MUST include a cold long-context sweep.

### Required prompt lengths

```text
512
8K
16K
32K
64K
120K
160K
maximum practical point near the configured context limit
```

For a ~200K configuration, the final point SHOULD be approximately 190K if it fits reliably.

### Required generation settings

- 128 output tokens;
- temperature 0 / greedy decoding;
- EOS ignored so every request produces the full output length;
- prefix caching disabled;
- unique prompt content per run to prevent accidental cache reuse;
- exact rendered token count recorded, rather than assuming source text length;
- one warm-up run before measurements;
- six measured runs per context point;
- median reported as the primary number;
- IQR reported with the median.

The benchmark SHOULD record separately where possible:

- prefill time;
- decode time;
- decode tokens/sec;
- end-to-end tokens/sec;
- peak device-memory consumption.

A configuration that improves short-context speed but materially lowers usable context capacity MUST state that tradeoff prominently.

## 8. Speculative decoding

When speculative decoding is enabled, tokens/sec alone is insufficient.

Every result MUST record the speculative metrics exposed by the runtime, preferably per request, including:

- mean accepted-token / acceptance length;
- draft acceptance rate;
- number of draft tokens proposed;
- number of draft tokens accepted;
- acceptance by draft position/depth when available;
- acceptance histogram when available.

The exact speculative configuration MUST be captured, including:

- algorithm (native MTP, DFlash2, DSpark, etc.);
- speculative depth / number of draft tokens;
- drafter dtype and quantization;
- acceptance method;
- any acceptance thresholds;
- adaptive-depth settings if used.

When testing adaptive draft depth, report the observed depth distribution in addition to aggregate throughput.

## 9. Lossy or relaxed acceptance

Any optimization that can change the target model's output MUST be labeled **quality-sensitive**.

Examples include cascade/relaxed acceptance policies that do not preserve the target distribution exactly.

### 9.1 Output-divergence test

Run the same deterministic prompt set through:

```text
A. target model without speculative decoding
B. target model with normal lossless speculative decoding
C. candidate relaxed/cascade configuration
```

Use at least 500 prompts; 1,000 is preferred for publishable cascade results.

At temperature 0, report:

- exact full-output match rate vs A;
- percentage of requests with any divergence;
- median token position of first divergence;
- distribution of first-divergence position;
- output length used for the comparison.

B SHOULD reproduce A. If it does not, investigate the harness/runtime before drawing conclusions about C.

### 9.2 Quality regression test

A quality-sensitive candidate MUST also run a small fixed evaluation suite against the same target baseline.

Initial suite:

- IFEval;
- GSM8K;
- HumanEval+;
- MBPP+.

The purpose is not to benchmark Qwen 3.8 generally. It is to detect regression caused by the decoding optimization.

Report absolute baseline and candidate scores and their delta.

A useful result statement combines performance and fidelity, for example:

```text
+28.4% decode throughput
97.8% greedy-output identity
-0.2 points aggregate eval delta
```

Do not publish a relaxed-acceptance speed result without its fidelity result.

## 10. Repetition, noise, and warm-up

Benchmarks MUST include enough repetition to distinguish small improvements from normal variance.

For an optimization claiming less than approximately 10% improvement, paired/interleaved A/B testing is strongly preferred.

Before collecting measurements:

- ensure the GPU is not occupied by another workload;
- warm the model/runtime sufficiently to remove first-run compilation effects;
- avoid mixing compilation/setup time with request timing;
- ensure thermals and power limits are stable;
- record failures, retries, and OOMs rather than silently dropping them.

Outliers SHOULD remain in raw data. If an outlier is excluded from a summary statistic, the exclusion rule and reason MUST be stated.

## 11. Memory and capacity reporting

Every candidate SHOULD report:

- static model memory;
- drafter/speculator memory;
- peak allocated device memory during the benchmark;
- KV-cache capacity or runtime-reported available KV blocks;
- maximum successfully tested context.

This is especially important on the 32 GB B70. A throughput improvement that consumes several GB for a drafter can be a regression for long-context use even if decode speed increases.

## 12. Result naming and storage

Use a stable run identifier containing at least the date, target model, and candidate configuration.

Example:

```text
2026-09-08-qwen38-27b-gptq-mtp4-cascade-v1
```

Suggested repository layout:

```text
benchmarks/
  long_context.py
  quality_compare.py
configs/
  baseline.sh
  mtp4.sh
  cascade-mtp4.sh
results/
  2026-09-08-qwen38-27b-gptq-mtp4-cascade-v1/
    metadata.yaml
    collect_env.txt
    server-command.txt
    betterbench/
    vllm-bench/
    long-context/
    spec-metrics/
    quality/
```

Raw benchmark outputs SHOULD be committed when reasonably small. Large artifacts MAY be attached to a GitHub release or stored separately with a stable link and checksum.

## 13. Minimum publishable result table

Every community-facing result SHOULD include a compact table containing at least:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| BetterBench decode median | | | |
| 512-token decode tok/s | | | |
| 16K decode tok/s | | | |
| 64K decode tok/s | | | |
| 120K decode tok/s | | | |
| 160K decode tok/s | | | |
| Max-context decode tok/s | | | |
| Mean speculative acceptance length | | | |
| Peak device memory | | | |
| Max tested context | | | |

For quality-sensitive methods also add:

| Fidelity metric | Baseline | Candidate |
|---|---:|---:|
| Exact greedy-output identity | 100% | |
| Requests with divergence | 0% | |
| Median first divergence | n/a | |
| IFEval | | |
| GSM8K | | |
| HumanEval+ | | |
| MBPP+ | | |

## 14. Community publication

GitHub is the canonical source for results. Community posts should link back to the exact result directory, commit, or release.

Recommended discussion channels:

1. r/LocalLLM for Qwen/speculative-decoding and cross-hardware comparisons;
2. r/IntelArc for B70/XPU-specific findings;
3. r/LocalLLaMA for broader local-inference visibility;
4. vLLM upstream issue/PR when the change belongs in scheduler, verifier, speculative-decoding, or runtime logic;
5. vLLM XPU kernels upstream when the gain comes from an Intel-specific kernel;
6. Intel's relevant inference/runtime repository when the issue is packaging, XPU configuration, or Intel runtime integration.

When upstreaming code, include the maintainer-facing `vllm bench serve` result in addition to the community-facing benchmark.

## 15. Suggested community post format

```text
Intel Arc Pro B70 + Qwen 3.8 27B <quant>
<optimization>: <candidate throughput> tok/s, +<delta>% vs <baseline>

BetterBench 20-pass paired A/B:
  baseline: <value>
  candidate: <value>
  delta: <value>% [95% CI]

Long context:
  16K:  <value> tok/s
  64K:  <value> tok/s
  120K: <value> tok/s
  160K: <value> tok/s
  ~190K: <value> tok/s

Speculation:
  mean acceptance length: <value>
  draft acceptance: <value>%

Memory:
  peak VRAM: <value>
  max tested context: <value>

Quality/fidelity, if applicable:
  exact greedy-output identity: <value>%
  eval delta: <value>

Commands, environment, raw JSON, and patch:
  <canonical GitHub result link>
```

## 16. Publication checklist

A result is ready to share when all applicable items are true:

- [ ] baseline and candidate were run on the same host;
- [ ] exact model and runtime revisions are recorded;
- [ ] complete server arguments are recorded;
- [ ] environment capture is saved;
- [ ] BetterBench full/20-pass result is saved;
- [ ] paired A/B result is saved when appropriate;
- [ ] `vllm bench serve` result is saved;
- [ ] long-context sweep is complete;
- [ ] peak memory and max tested context are recorded;
- [ ] speculative acceptance metrics are saved;
- [ ] quality-sensitive methods include output-divergence testing;
- [ ] quality-sensitive methods include the fixed regression suite;
- [ ] raw results are available from the canonical GitHub location;
- [ ] headline numbers are medians, not cherry-picked best runs;
- [ ] any material capacity, quality, or stability regression is stated alongside the speedup.

## 17. Guiding principle

Optimize for a result another person can falsify.

A good benchmark result should make it easy for someone with another B70, 5090, Radeon, or datacenter GPU to run the same workload, substitute only the hardware/runtime, and determine whether the improvement survives.
