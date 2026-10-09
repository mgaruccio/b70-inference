# Community benchmark protocol on our current serving setup

Executed 2026-10-09 after the user explicitly redirected work from full quality
evaluation to running the community benchmarks on our serving setup.

## Results

Five measured requests per cell, following the public client's generic and
same-shape warmups. All endpoint input/output counts matched exactly. Every
measured request had zero prefix-cache hits.

| Input / output tokens | Our median | Published S+M1 reference | Descriptive difference |
|---|---:|---:|---:|
| 512 / 128 decode | 116.00 tok/s | 112.65 tok/s | +2.98% |
| 8192 / 128 decode | 108.29 tok/s | 103.63 tok/s | +4.50% |
| 130944 / 128 decode | 62.28 tok/s | 62.52 tok/s | −0.38% |
| 8192 / 1 input tokens / client TTFT | 1928.02 tok/s | 1696 tok/s | +13.68% |

Our decode ranges were 107.95–120.22, 68.12–112.60 and 60.20–69.24 tok/s,
respectively. Aggregate draft acceptance was 92.03%, 71.54% and 83.50%.
The 8K cell varies materially across prompts; do not hide its lower observations.
Prefill here means input tokens divided by client TTFT, not isolated engine
prefill throughput. Decode is `(completion_tokens - 1)/(end - first_generated)`.

**Interpretation:** competitive with the published speed-only reference, not a
clear across-the-board breakthrough or proof of community leadership. These
are different generated prompts and machines/settings, not paired causal gains.
This five-run community-protocol comparison is development evidence, not the
repository's complete standard-publication or quality-qualification package.

## Serving configuration — unchanged

Used `/home/mike/inference/launchers/start-qwen38.sh` as-is, SHA256
`63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

- One B70; Qwen3.8-27B GPTQ symmetric G128; `--dtype float16`; FP8 KV.
- MTP4, draft INT4 S+M1, mixed-GDN-v5, XPU graphs enabled.
- 212992 configured context, utilization .95, batch-token budget 8192, C1.
- Existing prefix caching and thinking **enabled**. Cold entropy-first requests
  produced zero cache hits; the benchmark requests temperature 0 and fixed
  output lengths using `--ignore-eos`.
- 275 W cap, CPU boost off, all 16 CPU maximum-frequency caps 3801000 kHz.
- Image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- This is the existing launcher, **not the experimental shared-KV overlay**.

The published S+M1 reference uses 230 W, 131072 configured context and .88
utilization with caching disabled. Our enabled caching did not provide measured
hits. Do not present the numerical differences above as an isolated kernel gain.
No production configuration was edited; the owned server was stopped after the
measurements, restoring the previously idle host. Host and kernel guards passed.

## Public client, prompts and compatibility correction

Used the existing cookbook copies, verified against the public Git blob IDs:

- `b70-realworld-context-harness.py`: `a4ca3c28c2d87436f80e93c42cb9f9712f3765e7`
- `b70-generate-exact-prompts.py`: `645c021089c82d671a0c332497cfe2ec503c3f57`

The official prompt generator ran inside the serving image against its local
`/model` tokenizer. Six entropy-first exact-length prompts per decode length
provided one shape warmup and five measured samples. A separate six-prompt 8K
set was generated for prefill so the enabled prefix cache could not reuse the
8K decode inputs. Exact prompts are retained, not reconstructed from a seed.

The first attempt failed during the generic warmup: the public client recognized
`delta.reasoning_content`, while this server emits `delta.reasoning`. The retry
adds only `or delta.get("reasoning")` to that lookup. It does not change request
sampling, timer placement, token accounting or throughput formulas. Both attempts
and the exact compatibility change are retained in the archive.

The exact historical benchmark prompts are not public. This run uses their
public generator/client and workload sizes, not an invented claim of replaying
the original hidden requests.

## Commands and artifacts

Host command:

```sh
python3 -B /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-community-protocol/run.py
```

For each cell the controller runs the public client with `--mode context`,
`--target <input> --output <output> --budget 8192 --reps 5 --model qwen38`,
`--root http://127.0.0.1:8000 --full-output-warmup --ignore-eos`.
`run.py` refuses to overwrite its output directory. The successful retry reused
the first attempt's retained decode prompt file; both are in the archive.

- `summary.json`: successful five-run cell summaries.
- `community-benchmark.tar.gz`: `current-serving/` (failed warmup) and
  `current-serving-02/` (successful run), including public source, exact prompts,
  raw timed SSE, per-request records, commands, server/container metadata,
  launcher copy, compatibility patch description, host guards and cleanup.
- `SHA256SUMS`: file integrity checks.

References:
- [Public S+M1 submission](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/submissions/vllm-qwen38-mtp4-draft-int4.json)
- [Published 8K and long-context cells](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27b/QWEN38-VLLM-XPU.md)
- [Public benchmark client](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/benchmarks/b70-realworld-context-harness.py)

The reference results are self-reported speed evidence with limited quality
validation. No announcement or quality-neutrality claim follows from this run.
