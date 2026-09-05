# B70 inference

Public notes, plans, and tooling for serving large local models on a single
**Intel Arc Pro B70** (Battlemage G31, 32 GB, `xe` driver).

This is the hardware-optimization workspace. Evaluation wiring, Prime
tasksets, and hosted NVIDIA cells stay in
[local-dev-model](https://github.com/mgaruccio/local-dev-model).

Two models share the card. Only one is resident at a time.

| Model | Engine | Spec | Current C1 decode | Notes |
| --- | --- | --- | ---: | --- |
| **Muse Glimmer 30B GPTQ** | vLLM XPU | DFlash n=20, GPTQ draft | **89.1 / 101.1 / 42.6** tok/s (GSM8K / HumanEval / MT-Bench, greedy, until stop) | Moved to [muse-glimmer-b70](https://github.com/mgaruccio/muse-glimmer-b70) |
| Qwen3.8-27B GPTQ-Int4 | vLLM XPU | MTP-4, FP8 KV | **35.686** tok/s at 212k active context; **95.432** short-context reference | Default: 212,992-token C1, no KV offload |

Public llama.cpp Muse-on-B70 numbers from others are ~27–29 tok/s. The vLLM stack is a different path (GPTQ W4A16 + graphs + DFlash n=20), measured until stop with visible answers.

## New Glimmer concurrency profile — 2026-09-05

**C8 / DFlash K3 / native 131072 context:** **303.1 aggregate e2e tok/s** at eight clients after long-context stress, on the matched 256-output-token workload. The same-session K4 baseline was **282.2** (**+7.4%**); the earlier native-context K4 result was 278.1. This is not per-stream decode and is separate from the completed-answer table above.

K3 passed 16/16 completed-answer smoke checks, eight concurrently resident ~64k prompts, dynamic queue/drain, and single-request ~129k retrieval, with zero preemptions. **Known limit:** the earlier K4 forced-length, near-capacity six-request empty-response failure was not retested or fixed. Keep this as a research profile, not an unrestricted production-safety claim.

- [K3 hill-climb, repeated measurements, and exact validation](docs/glimmer-b70-hillclimb-20260905.md)
- [Original public K4 configuration and reproduction](https://github.com/mgaruccio/muse-glimmer-b70/blob/main/docs/concurrency.md)
- [Original K4 concurrency/context measurements and caveats](docs/glimmer-b70-concurrency-20260905.md)
- [Separate C8/K3 launcher](scripts/start-muse-vllm-concurrent.sh) — localhost port 18080; original C1 and Qwen defaults unchanged.

## Start here

- **Muse Glimmer post + recipe:** [mgaruccio/muse-glimmer-b70](https://github.com/mgaruccio/muse-glimmer-b70)
- Suite protocol: [`docs/dflash-share-suite.md`](docs/dflash-share-suite.md)
- Host contract: [`docs/arc-pro-b70-planned-deployment.md`](docs/arc-pro-b70-planned-deployment.md)
- Parked llama.cpp / SYCL notes: [`docs/glimmer-b70-research-program.md`](docs/glimmer-b70-research-program.md)
- **Qwen research plan:** [`docs/qwen38-b70-research-plan-20260824.md`](docs/qwen38-b70-research-plan-20260824.md)
- **Qwen B70 default configuration:** [`docs/qwen38-b70-golden-config.md`](docs/qwen38-b70-golden-config.md) — active 212,992-token C1 no-offload profile; 95.432 tok/s short-context reference.

## Tooling
- `scripts/glimmer-phase0-instrument.py` — public `/v1/chat/completions`
  SSE client. Splits TTFT / prefill / decode, reasoning vs content, and
  per-slot DFlash acceptance from the llama-server log.
- `scripts/glimmer-phase0-run.sh` — C1 + C4 `n_max` matrix on the host.
  Safe flags only (`-ub 512`). Restores the 2-slot DFlash cell at the end.
- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/` — GDN metadata overlay
  and P0 MTP tracer for the pinned vLLM XPU image.
- `scripts/compare_qwen38_b70_p0_traces.py` — offline trace compare.
- `scripts/vllm-dflash-share-suite.py` — quoteable DFlash C1 suite (stop + content + checks). Protocol: [`docs/dflash-share-suite.md`](docs/dflash-share-suite.md).
- `scripts/qwen38-ctx-ceiling-sweep.sh` / `scripts/qwen38-200k-fill.sh` — host-only Qwen KV ceiling and 200k filled-completion soak.
- `scripts/reset-b70-gpu.sh` — manual B70 PCI-reset recovery for a stuck fan. Run `--status` first, then run `sudo scripts/reset-b70-gpu.sh --reset` only after every GPU workload has stopped.
- `scripts/set-b70-headless.sh` — make `multi-user.target` the host default and disable Plasma Login so it cannot block a B70 reset; run once with `sudo scripts/set-b70-headless.sh --apply`.

Original Muse C1/K20 launcher: `scripts/start-muse-vllm-dflash-c1-graph-draft-gptq.sh`. New C8/K3 native-context research launcher: `scripts/start-muse-vllm-concurrent.sh`. Older host copies under `~/inference/launchers/` are fallbacks if `/tmp` was wiped.

## What this is not

- Not a drop-in serving recipe for NVIDIA Blackwell / NVFP4 / SGLang.
- Not a benchmark leaderboard. Numbers are from one B70, one host, labeled
  C1 vs C4, e2e vs decode, with acceptance when speculation is on.
- Not the Prime/Harbor eval workspace.

## License

MIT. Model weights remain under their upstream licenses (Meta Muse Glimmer,
Qwen, Unsloth/SergiioB artifacts).
