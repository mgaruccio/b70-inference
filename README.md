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
| Qwen3.8-27B GPTQ-Int4 | vLLM XPU | MTP-4, FP8 KV | 57–62 tok/s short decode | Prefix cache helps TTFT, not decode |

Public llama.cpp Muse-on-B70 numbers from others are ~27–29 tok/s. The vLLM stack is a different path (GPTQ W4A16 + graphs + DFlash n=20), measured until stop with visible answers.

## Start here

- **Muse Glimmer post + recipe:** [mgaruccio/muse-glimmer-b70](https://github.com/mgaruccio/muse-glimmer-b70)
- Suite protocol: [`docs/dflash-share-suite.md`](docs/dflash-share-suite.md)
- Host contract: [`docs/arc-pro-b70-planned-deployment.md`](docs/arc-pro-b70-planned-deployment.md)
- Parked llama.cpp / SYCL notes: [`docs/glimmer-b70-research-program.md`](docs/glimmer-b70-research-program.md)
- **Qwen research plan:** [`docs/qwen38-b70-research-plan-20260824.md`](docs/qwen38-b70-research-plan-20260824.md)
- **Qwen 200k context (2026-08-30):** [`docs/qwen38-b70-200k-context-20260830.md`](docs/qwen38-b70-200k-context-20260830.md) — live stays 131k @ 0.88 / 64 seqs; C1 @ 0.95 filled **195,992 + 32** tokens

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

Muse vLLM launcher is `scripts/start-muse-vllm-dflash-c1-graph-draft-gptq.sh`. Older host copies under `~/inference/launchers/` are fallbacks if `/tmp` was wiped.

## What this is not

- Not a drop-in serving recipe for NVIDIA Blackwell / NVFP4 / SGLang.
- Not a benchmark leaderboard. Numbers are from one B70, one host, labeled
  C1 vs C4, e2e vs decode, with acceptance when speculation is on.
- Not the Prime/Harbor eval workspace.

## License

MIT. Model weights remain under their upstream licenses (Meta Muse Glimmer,
Qwen, Unsloth/SergiioB artifacts).
