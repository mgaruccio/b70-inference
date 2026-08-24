# B70 inference

Public notes, plans, and tooling for serving large local models on a single
**Intel Arc Pro B70** (Battlemage G31, 32 GB, `xe` driver).

This is the hardware-optimization workspace. Evaluation wiring, Prime
tasksets, and hosted NVIDIA cells stay in
[local-dev-model](https://github.com/mgaruccio/local-dev-model).

Two models share the card. Only one is resident at a time.

| Model | Engine | Spec | Current C1 decode | Notes |
| --- | --- | --- | ---: | --- |
| Muse Glimmer 30B Dynamic Q4 | llama.cpp Vulkan | DFlash `n_max=2` | 27–32 tok/s e2e; engine eval ~43 tok/s | C4 with DFlash collapses; see the research program |
| Qwen3.8-27B GPTQ-Int4 | vLLM XPU | MTP-4, FP8 KV | 57–62 tok/s short decode | Prefix cache helps TTFT, not decode |

The interesting claim: Glimmer is dense and writes *less* KV per token than
a naive 27B transformer, but short decode on this card is weight/dequant
bound. Qwen also only does full attention on 16 of 64 layers (the rest are
Gated DeltaNet). The 2× gap is runtime stack + unused 16-token DFlash
block + a multi-sequence DFlash bug, not “attention bytes.”

## Start here

- Host contract: [`docs/arc-pro-b70-planned-deployment.md`](docs/arc-pro-b70-planned-deployment.md)
- **Glimmer C1 + C4+ program:** [`docs/glimmer-b70-research-program.md`](docs/glimmer-b70-research-program.md)
- Glimmer campaign log: [`docs/glimmer-b70-handoff-20260824.md`](docs/glimmer-b70-handoff-20260824.md)
- **Qwen research plan:** [`docs/qwen38-b70-research-plan-20260824.md`](docs/qwen38-b70-research-plan-20260824.md)
- Qwen speed handoff: [`docs/qwen38-b70-speed-improvement-handoff.md`](docs/qwen38-b70-speed-improvement-handoff.md)

## Tooling

- `scripts/glimmer-phase0-instrument.py` — public `/v1/chat/completions`
  SSE client. Splits TTFT / prefill / decode, reasoning vs content, and
  per-slot DFlash acceptance from the llama-server log.
- `scripts/glimmer-phase0-run.sh` — C1 + C4 `n_max` matrix on the host.
  Safe flags only (`-ub 512`). Restores the 2-slot DFlash cell at the end.
- `patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/` — GDN metadata overlay
  and P0 MTP tracer for the pinned vLLM XPU image.
- `scripts/compare_qwen38_b70_p0_traces.py` — offline trace compare.

Host launchers live on the inference box under `~/inference/launchers/`,
not in this repo.

## What this is not

- Not a drop-in serving recipe for NVIDIA Blackwell / NVFP4 / SGLang.
- Not a benchmark leaderboard. Numbers are from one B70, one host, labeled
  C1 vs C4, e2e vs decode, with acceptance when speculation is on.
- Not the Prime/Harbor eval workspace.

## License

MIT. Model weights remain under their upstream licenses (Meta Muse Glimmer,
Qwen, Unsloth/SergiioB artifacts).
