# Arc Pro B70 planned local deployment

## Purpose

This is the evaluation target for comparing the planned one-card Intel deployment with the existing RTX/hosted serving rows. The host is `user@inference-host` (`inference-host`) and must remain a single-model server: the two 32 GB-class model profiles are not intended to be resident together.

## Hardware and host contract

- GPU: Intel Arc Pro B70 (Battlemage G31), 32 GB VRAM, PCI ID `8086:e223`.
- Host: CachyOS, `xe` driver, 32 GB system RAM.
- ReBAR is mandatory: `lspci -vv -s 0b:00.0` must report BAR 2 at `size=32G`; vLLM's XPU runtime must report one Arc Pro B70 device before an evaluation.
- The host mounts `/dev/dri` into containers. Sleep, hibernation, and idle-triggered suspend are disabled.
- Serving endpoints bind on all interfaces but are firewall-restricted: ports 8000 and 18099 accept only `192.168.8.0/24` or `tailscale0` traffic. The tailnet address is `inference-host` (`inference-host`).

## Model profiles

| Evaluation row | Artifact and quant | Runtime | Endpoint | Status |
| --- | --- | --- | --- | --- |
| `qwen38-b70-gptq-int4-mtp4` | `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`; GPTQ 4-bit symmetric, G128, preserved BF16 MTP tensors | pinned vLLM XPU image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` | LAN `http://inference-host.lan:8000/v1`; tailnet `http://inference-host:8000/v1` | serving verified; eval pending |
| `muse-glimmer-b70-kquant-dynamic-q4` | Meta `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf`, with Q4_K_M multimodal projector and DFlash draft | llama.cpp with Vulkan | LAN `http://inference-host.lan:18099/v1`; tailnet `http://inference-host:18099/v1` | artifacts verified; not loaded |

Qwen's source is a B70-specific vLLM-XPU recipe. The MIAI Lab NVFP4 32 GB recipe targets NVIDIA Blackwell and is not valid on Intel XPU. Muse's single-B70 profile is Meta's GGUF recipe; it is not a vLLM or SGLang deployment.

## Planned launch contract

The server has on-demand launchers under `~/inference/launchers/`:

```bash
# Run exactly one model at a time, ideally inside tmux.
tmux new -s qwen38 '~/inference/launchers/start-qwen38.sh'
tmux new -s muse '~/inference/launchers/start-muse-glimmer.sh'
```

Qwen **live** uses fp8 KV cache, **131,072**-token maximum context, a **0.88** GPU-memory target, MTP-4, and `--max-num-seqs 64`. A disposable C1 profile at **0.95 / 200,000 / max-num-seqs 1** booted and completed a **195,992 + 32** token request (see [`qwen38-b70-200k-context-20260830.md`](qwen38-b70-200k-context-20260830.md)). That is not the live launcher. Muse uses full Vulkan offload (`-ngl 99`), Meta's Dynamic Q4 GGUF, multimodal projector, and DFlash drafter. Do not treat third-party benchmark claims as results for this machine.

## Evaluation procedure

1. Record host evidence: kernel, `xe` driver, BAR size, exact runtime/image or llama.cpp revision, model revision/files, and model SHA-256 manifest.
2. Start one profile and prove its public serving boundary from `desktop-1` over both permitted networks:\n   - Qwen: `curl -f http://inference-host.lan:8000/health` and `curl -f http://inference-host:8000/health`, then a fixed OpenAI-compatible completion against `/v1/chat/completions`.\n   - Muse: repeat those LAN and tailnet health checks on port 18099, then the same fixed completion shape against `/v1/chat/completions`.
3. Capture cold-start time, model load/free-memory information, first-token latency, output tokens/s, and a fixed short/medium/long prompt set. Keep requests single-stream first; add concurrency only after recording the one-stream baseline.
4. Run the same pinned taskset, harness version, sampling values, and rollout concurrency as each comparison row. The repository's existing RTX/hosted configurations are comparison baselines, not settings to copy onto Intel XPU.
5. Store raw service logs, model hashes, request payloads, measured results, and evaluator output with the row name above. Stop the server before switching models.

## Sources

- Intel Arc Pro B70/Xe support: <https://dgpu-docs.intel.com/overview/supported-hardware/xe-driver-gpus.html>
- vLLM Intel XPU support: <https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/>
- B70 Qwen GPTQ recipe: <https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16>
- Meta Muse Glimmer GGUF artifacts: <https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF>
- Intel's B70/Muse deployment guidance: <https://www.intel.com/content/www/us/en/developer/articles/community/day-0-local-agentic-ai-with-metas-muse-glimmer.html>
