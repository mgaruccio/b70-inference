# Muse Glimmer B70 handoff — 2026-08-24

Campaign to get fast concurrent streams from Muse Glimmer on the local Intel Arc Pro B70 (`inference-host`, `inference-host.lan`). Qwen was stopped for this work; the card is single-model.

## Current live service

- **Host:** `user@inference-host` (`inference-host`), also tailnet `inference-host`
- **GPU:** Intel Arc Pro B70 (Battlemage G31, `8086:e223`), 32 GB, `xe` driver, ReBAR 32G
- **Process:** llama.cpp **Vulkan** `b10588` (`~/inference/src/llama.cpp/build/bin/llama-server`)
- **tmux:** `muse`
- **Endpoint:** LAN `http://inference-host.lan:18099/v1`, tailnet `http://inference-host:18099/v1`
- **Model id:** `muse-glimmer-30b`
- **Weights:** Meta `Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf` (19.7 GB, official 32 GB text build)
- **Draft:** `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` (1.6 GB), `--spec-type draft-dflash --spec-draft-n-max 2`
- **Flags:** `-ngl 99 -ngld 99 -c 262144 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja`
- **No mmproj** (text-only; leftover VRAM stays in KV)
- **Health:** `{"status":"ok"}` at 2026-08-24 10:12 EDT
- **First request after this DFlash flip:** 26.1 tok/s, TTFT 2.16 s (128-token SSE). Warm C1 in the overnight sweeps was 27–32 tok/s.

UFW: `18099/tcp` allowed from `192.168.8.0/24` and `tailscale0`.

## Why the first deploy felt slow

The overnight **production** cell was Vulkan **without** DFlash, `-np 8`, so four concurrent streams would not collapse. That cell is ~18 tok/s on a single chat. A normal 1–2 stream Pi/herdr session wants DFlash on. That flip was made at 10:07 EDT.

Glimmer **always thinks**. Those tokens count in tok/s but often do not appear as `content`. Perceived speed is lower than the engine number.

This card’s measured single-stream ceiling in our sweeps is ~32 tok/s. It will not feel like a 5090.

## Winner by job (same 128-token SSE sweep)

Public boundary: `POST /v1/chat/completions` with `stream=true`. Script: `~/inference/launchers/glimmer-stream-sweep.py`.

| Engine | C1 | C2 agg / each | C4 agg / each |
| --- | ---: | ---: | ---: |
| **Vulkan + DFlash n_max=2** (live) | **27–32** | **48 / 24** | 12 / **3** (unusable) |
| Vulkan no DFlash, 8 slots | 18 | 37 / 18 | **42 / 10.6** |
| SYCL F16, no oneDNN, DFlash, `--no-warmup` | 26.6 | 34 / 17 | 24 / 6 |
| SYCL F16, no oneDNN, no DFlash | 14 warm | 34 / 17 | 40 / 10.3 |
| OpenVINO GenAI 2026.4 VLMPipeline GPU | 26.3 cold / **31.7 warm** | serialized ~32 agg | unlocked C4: 3/4 dropped |

**Use DFlash for 1–2 streams. Turn it off for 4+ live streams.**

## How to switch (on `inference-host`)

Launchers live on the host, not in this repo:

```text
~/inference/launchers/start-muse-glimmer.sh          # Vulkan (production)
~/inference/launchers/start-muse-glimmer-sycl.sh     # SYCL in DLE container
~/inference/launchers/start-muse-glimmer-ovgenai.sh  # GenAI 2026.4 HTTP wrapper
~/inference/launchers/start-muse-glimmer-ovms.sh     # OVMS (does not load this IR)
~/inference/launchers/glimmer-stream-sweep.py
```

```bash
# Fast 1–2 stream (current)
DFLASH=1 PARALLEL_SLOTS=2 SLOT_CTX=131072 ~/inference/launchers/start-muse-glimmer.sh

# Fast 4–8 stream (no spec)
DFLASH=0 PARALLEL_SLOTS=8 SLOT_CTX=32768 ~/inference/launchers/start-muse-glimmer.sh

# Stop / replace
tmux kill-session -t muse
```

Qwen is the other resident model. Do not run both. Qwen launcher: `~/inference/launchers/start-qwen38.sh` on port 8000.

## Paths tried and what broke

### llama.cpp SYCL (cookbook public recipe)

Cookbook: `~/inference/src/intel-arc-pro-b70-inference-cookbook/docs/muse-glimmer/MUSE-GLIMMER-B70.md`

- Built in `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04` with `icpx` 2026.0, `GGML_SYCL=ON`, `GGML_SYCL_F16=ON`.
- Binary: `~/inference/src/llama.cpp/build-sycl/bin/llama-server`
- Device enum works: `level_zero:gpu:0` Intel Graphics `[0xe223]`, 32.5 GB.
- **Must use `--no-warmup`.** Warmup decode caused `UR_RESULT_ERROR_DEVICE_LOST`.
- **262k unified or 128k + DFlash + `-ub 8192` hard-rebooted the box** (`uptime` reset). Same failure class the cookbook warned about.
- DLE image has **no oneDNN**. Intel apt `apt.repos.intel.com` returns **403**.
- oneDNN built from `uxlfoundation/oneDNN` with `DNNL_GPU_RUNTIME=SYCL`, linked `libdnnl.so.3`. First generate aborted: `GGML_ASSERT(ptr == pool_addr + pool_used)` in `ggml_sycl_op_mul_mat_sycl`. Do not ship that binary.
- Do **not** set `GGML_SYCL_DEVICE_ARCH=bmg-g31` (AOT). Reports say it compiles then crashes on B70; JIT + F16 is the safe path.

SYCL without oneDNN matches the cookbook C1 DFlash number (~26.8) but does not beat Vulkan for C2/C4.

### OpenVINO IR (Intel-native artifact)

Downloaded: `~/inference/models/OpenVINO/Muse-Glimmer-30B-int4-ov` (~20 GB).

- [`OpenVINO/Muse-Glimmer-30B-int4-ov`](https://huggingface.co/OpenVINO/Muse-Glimmer-30B-int4-ov) — INT4_ASYM, g64, **experimental**, needs OpenVINO **2026.3.1+**.
- `openvino/model_server:weekly` and `:latest-gpu` see CPU+GPU, then fail: **`Unsupported 'muse_glimmer' VLM model type`**.
- `--source_model` wants a git checkout / `graph.pbtxt`. A plain HF snapshot is not enough for OVMS.
- **OpenVINO GenAI 2026.4** on the DLE image **does** load `VLMPipeline(..., "GPU")`.
  - Load ~16–24 s.
  - 128-token generate: 26.3 tok/s cold, **31.7 tok/s warm**.
  - `LLMPipeline` fails (`input_ids` port missing — this IR is VLM).
  - PyPI stable max is 2026.3.0 (no `muse_glimmer`). Need `--pre` nightly `2026.4.0`.
  - Slim `python:3.12` image: missing `libOpenCL.so.1` / no GPU devices. Use DLE + `/dev/dri`.
  - Custom HTTP wrapper (`ov-genai-server.py`) is **not thread-safe**. Unlocked C4 dropped 3/4 connections. Locked C4 only serializes (~32 agg).
- Intel day-0 B70 numbers used OpenVINO + DFlash assistant tokens = 8. We did not find a working GenAI flag for that on this IR.

### vLLM XPU

Cookbook: experimental, slower than llama.cpp DFlash C1, chat leaks reasoning. Not re-run this session. Image still on host: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.

## Host artifacts to keep

```text
~/inference/src/llama.cpp/build/bin/llama-server          # Vulkan production
~/inference/src/llama.cpp/build-sycl/bin/llama-server     # SYCL, no oneDNN, usable with --no-warmup
~/inference/src/llama.cpp/build-sycl-dnnl/                # DO NOT SERVE (allocator abort)
~/inference/src/oneDNN/install/lib/libdnnl.so.3
~/inference/models/Muse-Glimmer-30B-GGUF/                 # Meta GGUF + draft + mmproj
~/inference/models/OpenVINO/Muse-Glimmer-30B-int4-ov/     # complete INT4 IR
~/inference/launchers/start-muse-glimmer.sh
~/b70-evals/muse-glimmer/                                 # sweep receipts if present
```

Sweep JSON from the desktop session (local `/tmp`): `sycl-*.json`, `dnnl-*.json`, `ovgenai-*.json`.

## Sources that actually drove flags

- Meta GGUF card: <https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF> (`-c` is split across `-np`; scale `-c` with slots; DFlash `-md` / `-ngld`)
- B70 cookbook (SYCL measured): host `docs/muse-glimmer/MUSE-GLIMMER-B70.md` — F16 mandatory, DFlash `n_max=2`, 128k C1, 64k×4 memory envelope
- llama.cpp SYCL: <https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md>
- Intel day-0: <https://www.intel.com/content/www/us/en/developer/articles/community/day-0-local-agentic-ai-with-metas-muse-glimmer.html>
- OpenVINO IR: <https://huggingface.co/OpenVINO/Muse-Glimmer-30B-int4-ov>
- Planned host contract: `docs/arc-pro-b70-planned-deployment.md`

## First next-session actions

1. Leave Vulkan + DFlash + 2 slots up unless someone needs 4-wide. Do not “try SYCL 128k + ub 8192” again without expecting a reboot.
2. If the goal is still faster **concurrent** decode: do not spend more time on GenAI `VLMPipeline` (no batching) or OVMS weekly (no `muse_glimmer`). Next real bets are (a) a newer OVMS/GenAI continuous-batching build that names this architecture, or (b) a llama.cpp SYCL that can run oneDNN without the mul_mat pool assert, bisecting `b10588` vs cookbook `d2f83055d`.
3. Optional C1-only alternative: GenAI 2026.4 on DLE is 31.7 tok/s warm and skips the draft, but it is a custom wrapper and single-flight.
4. Restore Qwen with `tmux kill-session -t muse` then `~/inference/launchers/start-qwen38.sh` when this card is needed for Qwen evals again.
5. A Luna researcher spawned as `glimmer intel research` hung for hours and ignored interrupts. Do not wait on it.

## Non-goals / do not repeat

- Shadeform 5090 / SGLang NVFP4 is the wrong host for this request.
- Do not load mmproj unless vision is required.
- Do not treat cookbook 1301 tok/s prefill as a serve number; that was `llama-bench` pp4096 on their SYCL+F16+oneDNN box.
- Do not claim GenAI concurrent streams. The unlocked wrapper is unsafe.
