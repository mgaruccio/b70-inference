# Muse Glimmer 30B on one Arc Pro B70 (vLLM-XPU + DFlash)

Status: **live C1 cell**, 2026-08-27. This is the public writeup for the
quoteable numbers and the recipe that produces them.

Protocol and scoring rules: [`dflash-share-suite.md`](dflash-share-suite.md).
Session notes: [`glimmer-b70-vllm-xpu-handoff-20260826.md`](glimmer-b70-vllm-xpu-handoff-20260826.md).

## Quote these numbers

Greedy (`temperature=0`), generate until `stop`, one stream, public OpenAI
streaming API. tok/s counts **all** completion tokens (thinking + answer).
Decode is post-first-token; e2e includes TTFT. A row is quoteable only if it
stopped, emitted visible `content`, and passed its check.

| Prompt | Finish | Tokens | Time to first content | Decode tok/s | E2E tok/s | Accepted drafts / verify | Check |
|---|---|---:|---:|---:|---:|---:|---|
| GSM8K test[0] | stop | 586 | 6.10 s | **89.1** | 86.7 | 3.37 | **$18** |
| HumanEval/0 | stop | 887 | 8.44 s | **101.1** | 99.3 | 4.14 | **doctest pass** |
| MT-Bench 81 | stop | 1585 | 8.53 s | **42.6** | 42.3 | 1.22 | 4670-char post |

Quote the **table**, not a single average. Writing is the slow cluster;
math/code speculate better. Receipt:
`~/b70-evals/muse-glimmer/20260827T-dflash-share-suite/share-suite.json`.

Public llama.cpp-class B70 Muse numbers from others are ~27–29 tok/s. Same
card, no-spec vLLM GPTQ sky prompt was ~31.5 e2e tok/s (128-token window,
decode timer broken). Do not mix those labels with the table above.

## Hardware

- **GPU:** Intel Arc Pro B70 (Battlemage G31), 32 GB, PCI ID `8086:e223`
- **ReBAR:** `lspci -vv` BAR 2 must be `size=32G`
- **Host:** Linux, `xe` driver, Docker, `/dev/dri` into the container
- **Concurrency:** this recipe is **C1** (`--max-num-seqs 1`)

vLLM documents Arc Pro B-Series as validated XPU hardware and GPTQ as a
supported XPU quantization path:
[XPU models](https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/),
[quantization](https://docs.vllm.ai/en/stable/features/quantization/),
[speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/).

Muse Glimmer ships a 5-layer DFlash drafter (block 16, aux layers
`{1,13,25,37,49}`) and claims lossless verify:
[model card](https://huggingface.co/meta-models/Muse-Glimmer-30B),
[DFlash paper](https://arxiv.org/abs/2602.06036).

## Optimizations (what stuck)

Each keep/kill used the public streaming API. Sky 8×256 is the **internal
gate** (thinking-channel, forced 256). The table above is the **share
suite** (until stop, content required).

| Step | What | Result | Keep? |
|---|---|---|---|
| 0 | llama.cpp Vulkan / SYCL DFlash2 | ~27–32 tok/s C1; DFlash2 SYCL n=8 **11 tok/s**, killed | **No** |
| 1 | **GPTQ W4A16 target** (4-bit, G128, symmetric, `XPUwNa16LinearKernel`) | no-spec ~31.5 e2e, in Qwen's no-spec band | **Yes** |
| 2 | **Official vLLM DFlash** n=15, eager | sky decode **42.1–43.7** | **Yes**, then superseded by graphs |
| 3 | **XPU graphs** + depth n=20 (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, drop `--enforce-eager`) | sky 8×256 **48.0** decode with BF16 draft; n=32 **regressed** | **Yes** |
| 4 | **GPTQ 5-layer draft** + packed-QKV fallback overlay | sky 8×256 **52.9** decode / **51.4** e2e, +10.1% vs BF16 draft, acceptance 1.56 vs 1.58 | **Yes** (live cell) |
| 5 | KV-only fused context projection | ~0.5–0.9% of step time | **No** (under 5% gate) |
| 6 | `extract_hidden_states` for real-target draft recalibration | HiddenStateCacheSpec page-size assert | **Parked** |
| 7 | Adaptive draft depth | not measured; vLLM adaptive verify is DSpark-only in current docs | **Not started** |

Kernel overlay: stock `vllm-xpu-kernels` **0.1.12.3** is missing Muse GQA 32/2
paged-decode tuples. The live cell installs **0.1.13.2** at container start.

Packed-QKV overlay (`scripts/patch-vllm-dflash-gptq-context-kv.py`): DFlash's
fast context-KV path reads `qkv_proj.weight`; GPTQ only exposes `qweight`.
`DFLASH_KV_MODE=none` (live) runs each quantized QKV module and keeps K/V.
Do **not** set `timing` on the serving cell: `torch.xpu.synchronize()` moved
acceptance 1.560 → 1.306.

Eagle3 layer ids: config `{1,13,25,37,49}` becomes live `(2,14,26,38,50)`.
That +1 is vLLM's conversion. Do not add a second +1.

## Run it yourself

Pinned image:

`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
(vLLM `0.27.2rc1.dev77+gac7509e2b`).

### 1. Artifacts

You need both GPTQ directories on the host:

| Role | Directory | Notes |
|---|---|---|
| Target | `Muse-Glimmer-30B-GPTQ-Int4-sym-G128` | ~21 GiB. 4-bit G128 symmetric, `desc_act=false`, `lm_head=false`. Vision left off (`--language-model-only`). |
| Draft | `Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128` | ~1.6 GiB. 35 decoder linears quantized; encoder/norms BF16. Embeddings + lm_head are **shared with the target**. |

Upstream BF16 weights: [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).
The GPTQ trees used here are **local** (GPTQModel **7.3.2**). Draft requant
script: `scripts/quantize-muse-dflash-assistant-gptq.py`. Never overwrite the
BF16 assistant or the known-good GPTQ draft directory.

### 2. Host files

From this repo, on the inference host:

```bash
git clone https://github.com/mgaruccio/b70-inference.git
cd b70-inference

# spec JSON must use container paths (/draft, not the host path)
cp scripts/muse-dflash-spec-gptq.json /tmp/muse-dflash-spec-gptq.json
cp scripts/patch-vllm-dflash-gptq-context-kv.py /tmp/patch-vllm-dflash-gptq-context-kv.py
```

Edit `MODEL` / `DRAFT` if your weights are not under
`/home/mike/inference/models/`.

```bash
export MODEL=/abs/path/Muse-Glimmer-30B-GPTQ-Int4-sym-G128
export DRAFT=/abs/path/Muse-Glimmer-30B-assistant-GPTQ-Int4-sym-G128
export SPEC=/tmp/muse-dflash-spec-gptq.json
export PATCH=/tmp/patch-vllm-dflash-gptq-context-kv.py
export DFLASH_KV_MODE=none
```

Confirm ReBAR and the render node:

```bash
lspci -vv -d 8086:e223 | grep -E 'BAR|Kernel driver'
stat -c '%g' /dev/dri/renderD128
```

### 3. Launch

```bash
DFLASH_KV_MODE=none bash scripts/start-muse-vllm-dflash-c1-graph-draft-gptq.sh
bash scripts/wait-vllm-health.sh 420 8000
curl -s http://127.0.0.1:8000/v1/models
```

Expect served id `muse-glimmer-gptq`. Logs should show
`XPUwNa16LinearKernel`, `num_spec_tokens=20`, and
`patched: ... qwen3_dflash.py (mode=base`.

Serve flags that matter:

- `--quantization gptq --dtype float16 --kv-cache-dtype fp8`
- `--max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048`
- `--no-enable-prefix-caching --language-model-only`
- `--reasoning-parser muse_glimmer`
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` (do not pass `--enforce-eager`)
- spec: `method=dflash`, `num_speculative_tokens=20`, `quantization=gptq`

32k context at 0.85 GPU memory OOM'd this card with the BF16 draft. 8k is
the working C1 window.

### 4. Measure

```bash
python3 scripts/vllm-dflash-share-suite.py 3 2048 /tmp/dflash-share-suite.json
```

The process exits non-zero if any prompt is not quoteable. Do not publish a
headline from a partial run. Suite details:
[`dflash-share-suite.md`](dflash-share-suite.md).

Smoke test (not the share suite):

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-gptq","messages":[{"role":"user","content":"Say hi."}],"max_tokens":32,"temperature":0}'
```

Muse streams `delta.reasoning` then `delta.content`. A client that only
reads `content` will look hung until thinking finishes.

## Do not

- Resume SYCL DFlash2, Vulkan cap tuning, or Shadeform 5090 work for this cell
- Call `torch.xpu.synchronize()` on the serving path
- Retry vLLM `extract_hidden_states` on this Muse/XPU build
- Deploy `DFLASH_KV_MODE=kvonly` without a new ≥5% gate
- Overwrite the known-good GPTQ draft directory
- Compare 128-token thinking windows to this share suite
- Treat n=20 as universal: sky/writing accept ~1.2–1.6 drafts/verify; easy
  math/code accept ~3–4. Adaptive depth is future work.

## Sources

- vLLM XPU: https://docs.vllm.ai/en/stable/models/hardware_supported_models/xpu/
- vLLM quantization: https://docs.vllm.ai/en/stable/features/quantization/
- vLLM speculative decoding: https://docs.vllm.ai/en/stable/features/speculative_decoding/
- vLLM acceptance metrics (accepted drafts **exclude** the bonus token): https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/
- Muse Glimmer card: https://huggingface.co/meta-models/Muse-Glimmer-30B
- DFlash: https://arxiv.org/abs/2602.06036
- Intel B70 / Muse day-0: https://www.intel.com/content/www/us/en/developer/articles/community/day-0-local-agentic-ai-with-metas-muse-glimmer.html
- GPTQModel: https://github.com/ModelCloud/GPTQModel
