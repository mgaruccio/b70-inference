# Flint production serving on the B70: vLLM XPU + DFlash 2 (cutover 2026-11-20)

`flint-inference.service` on the B70 guest now serves `flint` on `:8080`. It replaces the
transient llama.cpp unit. Clients reach it at `http://192.168.8.150:8080/v1`, the Pi `b70/flint`
provider. The stack:

- vLLM 0.34 XPU;
- Flint GPTQ W4A16 g64 (`urakozz/flint-40B-A2.5B-W4A16-g64-AutoRound-GPTQ@76a12a04`);
- DFlash 2 drafter (`incoai/Flint-DFlash2@39b0a7f7`) at k = 4;
- the batch-invariant decode kernels and int8 draft head from
  `flint-vllm-xpu-hillclimb-20261119.md` (guest venv = `build-spec5`).

llama.cpp remains available as the `flint-llamacpp.service` rollback.

Results are development-tier and single-stream (see "Limits"). Raw artifacts are in
`results/20261120-flint-vllm-serving/`. Sources are in `scripts/flint-vllm-xpu/serve/`.

## Deployment

| Item | Setting |
|---|---|
| Unit | `/etc/systemd/system/flint-inference.service`, enabled, `Restart=on-failure`, `Conflicts=flint-llamacpp.service` |
| Launcher | `~/flint-vllm-spike/serve/serve-flint.sh`: `--host 0.0.0.0 --port 8080 --served-model-name flint --gpu-memory-utilization 0.97 --max-model-len auto --max-num-seqs 1 --max-num-batched-tokens 4096 --enable-prefix-caching -cc.mode=none`, `glm47` tool/reasoning parsers, DFlash 2 `num_speculative_tokens: 4`, process-local Intel runtime environment as in `maintenance.sh` |
| Sampling defaults | from the checkpoint's `generation_config.json`: T = 1.0, top-p 0.95, top-k 20. These equal RecursiveSI's recommended settings (model card, `sglang/README.md`, upstream `generation_config.json`) and the old llama.cpp flags. Clients should use `max_tokens` ≥ 32768 for hard problems (RSI); Pi `b70/flint` is set to 32768. |
| API key | `/etc/b70-flint/api-key` (root 0600, the same key as before) → `LoadCredential=` → `VLLM_API_KEY` in the launcher. Never on argv. |
| Auth | `flint_auth.FlintAuth` (`--middleware`) requires `Authorization: Bearer` on every route except `/health`, and allows loopback `/metrics` for the LCD collector. vLLM's own `--api-key` covers only `/v1`, `/v2`, `/inference` and `/cohere`; `/invocations` and the other routes would otherwise run inference without a key ([vLLM security docs](https://docs.vllm.ai/en/latest/usage/security.html)). |
| Rollback | `flint-llamacpp.service` runs the exact previous llama.cpp b12499 argv with `--api-key-file %d/api-key`. It is disabled and `Conflicts=`/`After=` flint-inference. |
| Capacity | `--max-model-len auto` selects **141,056 tokens** (4.83 GiB KV) at memory fraction 0.97 with 4096-token prefill chunks. 2.37 GiB of 31.89 GiB stays free on the device while serving. The model's documented limit, 262,144, doesn't fit with bf16 KV; see `results/20261120-flint-vllm-serving/context-sizing.txt` for the sweep. Pi `b70/flint` `contextWindow` is 141056. |

Operations (`serve/README` on the guest):

```
sudo systemctl disable --now flint-inference && sudo systemctl enable --now flint-llamacpp   # roll back
sudo systemctl disable --now flint-llamacpp && sudo systemctl enable --now flint-inference   # roll forward
journalctl -u flint-inference -f
```

`systemctl edit flint-inference` can set `FLINT_NUM_SPEC`, `FLINT_MAX_MODEL_LEN`, `FLINT_MAX_BATCHED`
and `FLINT_PREFIX_CACHING`. `maintenance.sh` experiments stop whichever live unit is active and
restart it on exit (`NO_RESTORE=1` leaves it stopped). `swap-build.sh` now also installs
`xpu_draft_head.py` and `qwen3_dflash2.py` (stock `qwen3_dflash2.py.orig` for older builds).

## Validation of the live unit (`serve/validate.py`, `serve/prefix_check.py`)

- **Auth, from the guest and the host:**
  - `/health` returns 200.
  - `/v1/models` without a key or with a wrong key returns 401, as do `/invocations`, `/tokenize`
    and `/docs`. Remote `/metrics` returns 401.
  - With the key, `/v1/models` returns `["flint"]`.
- **Parity:** 8/8 prompts are bitwise identical to the spec5 k = 4 benchmark build. Tokens are
  identical to non-spec greedy decode on 8/8.
- **Tool calls:** all PASS, streaming and non-streaming:
  - single call `get_weather(Paris)`;
  - parallel calls `get_weather` + `get_time` (Tokyo);
  - a tool-result turn whose answer uses the returned 17 °C;
  - `reasoning_content` is split out.
- **Long context, hidden-phrase retrieval at greedy, all found:**

  | prompt tokens | TTFT | decode tok/s | acceptance length |
  |---|---|---|---|
  | 1,607 | 0.26 s | 284 | 3.52 |
  | 5,644 | 0.81 s | 194 | 2.67 |
  | 10,556 | 1.63 s | 209 | 3.14 |
  | 15,233 | 2.47 s (chunked prefill) | 197 | 2.94 |

- **Prefix caching:**
  - Agent-style multi-turn with a 5k-token prefix: TTFT drops from 700 ms to 80–125 ms, with
    4,992 cached tokens per turn and unchanged answers and acceptance.
  - Three ~8k-token hidden-phrase documents, each sent cold, then repeated (cache hit), then with
    a new tail: 9/9 correct. TTFT is about 1.2 s cold and about 70 ms warm.
  - An earlier 10.5k run that missed was a 1,024-token budget exhausted by reasoning; it passes
    at 2,048.
- **Pi client end-to-end, from the host:** `pi -p --no-session --model b70/flint --tools
  read,edit,write,bash` fixed a planted bug in `calc.py` / `calc2.py` and reported the executed
  output (5, 42). The guest log shows the requests from `192.168.122.1`, all 200.
- **Rollback round trip:** `systemctl start flint-llamacpp` stopped vLLM and served llama.cpp
  with key auth. The eval below ran on it. `systemctl enable --now flint-inference` brought vLLM
  back (35 s).
  - The first switch raced: llama.cpp started while vLLM was still deactivating. The fix is
    `After=flint-inference.service` in the rollback unit, which orders the stop first in either
    direction.

## Quality: GPTQ (vLLM + DFlash 2) vs GGUF Q4_K_M (llama.cpp)

`serve/eval_flint.py` is the standard subset of BENCHMARKING_STANDARDS.md §9.2:
- GSM8K: 100 seeded test items;
- IFEval: 100 seeded items, scored with Google's `instruction_following_eval`;
- HumanEval+: all 164 (`evalplus`).

Both backends ran through the same `:8080` chat API with thinking on, greedy decoding,
`max_tokens 4096` and concurrency 1. `serve/eval_paired.py` gives the paired per-item comparison.

| | vLLM | llama.cpp | only vLLM correct / only llama.cpp correct | hit 4,096-token limit (vLLM / llama.cpp) |
|---|---|---|---|---|
| GSM8K | 89 / 100 | 91 / 100 | 2 / 4 | 8 / 7 |
| IFEval strict (prompt level) | 85 / 100 | 89 / 100 | 2 / 6 | 12 / 6 |
| IFEval loose (prompt level) | 88 | 91 | | |
| HumanEval base pass@1 | 94.5% | 91.5% | | |
| HumanEval+ pass@1 | **146 / 164 (89.0%)** | 142 / 164 (86.6%) | 9 / 5 | 7 / 13 |
| total wall time, 364 items | **29.8 min** | 155.9 min | | |

None of the differences is significant (exact McNemar on the discordant pairs, p ≈ 0.29–0.69).
Excluding items that either backend truncated, GSM8K is 87 vs 88 of 90 and IFEval is 84 vs 84 of
88. The IFEval gap is the token limit, not instruction following. Conclusion: no measurable
quality regression from the GGUF → GPTQ switch at this sample size.

## Context raised to the memory limit, and RSI settings check (2026-11-20, later)

On the deployed 141k config, long-context hidden-phrase retrieval at greedy found the phrase every time
(`results/20261120-flint-vllm-serving/validate/validate-ctx141k*`). There were no restarts or
memory errors, and short-prompt parity stayed bitwise identical to the 16k config.

| prompt tokens | TTFT | decode tok/s | acceptance length |
|---|---|---|---|
| 1,574 | 0.24 s | 294 | 3.65 |
| 16,109 | 3.1 s | 195 | 3.02 |
| 32,168 | 8.4 s | 142 | 2.39 |
| 65,797 | 27 s | 145 | 2.82 |
| 108,315 | 72 s | 114 | 2.80 |
| 137,209 | 115 s | 104 | 2.89 |

A 141,057-token prompt is rejected cleanly with HTTP 400. Warm multi-turn TTFT is still about
80 ms with prefix caching.

Comparison with RecursiveSI's recommendations
([model card](https://huggingface.co/RecursiveSI/flint), `sglang/README.md`,
`sglang/serve_h200.sh`, upstream `generation_config.json`):

| RSI | here |
|---|---|
| T 1.0, top-p 0.95, top-k 20 as server defaults | same (`generation_config.json`) ✓ |
| `max_tokens` ≥ 32768 for hard problems | Pi `maxTokens` 32768 ✓ (was 4096) |
| context 262,144 | 141,056 (the memory limit with bf16 KV) |
| keep `reasoning_content` on replayed assistant turns | Pi replays the reasoning field; vLLM maps `reasoning`/`reasoning_content` to the template's `reasoning_content` → `<think>` ✓ |
| `--chunked-prefill-size 4096` | `--max-num-batched-tokens 4096` ✓ |
| `--tool-call-parser flint --reasoning-parser flint` | `glm47` (vLLM has no `flint` parser; tool and reasoning tests pass) |
| `--enable-fp32-lm-head` | **not matched**: logits come from the bf16 lm_head |
| greedy "not validated" | our evals use greedy for comparability; serving defaults are RSI's |

## Limits and follow-ups

- **One sequence at a time** (`--max-num-seqs 1`, as llama.cpp `--parallel 1`). Concurrent
  clients queue. Raising it means verify batches with M > 8 leave the batch-invariant path; it
  needs its own benchmark.
- **Maximum context is 141,056 tokens**, not the model's documented 262,144, because bf16 KV
  doesn't fit more on 32 GB. Reaching 262k would need an fp8 KV cache, which is lossy and not
  enabled.
- **Spec vs non-spec logprobs** differ after 256-token attention-segment boundaries (numerical
  only; see the hill-climb doc).
- **Not tested here:** a guest reboot (the unit is enabled; earlier sessions avoided reboots
  because of the process-local Intel runtime workaround), sustained thermal load, and
  standard-tier (BetterBench / `vllm bench serve`) runs.
