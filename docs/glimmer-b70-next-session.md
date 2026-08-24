# Glimmer B70 — next session

Date: 2026-08-24
Public repo: this one. Phase 0: `docs/glimmer-b70-phase0-20260824.md`.
Program: `docs/glimmer-b70-research-program.md`.

## Live cell (leave it up)

Vulkan llama.cpp `70adb1b`, tmux `muse`, port 18099.

- Target: Meta Dynamic Q4_K_XL (19.7 GB)
- Draft: official DFlash Q4_K_M, `--spec-type draft-dflash --spec-draft-n-max 2`
- `-ngl 99 -ngld 99 -c 262144 -np 2 --kv-unified -fa on -ctk q8_0 -ctv q4_1 -b 512 -ub 512 --no-mmap --jinja`

This is the 1–2 stream cell. Do not put `n_max=2` on 4+ streams.

## What is true

Split numbers only. Decode = log `eval time`. Prefill = log `prompt eval`.

| | Prefill | Decode | Card |
|---|---:|---:|---|
| C1, 83 tok cached | not a prefill row (5 new) | **38 tok/s** | 273 W / 2800 MHz (275 W cap) |
| C1, 1569 tok | **432** | 31 | 244 W |
| C1, 6024 tok | **402** | 27 | 254 W |
| C4 `n_max=1` | not isolated (queueing) | **15.5 /stream** | **244 W** |
| C4 `n_max=2` | not isolated | 3.1 /stream | 132–134 W |
| C8 any `n_max` | not isolated | 2–4 /stream | ~132 W |

Acceptance at `n_max=2` C1 was 0.64 / mean 2.27 — that is a **2-token window**, nearly full. Official DFlash at block 16 is the weak one (Inco mean accept length **4.44**; one NVIDIA tester 38.7% / 6.77 on a clean run, 14% mixed).

C4/C8 collapse is **not** [llama.cpp#27117](https://github.com/ggml-org/llama.cpp/issues/27117). Acceptance stays healthy. The GPU waits on serial DFlash `process()`: per sequence, read back 5 target layers (hidden 6656), then `llama_encode` + `llama_decode` on the draft. `draft()` is batched; inject is not.

Default 128-token gens never reach `content`. `chat_template_kwargs.reasoning_strength=low` does.

## Next session — do not invert this

`70adb1b` has no DFlash2. [PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) merged 2026-08-23. **Bump llama.cpp once**, then:

### 1. DFlash2 on C1 (hours)

Need the new tree anyway. Measure the new drafter before rewriting inject.

- Build Vulkan from a rev that contains #27342. JIT, no `bmg-g31` AOT. Keep `-ub 512`.
- Draft: `incoai/Muse-Glimmer-30B-DFlash2-GGUF` Q4_K_M (1.6 GB). Still `--spec-type draft-dflash`.
- C1 only. `n_max` ∈ {2, 4, 8, 15}. Official sampling (temp 1.0, top-p 0.95, top-k 64).
- Report **prefill, decode, accept length, card W**. Same sky prompt plus one coding prompt.
- Inco claim to beat: Glimmer accept length **5.70** vs official DFlash **4.44** (block 16). Throughput claims are NVIDIA/SGLang; ignore them as B70 numbers.

Kill: binary will not load the GGUF, or greedy output diverges from official DFlash, or C1 decode **worse** than 38 at `n_max=2`. Then stay on official DFlash and go to step 2.

Do **not** take DFlash2 to C4 until step 2. It does not fix serial inject. Higher accept only means fewer ticks.

### 2. Batch `process()` (the C4 job)

On the **same** new tree, not on `70adb1b`.

- `common_speculative_impl_draft_dflash::process()` in `common/speculative.cpp`: one encode+decode for all live seqs, keep the 5 layer activations on device.
- Prove C4 `n_max=2` decode **and** card W come back toward the `n_max=1` 15.5 / 244 W cell.
- Then C8. Then, and only then, DFlash2 at C4.

Kill: still 132 W after a batched inject. Then C4 production stays `n_max=1` and we stop spending the session on spec.

### Cheap parallel (does not need a rebuild)

Ship `reasoning_strength=low` on the Pi/agent template. Otherwise 38 decode tok/s is invisible.

## Do not

- 128k + DFlash + `-ub 8192` (hard reboot)
- `GGML_SYCL_DEVICE_ARCH=bmg-g31`
- Score anything with `completion_tokens / wall`
- Call C4 `n_max=1` a proven win over no-spec until no-spec is split the same way
- DSpark (Glimmer accept 4.48 vs official 4.44; sequential head makes inject worse)
- Start custom Q4_K kernels this session (`iaprof` first)
- C2 / C3

## Instrument

`scripts/glimmer-phase0-instrument.py` (acceptance regex is fixed). Host copy: `~/inference/launchers/`. Receipts: `~/b70-evals/muse-glimmer/<date>-<cell>/`.
