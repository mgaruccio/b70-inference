# Qwen3.8 B70 next-session handoff — 2026-08-23

## Current state

- **Live service:** original BF16 MTP-4 launcher, `~/inference/launchers/start-qwen38.sh`.
- **Health:** LAN and Tailnet health checks passed after final restore.
- **Do not deploy the draft-INT4 overlay yet.** It is faster but failed the final greedy parity gate at ~8k context.
- **Main research brief:** `docs/qwen38-b70-mlxfast-research-progress-20260823.md`.

## Strongest measured service configuration

Draft-INT4, MTP-4, XPU graph capture capped at 64, with the existing 131,072 context / 0.88 memory / FP8 KV / MBT 8192 / C1 contract:

| Row | Short warm median | Medium | ~98k | Status |
| --- | ---: | ---: | ---: | --- |
| BF16 baseline | 44.03 tok/s | 16.91 tok/s | 0.639 tok/s | accepted baseline |
| Draft-INT4 cap64 MTP-4 | **54.67 tok/s** | **17.83 tok/s** | **0.644 tok/s** | stable, but quality-rejected |

The cap64 row passed repeated tokenizer-verified ~97,962-token public API controls and sustained forced 2,048-token completions. Cap128 failed startup. MTP2 was slower; MTP3 failed startup.

## Blocking quality result

Greedy comparison at `temperature=0`, `seed=12345`:

- short: exact BF16/candidate match;
- coding: exact match;
- ~8k context: content mismatch, both ended by `length`.

Treat this as a hard gate. The candidate must not replace BF16 until token-level traces identify the first divergent target/draft decision and the mismatch is eliminated or a different candidate is proven correct.

Artifacts:

```text
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-bf16/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-gdn/
~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260823T205000Z-greedy-comparison.json
```

## Implemented patch artifact

The repository now has a narrow, reversible metadata-only overlay derived from the static-buffer half of unmerged vLLM PR #43955:

```text
patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py
```

Relevant commits:

- `c479140 Add versioned B70 GDN metadata overlay`
- `a3415d7 Guard B70 GDN metadata arange capacity`
- `818e8f9 docs(qwen38-b70-gdn): correct runtime target path for GDN metadata overlay`

It reuses speculative metadata buffers and skips redundant copies. It does **not** alter `GPUModelRunner`, target-token logic, GDN state semantics, mixed-batch semantics, ABI, or compiled XPU kernels.

Apply only after the existing MTP and boundary patches, and target the imported runtime module:

```bash
TARGET="$(python -c 'import vllm.v1.attention.backends.gdn_attn as m; print(m.__file__)')"
python /patch_mtp.py
python /patch_boundary.py
python /patch_gdn_metadata.py --dry-run --path "$TARGET"
python /patch_gdn_metadata.py --path "$TARGET"
```

The overlay was stable at C1 p98k and combined with v5 at C2, but its 54.94 versus 54.67 tok/s C1 result is noise and its single C2 sample was slower. It is not a speed win yet.

## C2 result

`patch_gdn_mixed_split_v5.py` successfully handled one overlapping 8k prefill + speculative-decode C2 smoke. The service survived and both requests returned 200. This is a correctness result only; repeat it before claiming throughput improvement.

## External research direction

The relevant Apple work is [MLXFast](https://www.yukon.org/mlxfast), specifically the [Qwen3.8 MTP challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge), not generic MLX tuning.

The promoted winner reuses residual/RMSNorm chunk-sum output in MTP verification widths 4–9, eliminating standalone intermediate fills. Its Metal code cannot be copied directly because MLXFast uses affine G64 quantization whereas B70 uses symmetric GPTQ G128 W4A16. The transferable agenda is to profile and remove equivalent redundant buffers/reductions/launches in the B70 MTP verification path.

## First next-session actions

1. Read `docs/qwen38-b70-mlxfast-research-progress-20260823.md` and this file.
2. Investigate the greedy ~8k mismatch first with token IDs, target/draft logits or acceptance events, and trace the first divergence. Do not spend time on a full cohort until this is resolved.
3. Resolve Prime balance, then run the one-task Pi/Harbor smoke with unchanged harness/sampling/context only after parity passes.
4. Repeat the cap64 C1 and v5 C2 rows enough for confidence intervals.
5. Profile MTP verification at widths 4–9. Attribute time to W4A16 GEMMs, RMSNorm/reductions, GDN state movement, metadata preparation, graph replay, and CPU synchronization.
6. Only then prototype an XPU/SYCL fusion based on a measured B70 bottleneck. Do not port MLX Metal code or its custom proposal head blindly.

## Known blockers

- `prime eval run` did not start the one-task Harbor smoke: `Payment required. Insufficient balance.`
- The installed `vllm-xpu-kernels 0.1.12.3` package contains compiled GDN `.so` files but no kernel source/build headers; meaningful kernel work needs an upstream source checkout/build path.
- Mixed XPU GDN behavior currently depends on Python v5 workaround logic; a durable solution belongs upstream in SYCL/XPU kernel partition/merge work.
