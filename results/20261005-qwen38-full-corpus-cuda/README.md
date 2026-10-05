# Full-corpus native Qwen3.8 MTP: CE-only experiment

**Development-tier CUDA BF16 result, not production qualification, a material speedup claim, or a B70 result. No configuration promoted.**

Unlike the earlier 64/16 pilot, this run used the entire planned disjoint public corpus: **374 train / 64 dev / 64 test**, fresh unquantized BF16 teacher captures, **10 complete epochs and 3,740 actual optimizer updates**. Only the native MTP head was trained; shared embeddings, LM head and target trunk remained frozen. Test requests did not select the checkpoint or draft depth.

## Training result

| Checkpoint | Weighted dev BF16-export CE objective |
|---|---:|
| Stock, step 0 | 1.0881145561888124 |
| **Selected, step 1,496** | **0.8809706300655034** |
| Final, step 3,740 | 1.0149416052014806 |

The selected objective is 19.04% below stock. Selection considered stock plus each completed epoch on all 64 dev captures. Teacher-forced CE and argmax agreement are **not serving acceptance rates**. Later epochs worsened the dev objective relative to the selected checkpoint.

Training used LR 1e-6, seed 42, accumulation 1, recursive depth 4, eight roots, depth weights `[1,1,0.8,0.8]`, max sequence 2,048, logits chunk 64, and checkpoint/dev evaluation every 374 updates. FP32 masters/AdamW, BF16 compute/export; no state or KL objective. Total supervised tokens by depth: `[441270,29920,29920,29920]`; deeper supervision is much sparser. Peak training CUDA allocated/reserved memory: 13,020,583,424 / 15,107,883,008 bytes.

## Real held-out serving results

Each cell completed the same untouched 64 requests, concurrency 1, greedy seed 42, thinking disabled, natural EOS with a 512-token cap, prefix caching off. Order was no-spec then stock/candidate/candidate/stock separately at depths 4 and 8. Only the native MTP overlay changes within each paired depth. Acceptance counts exclude bonus tokens. Functional checks use the existing sandbox, not a full quality benchmark.

| Cell | Accepted / draft pass | Median decode tok/s | Median E2E tok/s | Functional pass / 64 |
|---|---:|---:|---:|---:|
| No-spec | — | 30.808 | 30.392 | 55 |
| D4 A1 stock | 3.2816 | 99.680 | 90.984 | 55 |
| D4 B1 candidate | 3.2073 | 100.231 | 90.496 | 56 |
| D4 B2 candidate | 3.2609 | 100.571 | 90.928 | 54 |
| D4 A2 stock | 3.2670 | 100.382 | 90.975 | 54 |
| D8 A1 stock | 4.8640 | 112.523 | 99.992 | 54 |
| D8 B1 candidate | 4.7541 | 117.245 | 100.823 | 55 |
| D8 B2 candidate | 4.9344 | 116.801 | 101.612 | 55 |
| D8 A2 stock | 4.7844 | 110.926 | 99.883 | 56 |

Depth-8 pooled acceptance: stock **12,368 / 2,564 = 4.8237**, candidate **12,624 / 2,608 = 4.8405** (+0.35%). Depth-4 pooled acceptance: stock 3.2743, candidate 3.2337 (lower). Averaging the two cell E2E medians gives D8 99.937 vs 101.218 tok/s (+1.28%) and D4 90.979 vs 90.712 (-0.29%). These are **means of cell medians**, not pooled request medians; no confidence interval or material gain is established. The large stock-MTP versus no-spec difference is not a trained-head gain.

## Fidelity and state caveats

Exact output-token matches out of 64:

| Comparison | D4 | D8 |
|---|---:|---:|
| Stock A1 versus stock A2 | 58 | 55 |
| Candidate B1 versus candidate B2 | 55 | 56 |
| Stock A1 versus candidate B1 | 53 | 58 |
| Stock A2 versus candidate B2 | 64 | 53 |
| No-spec versus stock A1 | 58 | 59 |
| No-spec versus candidate B1 | 56 | 60 |

Even stock repeats and no-spec versus stock are nonidentical despite nominal greedy settings. Do not attribute all divergence to training or claim lossless identity; runtime/numerical fidelity needs investigation. Functional scores vary from 54 to 56 across spec cells.

`state-drift.json` is a **teacher-forced diagnostic on four dev captures / 32 roots**, not free-running drift or acceptance. At depth 1, candidate relative L2 is worse (0.9726 vs 0.9481) and teacher-to-head KL is worse (0.0954 vs 0.0651). At depth 2 KL is 0.2067 vs 0.1096; at depth 4 it is 0.9401 vs 1.0024. CE improvement did not consistently improve state/distribution matching. Coordinate and state-boundary equivalence must be audited before using these diagnostics as distillation targets.

## Environment and executable process

- One physical H100 PCIe, 81,559 MiB, Scaleway Paris; driver 570.124.06.
- `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Serving: vLLM 0.27.1, torch 2.13.0+cu130, pinned image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`.
- Trainer: Python 3.12.3, torch 2.14.1+cu130, Transformers 5.15.1. Trainer source SHA256 `f75d7865a34c5b87975e78ae261ba375e57cd233daf14e2be327013eff5e5e3f`, repository commit `1daf7b31`.
- CUDA 13 forward-compatibility userspace libraries; unchanged kernel driver. All serving arms set `VLLM_ENABLE_CUDA_COMPATIBILITY=1` and `VLLM_USE_V2_MODEL_RUNNER=0`. Trainer sets compatibility LD_LIBRARY_PATH and OMP_NUM_THREADS=8.

Executed entrypoint, after bootstrap and capture-restart corrections:

```sh
bash /home/mike/b70-evals/20261005-qwen38-full-corpus-cuda/launch-pilot.sh resume
```

The inherited `pilot` filenames do not imply reduced data. The accompanying workflow/launcher snapshots preserve the exact subprocess construction. The full archive contains per-cell `command.json`, runtime inspection, rendered prompts, SSE, request results, metrics before/after, functional checks, teacher captures, training history and checkpoints. The process validated all 374/64 records through the actual native API, trained all epochs, selected on dev, served nine 64-request cells through the real API, and streamed the full R2 archive readback before requesting deletion. `verify-final.py` records the exact completeness/hash assertions. Local workflow exit file is `0`.

## Artifacts and cleanup

Local root: `/home/mike/b70-evals/20261005-qwen38-full-corpus-cuda/`.

R2 prefix: `r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-full-corpus-cuda/`.

Final archive: **14,339,297,280 bytes**, seven ordered parts, whole SHA256 **`86632b201f09facabf33824d75a062f364ab67eeb67b2e19a39413bfb6539244`**. `final.tgz.parts.json` gives individual object names, sizes and SHA256 values. The filename is inherited; the workflow writes an uncompressed tar, so use `tar -xf`, not forced gzip decoding. Large weights/captures are external, not committed. Private signed-URL manifests are excluded.

Completed task `b92ed6638` logged `FULL_CORPUS374_64_TEN_EPOCHS3740_UPDATES_NINE_TEST_CELLS_AND_FULL_R2_READBACK_VERIFIED` with the whole hash, then requested deletion of owned lease `b95a876b-1c5f-4826-b39b-baec038225ce`. A subsequent authenticated GET to `https://api.shadeform.ai/v1/instances` returned no matching ID or name. No new lease was created for publication.

## Retained failures and limits

- DenVR readiness timeout/cancellation, bootstrap CUDA driver incompatibility, invalid dev split label, and existing incomplete capture directory are retained separately. Dev labels were normalized to native `heldout` without changing IDs/messages/membership; saved train captures were reused, not replaced by a pilot.
- Earlier B70 trained-head attempts failed on missing parent/root-owned paths before serving. **No trained-head B70 result exists here.** Do not pool CUDA BF16 with quantized B70 numbers.
- This is a small public coding subset with unknown pretraining exposure. It is not BetterBench full/20-pass, `vllm bench serve`, a concurrency sweep, a long-context sweep, a 500-prompt fidelity study, or the full fixed quality suite. Host/power/environment fields retained in raw artifacts have not all been audited for standard-publication completeness.
- Consequently the standard publication checklist is **not satisfied**. This report publishes development evidence only; no production promotion or community-comparable performance claim.

The completed [source audit and proposed cached-corpus state/KL ablation](ALIGNMENT_AND_ABLATION.md) records the teacher-forced versus sampled-draft distribution gap and remaining live parity gates. No definite normalization/base-label bug was found; partial-acceptance row selection, branch positions and HF/vLLM numerical parity remain unvalidated. No ablation has been implemented or launched; a fresh compute authorization is needed for live checks after the original lease cleanup.

Primary sources recorded for the completed run: [NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html), [Shadeform asynchronous instance creation](https://docs.shadeform.ai/api-reference/instances/instances-create), [PyTorch optimization/epochs](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), [TorchSpec capture/training](https://pytorch.org/blog/torchspec-speculative-decoding-training-at-scale/), [official MBPP corpus](https://github.com/google-research/google-research/tree/master/mbpp). These inform methodology, not proof of a trained-head serving benefit.
