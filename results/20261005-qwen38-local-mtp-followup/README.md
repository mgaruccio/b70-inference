# Native Qwen MTP: verified 1,000-update local follow-up

**Development-tier training result; not a serving-speedup claim or production promotion.**

After two advertised-available cloud rentals failed readiness and were canceled, the native head was trained on the existing RTX 5080. There were **1,000 actual optimizer updates** over the original **64 train / 16 dev** public on-policy CUDA BF16 teacher captures. This was **not** the planned expanded 374/64 corpus. Only the 424,699,392-parameter native MTP head was trainable; shared embeddings and LM head were frozen, and the base trunk was not constructed locally.

## Result

The preregistered criterion selected **step 250**, not the final checkpoint:

| Checkpoint | Weighted dev BF16-export CE objective |
|---|---:|
| Original head, step 0 | 1.1607012402472683 |
| **Step 250, selected** | **1.1377569169518464** |
| Step 500 | 1.3386736867779565 |
| Step 750 | 1.2779928817281017 |
| Step 1,000 | 1.4059151993701626 |

Longer training on this small reused corpus degraded the dev objective. This is an overfitting warning, not evidence that more updates help. These teacher-forced objectives and token agreements are **not speculative acceptance rates**. At publication, the selected head's serving acceptance and throughput remain unmeasured; its separately launched B70 quantized-target transfer evaluation must not be pooled with the CUDA BF16 baseline.

Verified final export: **11 native MTP tensor keys changed**, and the trainer recorded all 1,000 updates. Across the run: 1,000 sequence presentations, 290,075 input tokens, 289,888 observed hidden rows, and 135,787 useful positions. Supervised token counts by depth were `[135787, 8000, 8000, 8000]`; deeper metrics have substantially smaller sample counts.

## Configuration and observed verification

- Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- RTX 5080 16GB; existing desktop processes left untouched.
- Python 3.14.7, PyTorch 2.14.1+cu130, Transformers 5.15.1.
- FP32 master parameters/AdamW with BF16 compute and BF16 checkpoint export.
- Four-step recurrence, depth weights `[1, 1, 0.8, 0.8]`, eight roots, LR `1e-6`, seed 42, accumulation 1, max sequence length 1,024, logits chunk 16, export/dev validation every 250 updates. No state-supervision objective in this run.
- Physical CUDA BF16 matmul smoke passed; one real optimizer update plus export/dev validation passed before the full run.
- Peak allocated GPU memory: 13,017,441,792 bytes (12.12 GiB); reserved: 14,069,792,768 bytes (13.10 GiB). Maximum process RSS: 9,673,292 KiB.
- Two regression cases reproduced retained-gradient checkpoint-validation failures on the original trainer; both passed after commit `0f734404`. The targeted CPU file reported **67 passed, 1 skipped**. Real CUDA export/dev validation then passed through 1,000 updates.

The fix clears obsolete gradients after each optimizer/scaler update, before export validation constructs another head. Earlier failed preflight and post-update validation attempts remain in the raw archive; they are not counted as successful full runs.

Exact executed trainer argv is in `command.json`. Top-level reproducible execution was:

```sh
bash /home/mike/b70-evals/20261005-qwen38-local-mtp-followup/verify-and-resume.sh
```

This exercises the real trainer CLI, reads the verified teacher captures, and checks actual updates and exported tensor changes. All local weights, logs, commands, captures and failures are retained rather than deleting evidence.

## Artifacts

Local root: `/home/mike/b70-evals/20261005-qwen38-local-mtp-followup/`.

R2 prefix: `r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-local-mtp-followup/`.

- Full raw archive `final.tgz`: 8,496,967,680 bytes; SHA256 `410218b207e0f60cb1dad74100692423aad5a8b7df8626e735921f21f0e22363`.
- Selected step-250 `selected-mtp.safetensors`: SHA256 `b27365c402a7083555fbc870c6caf9de60459e810c9987c311415d39c3686594`.
- Final step-1,000 `tuned-mtp.safetensors`: SHA256 `9540e1e6873f802ac8c30479a15ad92d6257bb1475f47da97d2422bfd26b1a10`.

The original single-part raw-archive upload failed because the file exceeds R2's single-part limit. No training was rerun for this upload failure. Repair used existing `rclone` multipart support, followed by a **complete remote readback whose SHA256 matched the local archive**:

```sh
bash /home/mike/b70-evals/20261005-qwen38-local-mtp-followup/archive-multipart.sh
```

Compact observed summaries accompany this report; the raw archive includes full training history, checkpoint metrics, original/fixed source snapshots, commands, environment and failure logs. It excludes private signed-URL manifests.

Primary sources informing the runtime/transfer approach: [PyTorch CUDA memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html), [PyTorch installation guidance](https://pytorch.org/get-started/locally/), and [Cloudflare R2 limits](https://developers.cloudflare.com/r2/platform/limits/). Expandable allocator segments were used as a fragmentation aid, not as a memory-capacity guarantee; R2 multipart is required for this archive size.

## Next decision

Measure the selected checkpoint through the actual native speculative serving path against fresh stock heads at depths 4 and 8, including no-spec and stock-overlay controls. If serving gains are absent, do not spend on more tiny-corpus updates. Collect a larger disjoint corpus on confirmed working capacity before the next training experiment. This run does not establish the proposed independent-head comparison, alternating recurrence, state distillation, dynamic recursion, or production readiness.
