# Million-position native quant-aware MTP experiment

**Development result: no demonstrated serving gain, despite a 7.92% improvement in the offline selection objective. Stock remains deployed.** This is a completed larger attempt, not evidence that the broader quant-aware-training hypothesis is exhausted.

## Fresh held-out ABBA

Checkpoint selection was frozen before accessing the fresh test set: `lr5e6_decay`, step **400**. All three training configurations were evaluated on dev; only this preselected checkpoint was tested against stock. Order: stock → tuned → tuned → stock. All **63 prompts × 4 cells = 252 requests** completed, with no skipped test prompts.

| Pooled metric | Stock | Tuned | Relative change | Family-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Accepted draft tokens / speculative pass | 2.54412 | 2.54438 | +0.010% | −2.634% to +3.135% |
| Decode tokens/sec | 86.5626 | 86.5458 | −0.019% | −1.913% to +2.303% |
| End-to-end tokens/sec | 75.2453 | 75.4934 | +0.330% | −1.424% to +2.366% |
| Draft acceptance fraction | 63.6030% | 63.6096% | +0.0066 percentage points | — |
| Median request latency | 3.67975 s | 3.66321 s | −0.45% | — |
| Median TTFT | 0.465583 s | 0.465580 s | Essentially unchanged | — |

Unconditional acceptance by draft position, stock → tuned: **84.583→85.041%, 69.166→68.494%, 55.497→55.593%, 45.167→45.311%**. The first-position improvement was offset by a second-position decline; the pooled total was flat.

Decode throughput is `sum(generated_tokens - 1) / sum(server request_decode_time_seconds)`. End-to-end throughput is `sum(generated_tokens) / sum(client request wall seconds)`. The pass denominator is native `spec_decode_num_drafts_total`, not every verifier forward. Native speculative counters can include terminal candidates discarded by the API. Completion-counter deltas matched each request's exact API token count; prefix-cache hits were zero.

### Variability and limitations

- Test data spans **10 fresh source-session families**. Bootstrap resamples families, retaining both runs together: 10,000 resamples, seed 42. This is limited evidence, not a comprehensive estimate of temporal or hardware variability.
- Exact token matches: stock repeats **29/63**, tuned repeats **26/63**; cross-head comparisons **29/63** and **31/63**. Natural output lengths therefore differ, including within stock.
- Only **18 prompts** had identical output tokens in all four runs. On that post-hoc diagnostic subset, acceptance improved 0.853% and decode speed 0.686%; both confidence intervals still include zero. It is not the primary result.
- No functional quality score or execution of generated tools. Output differences alone establish neither a quality regression nor equivalence.
- This is not a standard-publishable/community-comparable result: no full BetterBench, concurrency/long-context suite, complete environment capture, or continuous thermal trace. No production promotion.

## The actual million-position dataset

- New engineering sources yielded **784 fresh contexts / 95 families** after excluding all previous contexts and families. Offline, network-disabled Gitleaks v8.30.1 flagged none. Scanner clearance is not proof that all sensitive content is absent.
- Frozen fresh split: **667 train / 54 dev / 63 test contexts**, across **77 / 8 / 10 families**. Train IDs and families were checked disjoint from both old and fresh held-out splits. The test file remained local until dev selection was frozen.
- New generation admitted **573 train contexts** per seed (94 too long, skipped without truncation), and **38 dev contexts / 14,263 useful positions** (16 too long). Eight separate canary contexts passed before the larger generation.
- Training sampling seeds **42–45**, temperature 0.6, top-p 0.95, top-k 20, natural tool/EOS termination. Maximum response 1,792 within a **2,048-token complete-sequence cap**. These are multiple sampled continuations of a finite prompt set, not thousands of independent new prompts.
- Reused the existing **340 training contexts / 162,964 useful positions** from the verified native capture path, never the old dev/test data.

| Cache stage | Cumulative distinct trajectories | Cumulative useful positions | Exact duplicate records rejected in stage |
|---|---:|---:|---:|
| Previous train cache | 340 | 162,964 | 0 |
| Seed 42 | 913 | 381,853 | 0 |
| Seed 43 | 1,475 | 603,165 | 11 |
| Seed 44 | 2,046 | 829,851 | 2 |
| Seed 45 | **2,607** | **1,043,893** | 12 |

Final cache: **913 distinct prompt contexts, 2,607 distinct full trajectories, 126 training families, 3,409,232 input-token positions**. Deduplication uses exact complete token IDs plus loss mask. The useful-position count is not an epoch multiplier, but it does include shared prefixes across distinct sampled trajectories; it is not a count of unique prefixes. Every merged tensor passed the existing trainer's shape, dtype, finite-value, position, vocabulary, and supervision checks and the explicit train-only prompt/family allowlist.

## Training and predeclared selection

All settings warm-started stock native MTP; no architecture switch, no BF16 teacher, no core fake-quant QAT. Frozen embeddings and RTN-effective draft LM head; FP32 masters, FP16 XPU compute, BF16 native overlays, existing RTN packing at serving load. Native recursive depth 4, eight roots/sequence, two epochs, accumulation 8, **652 optimizer updates per setting**, seed 42, checkpoint interval 100 (plus step 0 and final).

| Training setting | Best step by common dev criterion | Common post-RTN objective |
|---|---:|---:|
| Stock | 0 | 1.109299 |
| LR 5e-6, depth weights `[1,.5,.25,.125]` | **400** | **1.021431** |
| LR 1e-6, depth weights `[1,.5,.25,.125]` | 652 | 1.027323 |
| LR 5e-6, depth weights `[1,1,1,1]` | 400 | 1.027449 |

**Common selection weights `[1,.5,.25,.125]` were used across every run**, recomputing the score from saved per-depth dev CE. The flat-loss run's differently weighted training objective was not compared directly. Stock was eligible. Three independently evaluated stock common objectives agreed within 0.000007.

Collection, deduplication, and all three training runs completed in **5h54m**. Fresh held-out ABBA took **31m48s**. Training peak XPU allocation was **17,707,010,560 bytes**, reserved **20,101,201,920 bytes**. Collection stopped after reaching the token target; only four of the six permitted seeds were needed. No cloud rental; no target verifier resident during training.

### What the offline improvement actually changed

Post-RTN dev values for stock → selected step 400:

| Depth | CE | Argmax agreement | Evaluated positions |
|---|---|---|---:|
| 1 | 0.51324 → 0.47791 | 83.895% → 84.330% | 14,263 |
| 2 | 0.61221 → 0.53435 | 81.250% → 80.592% | 304 |
| 3 | 0.76827 → 0.74831 | 79.605% → 79.605% | 304 |
| 4 | 0.78309 → 0.71417 | 77.961% → 80.263% | 304 |

Lower CE did **not** imply a comparable increase in correct top-token decisions. Deeper dev estimates also used only 304 sampled positions each. This is a concrete reason to investigate acceptance-aligned validation/selection before spending on another data-size increase; it is **not a proven diagnosis** of every runtime effect.

A next bounded diagnostic would compare candidate checkpoints on dev using joint accepted-prefix agreement and native serving acceptance, with stronger root coverage. It must not reuse this now-consumed test set for selecting another checkpoint. Quantized-feature training vs a BF16-trained control also remains untested.

## Runtime parity and verification

Same pinned target and runtime as [the preceding attempt](../20260915-qwen38-native-quant-aware-trace-tune/README.md):

- `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`.
- `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`, vLLM `0.27.2rc1.dev77+gac7509e2b`, torch `2.13.0+xpu`, Transformers `5.15.0`.
- Arc Pro B70 32GB; final observed power cap 275W. Native MTP4, C1, target GPTQ INT4 g128, existing draft RTN INT4 g128, FP16 runtime, FP8 KV, configured context 212,992.
- Same temporary reference bundle: balanced mode, graph sizes `[1,2,4,8]`, uniform-prefill guard, loopback binding. Prefix caching enabled but unique salts impose cold requests. Same tokenizer, template, four-tool profile, and per-context thinking mode.
- Test: greedy/seed 42, natural termination, response cap 1,024, complete sequence limit 32,768. Four synthetic warmups excluded per cell. Capture off for timing. Only the head overlay differs between A and B.
- All temporary servers stopped. Persistent launcher SHA256 stayed `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

Five new merger tests passed in the pinned CPU-only container, including real CLI tensor loading, duplicate rejection, held-out-family rejection, invalid tensor rejection, and preservation of originals. The merger subsequently validated and assembled the real cache through the same path. The unchanged native capture/trainer/benchmark path had already passed the preceding 107-test check and was exercised again here through actual API generation, XPU training, head loading, and ABBA.

Primary-source research retained the existing architecture: the [SpecForge README](https://raw.githubusercontent.com/sgl-project/SpecForge/main/README.md) did not specify a workload-mixture recipe or this pinned native-MTP/XPU path; the [EAGLE README](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/README.md) emphasizes template correctness and measured generation/wall-time comparisons. Neither justified changing the verified native architecture or claiming a prescribed corpus size guarantees improvement.

## Private artifacts / exact commands

Host root: `/home/mike/b70-evals/20260915-mtp-scale-attempt/`.

- `commands/b70-mtp-scale-attempt.sh`: exact collection/merge/three-setting training job, with disk reserve and unchanged-launcher checks; run from the lead host with fresh output paths (it SSHs to `inference-host`).
- `commands/b70-mtp-scale-heldout-abba.sh`: exact fresh-test stock/tuned ABBA.
- `commands/analyze_abba.py`: aggregate analysis; `python3 commands/analyze_abba.py <host-root>` writes an exclusive summary and refuses to overwrite an existing one.
- `merge-{baseline,seed42,seed43,seed44,seed45}.json`, `dataset/train/`, `fresh-dev/`, `selection.json`.
- `training-{lr5e6_decay,lr1e6_decay,lr5e6_flat}/`: logs, checkpoints, per-depth metrics.
- `heldout-{A1,B1,B2,A2}/`: exact launcher/config; corresponding `*-output/request-*/{request,response,measurement}.json`.

Only aggregate results and analysis code are published. Private prompts, per-request results, feature tensors, and weights remain outside Git. Nothing was promoted or deleted to manufacture a favorable result.
