# Saved KL checkpoint sweep: dev D8, no retraining

**Completed development-tier evaluation: all ten saved epoch heads, 64 dev requests per cell, 17 cells / 1,088 measured requests. Zero optimizer updates, no test-set selection or promotion.**

## Conclusion

The incumbent CE-selected **step 2,618 also ranked first by dev D8 acceptance**, but barely: **5.112026** accepted draft tokens/pass versus step1,870 **5.108731**, a **0.0645%** gap. Four fresh repeat cells of the exact same step2,618 head ranged **5.026656–5.093982**. That variation substantially exceeds the ranking gap. **No better saved head or robust checkpoint ordering was demonstrated.**

This supports keeping step2,618 as the experimental reference rather than replacing it on the strength of this sweep. It does not support more identical training epochs: final epoch10 had lower acceptance, although its one-cell throughput was higher. Acceptance and throughput are distinct metrics, and the predeclared selection criterion was acceptance. No post-hoc throughput winner is promoted.

## All ten epoch heads

Same-host, same64 dev prompts, native D8, full BF16 target/shared heads, unchanged decoding/verification. Accepted/pass below excludes the bonus token. Each throughput entry is a per-cell request median, not a pooled median.

| Step / epoch | Accepted draft tokens/pass | Decode tok/s median | E2E tok/s median | Functional /64 |
|---|---:|---:|---:|---:|
| 374 /1 | 5.025122 | 120.1326 | 103.3469 | 51 |
| 748 /2 | 5.074407 | 122.6766 | 103.3354 | 52 |
| 1122 /3 | 4.983962 | 124.5112 | 106.1716 | 51 |
| 1496 /4 | 4.984702 | 122.0218 | 104.7628 | 52 |
| 1870 /5 | 5.108731 | 123.4808 | 107.9664 | 52 |
| 2244 /6 | 4.932486 | 123.6405 | 104.6215 | 51 |
| **2618 /7** | **5.112026** | 122.6804 | 105.1871 | 52 |
| 2992 /8 | 4.884161 | 124.7600 | 107.0752 | 51 |
| 3366 /9 | 5.068740 | 127.7777 | 107.9019 | 51 |
| 3740 /10 | 4.992695 | 128.6329 | 109.6710 | 51 |

Execution order was fixed before launch by seed42: **2992,1496,1122,3366,2244,2618,3740,1870,374,748**. All ten ran; no reduced subset or new training was substituted.

## Stock controls and confirmation

| Cell | Accepted draft tokens/pass | Decode tok/s median | E2E tok/s median | Functional /64 |
|---|---:|---:|---:|---:|
| No-spec | — | 30.7001 | 30.1817 | 52 |
| Stock D8 start | 4.999190 | 115.7943 | 102.0837 | 51 |
| Stock D8 end | 4.994337 | 115.7046 | 101.9015 | 51 |
| Confirmation A1 (2618) | 5.026656 | 124.1036 | 104.5754 | 51 |
| Confirmation B1 (2618) | 5.031528 | 124.3101 | 105.3355 | 51 |
| Confirmation B2 (2618) | 5.031528 | 124.2628 | 105.1270 | 51 |
| Confirmation A2 (2618) | 5.093982 | 123.4500 | 106.6821 | 51 |

Because the dev-acceptance winner equals the prior CE-selected head, **A and B load exactly the same checkpoint**. This is a same-head repetition/noise diagnostic, not evidence of an A/B improvement. The original sweep cell for2618 was higher in acceptance than all four repeats. Do not interpret its narrowly higher rank as a reproducible gain over1870.

Pooled four repeats: **5.045685** accepted/pass; pooled two stock controls: **4.996762** (**+0.979%**). Mean of repeat E2E cell medians **105.4300**, stock **101.9926 tok/s** (**+3.37%**, descriptive). No confidence interval, statistical significance or functional-quality improvement established. These dev measurements are not pooled with the earlier lease's held-out test results.

vLLM metric definitions: draft acceptance rate is accepted/proposed draft tokens; the reported accepted/pass excludes bonus. vLLM's acceptance-length convention adds one bonus token. Both denominators/counters and per-position acceptance are retained in `sweep.json`; neither is an output-fidelity guarantee.

## Fidelity and selection caveats

All measured heads used ordinary native MTP verification, not relaxed acceptance. Nonetheless, the known numerical/runtime fidelity caveats remain. Exact output identity versus the dev no-spec control was **54/64** for each stock D8 control; checkpoint2618 **57/64**, checkpoint1870 **57/64**, final3740 **56/64**. The four identical-head confirmation cells matched no-spec **57,57,57,54 /64** respectively. Per-request output lengths, first divergence and functional checks are in `sweep.json` and the archive. This does **not** close strict parity/fidelity gates.

The prior full KL experiment required explicit user approval of three BF16 near-tie recurrence argmax exceptions and a stock-D4/no-spec baseline divergence. Those are documented in [its report](../20261005-qwen38-full-corpus-kl-cuda-retry1/README.md), not erased by this sweep. This task measures dev ranking and reports fidelity; no production qualification or new parity waiver is claimed.

Selection was predeclared: maximize pooled accepted draft tokens per verification pass, bonus excluded; exact ties use median E2E throughput then earlier step. The dev set was already used for CE checkpoint selection, so this is exploratory model selection, **not blind confirmation**. The held-out test set was not deployed or used at all. The ranking is noisy and functional pass scores do not establish exact target-distribution preservation.

## Actual process and environment

User-authorized evaluation-only lease cap: **four hours / $13.20**, one Scaleway Warsaw H100 PCIe80GB, physically verified81,559MiB. Frozen unquantized `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; vLLM0.27.1 pinned image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`. CUDA forward compatibility, `VLLM_USE_V2_MODEL_RUNNER=0`, target/KV dtype, context8192, concurrency1, prefix-cache policy and decoding were inherited unchanged from the existing compiled serving path. Exact Docker argv, image inspection, environment, driver/GPU/CPU/RAM capture and per-request API payloads are in the archive.

The only intentional D8 candidate difference is the restored MTP weight file, loaded through the existing `B70_MTP_WEIGHTS` overlay. No persistent launcher or application source changed. Remote runtime dependencies stay outside interactive Pi.

Real E2E process, predeclared in `pilot-protocol.json`:

1. Restore only eleven saved BF16 heads (stock step0 plus ten epochs) and historical training metadata from the previous verified archive; verify all seven source parts and whole SHA, tensor schemas and each restored head's hash. No captures, train/test records or teacher regeneration.
2. Serve real `/v1/chat/completions` requests; functional canary and unmeasured warmup for each cell, then all64 canonical dev requests with512-token budget and original sampling. Retain full output/token IDs, timing, `/metrics` deltas, sandbox checks and summaries. Setup/model loading is excluded from request timing.
3. Dev no-spec control, stock D8 start, ten epoch heads in seeded order, stock D8 end; rank on dev acceptance.
4. Four fresh A/B/B/A cells against previous selected2618. Here A=B because the acceptance winner is2618; all repeats were still executed.
5. Upload raw artifacts, fully stream-readback the final archive and assert17 complete64-request cells, all ten epochs in ranking, four confirmations, zero new optimizer updates and no test data.
6. Delete only the owned lease after verification, then confirm absence via provider API.

Entry points:

```sh
bash /home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep/launch-owned.sh
# Same-lease continuation after bootstrap backup repair; never recreate:
bash /home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep/launch-ablation.sh
# Remote actual serving orchestration:
~/qwen-mtp-env/bin/python ~/qwen-mtp-code/sweep.py
```

Source application commit was `9f9c01e2`; public disposable workflow snapshots accompany this report. `run-pilot.py` is imported only for its existing Server/evaluate/archive helpers; its training/capture/test main is never invoked. The new `sweep.py` asserts dev-only inputs and zero updates. `cache-restore.json` identifies every head and its source hash.

Fresh primary-source research: [vLLM0.27.1 metrics](https://github.com/vllm-project/vllm/blob/v0.27.1/vllm/v1/spec_decode/metrics.py), [official speculative-decoding example](https://docs.vllm.ai/en/v0.27.1/examples/features/speculative_decoding/), [Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B). Conclusions: exclude bonus when comparing accepted draft tokens, keep accepted/proposed and accepted/pass distinct, measure throughput with unchanged workload, and report fidelity separately.

Local preflight tester exercised the17-cell orchestration with a fake workflow, deterministic ranking/order, dev64/no-training/no-test calls, and synthetic eleven-head safetensors restore; links, schema changes and omitted inventory were refused. Shell syntax and Python compilation passed. These were preparation checks, **not GPU proof**; the real E2E completion above supplies serving evidence. See `preflight-notes.txt`.

## Failure retained, archive and cleanup

Initial task `b2d382e3b` failed **after** physicalGPU/CUDA/model/source-patch checks and complete eleven-head restore. The inherited bootstrap backup tried a single PUT containing all weights and hit the object-size limit. It performed no serving cells. The redundant bootstrap weight files were excluded (they already exist in the fully verified source archive); bootstrap metadata backup then succeeded. The same owned capped lease continued; it was not recreated. Final backup retains the checkpoint files through the existing multipart uploader.

Continuation task **`bd25dce21` completed exit0**. The `failure.json` scp message in completion output means no sweep failure file existed, not that any measured cell was dropped. Original bootstrap failure log is retained in `bootstrap-backup-attempt1/`. Archive creation/verification/cleanup output is in `completion-and-readback.log`.

Local root: `/home/mike/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep/`.

R2 prefix: `r2:ml-archive/2026-10-06/cache/b70-evals/20261006-qwen38-kl-dev-checkpoint-sweep/`.

Final uncompressed tar (inherited `.tgz` extension): **9,375,856,640 bytes**, five parts, SHA256 **`555ab7c3702de8871d7b85f89405314aa016cf8c880bb9cb1fe9759ca1f1719d`**. All part sizes/hashes and the whole archive passed streamed readback before deletion; see `final.tgz.parts.json`.

Owned lease **`608d5613-f8d3-4174-8b0f-6d32d18af4ff`** was deleted after backup verification. Provider instances API at **2026-10-06T17:35:20Z** confirmed it absent; `cleanup-confirmation.json` retains that observation. Private manifests, credentials and signed URLs are not published. No other rental, training arm or deployment was started.

This is a bounded development result, not standards-complete community qualification: no BetterBench/long-context/concurrency qualification or statistical win is claimed. The likely next improvement to investigate remains serving-like recurrence training rather than simply adding epochs; that implementation/rental is **not authorized or launched by this evaluation-only run**.
