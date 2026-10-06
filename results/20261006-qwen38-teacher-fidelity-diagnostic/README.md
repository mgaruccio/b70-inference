# Qwen3.8 teacher execution-path fidelity diagnostic

**Development only; zero optimizer updates; not training-ready.** This is a correctness diagnosis, not a throughput benchmark, performance improvement, full-corpus training experiment, test-set selection, or production promotion. All original numerical and exact-argmax gates remain in force; no fresh waiver was granted.

## Findings

The archived offline comparison reproduced the prior 24/60 failing histories (46/240 rows). For **all 240 scored rows**, the aligned native row beat both in-bounds adjacent rows by cosine and relative L2. This bounded check provides no evidence of a ±1 row shift; it cannot localize divergence along an HF prefix because only four HF rows were archived.

On the new, single Paris H100 lease, changing native fresh prefill from the compiled runtime bundle to the **same eager bundle used by the incremental verifier** reduced numerical failures from 8/12 histories to 3/12, with exact argmax agreement at all 48 eager/verifier rows. The remaining failures keep the gate open. This identifies an execution-configuration contribution, **not** a particular RMSNorm/GDN kernel defect or rejection-cache bug.

HF full-prefix versus uncropped hybrid-cache execution itself failed numerically on 14/60 histories, although all 240 argmaxes agreed. Thus switching this teacher to cached execution alone is not a demonstrated fidelity fix. HF uses FP32 recurrent state; native GDN `auto` resolves to BF16 for this BF16 model. State precision and GDN backend were deliberately **not** changed in this experiment.

| Actual/reference pair | Histories / rows | Numeric failing histories | Argmax differing rows | Maximum relative L2 |
|---|---:|---:|---:|---:|
| HF full / HF cached | 60 / 240 | 14 | 0 | 0.073100 |
| HF full / native compiled | 60 / 240 | 27 | 0 | 0.164960 |
| HF full / native eager | 60 / 240 | 25 | 1 | 0.083053 |
| HF cached / native compiled | 60 / 240 | 26 | 0 | 0.127914 |
| HF cached / native eager | 60 / 240 | 24 | 1 | 0.123410 |
| Native compiled / native eager | 60 / 240 | 24 | 1 | 0.154132 |
| Native compiled / incremental verifier | 12 / 48 | 8 | 1 | 0.072673 |
| Native eager / incremental verifier | 12 / 48 | 3 | 0 | 0.059239 |
| HF full / incremental verifier | 12 / 48 | 8 | 1 | 0.066244 |
| HF cached / incremental verifier | 12 / 48 | 8 | 1 | 0.073569 |

Every target-pair argmax disagreement is the same `verifier-001`, position 202: HF full/cached and native compiled choose token 1005 at a BF16 tie, while native eager and the incremental verifier choose 1445 with margin 0.125. This remains a failure, not a waiver. The frozen original head's last-row argmax matched the actual public API token for all 120 capture-on prefill requests (`raw/native-head-sampling.json`); this last-row check does not turn interior-row agreement into a public API guarantee.

Unchanged thresholds: relative L2 ≤ 0.03, cosine ≥ 0.999, max-absolute-error/RMS ≤ 0.25, plus exact argmax. See `raw/paired-comparisons.json` for every row, top-two logits, and margins. HF-internal status is explicitly separate from native serving parity.

## What actually ran

- **One** Scaleway Paris H100 PCIe 80 GB, 81,559 MiB physical device, $3.30/hour; approved cap two hours/$6.60. No second rental.
- Original BF16 `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; no quantized teacher or weight remapping.
- Serving image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` (vLLM 0.27.1); V1 runner, CUDA compatibility enabled. HF torch 2.14.1+cu130, Transformers 5.15.1, `USE_HUB_KERNELS=NO`, native torch GDN fallback. Exact runtime and kernel resolution retained.
- **Six native cells:** stock D4/D8 live trace cells (two canonical train/dev requests each), followed by compiled/eager target-only prefill cells, each capture-off on two controls and capture-on on all 60 exact histories. All four native acceptance classes were observed, including actual rejection/corrected-continuation histories.
- Data: 48 fixed archived genuine greedy proposal prefixes plus **12 newly captured** incremental verifier prefixes on this lease. No test requests. Fixed prefixes are reused diagnostic input, not a reduced training corpus.
- HF full replay and actual hybrid-cache prefill-plus-four-decode calls on each of 60 prefixes (240 scored rows per path), detached frozen target and original BF16 LM head.
- Capture on/off exact output identity passed for both native modes. Original shared-head/API last-row identity passed for 120 requests. All exact prefix and scored-position assertions passed.
- Local CPU: **31 tests passed, 3 subtests passed**. Tiny real four-layer hybrid BF16 model: full/cached numeric and argmax pass, cache-state metadata exercised; this is not native parity proof.
- H100 bootstrap: **131 tests passed, 1 skipped, 3 subtests passed**; pinned source guards and physical/runtime checks passed.
- Diagnostic driver completed in 759.93 seconds. Its exit 0 means all planned diagnostic comparisons and archival completed, **not** that any fidelity gate passed. HF CLI returned 1 after writing its complete internal-gate-open report, as designed. Both stock D4/D8 native MTP recurrence checkers returned 1 with `gate_a_closed=false` and `numeric_or_argmax_failure`; their full failures are retained separately under `raw/parity-stock*/parity/`. This run did not rerun complete-overlay or no-spec output-identity cells, so it cannot close that wider Gate A either.
- **Zero** optimizer updates; no training, full-64 dev serving sweep, performance selection, loss tuning, tolerance change, or persistent serving change.

## Exact process and artifacts

The external operating root was `/home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic`. The defined public-boundary journey, fresh official research URLs, preconditions, fixed configuration, cleanup, and expected outcomes were recorded before launch in `pilot-protocol.json`.

Executed entrypoints:

```bash
bash /home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic/launch-owned.sh
bash /home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic/launch-diagnostic.sh
python3 /home/mike/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic/readback-and-delete.py
```

Bootstrap and diagnostic are **separate phases**: bootstrap never counts as the requested fidelity test. `workflow/` retains the actual disposable caller scripts, including explicit `diagnose-fidelity.py` invocation. `raw/*-command.json`, per-cell `command.json`/config, logs, exact request/response JSON, prefix manifests, source SHA-256, model path, installed packages, and GPU/runtime captures retain the executed steps. Lead source was `7a46e899`, including integrated diagnostic commit `4dcf5c57`. The model's strict loader/revision-path guard and `raw/model.json` establish the pin; the diagnostic's optional `checkpoint_revision` field is null and is not independent pin evidence.

The offline check used `qwen38_teacher_fidelity.py offline` with the restored `teacher-inputs.json`, `teacher-histories`, `prefill-replay-on/features`, and `verifier-histories`; it exited 1 because the original gate remained open. Full report: `offline-fidelity.json`. Prior archive parts and whole SHA-256 were verified before extraction; no teacher neighbor rows were invented.

Raw tensors are intentionally excluded from git. The complete final archive is at:

`r2:ml-archive/2026-10-06/cache/b70-evals/20261006-qwen38-teacher-fidelity-diagnostic`

Archive size **347,944,960 bytes**, SHA-256 **`d8fe3260e5a018fc5cd590c5be8fbf83682aa195acadb008e8ffc39d0100f0f4`**. The `.tgz` name denotes the existing uncompressed tar workflow. No credentials or signed upload URLs are published.

Full-object readback verified the archive's size and SHA-256 before explicitly deleting owned lease `664d2a4a-e669-4996-87a7-c0f9780e6303`. Provider API absence was confirmed at **2026-10-06T23:33:23Z**. Request-to-absence span was **0.42058 hours**, approximately **$1.39** at the listed rate (not an invoice), below the approved cap. See `archive-readback.json`, `cleanup-confirmation.json`, and `lease-cost-estimate.json`.

The first cleanup attempt stopped before readback/deletion because it assumed multipart names; the actual 347 MB archive used the existing uploader's single-object `final.tgz` format. The corrected filename guard also requires single-object size/hash equality with the index. Full readback then passed and deletion followed; the failed attempt is retained in `readback-first-attempt.log`. Local smoke-launch/reporting errors were corrected before rental and did not exercise or alter the GPU path.

## Decision

Independent read-only result review confirmed the stated counts and concluded: eager matching does not close the gate; this run is not training-ready. Publication assertions independently recomputed all ten pairwise summary counts from their raw rows and checked the archive/cleanup records. Executed trainer, diagnostic, and live-parity source SHA-256 values matched the lead checkout.

**Do not train yet.** Matched eager execution removes part, not all, of the discrepancy. Cached HF replay does not close the native-serving teacher gate. Keep the separate pairwise results, original gates, and this lease's evidence. Any further paid experiment, state-precision/backend change, or training requires a separate authorization. No performance ceiling or improvement has been established.
