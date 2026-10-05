# Full-corpus Qwen3.8 KL attempt: blocked before training

**Development-tier failed live trial. Zero optimizer updates; no KL acceptance or throughput result; no promotion.**

User authorized one fresh Scaleway H100 lease capped at four hours / $13.20 to verify live state alignment and then run the complete cached-corpus KL ablation. The lease was created, physically verified, bootstrapped, and cleaned up after the failure archive passed full remote readback verification. No second lease was created.

## Observed outcome

- Physical GPU: NVIDIA H100 PCIe, 81,559 MiB.
- All seven parts of the earlier full CE archive were streamed and verified: 14,339,297,280 bytes, SHA256 `86632b201f09facabf33824d75a062f364ab67eeb67b2e19a39413bfb6539244`.
- **All 374 train / 64 dev captures restored**, with no recapture, quantized-teacher substitution, or reduced corpus. See `cache-restore.json`.
- Exported the complete stock MTP overlay through the real trainer CLI. This is not an optimizer update.
- The eager no-spec server started and both selected public train/dev requests completed through `/v1/chat/completions`, returning their expected 64 token IDs. Raw requests/responses are in the archive.
- The instrumented stock D4 container exited while installing the parity hook, **before serving any D4 request or producing numerical parity evidence**. D4 overlay and D8 controls were not reached.
- The fail-closed gate prevented the CUDA optimizer smoke, full 3,740-update training, dev checkpoint selection, state diagnostic, and nine held-out serving cells. **None ran.**

This was not a small training pilot. It was a failed setup/alignment gate for the requested full-data experiment.

## Concrete failure and local correction

`parity-stock4-server.log` records:

```text
RuntimeError: MTP parity: unsupported installed method: GPUModelRunner.propose_draft_token_ids
```

The parity hook generated method fingerprints using default `ast.dump` under Python 3.14. The serving container uses Python 3.12. The same official source method produces different default serialization of optional empty AST fields. Thus the guard falsely rejected the supported method before the live trace could run.

The archived image inspection identifies build commit **`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`**, tag v0.27.1, and the exact protocol image digest. Using that commit's unmodified `GPUModelRunner.propose_draft_token_ids` source reproduced:

| Python / serialization | SHA256 |
|---|---|
| 3.14.7, default `ast.dump` (old expected value) | `4b724b4fbd544f7738baa161925d9aabaf35c1c79918c09d6fde3054b9406a83` |
| 3.12.12, default `ast.dump` | `afb7a8fad368123623978e9735edf9ecf40884a3498139ff6939c01da8705638` |
| 3.14.7, `show_empty=True` | `afb7a8fad368123623978e9735edf9ecf40884a3498139ff6939c01da8705638` |

The fix fingerprints an explicit JSON representation of declared AST semantic fields rather than interpreter-dependent `ast.dump` defaults. Empty fields and literal values are retained; only source locations remain excluded. The three method fingerprints were recomputed from the exact image build source. Whole-file proposer/eagle guards, exact vLLM version checks, and fail-closed semantic-change detection remain intact. This is **not** a bypass of the failed guard.

Local verification passed the complete source guard under Python 3.12 and 3.14 against that build, both before and after applying the actual existing training/native-capture patches. A fixed-digest regression also changes a method's literal to ensure a semantic change still fails. This closes the reproduced serialization defect **locally**; the corrected installer and real GPU parity checks have **not** been rerun in the container.

After the correction, the complete isolated Python 3.12 torch suite reported **103 passed, 1 skipped, 3 subtests passed**. Python 3.14 additionally ran all 11 guard/control tests successfully; its 11 torch-dependent tests were skipped because that interpreter lacks the external ML packages. Those tests use mocked API responses and are not additional live serving evidence. `cross-python-tests.log` retains the exact output.

Primary sources: [Python 3.14 `ast.dump` documentation](https://docs.python.org/3.14/library/ast.html#ast.dump) (the `show_empty` option was added in 3.13), [exact image-build runner source](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/worker/gpu_model_runner.py), and [exact MTP source](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5_mtp.py).

## Intended experiment, not completed results

Full cached 374/64 corpus, stock initialization, frozen BF16 target/shared embedding/shared output head, CE + teacher-to-student KL weight 1 / temperature 1, LR 1e-6, seed 42, depth 4, eight roots, weights `[1,1,0.8,0.8]`, max length 2,048, accumulation 1, logits chunk 64. **Ten complete epochs / 3,740 actual updates**, all 64 dev records each epoch, selection on minimum BF16-export dev CE rather than auxiliary loss. No state-matching arm in this attempt.

Serving was intended to retain no-spec plus stock/candidate/candidate/stock at D4 and D8, 64 requests each. The test set is a reused disjoint serving regression set, not newly blind confirmation. Baseline/differences/real-system process are recorded in `pilot-protocol.json` and the [alignment/ablation design](../20261005-qwen38-full-corpus-cuda/ALIGNMENT_AND_ABLATION.md). The teacher-forced versus sampled-draft distribution gap remains even with KL.

Runtime: full unquantized `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; vLLM 0.27.1 pinned CUDA image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`; CUDA userspace compatibility and `VLLM_USE_V2_MODEL_RUNNER=0` unchanged. Diagnostic cells use eager mode, with graph/async/batched parity explicitly unqualified. Executed source was main commit `bf797928`; the serialization correction was made after this failed trial.

The KL implementation is default-off and was independently reviewed for detached teacher, future-row masks, per-depth pair normalization, connected recursive/base-KV gradients, chunk lifetime and unchanged CE checkpoint selection. Before launch the integrated external CPU suite reported **102 passed, 1 skipped, 3 subtests passed**. The skip requires installed vLLM source. Earlier missing-RTN-fixture failures are retained; reruns used exact source copies plus retained original fixtures in an isolated test tree, preserving unrelated lead-checkout deletions. A separate D4 review found root-owned trace readability and post-output-cap comparison defects; both were fixed before this attempt. Those fixes did not expose the cross-Python fingerprint defect in synthetic tests.

## Commands, artifacts and cleanup

Executed entrypoint:

```sh
bash /home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda/launch-owned.sh
```

Original task `bc4b68d53` terminated with **exit 1**. The local `remote-pilot.log`, `failure.json`, startup log and public workflow snapshots accompany this report. Exact per-cell Docker argv, image inspection, environment, stock export, no-spec request/response token IDs, restored captures and logs are retained in the raw archive.

Local root: `/home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda/`.

R2 prefix: `r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-full-corpus-kl-cuda/`.

Failure archive `final.tgz`: **2,030,704,640 bytes**, whole SHA256 **`5576ffad397ae70ec07491115fecdc8de9c1981d7e19fa9d1de2774db2f03b68`**. It is an uncompressed tar despite the inherited extension. `final.tgz.parts.json` records its single object. `INCOMPLETE_RUN_FINAL_ARCHIVE_FULL_READBACK_VERIFIED` means archive integrity only—**not completed training or a passed live gate**.

After that verification, the script requested deletion of owned lease `7f8c31c1-ca7b-4e9b-a7f4-515e2138aa00`. An authenticated subsequent provider instance-list query found neither that ID nor `qwen38-mtp-h100-kl-20261005`. Private signed manifests and credentials are not published or included in the run archive. No persistent inference launcher was changed; no experimental configuration was promoted; no quantized B70 result is pooled with this CUDA attempt.

## Next blocker

A fresh, separately authorized lease is required to rerun the corrected installer and actual D4/D8 state/position controls before the full KL ablation. Do not report the source-only correction as passed GPU parity or the stock export as training. No additional rental has been initiated.
