# Qwen3.8 greedy recurrence correctness/profiling — gate failed

Development result, not a performance benchmark or a training trial. The new default-off path is implemented, but **not training-ready**. No optimizer updates, full training, test-set selection or promotion occurred. No performance improvement is claimed.

## Scope and environment

User approved implementation and one H100 correctness/profiling lease, capped at four hours/$13.20. Warsaw capacity disappeared before the create request; provider absence was confirmed and that attempt cost $0. User then approved finding another H100 within the same cap. One Scaleway Paris H100 PCIe 80GB was created (`8e806760-a07c-47dd-be04-5a9100a005b3`), $3.30/hour, provider automatic time/spend limits enabled. Cleanup status is recorded separately below.

- Original BF16 `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; no quantized teacher.
- Trainer: Python3.12, torch2.14.1+cu130, Transformers5.15.1; original multimodal checkpoint loaded strictly, frozen text-only forward.
- Serving: vLLM0.27.1, pinned image `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`, CUDA compatibility enabled, V1 model runner.
- Full archive restore: **374 train /64 dev**, seven parts, 14,070,067,200 bytes, SHA256 `27ba5a05d762c90a6fbba164486dce636dac125214fb62ca40aaaad17c950458`. All438 records passed contiguous-position checks. No test data restored, no full-corpus recapture.
- Implementation: `92c68398` (worker `cb599274`); [design and official research sources](../../docs/qwen38-serving-recurrence-design.md).

The treatment changes only deeper forward KL to the current BF16 greedy draft history and fresh frozen-teacher replay. Teacher-forced CE and captured depth1 KL remain unchanged. This validates within-block draft conditioning, not a new distribution of accepted serving contexts.

## Actual public-boundary checks

Exact commands, inputs, environment and logs are in `workflow/` and `raw/`; `pilot-protocol.json` contains the predeclared process.

1. External CPU suite passed **122 tests,1 skipped,3 subtests** in the clean worker worktree. Lead checkout:120 passed,2 missing historical-fixture failures,1 skipped,3 subtests. Unrelated historical deletions were preserved; source/tests match the clean worktree. Final offline caller preflight:22 parity tests passed, including all four acceptance-class reconstructions.
2. The disposable H100 bootstrap suite passed **122 tests,1 skipped,3 subtests**; skip requires installed vLLM in that test interpreter. Docker source-patch/CUDA checks passed.
3. Actual trainer CLI `--validate-only`, off and on `--greedy-kl`, on three real train and three dev records spanning short/median/long lengths. Depth4,8 roots,weights `[1,1,.8,.8]`,KL1,T1,seed42,max_length2048,chunk64. **Zero optimizer steps**. Student gradients finite/nonzero, frozen weights receive no gradients; CE-anchor checks passed; stock MTP tensor identity and export/reload equality of192 proposal tokens per arm passed. These six records are diagnostics, not shortened training.
4. Five native serving diagnostic cells: no-spec,stockD4,stock-overlayD4,stockD8,stock-overlayD8; two canonical train/dev requests per cell, temperature0,seed42,64 output-token cap. All four acceptance classes observed. Stock/complete-stock-overlay outputs identical at D4/D8. **Strict recurrence checks failed in all four speculative cells**, on the dev request; D4 differs from no-spec, while D8 matches these two diagnostic requests. No fresh near-tie waiver.
5. Actual implementation teacher replay:48 greedy branch histories from six real records plus12 actual native verifier histories, including rejected suffixes and corrected-cache continuations. **60 histories /240 scored teacher rows**. Two separate native `/v1/completions` exact-token prefill cells, speculation/prefix caching off, one request at a time, `max_tokens=1`, capture off/on:60 requests each. Capture-on/off outputs identical.
6. Teacher comparison failed existing numerical and argmax gates. Workflow stopped before the planned five full64-dev serving cells: **0/5 completed,0/320 measured full-dev requests**. Do not treat bounded diagnostic calls as throughput measurements.

## Gate failure

Existing tolerances were retained: relative L2≤0.03, cosine≥0.999, maximum absolute error/reference RMS≤0.25, plus exact argmax.

| Comparison | Histories | Numeric-failing histories | Argmax disagreement |
|---|---:|---:|---:|
| HF fresh teacher vs native fresh prefill |60|24|1 of240 rows|
| Native fresh prefill vs actual incremental verifier |12|8|0 of48 rows|
| HF fresh teacher vs actual incremental verifier |12|8|1 of48 rows|

HF/native-fresh maximum row relative L2 was **0.170953**, maximum absolute error/RMS **0.929428**. The one teacher argmax mismatch occurs on `mbpp-601`, `verifier-001`, position202: native token1445=25.375 versus1005=25.25; HF ties1445/1005 at25.375 and chooses1005. Full rows/top-two logits are retained in `raw/teacher-parity.json`.

This is not established to be a target-indexing or cache bug. Exact IDs and position arithmetic were checked. Importantly, **native fresh prefill also disagrees numerically with native incremental verification** on eight histories: the fidelity gap is not solely an HF-versus-vLLM comparison. Numerical execution-path differences are a plausible cause, not a proven diagnosis. Successful CPU contracts and export equality do not close this gate.

`strict_gate_a_closed=false`, teacher numeric/argmax gates false, training not ready. Prior experiment's qualified BF16 exceptions are not fresh authorization to weaken these checks.

The separate student recurrence diagnostic passed hidden-state tolerances but failed greedy recurrent argmax at D4 position148 and D8 positions130/143 on the dev request. Live-input HF and frozen-head projections agree with native; the free-running recurrence does not. These are distinct from the full-target teacher comparison above. HF/native-fresh numerical failures cover46/240 scored rows.

Selective read-only final review found no blocking misstatement: the observed prefixes/positions support the intended teacher alignment, but do not establish an indexing/cache bug or training readiness. Retained evidence should guide a bounded execution-path investigation, not a premature rejection-cache rewrite.

## Profiling, not a full-run budget

Fresh teacher replay measured **0.258–0.354 seconds/root** across the six diagnostic records; peak CUDA allocation61,071,603,200 bytes. The naive29,920-root teacher-forward-only extrapolation is **2.15–2.94 hours**. It excludes CE/backward/optimizer, control arm, dev selection, serving repeats and setup. It is not a complete training cost estimate or authorization, and the backend currently fails fidelity gates.

## Artifacts and cleanup

Raw tensor traces, original restored captures/heads and executed artifacts are retained in the existing multipart archive workflow, not Git. Public reports/logs and exact disposable scripts are here. `raw/reference-heads/` metadata describes historical training, not updates in this validation.

Final archive **readback verified**:3 parts,4,848,547,840 bytes,SHA256 `5b42d82d793fa593e59fcb5d7bd6c1e73ea86bfb4a73a68229db904e0fc293f8`, at `r2:ml-archive/2026-10-06/cache/b70-evals/20261006-qwen38-serving-recurrence-validation`. The owned lease was explicitly deleted only after full readback; provider API absence confirmed **2026-10-06T22:06:58Z**. See `archive-readback.json` and `cleanup-confirmation.json`. Create-request-to-absence wall time≈0.599 hours, roughly$1.98 at the listed hourly rate; this is **not a provider invoice**. No credentials or signed transfer manifests are published.

## Next bounded step

Isolate fresh-prefix versus incremental teacher numerical fidelity, including the hybrid recurrent-state path, on these same retained histories. Keep target conditioning and greedy policy fixed. Do not begin full training, retune KL, loosen tolerances or claim an acceptance gain until the backend is qualified and a separate complete-schedule budget is approved.
