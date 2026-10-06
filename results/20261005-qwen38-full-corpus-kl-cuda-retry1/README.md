# Full-corpus Qwen3.8 CE+KL: completed qualified development trial

**All 374 training / 64 dev captures, ten complete epochs, 3,740 optimizer updates, nine 64-request held-out serving cells. No promotion or production qualification.**

The user authorized the full cached-corpus experiment on one fresh H100 lease capped at four hours / $13.20. The completed Warsaw retry reused that same lease through setup corrections; no additional instance was created during corrections. All captures were restored from the original unquantized BF16 CE run, with full archive readback verification, rather than recaptured or reduced.

## Result

Minimum weighted dev BF16-export **CE** selected step **2,618**, objective **0.7815388148**, versus stock **1.0881145562** (28.18% lower). Final step 3,740 objective was **0.8065499501**. CE-only's previously selected objective was 0.8809706301; these offline improvements do not establish serving quality. Checkpoint selection remained CE, not the combined CE+KL training objective.

| Held-out cell (64 requests each) | Accepted / draft pass | Median decode tok/s | Median E2E tok/s | Functional pass /64 |
|---|---:|---:|---:|---:|
| No-spec | — | 30.6762 | 30.2749 | 56 |
| D4 A1 stock | 3.2695 | 100.0472 | 90.6864 | 54 |
| D4 B1 candidate | 3.3044 | 100.1626 | 90.8688 | 54 |
| D4 B2 candidate | 3.2638 | 100.6310 | 90.4687 | 56 |
| D4 A2 stock | 3.2816 | 99.0030 | 90.4724 | 55 |
| D8 A1 stock | 4.8640 | 112.3281 | 99.8671 | 54 |
| D8 B1 candidate | 4.9782 | 119.9349 | 104.6359 | 55 |
| D8 B2 candidate | 4.9670 | 119.8488 | 104.2214 | 54 |
| D8 A2 stock | 4.8640 | 112.3561 | 99.8361 | 54 |

Acceptance excludes bonus tokens. D8 pooled stock **12,306 / 2,530 = 4.8640316206**; candidate **12,327 / 2,479 = 4.9725695845** (**+2.23%**). Mean of the two per-cell E2E medians: stock **99.8516**, candidate **104.4286 tok/s**, **+4.58%**; decode **+6.72%**. This is a descriptive summary, not a pooled request median or confidence interval. D4 pooled acceptance increased 0.25%, E2E 0.10%: essentially flat.

The D8 signal is better than the earlier CE-only run's roughly 1.28% E2E result, but those are different leases/regions and cannot be treated as a controlled CE-versus-KL causal comparison. Within this run, stock/candidate ABBA controls share the same hardware and software. No confidence interval, statistical significance, material-win threshold, or functional-quality improvement is established. This reused disjoint test set is a regression set, not newly blind confirmation.

## Fidelity and explicitly approved qualifications

**Strict Gate A did not close.** The user approved a qualified trial only after seeing each concrete failure. The unchanged source parity checker continues to report `numeric_or_argmax_failure`; the external orchestration records narrow exceptions, not a strict parity pass.

Five eager public API control cells (no-spec, stock D4, complete stock-overlay D4, stock D8, complete stock-overlay D8), each with the same train `mbpp-601` and dev `mbpp-511`, returned 64 tokens. All four speculative cells passed structural token/position/feedback/slot checks and numerical hidden-state tolerances. Across the cells, prefill, full acceptance, partial rejection and zero acceptance were observed. Full stock overlays matched corresponding stock outputs exactly at D4 and D8.

Three recursive HF-versus-native sampled argmax exceptions, all on the same dev request, were disclosed and authorized, including matching overlay controls:

- D4 round15/depth2: native-state logits token307 **19.625**, token220 **19.5**; recursive replay tied both at **19.5**, choosing220. Recursive relative L2 **0.00612**.
- D8 round8/depth4: native-state logits307 **27.125**,279 **27.0**; recursive replay tied both at **27.125**, choosing279. Relative L2 **0.01251**.
- D8 round10/depth5: native-state logits15/16 tied at **38.0**; recursive replay favored16 **38.25** over15 **38.0**. Relative L2 **0.01357**.

Live-input HF replay and the frozen head on native states retained native argmax at all these rows. A read-only source audit found no replay/index/position defect. These observed BF16 near-ties explain the flips but do not satisfy exact-argmax parity. Other mismatches and numeric/structural failures remained blocking. Raw margin evidence and failed gate reports are retained.

A separate no-spec identity gate failed: stock D4 diverged on `mbpp-601` starting at token3, 61/64 positions different. D8 matched no-spec on both diagnostic requests. Stock/stock-overlay equality held exactly at both depths, so the divergence was not introduced by loading the overlay. The user explicitly authorized proceeding with this baseline fidelity limitation. `live-gate.json` therefore has `strict_gate_a_closed=false` and `no_spec_fidelity_passed=false`, even though qualified `training_allowed=true`.

Held-out serving also shows repeat nonidentity: D4 stock repeats **58/64**, candidate repeats **55/64**; D8 stock repeats **64/64**, candidate repeats **58/64**. Stock/candidate token identity was D4 A1:B1 **64/64**, A2:B2 **53/64**, D8 A1:B1 **58/64**, A2:B2 **64/64**. No-spec versus D8 stock was **54/64**, candidate **55/64**. See all per-request results in `comparison.json`. Do not claim exact output fidelity from functional pass counts or the diagnostic overlay control.

## Training and state diagnostic

Stock initialization; CE + **KL(teacher || student)** weight1, temperature1, FP32 temperature-squared scaling, detached teacher, frozen shared BF16 embedding/output head. LR1e-6, seed42, recursive depth4, eight roots, depth weights `[1,1,0.8,0.8]`, max length2048, accumulation1, logits chunk64; validation/checkpoint every374 updates, all64 dev records. Source: main `1c356660`, plus published external orchestration snapshots. No state-matching objective was added.

Counts: 3,740 sequences seen; 985,210 input tokens; CE labels and valid KL pairs per depth **[441270,29920,29920,29920]**. Peak CUDA allocated **13,020,421,632 bytes**, reserved **15,109,980,160 bytes**. The one-update CUDA smoke preceded the separate full 3,740-update training run and is not counted toward its epochs.

Teacher-forced diagnostic on four dev captures, 32 roots/depth: stock→candidate next-token KL at depths1–4 **0.06508→0.01836, 0.10960→0.08886, 0.21657→0.09862, 1.00237→0.60358**. Depth4 state cosine **0.4403→0.4986**, relative L2 **1.2319→1.1280**. This small diagnostic is not free-running serving drift or test acceptance. The branch training input remains ground-truth teacher-forced, whereas serving uses sampled drafts: the off-policy distribution gap remains.

## Environment, process and retained failures

Development tier. Scaleway Warsaw H100 PCIe 81,559MiB, driver570.124.06; unquantized `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Serving vLLM0.27.1, image `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`, container torch2.13.0+cu130; trainer torch2.14.1+cu130, Transformers5.15.1, Python3.12.3. CUDA forward-compatibility and `VLLM_USE_V2_MODEL_RUNNER=0` unchanged. Parity controls eager-only; measured serving cells use the existing compiled workflow. Graph/async/batched parity and actual KV-value equality are not qualified. No B70 quantized numbers are pooled here; no persistent inference launcher changed.

Real E2E process: restore/verify all374/64 cached captures; execute public `/v1/chat/completions` diagnostic controls and CUDA HF replay; require coverage and qualified controls above; run actual trainer CLI for smoke then ten full epochs; select the minimum dev BF16 CE checkpoint; run 64 canonical held-out requests/checks per no-spec/D4/D8 ABBA cell; retain responses, token IDs, functional checks, acceptance counters, timing and actual Docker/CLI argv; upload all artifacts, verify complete remote readback, then delete only the owned lease. Exact commands and environment are in the published workflow and archive, not mocked/unit substitutes.

Entry points on the funded lease:

```sh
bash /home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1/launch-owned.sh
# Corrections resumed the SAME lease, never launch-owned again:
bash /home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1/launch-bootstrap.sh
bash /home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1/launch-ablation.sh
```

Actual training/request/check command arrays, logs, selected-head files, controls and replay tensors are in the raw archive. The successful terminal task was **`bacde10cb`, exit0**. Earlier failures remain evidence, not ongoing jobs: first attempt's Python AST guard defect (separate report); billing-rejected create attempts allocated no instance; funded bootstrap's missing nested hook test fixture; completed D4 strict-parity failure; resume export/report overwrite guards; D8 strict-parity failures; no-spec identity failure. None of these failed attempts performed full training. The nested deployment fixture and resume orchestration were corrected; numerical/fidelity failures were not silently changed into passes.

Integrated external CPU verification before rental: **103 passed,1 skipped,3 subtests passed**, including KL and parity tests. Installed-vLLM check was skipped there; actual container source guards subsequently executed during bootstrap. The worker's Pi-runtime skipped tests were not ML validation; the external runtime superseded them. Raw verification output is in the prior [attempt report](../20261005-qwen38-full-corpus-kl-cuda/README.md). Live E2E above supplies actual GPU evidence and explicitly reports its failures.

Fresh external research used [pinned vLLM MTP source](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/model_executor/models/qwen3_5_mtp.py), [proposer](https://github.com/vllm-project/vllm/blob/6e448d0ea9bf3d88d898b65449ca6dc2aec170ac/vllm/v1/spec_decode/llm_base_proposer.py), [PyTorch distillation tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html), [Python AST docs](https://docs.python.org/3.14/library/ast.html#ast.dump), and [NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). Conclusions: match actual serving normalization/positions; teacher-to-student detached T² KL; preserve interpreter-independent semantic source guards; explicitly enable supported data-center CUDA userspace compatibility. Further alignment limitations are in the [audit/design](../20261005-qwen38-full-corpus-cuda/ALIGNMENT_AND_ABLATION.md).

## Archive and cleanup

Local root `/home/mike/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1/`.

R2 prefix `r2:ml-archive/2026-10-05/cache/b70-evals/20261005-qwen38-full-corpus-kl-cuda-retry1/`.

Final uncompressed tar (inherited `.tgz` extension), **14,070,067,200 bytes**, seven parts, SHA256 **`27ba5a05d762c90a6fbba164486dce636dac125214fb62ca40aaaad17c950458`**. `final.tgz.parts.json` lists per-part bytes/hashes. Complete streamed readback verified all parts, whole hash, full training counts and nine complete serving cells before deletion.

Owned instance **`f0d6062c-aa85-4b9d-ab31-87e4a49bfe8e`** deletion was requested only after verification. Subsequent authenticated provider instances API query at **2026-10-06T05:57:19Z** confirmed it absent; see `cleanup-confirmation.json`. Private signed manifests, credentials and private billing responses are not published. No new rental, state-loss arm or follow-up experiment has been started.
