# Inkling 2 Mini: per-batch-size DSpark verification schedule on the B70

**Development tier, 2026-11-25.** One Arc Pro B70 at 275 W. Same build, model, plugin and launcher as the [live 409K/C32 server](../20261125-inkling2-sparse-conv/#production-c32-at-409000-tokens): vLLM `4932e8596a`, plugin `e5f6619`, FP8 KV, CUTLASS Xe2 MoE, hybrid prefill, sparse conv history, DSpark k=7 probabilistic drafting, 2,048-token budget, 64-token pages, utilization 0.98. Test engines ran one at a time on loopback `:18080`. The only intentional change is one speculative-config key:

```json
"num_speculative_tokens_per_batch_size": [[1, 2, 7], [3, 32, 3]]
```

The DSpark drafter still produces its fixed 7-token block. With this key, the vLLM scheduler verifies 7 drafts per request when 1–2 requests are running and 3 when 3–32 are running. Model Runner V2 truncates the rest (`draft_tokens[:, k:] = -1`). Verifying a prefix of the draft keeps rejection sampling lossless.

## Origin: inference-optimisation-skill review

We reviewed [sybil-solutions/inference-optimisation-skill](https://github.com/sybil-solutions/inference-optimisation-skill). Its process additions now appear in [BENCHMARKING_STANDARDS.md](../../BENCHMARKING_STANDARDS.md) §7, §8 and §10.1:

- ceilings first;
- interleaved controls with a "beat both neighbours, no >3% regression" acceptance rule;
- a long-context retrieval gate;
- a per-concurrency draft-length sweep.

| Skill lever | Status for Inkling on B70 |
|---|---|
| Sweep draft length per concurrency | **Adopted (this result).** It was never tried before: the checkpoint rejects `num_speculative_tokens != 7` (the Nov 21 k=3/k=5 boots failed), so only the scheduler-side schedule can vary the verified length. |
| Adaptive verification ([vLLM docs](https://docs.vllm.ai/en/stable/features/speculative_decoding/adaptive_verification/), [blog / PR #47808](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification)) | **Blocked.** `Inkling2DSparkSpeculator.__init__` asserts `not self.enable_adaptive_verification`. The engine failed at boot (run `run-20261125-200953.lX4mWy`). Support would need speculator, varlen-graph XPU attention and conv-state work. |
| Prune the draft LM head's vocabulary | **Open, needs code.** Per step, the drafter reads the full BF16 `unembed` (0.93 GB) for base logits, plus 7 × 103 MB folded Markov projections. Upstream's reduced-vocab path (`draft_id_to_target_id`) is not wired for Inkling's Markov head. `dspark_draft_topk` is also asserted off. |
| Decode graphs, prefix caching, chunk size, persistent JIT caches | Already in use (FULL_DECODE_ONLY, prefix caching on, 2,048 budget, Triton cache). Startup to `/health` was 23–26 s on every arm. |
| Long-context needle gate | **Adopted.** See below. |

**Decode ceiling at C1, from checkpoint tensor sizes.** Each target forward reads:

- routed experts: 15.67 GB × 8/128 = 0.98 GB;
- attention and other dense weights: 1.37 GB;
- BF16 `unembed`: 0.93 GB;

for about 3.3 GB per forward. A k=7 verify reads a union of routed experts across 8 rows, up to about 6.3 GB under independent routing. The drafter adds about 2.3 GB per step.

At roughly 560 GB/s measured streaming bandwidth, with the C1 chat acceptance length of 3.1, step bytes of about 7.6–10.9 GB give a practical ceiling of roughly 160–230 tok/s. Measured C1 is 109 tok/s, about 48–69% of that, so the remaining C1 gap is kernel efficiency and draft bytes, not draft length. The ceiling is an estimate: the expert union assumes independent routing, and the drafter read count comes from code inspection, not a profile.

## Sweep (static k at all batch sizes)

`QUICK=1 bash hillclimb.sh k3 ctl k5 k2 ctl k4` ([run](runs/run-20261125-201409.epzsol/)). The table shows each arm against the mean of the two k=7 controls. The arm directory prefixes in that run are wrong (a loop-variable bug, since fixed); the arm names are correct.

| Workload | ctl k=7 | k5 | k4 | k3 | k2 |
|---|---:|---:|---:|---:|---:|
| Chat C1 tok/s | 109.1 | +1.9% | −1.4% | −6.0% | −13.4% |
| Chat C8 aggregate | 280.8 | +7.7% | +11.3% | +9.6% | +10.9% |
| Chat C32 aggregate | 649.3 | +7.3% | +10.1% | +16.2% | +17.2% |
| Random C4 | 140.4 | +15% | +32% | +15% | +19% |
| Random C16 | 283.0 | +8.6% | +10.5% | +11.8% | +11.6% |
| Random C32 | 352.5 | +5.5% | +6.9% | +13.2% | +12.5% |

Greedy completion token IDs matched exactly across all arms (8/8 fixtures, 513–32,771-token prompts, cold and warm). No arm lost KV capacity; the controls had 8.21 GiB / 410,844 tokens and the k arms 8.29–8.40 GiB.

A three-segment schedule, `[[1,2,7],[3,12,4],[13,32,3]]`, **did not boot at 409,000 tokens**. Capturing decode graphs for three query lengths left 8.17 GiB, 40 tokens short of one maximum-length request (run `run-20261125-211522.FS58NC`). The two-segment schedule keeps the cap.

## Confirmation (ABAB, full suite)

`SCHEDULE='[[1,2,7],[3,32,3]]' bash hillclimb.sh sched ctl sched ctl` ([run](runs/run-20261125-211712.Xj9pZ8/), [summary.json](runs/run-20261125-211712.Xj9pZ8/summary.json)).

| Metric | k=7 control (mean of 2) | Schedule (mean of 2) | Change | Worst neighbour pair |
|---|---:|---:|---:|---:|
| Chat C1 decode tok/s (6 prompts, 256 tok) | 109.0 | 109.1 | +0.1% | +0.1% |
| Chat C1 at +30K context (3 prompts) | 92.3 | 92.3 | 0.0% | −0.1% |
| Chat C8 aggregate output tok/s | 282.7 | **313.7** | **+11.0%** | +8.1% |
| Chat C32 aggregate output tok/s | 642.4 | **765.7** | **+19.2%** | +18.6% |
| Random 512/128 C1 output tok/s | 68.5 | 68.4 | −0.1% | −0.2% |
| Random C4 | 155.4 | **176.9** | **+13.8%** | +9.8% |
| Random C8 | 201.3 | **236.7** | **+17.6%** | +13.5% |
| Random C16 | 287.1 | **314.8** | **+9.6%** | +8.2% |
| Random C32 | 351.1 | **398.6** | **+13.5%** | +13.1% |

How the numbers were produced:

- Chat rows use streaming `/v1/chat/completions` on the `live_bench.py` prompt panel, at temperature 0 with EOS ignored. Concurrent chat runs 2N requests on N workers; aggregate = completion tokens ÷ wall time.
- Random rows are `vllm bench serve` medians of 3 trials, with seeds fixed by (concurrency, trial).
- Token counts come from server usage stats.

**Verdict.** Every target metric (C4–C32) beats both neighbouring controls, and nothing regresses by more than 0.2%, so the schedule is accepted under §10.1. C1 is unchanged by design (batch size 1 still verifies 7).

Other results from the confirmation:

- **Latency (random C32, median of 3 trials):** TTFT 461/458 ms vs 515/517 ms; median TPOT 58.5/56.1 ms vs 64.8/64.2 ms.
- **Latency (random C8):** TPOT 23.8/24.9 ms vs 30.7/28.7 ms.
- **Acceptance:** acceptance length fell (C32 chat 2.4 vs 2.7) while throughput rose; fewer rejected verify rows is the point.
- **Correctness and capacity:**
  - greedy parity was 8/8 exact in every arm;
  - there were zero preemptions;
  - both schedule arms booted with 8.23 GiB / 412,065 KV tokens, versus 8.21 GiB / 410,844 for the controls.
- **Needle gate at 207,257 tokens:** all four arms answered `A=O0LA3QI3, B=ULYB3M8Z, C=FY4CHOUN` exactly.

**Near-full-context needle gate at 388,513 tokens** (95% of the 409,000 cap): `NEEDLE_ONLY=1 SCHEDULE='[[1,2,7],[3,32,3]]' bash hillclimb.sh sched ctl` ([run](runs/run-20261125-220628.k6mVuI/)). Both arms answered `A=8FJJEJY3, B=IOHC29QI, C=BB9LGSRL` exactly. TTFT was 230.7 s for sched and 230.9 s for ctl. This is the first retrieval check of the live configuration near its cap; earlier 409K checks covered capacity only. An earlier "400K" attempt sent 414K tokens and got HTTP 400 because of a harness sizing bug, since fixed; it is not a model failure.

## Reproduce

1. On `b70-dev`, stage `checks.py` and `hillclimb.sh` in `~/inkling-vllm-xpu/adaptive-verif/`. They reuse the `sparse-69444` venv, sources, plugin and `sweep.py`.
2. Run the commands above.
3. The script then does the rest:
   - refuses to start unless the live service is idle;
   - stops `inkling-inference`;
   - runs each arm on `127.0.0.1:18080`;
   - restores production, verifying it with `~/cmp/live/live_check.sh` (authenticated chat and tool call) and an unauthenticated 401 check.
4. Summarize with `python3 -I summarize.py RUN_DIR`.

Local copies omit the per-request `--save-detailed` bench JSON and server logs. Full raw evidence stays on the guest under `~/inkling-vllm-xpu/adaptive-verif/run-*/`. Production was restored and passed smoke after every run.

## Adaptive verification probe (rejected)

The patch, in a copied tree on the guest (`~/inkling-vllm-xpu/av-probe`), made three changes:

1. Dropped the `not self.enable_adaptive_verification` assert in `models/inkling2/speculator.py`.
2. Declared the Inkling sconv metadata builder `AttentionCGSupport.ALWAYS`. It already derives per-token metadata from the device `query_start_loc`.
3. Added `confidence_head = None` on the DSpark decoder, so vLLM uses its online acceptance estimator.

The run was `QUICK=1 SCHEDULE='[[1,2,7],[3,32,3]]' bash hillclimb.sh av sched av sched` ([run](runs/run-20261125-224634.VsC55j/)).

- **Boot:** it booted at 409,000 tokens with 8.23 GiB / 411,841 KV tokens. Startup took 43–53 s instead of 30 s because of step-cost profiling.
- **Throughput:** compared with the schedule (mean of 2 arms each):

  | Metric | Change |
  |---|---:|
  | Chat C1 | −0.6% |
  | Chat C8 | −3.0% |
  | Chat C32 | −2.4% |
  | Random C1 | −5.4% |
  | Random C4 | +14.6% |
  | Random C8 | −1.6% |
  | Random C16 | +1.6% |
  | Random C32 | −1.0% |

- **Greedy reproducibility:** outputs were not reproducible. The two schedule arms matched each other on all 8 fixtures. The adaptive arms diverged on the 32,771-token fixture: from the schedule at token 2 (cold) and 35 (warm), and from each other at token 40. Every static-k arm matched exactly despite different verify widths, so the trimmed batch changes the target computation somewhere. That was not investigated.
- **Logprobs:** vLLM rejects output logprobs with adaptive verification.

**Rejected.** The only gain is at C4, where a single k=3 segment is short. A 3–6 request segment at k=4 or k=5 would cover it, if the extra graphs fit under an optional reduced `--max-model-len`.

## Production

Promoted on 2026-11-25 with [`promote.sh`](promote.sh), which is the sparse-conv promotion script with its artifact labels renamed. On `b70-dev`:

```bash
cd ~/inkling-vllm-xpu/adaptive-verif && timeout 1200 bash promote.sh serve-inkling-dspark-schedule.sh
```

The script checks that the service is idle, backs up the launcher, installs the [new launcher](../../scripts/inkling-vllm-xpu/serve/serve-inkling.sh) and restarts the service. It rolls back automatically if any check fails. The checks passed:

- The running engine reports the schedule in its speculative config.
- Authenticated `/v1/models` showed the 409,000-token cap.
- `live_check.sh` passed: chat answered `95` and the `echo_value` tool call succeeded.
- C1 decode was 107.0 tok/s short and 90.7 tok/s at +30K, unchanged from before.
- Unauthenticated `/v1/models` returned 401.
- A 408,744-token prompt plus 128 generated tokens completed in 248.6 s, with zero preemptions and 100 accepted draft tokens.
- KV was 8.23 GiB / 412,065 tokens.

Guest artifacts are in `~/inkling-vllm-xpu/sparse-69444/promote-dspark-schedule.QdmtCT/` and `~/cmp/live/20261125-222347-promoted-dspark-schedule/`. The rollback launcher is `~/inkling-vllm-xpu/serve/serve-inkling.sh.pre-dspark-schedule-20261125-222319`, and the live launcher's sha256 is `e8fef2d9…afd29`.

## Not done

- No separate live-traffic C8/C32 benchmark after promotion; the gains above come from the matched loopback engines.
- This is not a publishable-tier result: no BetterBench, no full long-context sweep, and 2 control/candidate pairs.
- Only greedy (temperature 0) workloads were measured; sampled chat has different acceptance.
- Schedule boundaries were chosen from static-k sweeps, not tuned per batch size.
