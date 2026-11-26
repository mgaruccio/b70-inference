# Inkling 2 Mini: reduced DSpark draft vocabulary on the B70 (rejected as default)

**Development tier, 2026-11-26.** One Arc Pro B70 at 275 W.

- **Baseline:** the live 409K/C32 configuration (vLLM `4932e8596a`, plugin `e5f6619`, FP8 KV, CUTLASS, hybrid prefill, sparse conv history, DSpark k=7 probabilistic) plus the [per-batch-size schedule](../20261125-inkling2-draft-length-schedule/) `[[1,2,7],[3,32,3]]`.
- **The one change:** `INK2_DRAFT_VOCAB=<token list>`, an opt-in plugin flag (branch `draft-vocab` of `vllm-inkling2-xpu`).
- **Context length:** all arms ran at `--max-model-len 400000`, because larger draft vocabularies need some KV memory (see capacity below). This doesn't change throughput.

## What the flag does

Without the flag, each decode step the drafter reads the full BF16 `unembed` (201,024 × 2,304, 0.93 GB) for its base logits, plus 7 folded Markov projections (201,024 × 256, 103 MB each).

An old eager profile (`~/inkling-vllm-xpu/hc/20261121-181816-prof-s1/`) timed these at:

- 1,566 µs for the base logits;
- 7 × 178 µs for the Markov projections;
- about 2.8 ms per step in total, already at about 580–590 GB/s.

With the flag, the drafter keeps only the listed LM-head rows. It computes logits over those rows and scatters them into a full-vocabulary buffer that is −inf everywhere else. The target still verifies against the full vocabulary, so rejection sampling is exact and output is unchanged; only acceptance can fall.

On the B70, `tests/test_draft_vocab.py` passes 4/4 ([log](runs/run-20261126-002038.q1uUAR/pretest.log)). It checks that kept columns are bit-identical to the full-vocab draft and that all other columns are −inf.

## Vocabulary

To build the lists:

1. Rebuild the prompt mix with `fetch_prompts.py`: 2,679 prompts from UltraChat, Magicoder Evol-Instruct, GSM8K and non-English Aya (`prompts.json` sha256 `74783ce5…c747`).
2. Run `collect.py` against the live server with default sampling and reasoning on. It collected **1,765,338 generated token IDs**.
3. `build_vocab.py` ranks tokens on 80% of requests, always keeps all 60 added/special tokens, and reports coverage on the held-out 20% ([coverage.json](vocab/coverage.json)).

Only 44,855 distinct tokens appeared in the 80% split, so the "49152" list is every observed token (44,910).

| Kept tokens | Held-out coverage | Chat | Code | Math | Multilingual |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 92.9% | 93.5% | 95.6% | 96.6% | 83.4% |
| 32,768 | 96.6% | 97.3% | 98.3% | 98.4% | 90.1% |
| 44,910 | 97.8% | 98.4% | 99.0% | 98.8% | 92.7% |

## Interleaved sweep

```bash
QUICK=1 MAXLEN=400000 SCHEDULE='[[1,2,7],[3,32,3]]' \
  PRETEST="bash $HOME/inkling-vllm-xpu/draft-vocab/run-tests.sh" \
  bash hillclimb.sh sched v32768 sched v16384 sched v49152 sched
```

[Run](runs/run-20261126-002038.q1uUAR/), [summary.json](runs/run-20261126-002038.q1uUAR/summary.json). The harness is [`hillclimb.sh`](../20261125-inkling2-draft-length-schedule/hillclimb.sh).

The table shows each arm against the mean of its two neighbouring `sched` controls. "both" means the arm beat both neighbours.

| Metric | 32,768 | 16,384 | 44,910 |
|---|---:|---:|---:|
| Chat C1 decode (109.0 tok/s control) | **+5.0% (both)** | **+5.4% (both)** | **+6.9% (both)** |
| Chat C8 aggregate | +1.5% | −2.8% | −1.8% |
| Chat C32 aggregate | +0.9% | −2.2% | −0.6% |
| Random 512/128 C1 | −1.1% | −22.8% | −1.4% |
| Random C4 | −7.7% | −8.1% | −2.5% |
| Random C8 | −7.6% | −20.7% | −12.2% |
| Random C16 | −10.3% | −19.4% | −10.5% |
| Random C32 | −8.0% | −16.8% | −7.7% |
| Random C32 acceptance length (control ≈ 2.11) | 1.78 | 1.50 | 1.81 |
| KV available / tokens (control 8.23 GiB / 411,695) | 8.18 / 409,528 | 8.24 / 412,605 | 8.12 / 406,514 |

Greedy token IDs matched the first control exactly in every arm (8/8 fixtures).

On random token-ID prompts the target often reproduces rare tokens from the prompt. The reduced drafter can't propose those tokens, so acceptance falls and concurrent throughput drops 8–21%. Copy-heavy real traffic (agents quoting files, IDs, hashes, other languages) is at similar risk. On the chat panel, C1 gains 5–7%. That is below the ~9% ceiling the profile implied, because acceptance fell from 3.1 to 3.0.

**Verdict.** Under [§10.1](../../BENCHMARKING_STANDARDS.md), every size beats both neighbours on its target metric (chat C1), but each regresses tracked metrics by well over 3%. **Rejected as a default.** The code stays on the plugin's `draft-vocab` branch as an opt-in flag; it is inert when `INK2_DRAFT_VOCAB` is unset. A per-request extension (static set ∪ tokens already in the request's context) would address the copy failure mode, but it needs per-request draft-vocab masks. It is not done.

Production was restored and passed `live_check.sh` after the run. The live launcher was not changed.
