# Widths 3 and 7: fidelity rejected, timing blocked

**Development correctness screen completed. Neither wider width is timing-eligible. Investigation continues; no promotion.**

The lead ran fresh A (same candidate runtime, no draft), W3 (`n_max=3/min=0`) and W7 (`n_max=7/min=0`) on `inference-host`, with the unchanged frozen requests and existing five-width/single2 runtime. The original service was restored and independently verified after the screen. No runtime source, binary, model, hardware cap, or persistent launcher was changed.

## Observed public-boundary results

A passed all 25 checks and reproduced the frozen no-draft reference. Both W3 and W7 passed all 20 quality cases and the retrieval anchor, including exact text/finish/usage. Both failed the short and long ordinary-output comparison and both corresponding actual-emission-ID diagnostics.

| Arm | Prompt tokens | First differing diagnostic output ID (1-based) | Draft tokens proposed | Accepted |
|:---|---:|---:|---:|---:|
| W3 | 55 | 116 | 127 | 84 |
| W3 | 8270 | 14 | 128 | 84 |
| W7 | 55 | 116 | 225 | 94 |
| W7 | 8270 | 14 | 252 | 91 |

These positions come from actual `choices[0].logprobs.content[*].id`, checked against verbose IDs and pieces, not retokenized ordinary text. All four diagnostic responses exercised real rejected proposals. No probability or raw-F32 equality claim is made.

**Bookkeeping caveat:** the controller increments diagnostic/rejection behavior counters only after the equality comparison succeeds. Consequently the rejected-width reports say those behaviors were not exercised and add a behavior failure. That is not evidence that requests or rejections were absent: their saved raw diagnostic responses and comparisons contain the counts above. The four explicit output mismatches independently disqualify each width; the bookkeeping limitation does not change selection.

`qualification-r1/qualification.json` has an empty qualified-width list. The explicit timing entrypoint therefore records a blocked result; no warmup or measured timing request for this sweep is permitted. Qualification request timings must not be presented as performance comparisons.

## What this does and does not localize

The pinned LiLiCorr code always evaluates the checkpoint's full eight-position draft block, even at `n_max=1`, then consumes only the requested proposals. The trained usable maximum is seven. This makes larger widths worth investigating, but does not make them automatically correct or fast.

The existing per-column router workaround is restricted to **two target columns**. Draft maxima 3 and 7 permit nominal target verification batches of **four and eight columns**, not three and seven. Router output dimensions 240/304/368/432/496 are another, separate dimension.

Wider `n_max` also changes recurrent-state depth, output capacity, rollback, and potentially the prefill scan path. The common divergence positions suggest a shared numerical/state-path issue but do not establish which component causes it. No arithmetic/kernel patch has been made on this evidence alone.

The next bounded isolation is same-width W3 with `n_max=3/n_min=4/p_min=0`, retaining the wider context/state/drafter configuration while discarding proposals. Its actual startup and cumulative server counters must prove no clamp, zero proposals/acceptances, and positive drafter execution. It will compare the same four short/long ordinary and diagnostic requests against A. This control is being prepared separately; it is not a result of this case. `n_min=8` cannot suppress W7 because the trained limit clamps it to seven.

## Existing width-1 timing clues

Read-only analysis of the prior width-1 C1/C2 logs found 31 cumulative speculative-stat records per cell. The last eight align with the declared four short and four long requests; consecutive counter deltas matched every measured HTTP proposal/acceptance count. Across six measurements per point:

- Short: median draft-generation duration **1555.3805 ms**, 69 draft calls, approximately **22.5417 ms/call**.
- Long: **1601.034 ms**, 70 calls, approximately **22.8719 ms/call**.
- Median server prediction time minus the corresponding draft duration was **2675.653 ms short / 2751.8095 ms long**. This residual includes verification, feature/state work, sampling and other loop work; it is not a pure target-kernel profile.

The unchanged no-draft candidate's earlier generation-time medians were 2830.955/2942.066 ms. These measurements explain why paying for a full draft block to use only one proposal is unattractive, but do not identify the wider-width fidelity defect.

## Execution and retained evidence

Remote directory: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width-sweep`.

```bash
bash cpu-checks.sh
python3 -B run-qualify.py --lead-run-qualification
python3 -B analyze-qualify.py --lead-after-qualification-restore
python3 -B run-timing.py --lead-run-time-eligible
python3 -B analyze-timing.py --lead-after-timing-restore
```

The final CPU gate passed 18 controller/contract fixtures plus three qualification-selection tests. Earlier failed preparation attempts are preserved under `worker-scratch/`: stray import-time `EOF`, fixture path/count/source-assert errors, and an otherwise passing test process that exited before freezing evidence. These were preparation failures, not hidden inference retries. Width-aware startup validation, current-phase name collisions, exact consumed-reference pins, rejected-sibling selection, and successful cleanup were corrected before live work. No live qualification request was retried or replaced.

Artifacts include frozen preparation evidence, command logs, all 75 public checks and full mismatches under `qualification-r1/{A,W3,W7}`, native/server logs, actual PID environment/argv checks, source/binary/hardware checks, restoration smoke and independent live-restoration records, and blocked timing records. The accepted original remains the service; all three qualification containers are stopped and preserved.

The 439,011,667-byte cumulative original stdout prefix is stored losslessly as `qualification-r1/ORIGINAL/stopped-prefix.stdout.log.gz` in Git to stay below the hosting file-size limit. Its uncompressed original remains on the inference host. `stopped-log-prefix.json` records the hash and size of the decompressed bytes; compression does not alter any measurement or comparison.
