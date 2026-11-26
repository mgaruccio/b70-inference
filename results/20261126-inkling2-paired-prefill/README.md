# Inkling Mini on B70: paired prefix kernels

Development tier, isolated 409K loopback service. Same pinned environment as [the individual probes](../20261126-inkling2-verifier-kernels/README.md). Plugin `08592528` integrates the exact gather (`e292941d`) and merge (`a19cc60`) with independent, default-off flags. Control has only `INK2_FUSED_PREFIX_MERGE=1`; candidate adds `INK2_FUSED_PREFIX_KV_GATHER=1`. All other settings unchanged, including 409,000 context, C32 cap, FP8 KV, and the live two-segment DSpark schedule.

## First incremental comparison

`PRETEST='bash check-paired.sh' C8_DIAG=sandwich timeout 5400 bash hillclimb.sh paired-ctl paired-cand paired-ctl`, run `run-20261126-193701.bwiKiC`. Seven XPU cases passed without skips, including actual dispatch and bitwise results for all four flag combinations. One warmup and three scored distinct cold 131,072-token prompts per arm; actual cold TTFT medians **43.880 / 42.282 / 43.883 seconds**. Adding gather on top of merge saves another ~1.60 s (3.64%); this ABA does not directly measure both vs stock. Cold IDs matched and all cached-token counts were zero. C1 decode stayed ~109 tok/s. Fresh-engine C8 medians 316.47 / 317.49 / 322.48 tok/s, post-prefill 324.09 / 323.20 / 313.01. No C8 preemptions; no >3% throughput loss for candidate vs either neighbor on those panels. Artifacts in `sandwich/`.

## Confirmation gate

On the same staged build, `QUICK=0 timeout 7200 bash hillclimb.sh paired-ctl paired-cand paired-ctl`. Repeat cold/greedy comparison, +30K chat, concurrency 1–32 and candidate needle retrieval at both 200K/375K. Require explicit JSON passes, qualifying target/guardrails and verified restoration. No live promotion. A direct both-vs-stock serving comparison is separate from the incremental test.

The first full confirmation attempt exited 124, with no captured output and no newer run directory in the guest's verify-kernels tree. The production unit was active afterward. This is an unscored timeout, not evidence about retrieval, correctness or speed; the same ABA was resubmitted.

Partial confirmation (`run-20261126-195617.cFDTuY`, `confirm/`): all eight greedy fixtures matched; chat C1 was 109.1 / 109.1 / 109.1 tok/s; C8 was 323.9 / 316.7 / 309.4. Candidate passed retrieval at 207,257 input tokens. A stale guest harness omitted cold TTFT for these arm names and sent the 388,513-token fixture to control only. Do not count this as an independent cold repeat or a candidate long-needle pass. Production restoration was verified. For the corrected rerun, the local harness was copied under its content hash and checked before execution.

Corrected full ABA (`run-20261126-203429.sss85g`, `full/`): cold TTFT **43.916 / 42.291 / 43.906 s**, a repeatable 3.7% reduction beyond merge alone. All cold IDs matched with zero cached tokens and no preemptions; all eight greedy fixtures matched. The candidate passed both retrieval checks at **207,257 and 388,513** actual prompt tokens. Production restoration passed. Chat C1 and +30K were unchanged. The C8 guardrail did not clear: chat 334.3 / 313.5 / 304.5 tok/s and random 228.9 / 236.9 / 247.8; the candidate was between controls but below the faster neighbor by 6.2% and 4.4%, respectively. Not accepted as default.

Next development comparison: score the two-kernel bundle against both flags off on the same build (`QUICK=0 bash hillclimb.sh paired-stock paired-cand paired-stock`). This measures net user-visible benefit, not attribution. Workloads and gates as above; do not add the earlier single-lever percentages or waive the C8 guardrail.
