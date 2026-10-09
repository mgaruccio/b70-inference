# Shared-KV HumanEval+ diagnosis and announcement qualification

User request: “address whatever the humaneval+ issue is and then let's get the
code pushed and then compare our build with current community best for the B70
and see if we have something to announce”. No production change or public
announcement is authorized by this investigation.

## Initial evidence and predeclared process

Prior quality commit `bac718884c7af20ae943e3c12534e911e07b2876` is an ancestor of
HEAD; its archived working-tree files are absent, not lost. Read git objects or
the existing remote copies. Do not restore or stage unrelated archival deletions.
Same build-06 library `e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec`
had target/native/candidate HumanEval+ counts 140/139/136 out of 164. Candidate
regressions against native were `/1`, `/19`, `/130`; controls `/11` and `/126`
also expose native/target differences. Two candidate failures include markdown
fences; `/130` contains an incorrect algorithm. These observations do not yet
establish kernel causality. Do not strip fences, edit prompts, relax tolerances,
change precision or special-case benchmark tasks to erase the flag.

Fresh primary research before execution:
- <https://raw.githubusercontent.com/evalplus/evalplus/v0.3.1/README.md>:
  official saved-sample evaluation and Docker execution path.
- <https://raw.githubusercontent.com/evalplus/evalplus/v0.3.1/docs/execution.md>:
  timeout/OOM count as failure; avoid evaluator overload. Keep original time
  limits, full tests and serial scoring, not Mini or relaxed timeouts.

1. Re-score all saved HumanEval+ completions in target/native/candidate arms with
   the exact cached evaluator image, unchanged frozen dataset and scoring code.
   Use network-none, read-only, non-root, capability-free containers with bounded
   CPU/memory/PIDs and a writable output directory plus tmpfs only. Save command,
   source/image hashes, logs, all scores, exit and owned-container cleanup.
2. Preserve five selected primary rows verbatim from the frozen prepared JSONL.
   Exercise the existing real HTTP generation path using unchanged seed 42,
   temperature 0, 4096-token cap and EOS handling. Run native/candidate/candidate/
   native cells, four complete five-task repetitions per cell (eight per arm),
   plus the existing paired 64K token-ID diagnostic. Reuse current pre/post boot,
   boost-off, CPU-cap, 275W, launcher/source and kernel-error guards. Same image,
   model, precision, MTP4, 212992 context, 8192 batch, C1 and prefix-off contract.
3. Score every repetition through the same pinned scorer. Compare per-task pass,
   fence/format errors, output IDs, first divergence, finish reason and within-arm
   variation. Targeted results diagnose; they do not qualify the full model.
4. Change code only for a demonstrated mechanism. Require fixed-input operator
   checks and real HTTP reproduction, then the complete frozen quality suite and
   output-divergence process before claiming the quality issue resolved. Preserve
   failures; no benchmark-specific routing or post-hoc quality margin.
5. Push reviewed code/evidence only. Announcement readiness additionally requires
   the repository's full BetterBench, vLLM serving, context/concurrency, acceptance
   and capacity package and a dated, reproducible community comparison. A larger
   unpaired tok/s number is not evidence of community leadership.

Evaluator image: `sha256:4638874848e6aace7d1b5b8a1f1bdeb9993eb9edffdf8035dbef6057b23949c9`.
Original evaluator source SHA256:
`8544176f922079868d022f100de2f2e424a4617be321ae1df08dbbd2fdb11c48`.
Remote root: `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-shared-kv-quality`.
Runs remain outside Pi. Lab preview is unavailable (socket ENOENT); the user
explicitly requested this diagnosis, and ordinary scoped CLI execution is used.


## Saved-output scoring result

Fresh scoring reproduced all 492 saved per-task base/plus statuses exactly:
target 148/140, native 147/139, candidate 144/136 (each out of 164).
All three containers exited 0, auto-removed, and passed postconditions. This
rules out a scoring discrepancy in this rerun, not an inference regression.
The repeated-scoring wrapper uses the same Docker/evaluator argv as this
successfully exercised path, verified structurally after path-variable renaming.
It requires all four generation cells and exactly five declared IDs per repeat;
all 80 diagnostic samples must be retained, including failures.

## Corrected numerical diagnostic

The original operator probe forced the native helper to 32 splits. Actual
serving passes `num_splits=None`; the public C++ source is not proven to match
the installed wheel. The 416-length partition predicate is therefore scoped
to the forced-32 comparison, not serving auto. The proposed split-0 arithmetic
replacement was withdrawn because it did not preserve native partial rounding.
No kernel change is justified by that proposal.

Before interpreting numerical neutrality, `probe-auto.py` compares the unchanged
installed auto helper, the prior forced-32 control, and unchanged build-06 through
`flash_attn_varlen_func`. Predeclared: 21 lengths including the 139/149/280-token
failure contexts, two seeds, actual-length versus 212992 max-length metadata
(84 eager cases), plus six device-length changes in graphs captured at 65541.
Original rtol=0.02/atol=1e-4 remain the pass gate; bitwise differences and per-token
counts are separately recorded. This supplements rather than replaces the real
HTTP generation and full-quality gates. The existing guarded operator runner
uses isolated copied inputs; no original dependency or serving code is modified.


## Repeated diagnostic results

All four cells and all 16 scoring invocations completed; all 80 generations
stopped normally. Native/candidate passes out of eight observations per task:
`/1` 5/4, `/11` 6/6, `/19` 2/0, `/126` 8/8, `/130` 8/8. The remaining `/1`,
`/11`, `/19` failures were fenced outputs. Native outputs varied within a server
and between restarts; corresponding native-restart outputs matched 11/20, versus
16/20 between candidate restarts. These are correlated repeated observations of
five selected tasks, not 80 new benchmark tasks or proof of non-regression.

The automatic-path operator test passed all 84 eager and six graph cases at the
original tolerance. At 139/149/280 all tested routes matched bitwise, including
fixed-metadata graph replay. Differences at longer lengths establish that the
old forced-32 control cannot stand in for all serving-auto configurations. Neither
observation proves a cause for the historical model-output quality delta.

`diagnostics.tar.gz` retains the three saved rescoring outputs, four repetition
cells including scores, and automatic-path operator artifacts. Reproduce the
summary with `python3 compare-repeats.py --root <extracted-archive-directory>`.
The inherited top-level serving summary describes the auxiliary 64K diagnostic;
the five-task workload is recorded in `diagnostic_workload`, each generation's
metadata, and `repeat-comparison.json`. No throughput claim uses these cells.

## Full-suite follow-up (predeclared; paused by user)

The user redirected work to community benchmarks on the existing serving setup.
The target full run was intentionally interrupted (generation exit -15), its
server cleaned up, and the remaining full arms were not launched. This plan was
not completed and must not be treated as quality clearance or resumed implicitly.
The new full-run wrapper's generation/interrupt cleanup was exercised; its full
four-suite scoring path was not reached. Completed benchmark results are in
[`../20261009-qwen38-community-protocol/`](../20261009-qwen38-community-protocol/).

Run `run-full-quality.py` serially for target, native A1, candidate B1, candidate
B2, native A2, each with a fresh output directory. It invokes the unchanged
original `run-quality.py` without `--limit`, followed by the original evaluator
with `--task all`. Every arm must retain 2466 generations: 2402 primary tasks
(541 IFEval, 1319 GSM8K, 164 HumanEval+, 378 MBPP+) plus 64 frozen repeats. The
original full 64K diagnostic (six prompts, two repetitions each) also remains.
No prompt, precision, sampling, completion sanitization, per-test timeout or
kernel change is introduced. The 7200-second outer CPU scoring watchdog covers
all four suites; it does not change individual EvalPlus timeouts.

Retain commands, source hashes, generation records/token IDs, complete scores,
logs, guards and cleanup under `full-*/native/`. Compare both native/candidate
pairs and the target reference; do not select a favorable run or erase the
historical losses. Program completion is not a quality approval. Full-suite
results and standard performance/community comparison gates remain outstanding.
