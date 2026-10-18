# W3 suppression matches the no-draft baseline

**Development isolation result; no timing comparison or promotion.**

The lead ran the declared control on `inference-host` after all 11 remote CPU fixtures passed. Compared with active W3, only the disposable container name and `--spec-draft-n-min 0` → `4` changed. The runtime, weights, maximum draft count 3, recurrent-state depth 3, output capacity 4, block size 8, and remaining settings stayed fixed. Actual startup and PID evidence passed without normalization or clamping.

All four frozen requests matched both the original frozen A and the fresh width-sweep A:

- Short ordinary response: exact text, finish and usage.
- Short diagnostic: exact actual emission IDs, pieces, text, finish and usage.
- Long ordinary response: exact text, finish and usage.
- Long diagnostic: exact actual emission IDs, pieces, text, finish and usage.

These same comparisons failed for active W3. Its diagnostic first differences were output IDs 116 (55-token prompt) and 14 (8270-token prompt). Both diagnostic inputs here retained the original prompt tokens and generated 128 tokens; output IDs were not reconstructed by retokenizing text.

## Suppression was observed, not assumed

The final cumulative server statistics explicitly recorded:

```text
#calls(b,g,a) = 4 504 0
#gen drafts = 0; #acc drafts = 0
#gen tokens = 0; #acc tokens = 0
dur(b,g,a) = 0.004, 11930.648, 0.000 ms
```

Thus the drafter executed 504 generation calls while all proposals were discarded. Every captured cumulative counter record had zero generated/accepted tokens. All four HTTP responses omitted draft counters; those omissions were **not** treated as zero evidence. The duration above demonstrates execution only, not a performance comparison.

## Interpretation and limit

W3's wider allocation and prefill configuration can reproduce A when no proposals enter verification. This narrows the observed discrepancy to the active proposal/verification/rollback path. It does **not** identify the responsible arithmetic, prove a router cause, establish raw-F32/logit equality, or establish the corresponding result for W7.

The next isolated experiment will test extending the existing guarded router per-column decomposition to 2–8 target columns. Draft maxima 3 and 7 allow target batches of four and eight columns. That experiment must pass native checks and the complete public correctness screen before timing; it is not a verified fix or a result of this control.

## Execution and cleanup

Remote working directory:
`/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width3-suppression`

```bash
bash cpu-checks.sh
python3 -B probe.py --lead-run-suppression
python3 -B probe.py --lead-after-restore
```

The controller and independent analysis completed successfully. The accepted `nemotron-bolt-probe` identity, configuration, full canonical mount objects, health, stop smoke, sole live DRM ownership, hardware settings, protected source maps and five-binary manifests passed restoration checks. The suppression container is stopped and preserved. No runtime source, binaries, persistent launcher or model weights changed. No request was retried or replaced.

Evidence is in `preparation/`, `worker-scratch/cpu-9mlNadxU/`, `controller.log`, `analysis.log`, and `run-r1/`, including all raw request/response bodies, startup/PID records, native gates, cumulative server logs, comparisons and independent restoration. The cumulative original stdout prefix is retained in Git as `run-r1/ORIGINAL/stopped-prefix.stdout.log.gz`; the uncompressed remote original is preserved, and `stopped-log-prefix.json` identifies the decompressed bytes.
