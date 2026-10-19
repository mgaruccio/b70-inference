# W3 development performance result

**W3 is slower than matched no-draft A on both tested prompts. Do not promote it on the basis of this run.** This is a bounded, descriptive development result, not a publishable benchmark suite, general-model speed claim, confidence interval, or significance claim.

## Successful measurement

Run `run-r3` completed all four cold cells in order `VSPF-A1 → VSPF-W3a → VSPF-W3b → VSPF-A2`. Independent raw analysis reports `status: analyzed`, no errors, and verified restoration of the original `nemotron-bolt-probe` service. The model/build is still the previously qualified **r2** singleton candidate; `r3` identifies the benchmark attempt, not a new model or runtime build.

Both arms use the qualified r2 binaries and matched launch configuration; the intentional difference is no-draft versus W3 speculation. Each cell first passed all 25 public fidelity/function/diagnostic checks. Each of the two prompts then received one excluded warmup and three measured requests: **six measurements per arm per prompt, 24 total**, with greedy 128-token output and `cache_prompt=false`. No measured sample was removed or retried.

### Server-reported generation throughput

Values are medians; parentheses contain type-7 IQR width, in tok/s.

| Prompt tokens | No-draft A | W3 | W3 change |
|---|---:|---:|---:|
| 55 | 44.9053 (0.0262) | 32.7754 (0.0250) | **−27.01%** |
| 8,270 | 43.1840 (0.0137) | 30.6553 (0.0584) | **−29.01%** |

The headline is the runtime's final `timings.predicted_per_second`, preserved unchanged. It is not reconstructed from streamed chunks, the first-content proxy, or 128 divided by server generation duration.

### Client latency and throughput

| Prompt tokens | Metric | No-draft median | W3 median | W3 change |
|---|---|---:|---:|---:|
| 55 | Full request elapsed | 3.2879 s | 4.3619 s | +32.66% |
| 55 | First content | 0.4587 s | 0.4838 s | +5.46% |
| 55 | Output tokens / full request elapsed | 38.9306 tok/s | 29.3451 tok/s | −24.62% |
| 8,270 | Full request elapsed | 11.5226 s | 13.4460 s | +16.69% |
| 8,270 | First content | 8.5798 s | 9.3000 s | +8.39% |
| 8,270 | Output tokens / full request elapsed | 11.1086 tok/s | 9.5195 tok/s | −14.30% |

Full-request elapsed IQR widths were 0.04358/0.04390 s (A/W3, short) and 0.00537/0.01159 s (A/W3, long). Server prompt-processing medians were 456.537/481.249 ms (short) and 8,574.302/9,294.334 ms (long). Complete quartiles and per-cell summaries are in `run-r3/analysis-r1/analysis.json`.

Baseline generation medians were stable across the two loads: A1/A2 = 44.9011/44.9350 tok/s short and 43.1858/43.1821 tok/s long. This is descriptive evidence, not a statistical independence or significance claim.

## Fidelity and speculation

- **100/100 prevalidation jobs passed:** 25 each for A1, W3a, W3b, A2. Existing native/startup, actual PID/container, immutable source/binary, and diagnostics-off checks also passed.
- **24/24 measured responses matched their no-draft reference**, including 12/12 W3 responses. W3 comparisons use the fresh A1 reference. Prompt/output counts, finish reason, usage, and zero cached prompt tokens were checked.
- W3 short: every measured request reported 127 proposed and 84 accepted draft tokens (**66.14%** accepted/proposed).
- W3 long: every measured request reported 133 proposed and 82 accepted draft tokens (**61.65%** accepted/proposed).
- Baseline draft counters were absent and remain unavailable, not zero. Acceptance-by-position, acceptance histograms, and mean accepted length per verification cycle are not inferred from these aggregate counters.
- No raw-F32/vector equality is claimed by this performance experiment. W7 remains excluded.

The measured W3 bundle does not recover its additional cost despite accepting drafts. This experiment does not isolate drafter overhead from individual verifier/kernel costs.

## Verification and exact execution

Host: `mike@inference-host`, Intel Arc Pro B70. Target: NVIDIA Nemotron-3.5-Bolt-74B-A9B 24GB GGUF, with the frozen qualified model/serving configuration. Exact Docker commands, PID argv/environment, image identity, hardware, OS/kernel/driver/CPU, source IDs, and binary hashes are retained in the run artifacts rather than reconstructed here.

The successful capture/analysis command on the host was:

```bash
set -euo pipefail
cd /home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing
bash cpu-checks.sh > cpu-r3.log 2>&1
python3 -B controller.py --lead-run-timing 2>&1 | tee controller-r3.log
python3 -B analyze-timing.py --lead-after-timing-restore 2>&1 | tee analyzer-r3.log
```

SSH used `BatchMode=yes`, `ConnectTimeout=30`, `ControlMaster=no`, `ControlPath=none`, `ForwardAgent=no`, and keepalives. Local and remote CPU fixtures passed **14 tests**. Before the successful attempt, the real preflight and independent live restoration were exercised without stopping/creating containers; the saved hardware snapshot was checked against the actual preflight observation. The corrected container validator was also exercised against the actual, unmodified stopped r2 container, not a rewritten Docker record.

Successful terminal markers:

- `W3_ABBA_TIMING_CAPTURED_RESTORED_ANALYSIS_PENDING`
- `W3_ABBA_TIMING_ANALYZED_RESTORED_DEVELOPMENT_DESCRIPTIVE_ONLY`

Both `run-r3/restoration.json` and `run-r3/analysis-live-restoration.json` passed. A final live listing showed only `nemotron-bolt-probe` running. Persistent launchers, model files, builds, and original qualification artifacts were not changed. Experimental containers remain stopped for inspection.

## Failed attempts and limitations

Attempts r1 and r2 stopped **before native startup and before any timing requests** because of adapter-only Docker validation defects: fresh hostname comparison, then bind-list ordering. R1 also omitted the hardware snapshot required by its restoration verifier. Both attempts' errors, raw records, and exact source snapshots are retained; their failed restoration-verifier status is not rewritten. After each failure, the existing qualified gate independently verified the original service and a fresh READY smoke passed. These failures are not W3 performance samples.

The final run uses the same declared workloads, gates, warmups, sample counts, and launch configuration; only adapter lifecycle/evidence validation and fresh attempt names changed. Two shell-dispatch attempts also failed before executing their requested check because the shell was fish; explicit Bash execution corrected them without changing benchmark code or criteria.

This tests two reused, cache-disabled prompts at concurrency one, not unique-prompt generalization, BetterBench, `vllm bench serve`, or a full context sweep. It does not establish maximum usable context or peak VRAM. Exposed energy, frequency, and temperature samples are retained; unavailable occupancy/bandwidth/peak-VRAM fields remain unavailable. No outlier exclusion, confidence interval, or promotion was performed.

## Evidence locations

- Canonical local copy: this result directory.
- Complete host copy: `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing/` on `inference-host`.
- Successful raw data and independent analysis: `run-r3/`, especially `analysis-r1/analysis.json`, per-cell `prevalidation.json`, `timing-rows.json`, native logs, and measured request `.http-body`, `.events.json`, `.result.json`, and `.telemetry.json` artifacts.
- Failed attempts: `run-r1/`, `run-r2/`; exact corresponding code: `attempt-r1-source/`, `attempt-r2-source/`.
- Recovery checks: `aborted-r1-original-*`, `aborted-r2-original-*`, `aborted-r2-container-check.json`, and `preflight-check-r2/`.

Git stores the 88 raw log files as `.log.gz` copies (126,057,896 compressed bytes).
Every decompressed archive was SHA-256 checked against its original before
commit; the uncompressed originals also remain locally and on the host. To
restore a log in a fresh checkout, use `gzip -dk path/to/file.log.gz`. All other
raw API, telemetry, JSON, and source artifacts retain their original bytes.

Disk space did not cause a failure. Approximately 7.48 GiB remained after the successful run; no host cleanup or deletion was needed.
