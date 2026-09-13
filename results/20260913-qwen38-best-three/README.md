# Best MTP4 / DSpark / DFlash2 comparison

Status: **completed successfully** (2026-09-13 02:22:43–03:05:29 UTC). Development tier only. No production promotion. All three cells exited0; no failed cells or excluded samples.

## Authorized scope and baseline

User: “let's get a comparison run of our best mtp, dspark, and dflash config, I'd like to see where they all stand as of now”. User selected **Best configurations** rather than normalized context/batch settings.

Baseline is the current best MTP4 bundle. Same B70 / 275 W, target checkpoint, FP16 GPTQ compute, FP8 target KV, C1, greedy seed42, thinking off, prefix cache off, input payloads and 128 forced output tokens. Deliberate bundle differences include runtime, draft quantization, runner, context allocation, prefill batch and graph shape. This user-selected development comparison intentionally does not satisfy the same-maximum-context rule for a controlled/publishable comparison in `BENCHMARKING_STANDARDS.md`. Do not attribute a difference to algorithm alone.

| Bundle | Context cap | Prefill batch | Runtime | Speculation |
|---|---:|---:|---|---|
| MTP4 | 212992 | 8192 | ac7509e2b / f01e24f6 image | K4, BF16 draft with INT4 S+M1, v5 mixed split |
| DSpark Split-K | 65664 | 2048 | 73029d424 / 7a558f63 image, V2 | K7, corrected BF16 draft/KV, standard rejection, greedy draft, adaptive off, native noncausal Split-K |
| DFlash2 INT4 | 180224 | 2048 | 73029d424 / 7a558f63 image, legacy V1 | K7 full verification, partial RTN INT4/G128 draft, BF16 draft compute/KV, cache group8, adaptive off |

Frozen `*-reference-launch.json` files are byte copies from:
- `../20260911-qwen38-step-profile-64k/mtp4-production-shape/launch-argv.json` (55.698 tok/s historical 64K median).
- `../20260912-qwen38-dspark-noncausal-split-k/candidate-01/launch-argv.json` (30.660 tok/s historical 64K median).
- `../20260910-qwen38-dflash2-adaptive-verification/baseline/launch-argv.json` (46.14 tok/s historical 64K median). This later cache-group8/batch2048 configuration supersedes the older 64K/batch8192 DFlash measurement. Cap3/adaptive were not consistently faster through64K.

No new kernel, acceptance-policy change or tuning sweep. MTP's production prefix cache is disabled for the common cold workload. Replay only renames the disposable container, redirects writable `/output`, and makes historical `/profile` source read-only. No production launcher modifications. DSpark source must match final qualified SHA256 `e19d3fa94a11942aeb9db4ca4552ab4357373c14d7095f7612761a05b1d924ff`.

## Fresh research affecting the approach

Fresh official-source research performed before implementation:
- https://docs.vllm.ai/en/latest/features/speculative_decoding/ — speed is workload-dependent; numerical/non-determinism caveats prevent equating standard rejection with guaranteed identical strings. Preserve the empirically tuned depths rather than reset to documentation defaults.
- https://docs.vllm.ai/en/latest/features/per_request_metrics/ — TTFT, generation interval and total request latency are distinct. Retain existing raw streaming timing/usage, do not conflate decode throughput with end-to-end throughput.
- https://docs.vllm.ai/en/latest/features/speculative_decoding/acceptance_metrics/ — acceptance is diagnostic, not throughput. Use the pinned versions' existing Prometheus draft/accepted/position counters; do not add newer runtime flags.

## Defined real end-to-end process

Environment: `inference-host`, explicit Bash, existing pinned disposable Docker images and local model/draft files. No ML dependencies in Pi. Preconditions: no running containers/render-device holders, stopped `glimmer-tb21-prefix-c8`, production launcher SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`, power275000000 microwatts. Initial read-only host check passed.

1. Stage this directory only under `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-best-three`. Existing historical inputs stay unchanged.
2. Sequential order **MTP4 → DSpark → DFlash2**, one server load each; no opportunistic retuning or silent retries. This is not interleaved statistical qualification.
3. `python3 -u run-comparison.py --cell CELL --out CELL-01` invokes the existing step-profile driver's lifecycle, pinned runtime capture and cleanup without profiling. Replay the exact selected serving arguments.
4. Existing `run-acceptance-diagnostics.py` public API gates cover health/model listing, completions/chat, three canaries,131 finite-logprob boundaries, eight sandboxed functional tests, repeatability collection. Require all gates to pass. DSpark must log actual Split-K dispatch.
5. Existing unchanged `qwen38_long_context_bench.py` hits the real `/tokenize` + `/v1/completions` streaming interface at exactly **512,8192,32768,65536 input tokens**,128 output tokens, temperature0,seed42,ignoreEOS. One warmup and six measured trials per length. Deterministic nonce content differs per trial, matches across configurations. No mocked service or synthetic operator timing.
6. Require reported rendered/input and output counts, complete SSE streams, finish reason length, six valid measurements per point and speculative counter snapshots. Retain all raw requests/SSE/metrics/output, gates, image/runtime captures, exact argv, input-source hashes, errors and cleanup in `CELL-01/`; retain console output and exit codes. No sample exclusion.
7. Summarize median/inclusive IQR decode proxy, TTFT, whole-request latency, draft acceptance, payload identity and output-text identity against MTP4. Client decode proxy is not GPU-only time; verify formula from existing client before reporting. Exact text equality is not token-ID equality or broad quality qualification.
8. Each cell removes only its owned disposable container. Final check verifies no running containers/render-device holders and unchanged production launcher/power. Copy evidence back, parse/check summaries, document failures and results, then commit and push the finished campaign.

Limits: four common context points, C1 only, one sequential load per bundle, no confidence interval, full BetterBench, full quality suite, near-limit capacity sweep or peak-memory qualification. DSpark's64K prompt is exactly at its configured128-output boundary; MTP/DFlash have headroom. Output variability already exists; no bitwise-equivalence claim. Report configured capacity separately from the largest successfully tested request in this campaign.

Lab `preview_action` was attempted and failed because `/tmp/prime-lab-1000/2c31a91a094d7226455d297a/lab.sock` is unavailable; CLI execution proceeds under the user's explicit scope confirmation.

## Observed results

**MTP4 remains the strongest long-context decode bundle.** DFlash2 is competitive at512/8K, but the small gaps are not statistically established in this sequential six-sample run. DSpark Split-K remains substantially slower at every tested length despite its previously qualified improvement over plain DSpark.

### Decode throughput, median (inclusive IQR), tok/s

| Input tokens | MTP4 | DFlash2 INT4 | DSpark Split-K | DFlash vs MTP | DSpark vs MTP |
|---:|---:|---:|---:|---:|---:|
|512|70.84 (4.87)|73.42 (6.41)|50.89 (5.19)|+3.65%|−28.16%|
|8192|63.25 (2.04)|62.08 (11.91)|42.26 (5.79)|−1.85%|−33.19%|
|32768|60.64 (1.39)|50.75 (0.83)|37.58 (2.75)|−16.31%|−38.03%|
|65536|54.45 (2.23)|45.22 (8.94)|31.20 (0.56)|−16.95%|−42.71%|

These are the unchanged client's **post-first streaming burst-rate proxy**, `(128−1)/(stream_end−first_nonempty)`, not individual-token ITL or device-only decode timing. All six samples and unrounded medians/IQRs are in `comparison.json`. Warmups are excluded; no outliers were dropped.

### TTFT / whole-request latency, medians, seconds

| Input tokens | MTP4 | DFlash2 INT4 | DSpark Split-K |
|---:|---:|---:|---:|
|512|0.289 / 2.082|0.283 / 2.014|0.324 / 2.808|
|8192|4.197 / 6.209|4.356 / 6.396|4.934 / 7.955|
|32768|21.118 / 23.206|21.192 / 23.694|23.479 / 26.866|
|65536|53.866 / 56.195|52.513 / 55.323|57.086 / 61.159|

At64K, DFlash's faster TTFT offsets its slower decode for this128-output request: whole-request median is0.872s lower than MTP (~1.55%). This small sequential difference is not a confidence-qualified win. Do not infer whole-request gains from decode rankings, or extrapolate to longer outputs/concurrency.

### Measured speculative acceptance

Counts aggregate only the six measured requests at each point; use observed proposals, not nominal K×rounds for DSpark's clipped64K boundary.

| Input | MTP accepted/proposed (rounds) | DFlash accepted/proposed (rounds) | DSpark accepted/proposed (rounds) |
|---:|---:|---:|---:|
|512|507/1084 (271),46.77%|537/1659 (237),32.37%|510/1883 (269),27.08%|
|8192|488/1132 (283),43.11%|510/1841 (263),27.70%|476/2114 (302),22.52%|
|32768|510/1048 (262),48.66%|504/1890 (270),26.67%|477/2079 (297),22.94%|
|65536|521/1024 (256),50.88%|511/1778 (254),28.74%|468/1887 (287),24.80%|

At64K, accepted draft tokens/round are2.035 MTP,2.012 DFlash,1.631 DSpark. DSpark proposes6.575/round on average at its exact boundary, versus4 MTP and7 DFlash. These runtime counters can include terminal over-generation; accepted+rounds is not necessarily final user-visible output count. Position counters remain in every raw metrics snapshot.

### Correctness and fidelity

- Each bundle passed3/3 canaries,131 finite-logprob boundaries,8/8 sandboxed functional tasks, and repeatability collection (one distinct prose and one code output in each short repeat family).
- All84 cold streams passed:72 measured +12 warmups, with exact input counts,128 output tokens, complete SSE and finish reason `length`.
- All28 cold request payloads (including warmups) are identical across the three bundles. All19 fixed short output-token-ID arrays match MTP for both alternatives.
- Measured long-sweep **text** matches versus MTP at512/8192/32768/65536: DFlash **2/6,0/6,4/6,2/6**; DSpark **2/6,1/6,4/6,1/6**. Aggregate8/24 for each alternative, but on different requests. Do not call this token-ID equality or quality equivalence.
- No fresh target-only or long-prompt repeated-baseline controls were run. Prior unchanged-baseline variability and different runtime bundles prevent assigning all divergence to a particular drafter/patch. Full quality qualification remains unperformed.

### Runtime/capacity observations

| Bundle | Runtime | Available KV memory | Runtime KV capacity | Graph memory | Configured cap | Largest tested input+output |
|---|---|---:|---:|---:|---:|---:|
|MTP4|vLLM0.27.2rc1.dev77+gac7509e2b|8.23GiB|226397 tokens|0.15GiB|212992|65536+128|
|DFlash2|vLLM0.28.1rc1.dev278+g73029d424|7.17GiB|184811 tokens|0.16GiB|180224|65536+128|
|DSpark|vLLM0.28.1rc1.dev278+g73029d424|7.22GiB|74528 tokens|0.08GiB|65664|65536+128|

All runtime captures report Torch2.13.0+xpu and XPU available. Exact image digests/configuration are retained per cell. The common cold client SHA256 is `a01a99b21f36ef446d220df66a4e739a02b3ab18dfe99dec8beb333719276907`. DSpark logged the expected native noncausal dispatch. Available KV and graph allocation are not peak-memory measurements; this run does not requalify any larger context capacity.

## Execution, verification and artifacts

Executed from the remote campaign directory:

```bash
bash run-all.sh
# Calls, in order, with timeout90m per cell:
python3 -u run-comparison.py --cell mtp4 --out mtp4-01
python3 -u run-comparison.py --cell dspark --out dspark-01
python3 -u run-comparison.py --cell dflash --out dflash-01
```

Observed UTC windows: MTP02:22:43–02:37:02; DSpark02:37:02–02:50:57; DFlash02:50:57–03:05:29. All exit0 (`cell-exits.tsv`). No profile instrumentation was enabled; DSpark's `B70_STEP_TIMING` flag is only the existing import-shim activation for Split-K.

Evidence: `CELL-01/length-LENGTH/long-context/` retains requests/rendering/SSE/metrics and summaries; each cell root retains API gates, logprobs, outputs, source hashes, launch argv, image/runtime captures, server logs and cleanup. `*.command.txt`, `*.console.txt`, `*.preflight.txt`, `*.cleanup.txt`, `host-final.txt` preserve shell execution and host checks. No historical evidence was overwritten.

Local verification:

```bash
python3 results/20260913-qwen38-best-three/analyze.py
bash -n results/20260913-qwen38-best-three/run-all.sh
```

Analysis validates all three cell/gate statuses,84 streams, token counts, identical payloads, finite timing data, medians, observed speculative counters and unchanged per-cell host snapshots. All1490 JSON files parsed successfully after retrieval (including generated comparison); AST checks passed for both authored Python scripts. No new ML dependencies were installed.

Final host check at2026-09-12T23:05:59-04:00 confirmed no running containers/render-device holders, unchanged launcher hash and275000000-microwatt cap. No production promotion or persistent changes.

Publication checklist: raw data/exact commands/runtime identity/speculative metrics/medians/cleanup retained; full BetterBench, vLLM bench serve, full capacity sweep, peak memory and full quality/output-equivalence suite **not run**. This remains a development-only result.
