# Profile-first EAGLE-4 drafter experiment

**Development-tier engineering, not a standard-compliant benchmark or quality certification.**

## Outcome

**EAGLE-4 did not beat Intel's public recipe.** A depth-4 draft tree completed the real serving checks and reached **12.403 output tok/s**, versus **17.453 tok/s** for a fresh public MTP4 control at the approved matched memory fraction 0.9. EAGLE is **28.93% lower** in end-to-end output throughput; its median TPOT is **1.684×** MTP's. The attempted draft-only graph optimization failed real-server qualification and was reverted. Nothing was promoted to a persistent deployment.

The depth-4 CLI measurement is better than the [previous restored EAGLE measurement](../20261025-intel15-eagle4/RESTORATION.md) (8.467 tok/s, median TPOT60.575ms), but **that is a historical, non-interleaved comparison, not a causal 46.5% tree-tuning claim**. Today's same-input depth14/depth4 HTTP comparison is mixed: depth4 decode is 5.88% slower at8K and 5.44% faster at32K. A fresh depth14 CLI control was not run.

## Profile first, then select a change

Before optimizing, the restored source `24f1c632ad9c641cd3dcd0391c9ed8112e79763f` was profiled on the actual B70 through HTTP. Both input classes were warmed. Profiling started only after the first returned token, with CPU/XPU activities, shapes/stacks/memory disabled, and a12-worker-iteration limit. Temporary `record_function` labels added no computational changes or synchronization. The unprofiled control used unmodified source, no labels and no profiler.

| Diagnostic per decode round | 8K profile | 32K profile |
|---|---:|---:|
| Observed rounds | 12 | 12 |
| Draft steps | 9.083 | 8.917 |
| Inclusive host draft growth, ms | 56.370 | 55.434 |
| XPU work launched inside draft growth, ms | 25.668 | 25.203 |
| Inclusive host proposal stage, ms | 82.313 | 81.396 |
| Total XPU duration sum, ms | 79.048 | 95.591 |

At32K, draft growth includes ~14.45ms of GEMM kernels and7.87ms of draft attention per round. The target also spends ~10.39ms on tree-prefix materialization and14.94ms on native attention plus its reduction. Draft launch overhead and unnecessary depth were promising candidates, but the target verification cost remains substantial.

**These are attribution diagnostics, not benchmark latencies.** Profiling inflated mean round wall time to~376ms. Nested host spans are inclusive; do not sum them. XPU duration sums are not busy-time unions. Kernel attribution uses the correlated host launch timestamp, not merely temporal overlap with asynchronous device execution; all captured kernel/copy events found a matching runtime launch. Raw traces and the analysis are in [`profiles/`](profiles/) and [`commands/summarize-profile.py`](commands/summarize-profile.py).

## Experiments and qualification

1. **Unchanged eager control:** six measured512-output requests at each of8192/32768 saved input tokens, plus excluded protocol/warmup requests and three semantic chat canaries. All succeeded.
2. **Draft-only graph replay:** the opt-in in `f38a68c2b2495f25dc21823c9d9e066c8a8c1a68` enabled the existing per-step XPU capture machinery without changing eager target execution or the tree. CPU wiring tests passed20/20; the real-XPU native suite passed100/100 with zero failures/errors/skips in241.324s. Independent source review found no blocking contract violation. **Nevertheless, the real server failed:** four excluded512-output requests and the first measured8K request succeeded, then the second measured8K request stopped delivering tokens after7 returned token IDs and reached the600s socket timeout. The logs contain no Python exception identifying a root cause. The cleanup trap stopped the server. CLI and the planned post-measurement replay trace never ran. **Reject this candidate; microkernel/toy graph tests did not qualify the real draft forward.** Commit `2308f3a` reverted the opt-in; current production files and tests match24f1c63 exactly.
3. **Final bounded fallback:** use the existing configuration knob `tree_depth=4`, reducing growth from~9 observed steps to3 (confirmed by server logs). Keep width16,15 proposals, top-k4 and gain0.0. Target and draft are eager. All16 HTTP probe requests, all three semantic canaries and all10 measured CLI requests succeeded. This changes the proposal set, not the target acceptance rule; no relaxed acceptance or precision change was introduced.
4. **Fresh Intel control:** rerun the released-image public MTP4 recipe at0.9 with the same HTTP fixture, canaries and exact CLI workload. All16 probe requests, three semantic canaries and10 measured CLI requests succeeded. The published0.8 control remains unchanged in the [previous results](../20261025-intel15-eagle4/RESTORATION.md).

## Fresh CLI comparison

Each arm: one excluded warmup, ten measured requests,32768 input/512 output tokens, temperature0, seed0, ignored EOS, configured concurrency1, prefix caching off. Both arms produced327680 input and5120 output tokens with zero errors. Output throughput includes prefill; TPOT excludes the first token. Speculative counters below are from the CLI measured window, not whole-invocation Prometheus deltas.

| Metric | Intel public MTP4, mem0.9 | EAGLE depth4, mem0.9 |
|---|---:|---:|
| Successful / failed requests | 10 / 0 | 10 / 0 |
| Median TTFT, ms | 20604.8733 | 20610.2312 |
| Median TPOT, ms | 15.48004 | 26.07598 |
| P90 TPOT, ms | 20.42989 | 73.09138 |
| Mean TPOT, ms | 17.08273 | 40.44480 |
| Output throughput, tok/s | 17.453032 | 12.403237 |
| Measured duration, s | 293.358766 | 412.795481 |
| Acceptance, % | 83.87234 | 11.03734 |
| Mean acceptance length | 4.35489 | 2.65560 |
| Rounds | 1175 | 1928 |
| Proposed / accepted tokens | 4700 / 3942 | 28920 / 3192 |

Tree acceptance fraction divides by15 branch proposals, whereas MTP divides by4 chain proposals; those fractions are **not directly comparable**. Acceptance length includes the bonus token. The shallow tree reduced per-round work but still needed substantially more target verification rounds than MTP.

The CLI's retained `max_concurrent_requests=2` is its one-second-bucket peak statistic, not the configured request limit: the pinned benchmark counts both adjoining serial requests in a shared second (`vllm/benchmarks/serve.py:693-706`). Do not relabel the run as configured concurrency2 or silently rewrite the raw field. Similarly, SSE-event ITL is not per-token TPOT under multi-token speculative streaming.

## Same-input HTTP comparison

Six measured requests per cell;512 output tokens, identical saved token-ID prompts and sampling. Below is median post-first-token decode time, with IQR in parentheses, seconds. Full timing arrays, TTFT/full latency, counters and outputs are in [`summary.json`](summary.json) and compressed raw probe documents under [`checks/`](checks/).

| Input tokens | Unchanged EAGLE depth14 | EAGLE depth4 | Fresh public MTP4 mem0.9 |
|---|---:|---:|---:|
| 8192 | 14.183557 (0.985647) | 15.017772 (0.546559) | 12.405521 (0.527657) |
| 32768 | 14.232679 (1.463653) | 13.458020 (0.969949) | 11.105261 (0.184260) |

The counter-based mean acceptance length fell from4.403 to3.648 at8K and from4.552 to3.931 at32K. More rounds offset the cheaper draft:700→844 total rounds at8K and677→782 at32K. This is not a uniform EAGLE improvement and does not beat MTP at either saved-input point.

## Configuration and limits

- B70 PCI0xe223, pinned `intel/llm-scaler-vllm:1.5` local image ID `sha256:52218ad85513ab6686d4c090c83c2bd8c5b02423c63aa4dabd41837fe641fe3b`; Torch2.12.0+xpu and Triton3.7.1. Target `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Both arms retain TP1, sym_int4 target weights, FP16 compute, FP8 E4M3 target KV, FP16 SSM, max length40000, batch-token limit8192, max sequences64, requested block64/runtime1024, eager execution and no prefix caching. No target checkpoint, quantization or context reduction was made.
- EAGLE retains FP16 draft KV/window4096, width16,15 proposals, top-k4, gain0.0, V2/CUSTOM and engine6cae86c. Only configured depth14→4 changes between the successful EAGLE arms. Width8 was not substituted: the qualified native FP8 long-context path is width16.
- Intel uses the released legacy runner and native MTP method `qwen3_5_mtp`/4. EAGLE requires its engine hooks and verifier. This is an installed-configuration comparison, **not a drafter-only causal experiment**.
- EAGLE's logged available KV budget/capacity remains5.51GiB/41978 tokens; fresh MTP0.9 logs7.02GiB/153454 tokens. These are initialization capacities, not measured peak allocations or a claim that the configured context was increased. EAGLE still does not fit the requested configuration at the published0.8 budget.

**Correctness remains unproven beyond the retained checks.** All successful arms answered the chat canaries with437, a valid Python add function and JSON `{"answer":42}`. But depth4 matched none of the six unchanged-EAGLE output sequences at either context. First differing positions versus unchanged repetition0 were `[188,16,188,43,188,55]` at8K and `[244,1,244,1,2,1]` at32K. Unchanged EAGLE itself produced six distinct outputs at each context; depth4 produced five. These results neither establish exact greedy equivalence nor certify quality non-regression. No BetterBench/full quality suite, context/concurrency sweep, interleaved crossover or matched thermal study was performed. No persistent deployment is recommended from this attempt.

## Executed journey, commands and retained evidence

The predeclared process and fresh primary-source research are in [`PLAN.md`](PLAN.md). The exact experiment launchers/runners are in [`commands/`](commands/); they are temporary benchmark commands, not persistent inference launchers. Source/weights remain external; no EAGLE source or weights are redistributed here.

Coordinator execution, in serial GPU order:

```bash
bash /tmp/20261026-eagle4-profile/drive-profile.sh
bash /tmp/20261026-eagle4-profile/drive-eager-control.sh
bash /tmp/20261026-eagle4-profile/drive-candidate.sh       # rejected replay attempt, exit1
bash /tmp/20261026-eagle4-profile/drive-depth4.sh
bash /tmp/20261026-eagle4-profile/drive-mtp-final.sh
```

Servers were accessed only through loopback `http://127.0.0.1:18110`: require GET `/health`; run `tests/intel15_http_probe.py run --repeats 6 --fixture <run-root>/fixture.json --baseurl http://127.0.0.1:18110 --model Qwen3.8-27B --timeout 600`; then run the three chat canaries. The final CLI command is preserved verbatim in `commands/run-arm.sh` and both benchmark logs: `vllm bench serve`, random32768/512, ten prompts/one warmup, seed/temperature0, ignored EOS and maximum concurrency1. Only the rejected replay arm ran `commands/run-native-graphs.sh`; its XML is independently parsed as100 tests with no failures/errors/skips.

Full original artifacts: `inference-host:/home/mike/b70-evals/20261026-eagle4-profile/`, mirrored coordinator `/tmp/20261026-eagle4-profile/`. The MTP benchmark succeeded, but its first coordinator copy failed because the local parent directory did not exist; creating that directory and copying the already completed artifacts repaired collection without rerunning or replacing measurements. The raw results, failure evidence, server logs, semantic responses and two original compressed XPU traces are retained in this directory. Existing outputs intentionally refuse overwrite; use a new run root/label for reproduction.

Final cleanup was freshly checked: **no running containers, no listener18110 and no matching GPU-serving processes**;73.9GiB remained free. See [`checks/final-cleanup.txt`](checks/final-cleanup.txt). No EAGLE upstream push or persistent launcher change was made.
