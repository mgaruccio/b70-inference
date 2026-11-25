# Inkling 2 Mini: upstream vLLM and sparse conv history on B70

**Development tier; not a production rollout.** Compare the production source (3146a9df41, including its existing XPU head patch), clean upstream main (0085facc58, includes [#69186](https://github.com/vllm-project/vllm/pull/69186)), and upstream plus both commits of [#69444](https://github.com/vllm-project/vllm/pull/69444) and the XPU plugin opt-in. Keep the model, XPU runtime, FP8 attention KV, CUTLASS, hybrid prefill, 327,680-token limit, 2,048-token scheduler budget, and DSpark k=7 probabilistic drafting fixed. The [official XPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/#intel-xpu) separates the XPU kernel wheel from vLLM source; reuse the existing Torch/Triton/XPU kernel dependencies in a copied environment so the comparison isolates Python source changes.

On `b70-dev`, stage the three source trees and plugin under `~/inkling-vllm-xpu/sparse-69444` as specified by `maintenance.sh`; stage `parity.py` and the [existing sweep helper](../20261125-inkling2-concurrency/sweep.py) there too. Run `timeout 2400 bash ~/inkling-vllm-xpu/sparse-69444/maintenance.sh`. Save the live launcher, verify it is idle, and start only one test engine at a time on `127.0.0.1:18080`. The completion API checks matching fixtures around the chunk boundary, cold and warm prefixes including extensions, greedy token IDs/logprobs, and actual DSpark accepts/rejects. The short C1/C32 trials check gross throughput changes; C64 checks startup, the standard short workload, a near-limit request and mixed traffic. The script retains exact launch commands, package pins, server logs and raw API/benchmark results in its printed guest run directory and brings the original service back on exit.

The staged plugin must include the same built `_inkling_nvfp4_xe2*.so` and installed plugin entry-point metadata as production. Source archives alone do not include them. If a later arm fails, set `OLD_BASELINE` to the verified `old-c32` artifact directory from the previous run; this reuses only the completed production-reference arm.
All arms used the same production CUTLASS extension with the same Torch/XPU dependencies; the extension was not rebuilt against the newer vLLM source.
If all three C32 arms passed, set `VERIFIED_C32_RUN` to their parent run directory to resume at C64. The C64 short trial gets 300 s rather than the existing sweep's 100 s timeout because it issues 64 warmups before its 256 measured requests.

## C32 results

| Variant | Available KV memory | Effective KV tokens | C1 output tok/s | C32 output tok/s |
|---|---:|---:|---:|---:|
| Production reference | 8.68 GiB | 353,884 | 59.9 | 349.2 |
| Clean upstream | 8.23 GiB | 335,525 | 59.8 | 352.5 |
| Upstream + sparse conv | 8.23 GiB | 409,352 | 59.9 | 347.2 |

At equal 8.23 GiB, the sparse build raised vLLM's effective 320K cache-capacity estimate by 22.0%. Both paired comparisons matched all 864 greedy token IDs across 18 requests and kept prefix hit lengths. Max/mean absolute selected-token logprob deltas were 0.07413/0.00085 (old → upstream) and 0.12504/0.00090 (upstream → sparse). DSpark accepted 530/2,359, 530/2,352, and 528/2,352 proposed tokens on those fixtures, respectively. There were zero preemptions across the parity and short throughput trials. Throughput entries are single repetitions and do not establish a performance delta.

Guest evidence: `~/inkling-vllm-xpu/sparse-69444/run-20261125-150539.7XFdNr/` (includes the matched production reference from `run-20261125-145945.id4jJh/`). C64 booted in that run with 7.40 GiB and a 367,562-token effective cache, reached 64 active requests, then the client hit its 100 s timeout while still making progress. That trial does not count as a completed benchmark. Production restored and passed authenticated smoke afterward.

## Completed C64 verification

`VERIFIED_C32_RUN=$HOME/inkling-vllm-xpu/sparse-69444/run-20261125-150539.7XFdNr timeout 1500 bash ~/inkling-vllm-xpu/sparse-69444/maintenance.sh`

- The sparse build again booted at 320K with 7.40 GiB available and 367,562 effective KV tokens; the original C64 build did not fit at that memory budget.
- The 512-input/128-output trial completed 256 requests, peaked at 64 running, and had zero preemptions. It delivered 355.8 output tok/s versus 347.2 for the separate sparse C32 trial (+2.5% in one run each), at p95 TTFT of 10.24 s versus 3.99 s. Median TPOT was 138.9 ms versus 60.9 ms.
- A separate 327,424-token prompt plus 128 generated tokens completed in 178.4 s with zero preemptions.
- One 245,760-token prompt with four 8,192-token prompts peaked at five running and had zero preemptions. The short requests still took 114.6–121.4 s to first token. This establishes overlap, not interactive latency under long prefill.

Raw guest evidence: `~/inkling-vllm-xpu/sparse-69444/run-20261125-151425.GaTFl2/` (`sparse-c64/{server.log,launch.txt,c64-r1.json,c64-r1.summary.json,near-limit.json,mixed.json}`, `restored.log`). Production was restored with the unchanged launcher; authenticated smoke passed, unauthenticated `/v1/models` returned 401, and `/health` returned 200.

Local branches (not published): vLLM `local/b70-sparse-69444` at `4932e8596a` on main `0085facc58`; plugin `local/b70-sparse-69444` at `e5f6619` on `27474f21`. The experimental plugin only opts in with `INK2_SPARSE_CONV_HISTORY=1`.
