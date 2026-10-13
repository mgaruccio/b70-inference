# Native INT4 prefill: exact shapes and dispatch

**The approved bounded diagnostic passed. Production restored unchanged. No performance candidate or speedup is claimed.**

## What is now established

The exact32768-token profile executes **6656 ×4, 3328, 2816**, matching the previous cache-on/align baseline. All **1590 native INT4 operator calls** have input dimensions. The trace contains **9888 actual XPU kernels**, with **zero** unmapped generic GEMMs or native kernels missing dimensions. There are19 distinct `(M,K,N)` geometries.

Correlated INT4 device activity totals **11552.280ms**. The largest two `(K,N)` projection geometries account for **69.90%** of it across their actual M values:

| K | N | Operator calls | Correlated device ms | Share of INT4 device activity |
|---:|---:|---:|---:|---:|
| 5120 | 34816 | 390 | 5344.187 | 46.26% |
| 17408 | 5120 | 390 | 2731.272 | 23.64% |
| 5120 | 16384 | 288 | 1871.110 | 16.20% |
| 6144 | 5120 | 390 | 960.222 | 8.31% |
| 5120 | 14336 | 102 | 594.560 | 5.15% |
| 5120 | 248320 | 24 | 26.674 | 0.23% |
| 10240 | 5120 | 6 | 24.255 | 0.21% |

These are **GPU-event durations correlated to native dispatcher input shapes**, not CPU scope timings, wall latency, or internal tile dimensions. Device sums can overlap. Dimensions alone do not prove layer ownership. Detailed input metadata and M-specific counts/timings are in `dispatch-analysis.json`.

## oneDNN dispatch evidence and limits

The full server log contains111 native u4 creation/dispatch records:

- 74 selected GEMM/matmul creation records report **`jit:gemm:any`**.
- 37 dispatch records reject/skip **`jit:xe_hp:gemm:any`**: 30 report `unsupported datatype combination`, 7 report `skipping or dispatching to another implementation` (M1).
- Actual prefill geometries at M6656/3328/2816 have both selection records and matching trace dimensions.

The [source at the runtime-reported oneDNN commit80afa710](https://github.com/uxlfoundation/oneDNN/blob/80afa71049cd69a3df32adcccb623b12cd7baa22/src/gpu/intel/gemm/jit_xe_hp_systolic.cpp) checks equal f16/bf16 A/B types or8-bit integer A/B for that older systolic implementation. Our f16/u4 contract does not satisfy that check. This explains the reported rejection; it does **not** prove the selected `jit:gemm:any` backend lacks XMX/DPAS, reveal its tiles, or establish a dispatch bug.

There are **zero u4 creation records inside the profiled window**. The records above span startup/warmup; they are selection/creation evidence, not per-execution timing or a new dispatch on each kernel. No oneDNN execution synchronization, unitrace, graph disabling, FP8 query MMA, GDN retuning, K64 widening, or new chunk budgets were used.

## Real API execution and restoration

Host boot remained `1dce0407-8f83-4873-a04e-8f038212d37e`. Installed stack: Torch2.13.0+xpu, vLLM0.27.2rc1.dev77+gac7509e2b.xpu, vllm-xpu-kernels0.1.12.3, oneDNN3.12.0 commit80afa710, Intel runtime driver1.15.39122. The pinned image and compute baseline in [README.md](README.md) remained unchanged.

Commands:

```bash
bash run-compatibility.sh
bash run-host.sh
python3 -B analyze-shapes.py native-01 > dispatch-analysis.json
python3 -B verify-public.py
python3 -B test-controller.py
python3 -B test-shapes.py
```

- Tiny synthetic graph precheck: finite/equal eager/captured/replayed output, exact native dimensions, four XPU kernel events. This is not real-model performance evidence.
- Warmup and single profiled API requests: HTTP200, exact32768+1 usage, finite logprobs, no reported prefix hits. Diagnostic elapsed times were78.260s (cold warmup) and20.074s (profiled); these are **not** baseline/candidate measurements.
- Original production was restored automatically; its first cold5+1 response took **60.923s**. Do not conceal that cold initialization cost.
- Same boot, boost off, all16 CPU caps3801000kHz, power275W, and unchanged launcher/K64/guard hashes. Focused kernel checks found no faults. Only `qwen38` remained running.
- Independent public `/health` and `/v1/models`: HTTP200, qwen38/context212992. The real text canary `The sum of two and three is` returned ` five`, exact7+1 usage and finite logprob in **0.084s** (`public-restored.json`). This tiny check is not a throughput benchmark or quality evaluation.
- Five lifecycle tests and three shape/correlation/dispatch tests passed (`tests.log`). The restoration timeout hole found in review was reproduced and fixed before the approved production stop; stop/wait failures cannot skip a fresh state check. Unknown/surviving containers still block overlap.

## Artifacts and next target

`native-01/` retains the exact launchers/controller/helper, installed profiler/native wrapper source, requests/responses, metrics, scheduler ledger, full logs, host checks and3.2MB compressed native trace. `compatibility/` retains the graph precheck. `shapes.json` is the original host analysis; `dispatch-analysis.json` adds u4 dispatch parsing and grouped geometry shares. Artifacts were retrieved over ordinary SSH/SCP; no network settings were changed.

The strongest measured target is now the **native JIT path for K5120/N34816 and K17408/N5120 at actual M6656/3328/2816**, not another budget sweep or a blind forced older-systolic path. Next inspect the selected native JIT strategy for these exact contracts and propose one supported change before a paired real-weight/API comparison. No supported tile knob or achievable gain is established here, and the previously slower Triton/dequant prototypes are not rerun.
