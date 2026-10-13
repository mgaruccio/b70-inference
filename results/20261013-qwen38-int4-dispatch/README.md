# Qwen native INT4 shape/dispatch diagnostic

**Development diagnostic only. No candidate, latency comparison, speedup claim, or production promotion.**

## Question and unchanged baseline

The [previous native profiles](../20261012-qwen38-prefill-profiling/execution.md) establish that INT4 GEMMs dominate prefill but do not retain their exact dimensions or prove oneDNN dispatch selection. Capture those facts before choosing a replacement/tuning candidate; do not repeat a chunk-budget sweep.

Host: `inference-host`, pinned image `f01e24f6` (full digest in the raw artifacts). Retain C1, context 212992, utilization .95, guarded budget8192, FP16 activations, FP8 KV, prefix cache/Mamba align, native MTP4 with INT4 S+M1, mixed GDNv5, K64, XPU graphs, and 275W. The original launcher must remain SHA256 `5701b40e355e99b65c776dd6c9e499980ccc3be758f17288c30f840f86e99a1d`.

The only compute-path instrumentation changes are iteration logging, Torch input-shape capture, and `ONEDNN_VERBOSE=profile_create,dispatch` with timestamps. The temporary diagnostic binds localhost; the restored service uses its original bind. No persistent settings or networking changes are made.

## Research and compatibility

- [oneDNN 3.12 verbose modes](https://uxlfoundation.github.io/oneDNN/v3.12/dev_guide_verbose.html): creation timings and dispatch diagnostics are not device execution timings. Do not use `profile_exec`, `profile`, or `all`: GPU execution verbosity introduces synchronization incompatible with the previously observed command-graph recording path.
- [PyTorch profiler implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/profiler/profiler.py): input-shape tracing is diagnostic instrumentation with overhead/reference-retention caveats, not a latency benchmark.
- [vLLM profiling guide](https://docs.vllm.ai/en/v0.27.0/contributing/profiling/): start/stop controls require profiler configuration at startup. The current production API had no profiling routes. Installed `config/profiler.py`, `profiler/wrapper.py`, and `v1/worker/xpu_worker.py` were inspected; the installed code, not an assumed upstream source match, governs supported options.

A tiny **synthetic** native INT4 probe (M32/K128/N128) first exercises eager execution, graph capture and graph replay with the proposed instrumentation. It must retain exact native input dimensions and real XPU events, and require finite/equal eager/replayed output. This is a compatibility test, not real-weight or serving-performance evidence. Production is not stopped for this probe.

## Defined real API test and cleanup

1. Require the compatibility pass, only `qwen38` running, pinned image, unchanged launcher/K64/guard hashes, same boot, boost off, all16 CPU caps3801000kHz, 275W, and clean focused kernel health. Retain installed profiler/native-op source.
2. Stop only Qwen. Start a disposable copy of the unchanged compute configuration with the instrumentation above; await `/v1/models` advertising `qwen38` and context212992. Verify native `/start_profile` and `/stop_profile` routes.
3. Warm one exact32768-token shape with a unique first token, greedy output1, ignored EOS, finite logprob, exact usage, and no reported prefix hits. Exclude this request from profiling.
4. POST `/start_profile`; submit **one** distinct-prefix exact32768-token request through `/v1/completions`; require32768+1 usage and finite logprob; POST `/stop_profile`. Keep stack/memory/FLOP recording disabled. Retain request, responses, metrics, iteration ledger, trace and full server log.
5. Require the iteration context-token sum32768 and actual XPU kernels. Correlate kernel external/runtime IDs to native operator dimensions. CPU scopes and oneDNN creation time are never counted as GPU execution. Report missing/unmapped events explicitly; do not invent internal tile geometry.
6. In `finally`, best-effort stop/wait only the owned diagnostic; cleanup failures cannot skip a fresh container-state check. A surviving/unknown GPU owner blocks another launch. If no container remains, restart the original immutable launcher. Verify a finite5+1 response, unchanged host invariants and clean kernel health.
7. Lead verifies the restored externally reachable API independently. Retain raw evidence on the host and here; leave original production running. Disposable containers use `--rm`; retained logs/traces are intentional result artifacts.

The controller is protected from SSH disconnect by `nohup` and bounded startup/API timeouts. A transport failure is not diagnostic completion and requires inspecting the existing host execution rather than launching a second copy.

## Exact commands

From this directory on the desktop:

```bash
bash run-compatibility.sh
python3 -B test-controller.py
python3 -B test-shapes.py
bash run-host.sh
```

The latter executes on the host:

```bash
python3 -Bu /home/mike/b70-evals/20261013-qwen38-int4-dispatch/run-diagnostic.py --output /home/mike/b70-evals/20261013-qwen38-int4-dispatch/native-01
python3 -B /home/mike/b70-evals/20261013-qwen38-int4-dispatch/analyze-shapes.py /home/mike/b70-evals/20261013-qwen38-int4-dispatch/native-01
```

## Execution status

- Synthetic graph compatibility: **PASS**; Torch2.13.0+xpu, oneDNN3.12.0, native operator dimensions retained, four actual kernel events, finite/equal output. Raw probe artifacts: `compatibility/`.
- Local targeted checks: five controller lifecycle/unchanged-launcher checks and three shape/correlation/dispatch checks pass. A cleanup-timeout restoration hole was reproduced and fixed **before** stopping production.
- Full 32K capture/restoration: **PASS**. See [execution.md](execution.md) for exact dimensions, correlated activity shares, dispatch limitations and independent public API verification.
