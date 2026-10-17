# Bolt prefill investigation

Development-only results. The user's focus moved from decode to prefill; no persistent launcher, model weights, precision, driver, network, power, or clock settings changed.

## Baseline and test process

Baseline: validated `llama.cpp-bolt-b6334e37-nvfp4-router-r1` (R1 + MoE reduction + router fusion), upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`, original GGUF, q8 KV, F32 recurrent state, context65536, batch/microbatch1024, single sequence, ESIMD1/FUSION1/graphs0. Host: `mike@inference-host`; image `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04`; oneDNN install unchanged. Exact server environment/mounts/argv and hardware state are retained in the run directories.

1. Against the already validated, resident, non-debug server, run the unchanged gate-v2 API suite; require20/20 before timing.
2. Reuse `probe-api.py` short and long modes:55/8270 rendered prompt tokens, cache disabled and verified `cache_n=0`, greedy seed42,128 forced outputs, one warm-up plus three measured repetitions. Exact8238-token `BOLT-7319-CEDAR` retrieval remains mandatory.
3. Report server prompt-processing time/throughput separately from streaming client TTFT and decode throughput.
4. Profile the same8270-token prompt with **one** output token. Start unitrace paused, warm the entire request, resume collection for one uncached request, then stop collection. This output-budget difference is intentional for attribution; its timings are **not** performance-comparison results.
5. Stop only the experiment-owned instrumented container; restart the original validated container and require a real API `READY` response. Verify unchanged hardware controls. Preserve logs and failures.

Public boundary: `http://127.0.0.1:18098/v1/chat/completions` on inference-host. Scripts: `prefill-baseline.sh`, `build-prefill-unitrace.sh`, `run-prefill-profile.py`. Run these **on inference-host**, using Bash. The Python profiler driver is bounded by `timeout --signal=TERM --kill-after=240s 1800s`; its `finally` block restores the original Bolt container. No unrelated model is started.

## Fresh uninstrumented baseline

`prefill-router-r1-baseline`:20/20 gate-v2 passes, exact8238-token retrieval passes, hardware-before/after identical. Native correctness uses the already accepted316-case baseline; the baseline script does not reclassify an unvalidated binary.

| Metric | Short55 | Long8270 |
|---|---:|---:|
| Median server prefill |469.046ms|8583.184ms|
| Median server prefill throughput |117.2593tok/s|963.5119tok/s|
| Median client TTFT |0.533152s|8.669923s|
| Median client decode |40.4950tok/s|39.1263tok/s|

Long prompt-processing repetitions:8578.174,8583.184,8609.551ms. No outliers discarded. Three samples are development evidence, not cross-run confidence intervals.

## Kernel attribution

Intel PTI tag `pti-1.0.0`, commit `887bba6e28ce84cc0d3813ef876e24add107c318`, reports unitrace2.3.0. Built in an isolated host experiment directory with six jobs, no system installation; Level Zero enabled, OpenCL/ITT/XPTI/MPI disabled. The tool's embedded commit string is empty because its build helper ran outside the Git directory; `pti-commit.txt` records the actual source commit. Build output and all warnings are retained.

`prefill-router-r1-unitrace` passes request checks (8270 input,1 output,cache_n0), restoration API smoke, and hardware equality. The instrumented request reports9318.432ms server prefill; do **not** compare this with8583.184ms as an optimization regression.

| Device kernel family | Calls | Summed device time | Share of summed device activity |
|---|---:|---:|---:|
| Grouped IQ2_XXS dequant/GEMM |808|1887.363ms|21.98%|
| F32 SSM scan |440|1497.786ms|17.44%|
| `gemm_kernel` |5284|1199.329ms|13.97%|
| NVFP4→F32 dequantization |720|784.817ms|9.14%|
| IQ2_XXS→F16 dequantization |3196|617.841ms|7.19%|

53,364 device events sum to8.587583s; their interval union is8.389287s over a9.313412s device span. Gaps total0.924125s; largest12.53175ms. Summed durations include overlap. These timings do not establish bandwidth saturation, occupancy, or the cause of gaps.

Host `zeCommandListAppendMemoryCopy` time is7.312935s. It includes synchronization/waiting for earlier device work: it is **not**7.3s of PCIe transfer, and must not be added to the device times. Generic `gemm_kernel` does not uniquely identify a model layer.

Raw attribution: `unitrace.1.txt.gz`, `llama-server.1.json.gz`, API payload/events/results, container metadata, and `analysis.json` in the profile directory. The Chrome trace originally landed at container `/llama-server.1.json`; it was copied from the stopped owned container and losslessly compressed with `gzip -n`. Reproduce the summary with:

```bash
python3 -B results/20261016-nemotron35-bolt-sycl/analyze-prefill-profile.py
```

## First candidate selected

Staged **prefill-only SSM transforms** on top of the router+MoE baseline. The scan is not the largest individual kernel, but it is a measured17.4% cost with a narrowly testable exactness-preserving opportunity: move redundant per-head softplus/exponential work out of the serial token recurrence. Keep the state update order, subgroup reduction, F32 arithmetic, and single-token decode path unchanged. No parallel-prefix reassociation or reduced-precision substitution.

The first candidate is now validated as a development result: **385 native checks**, real-model activation including1024-token batches, fresh **20/20** gate-v2 before timing, matched55/8270 benchmarks and exact8238-token retrieval. Initial dispatch failure is preserved; its translation-unit VMM guard was corrected without changing numerical tests. A fresh confirmation pair reduces long prefill **8641.700→8232.165ms (−4.74%)**, throughput **956.988→1004.596tok/s (+4.97%)**, and TTFT **8.729244→8.318865s (−4.70%)**. Short server prefill is neutral; short TTFT remains variable/slower, with logged RAM prompt-cache bookkeeping explaining the pre-evaluation delay. Full-output identity is not established, and same-build launch controls also vary. See [implementation, strict gates, all repetitions and caveats](../20261016-nemotron35-bolt-prefill-ssm/README.md). The candidate remains resident without changing persistent launchers. Grouped expert GEMM and NVFP4 dequantization remain measured opportunities for later work.

## Fresh primary-source research

- [Pinned SSM source](https://github.com/ggml-org/llama.cpp/blob/b6334e37e6ef286fc5091a41b6561b2113d915fb/ggml/src/ggml-sycl/ssm_scan.cpp): serial `n_tok` recurrence and repeated F32 transforms. Source alone was a hypothesis; the real-model profile above establishes measured cost.
- [Intel unitrace documentation](https://github.com/intel/pti-gpu/blob/pti-1.0.0/tools/unitrace/README.md): device timing, kernel submission and Chrome timeline collection; named-session pause/resume/stop excludes model loading and warm-up without modifying the model binary. No hardware-metric/sysctl changes used.
- [SYCL MoE prefill PR23142](https://github.com/ggml-org/llama.cpp/pull/23142): counting-sort/contiguous mappings can help other models. Inspection confirms this baseline already has `mmid_counting_sort_rows` and grouped dequant GEMM; no blind backport and no transfer of reported B60 gains to Bolt/B70.
- [NVFP4 support PR21227](https://github.com/ggml-org/llama.cpp/pull/21227): correctness support is not evidence of efficient batched prefill.
- [Intel VTune GPU Compute/Media Hotspots](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-1/gpu-compute-media-hotspots-analysis.html): potential deeper follow-up, not run here.
