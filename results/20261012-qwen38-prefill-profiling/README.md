# Native prefill profiling pilot

Development diagnostic only; no throughput claim or production promotion.


Final result: **both budget changes rejected; guarded production 8192 retained**.
See [execution.md](execution.md) for the completed matched comparison, actual
scheduler explanation, GPU attribution, observed public API checks and limitations.

## Scope and baseline

Capture actual scheduled chunks and executed XPU activity before another budget
comparison. The initial pilot is one cold 16,384-token prefill at budget 8192,
with prefix caching on / Mamba align. Production is temporarily replaced by a
localhost-only disposable copy, then restored unchanged in `finally`.

Keep the pinned `f01e24f6` image, K64 library, maintained prefill guard, native
MTP4 / INT4 S+M1, FP8 KV, C1, context 212992, utilization .95, graphs enabled,
and 275W. Only built-in iteration logging and Torch profiler configuration are
added. Shapes, stacks and memory collection are disabled; there is no unitrace
and no per-kernel synchronization.

## Research affecting the approach

- [Pinned XPU worker](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/worker/xpu_worker.py): native profiling uses CPU and XPU activities.
- [Pinned profiler configuration](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/config/profiler.py): disable stacks/shapes/memory, ignore frontend, capture immediately with a bounded iteration limit.
- [Pinned iteration logging](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/metrics/loggers.py): iteration details report scheduled context/generation tokens. The Prometheus iteration-token histogram is NOT a per-chunk ledger.
- [PyTorch 2.13 profiler](https://docs.pytorch.org/docs/2.13/profiler.html): use XPU device activity, not asynchronous CPU scopes; recording shapes can retain tensor references and increase overhead.

Installed versions/configuration and API wiring must be checked before launch.
Previous fatal MCA/CO risk is not declared resolved by a reboot. Require current
boot identity, boost off, all 16 cached caps 3801000 kHz, unchanged hashes/power,
and a clean focused kernel check. A decoder-enabled boot message is not a fault.

## Defined public-boundary test

1. Verify the above host prerequisites and that only production `qwen38` owns the
   GPU. Save the exact original launcher and its SHA256.
2. Stop only that container. Start the disposable localhost-only copy with the
   same compute configuration plus iteration logging and native Torch profiling.
3. Await `/v1/models` with `qwen38` and max length 212992. Through
   `/v1/completions`, warm the exact 16384-token shape using a distinct first
   token; allow cold compilation/one-time INT4 draft initialization to finish.
4. Save a server-log byte boundary, call POST `/start_profile`, send one cold
   exact-ID prompt (different first token, otherwise repeated token 15), greedy
   `max_tokens=1`, `ignore_eos=true`, `logprobs=1`; call POST `/stop_profile`.
5. Require HTTP success, exact 16384+1 usage, finite logprob and no reported cache
   hits. Parse actual iteration context tokens from the log window and require
   their sum to be 16384. Retain the unmodified trace/table/log/request response.
6. Inspect actual device events in the trace. CPU launch scopes, primitive
   creation and graph opacity are not kernel-execution evidence. If real GPU
   activity is absent, stop rather than invent attribution or enable unitrace.
7. Always remove only the owned disposable container and restore the saved,
   unchanged production launcher. Verify an external five-input/one-output
   finite-logprob request, unchanged boot/power/hash, and post-run kernel health.

Retain the frozen controller, launchers, raw server logs, metrics, responses,
iteration rows and compressed Torch trace in the run directory on inference-host.
Transfer large artifacts through the existing private R2 workflow if SSH bulk
transfer stalls; do not expose credentials or signed URLs in logs.

Only after the pilot proves device visibility and artifact retrieval should the
matched 8192/8320/6656, cache-on/off, 16K/32K comparison proceed. Unprofiled latency
and profiling overhead must be measured separately. No 9984 deployment, context
reduction or configuration promotion is authorized by alignment arithmetic.


## Matched campaign

After the pilot's GPU visibility and SHA256-verified private R2 readback pass,
`run-campaign.py --output <new-directory>` runs the 12 authorized combinations
of budget 8192/8320/6656, prefix cache on/off, and exact length 16384/32768 in a
seeded randomized cell order (20261012). Each disposable cell uses the same
public-boundary lifecycle above: one excluded shape warmup, five unprofiled
cold distinct-prefix responses, then one separately profiled response. The
original pilot used three unprofiled responses; its frozen controller is kept.
Two final cache-off 8192 cells (16K/32K) anchor baseline drift. Any cell or
restoration failure stops the campaign. Production is restored between cells
and at completion. This remains development evidence: one trace per cell and
five timings do not establish a statistically publishable one-percent effect.

Retain each cell's raw responses, exact scheduled-token ledger, native trace,
full logs, metrics and host checks. Compare actual chunk sequences, device
kernel counts/durations, unprofiled medians/ranges, profiling overhead and
baseline drift separately. Do not call summed GPU activity elapsed latency.


## Completed pilot observations (not the campaign conclusion)

Command: `python3 -u run-pilot.py --output <host-root>/pilot-8192-on-16k-01 --budget 8192 --prefix-cache on --length 16384`.
Host boot: `b46543b7-a459-4c7d-aeb4-7ed41a62ecd7`; installed Torch
`2.13.0+xpu`, vLLM `0.27.2rc1.dev77+gac7509e2b`.

- Actual context ledger: **6656, 6656, 3072**. Native trace annotations independently
  match it. Configured budget 8192 is not the executed chunk shape with Mamba align.
- Three cold unprofiled responses: 8.697573 / 8.722657 / 8.768153 seconds;
  profiled response 8.789807 seconds (+0.77% versus the median, descriptive only).
- 4,944 actual `kernel` events: summed device activity 8.721204 seconds; first-to-last
  device span 8.780327 seconds. Neither is substituted for public response latency.
- Correlation IDs map all 942 generic `gemm_kernel` events to their CPU launch
  operators: 795 INT4 GEMMs, **5.899451 seconds**, versus 147 dense GEMMs,
  **0.033447 seconds**. These are the corresponding GPU event durations, not
  operator CPU durations. K64 attention: 51 calls / 1.572140 seconds; named GDN
  kernels: 1,008 calls / 0.620386 seconds. INT4 GEMMs dominate, not dense GEMMs.
- CPU annotation spans are asynchronous launch scopes. 220 device kernels fall
  outside the non-empty context annotations; their work remains in aggregate
  groups but is explicitly unassigned in per-phase analysis. No tensor shapes
  were recorded, and no claim about individual GEMM geometry is made.
- Every response had exact usage and finite logprob. All host invariants remained
  unchanged and focused kernel checks found no matches. Production was restored;
  an external `http://inference-host:8000/v1/completions` 5+1 probe returned HTTP200
  with finite logprob in 0.725413 seconds (`public-after-pilot.json`). The controller's
  first restored cold probe took 61.283806 seconds; do not conceal that cold-start cost.

Full 27-member archive retrieved through private R2; 1,263,922 bytes, SHA256
`fcbc9fa7dff16f48c5d15ad1e9b49a5ff9b9e47b01b1aabd07c94c56b03c1b2f`.
Extraction is retained in `pilot-8192-on-16k-01/`, including the original frozen
controller. Temporary R2 object and signing/host-transfer files were removed.
`pilot-correlated-analysis.json` is generated by `analyze.py`; its targeted test
checks GPU-only durations, external/correlation ID mapping and asynchronous phase
assignment. The 12-cell comparison plus baseline-after anchors completed; see `execution.md` and `comparison.json` for the final results.
