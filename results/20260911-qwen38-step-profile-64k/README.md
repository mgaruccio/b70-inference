# Qwen 3.8 64K speculative-step profile

**Development campaign.** See [observed-results.md](observed-results.md) for actual
baselines, stage measurements, failures, and the evidence-selected MTP2 experiment.
No publishable benchmark or production promotion is claimed. All servers are
disposable; persistent launchers, model snapshots, and archived patches stay unchanged.

## Question and comparison contract

The question is whether there is a real 64K decode improvement over the starting
MTP4 service, and which speculative step is worth one later, measured optimization.
Run fresh unprofiled baseline cells, then profile the two existing bundles.
Only after attribution select one bounded optimization. Here that experiment is
MTP2 versus MTP4 via `run-mtp-depth.py`, retaining the original context-capacity settings.

| cell | image/runtime | speculation and unchanged patch stack |
| --- | --- | --- |
| `mtp4` | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` (vLLM `ac7509e2b`) | archived MTP nightly, MTP boundary, GDN mixed-split v5, draft LM-head INT4, draft MTP INT4, and prefill guard |
| `dspark` | `vllm/vllm-openai-xpu@sha256:7a558f63b703a2b19020eea66483830dc33becfa2503b83074755bcceb8110d4` (vLLM `73029d424`) | archived prefill runner, corrected DSpark overlay (`patch-dspark-native.py`, SHA256 `0640edc7a72c4b6650bb6846cdc988c87883dad7c0cb86d36684513a1c070643`), and XPU boundary patch |

This is explicitly a **whole-stack BUNDLE comparison**. The images, vLLM
revisions, speculative algorithms, and patch sets differ. A cross-cell delta
must not be attributed to one kernel or one patch, and MTP patches must not be
ported into DSpark.

Both cells use the same target and serving workload:

- model `/home/mike/inference/models/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`, GPTQ Int4 symmetric G128, FP16 compute;
- FP8 KV, no prefix caching, no CPU/KV offload, C1, XPU graphs enabled;
- `--max-model-len 65664`, `--max-num-batched-tokens 2048`, `--max-num-seqs 1`, port `127.0.0.1:8000`, served name `qwen38`;
- `enable_thinking=false`; explicit requests use temperature `0`, seed `42`, `ignore_eos=true`, and exactly 128 output tokens;
- MTP keeps capture sizes `[1,2,4,8]`; DSpark keeps `FULL_DECODE_ONLY` and capture sizes `[7,8]`.

The generated container names are stable and owned by this driver:
`b70-step-profile-mtp4` and `b70-step-profile-dspark`.

## Fresh primary sources and conclusions

The implementation uses native PyTorch CPU+XPU profiling already present in the
pinned images; it installs nothing:

- <https://docs.pytorch.org/docs/2.13/profiler.html>
- <https://docs.pytorch.org/docs/2.13/generated/torch.xpu.Event.html>
- <https://github.com/vllm-project/vllm/blob/73029d424/docs/contributing/profiling.md>
- <https://github.com/vllm-project/vllm/blob/73029d424/vllm/profiler/wrapper.py>

The pinned source was inspected read-only at `/tmp/vllm-pinned-73029d424`.
`XPUWorker` constructs the Torch wrapper with `activities=["CPU", "XPU"]`.
The wrapper's `delay_iterations` counts worker steps after `/start_profile` and
starts on the step whose active count equals the delay; `max_iterations` stops
when the recorded count becomes greater than the limit. Therefore the trace must
be checked for its actual number of steps rather than assuming an exact five.
The profiler's CUDA aggregate table is disabled here. Graph replay can hide inner
kernels, so a graph trace is attribution evidence, not an additive speedup proof.

## Profiling protocol

`--profile` starts the same graph-enabled launcher and adds a bounded Torch
configuration:

```text
profiler=torch
CPU+XPU activities
with_stack=false
record_shapes=false
with_memory=false
with_flops=false
torch_profiler_dump_cuda_time_total=false
ignore_frontend=true
warmup_iterations=0
max_iterations=5 (default)
delay_iterations=0 (default)
torch_profiler_use_gzip=true
```

The driver renders the same tokenizer-verified deterministic prompt shape as the
archived `scripts/experiments/qwen38_long_context_bench.py` (source SHA256
`a01a99b21f36ef446d220df66a4e739a02b3ab18dfe99dec8beb333719276907`). It sends
one real streamed 64K `/v1/completions` request. `/start_profile` is launched
only after the first non-empty SSE output event, so the initial prefill is not
intentionally profiled; `/stop_profile` is sent after a small output-event
window, while the server-side max iteration limit remains the hard bound. The
request still drains and validates the full 128-token response.

A 65,536-token prompt with a 2,048-token scheduler budget has 32 prefill chunks.
If a run needs a delayed start for diagnostic reasons, use an explicitly recorded
value such as `--profile-delay-iterations 35` and inspect the trace. The proposed
`delay_iterations=2` setting is rejected by the driver because it is the known
prefill-contaminated proposal. Regardless of configuration, reject any trace that
contains prompt/context work in the ranked sample. Rank native **Self XPU**
time and do not add a wrapper and its child kernel together. No CUDA aggregate
table is produced.

Traces are written beneath the run's remote `profile/` directory. They can be
large and are intentionally host-local; do not copy them into Git merely to make
a summary. No optional trace-summary infrastructure is included.

## Defined end-to-end journey

Run on the isolated inference host, outside the interactive Pi runtime. The
following preconditions are mandatory for **each** cell:

1. both immutable images, the target model, the frozen DSpark draft, and the
   archived patch paths below are already present;
2. `/dev/dri/renderD128` is available and the host has no running containers;
3. `glimmer-tb21-prefix-c8` is stopped; the power-cap file reads `275000000`;
4. `/home/mike/inference/launchers/start-qwen38.sh` has SHA256
   `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`;
5. every output directory named below is new. The driver refuses an existing
   owned name and does not remove a pre-existing stopped container.

The default archived remote paths are taken directly from the source launch
records:

```text
MTP4 root:       /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard/mtp4-long
DSpark root:     /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260910-dspark-v2-feasibility
DSpark campaign: /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-dspark-layer-norm
DSpark draft:    .../20260910-dspark-v2-feasibility/draft
```
The MTP4 source launcher is `results/20260909-qwen38-dflash2-rtn-standard/mtp4-long/launcher.sh` (its remote paths above are retained). The corrected DSpark source argv is `results/20260911-qwen38-dspark-layer-norm/norm-graph-64k/launch-argv.json`; its corrected remote overlay path is retained rather than replaced.

Use the archived unchanged client (or pass an unchanged copy with
`--long-client`):

```bash
R=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260911-qwen38-step-profile-64k
LONG_CLIENT=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260909-standard/mtp4-long/qwen38_long_context_bench.py
```

Run the cells sequentially, in this order. Do not run two cells concurrently;
port 8000 and the one-card host are intentionally exclusive.

```bash
# 1. Profile the starting MTP4 bundle.
python3 -u "$R/run-step-profile.py" \
  --cell mtp4 --profile --long-client "$LONG_CLIENT" \
  --out "$R/mtp4-profile"

# 2. Profile corrected DSpark with its own unchanged stack.
python3 -u "$R/run-step-profile.py" \
  --cell dspark --profile --long-client "$LONG_CLIENT" \
  --out "$R/dspark-profile"

# 3. Unprofiled MTP4: one warmup plus six measured 64K trials.
python3 -u "$R/run-step-profile.py" \
  --cell mtp4 --long-client "$LONG_CLIENT" \
  --out "$R/mtp4-throughput"

# 4. Unprofiled corrected DSpark: the same deterministic prompt/trial contract.
python3 -u "$R/run-step-profile.py" \
  --cell dspark --long-client "$LONG_CLIENT" \
  --out "$R/dspark-throughput"
```

The driver uses `--pull=never`, waits up to 1,800 seconds for startup, and uses
long bounded request/control timeouts. A normal benchmark delegates only the
long-context HTTP workload to the unchanged client with:

```text
--lengths 65536 --near-limit 65536 --confirm-prefix-cache-disabled
```

That client renders the prompt through `/tokenize`, preserves raw requests,
SSE, metrics, and rendered prompt IDs, and performs one warmup plus six measured
trials at 128 forced output tokens. The profile mode performs one separate
64K generation request, not the six-trial benchmark.

## Expected evidence and validation

A successful cell ends with `summary.json` containing `status: "passed"` and
`host_unchanged: true`. The driver fails closed unless it can validate:

- served model `qwen38` and `max_model_len=65664`;
- exact prompt/completion usage counts (`65536`/`128`), length finish, complete
  SSE, no parse/server errors, and no dropped measured trials;
- finite Prometheus metrics and non-decreasing speculative counters for drafts,
  draft tokens, and accepted tokens;
- profile start/stop responses, a first-output activation boundary, and at least
  one flushed trace file for profile cells;
- before/after launcher hash, power cap, running-container list, and Glimmer
  state, with removal attempted only for the observed image-matching owned name.

Each output directory retains, as applicable:

```text
asset-manifest.json, driver-args.json
launch-argv.json, launch-metadata.json, launcher.sh
image-inspect.*, container-observed.json, container-inspect.*
runtime-identity.json, collect-env.*
server.log, startup-attempts.json, models.*, model-selection.json, server-info.*
host-before.json, host-after.json, container-cleanup.json, failure.txt
long-client-argv.json, long-client-command.txt, long-client-output.txt
long-context/                          # benchmark raw prompt/request/SSE/metrics
rendering/, profile-request.*, profile-sse.*
profile-start.*, profile-stop.*, profile-boundary.json
metrics-before.*, metrics-after.*, profile-artifacts.json
profile/                                # host-local compressed Torch traces
```

Failures are evidence: `failure.txt`, partial streams, server logs, cleanup
records, and host snapshots are retained. Cleanup runs from `finally` on normal
errors and SIGINT/SIGTERM, and only calls `docker rm -f` after this invocation
observed the expected image under the owned stable name. It never stops Glimmer,
removes another container, deletes a model, or mutates an archived source.

These are development results and do not satisfy the repository's publishable
benchmark checklist: no BetterBench, `vllm bench serve`, quality suite, or
community claim is authorized here. Profiled throughput is diagnostic only; use
the separate unprofiled six-trial summaries for any later development
comparison, and report the MTP4-vs-DSpark result as a bundle.

## Validation performed in this checkout

No GPU, Docker server, remote command, install, push, or delegation was used
while preparing these assets. CPU-only checks were:

```bash
python3 -m py_compile results/20260911-qwen38-step-profile-64k/run-step-profile.py
python3 results/20260911-qwen38-step-profile-64k/run-step-profile.py --help
```

Both completed successfully. The requested preview/show-patch tool was not
available in this worker environment; the previously attempted preview failed
with `ENOENT /tmp/prime-lab-1000/2c31a91a094d7226455d297a/lab.sock`.
