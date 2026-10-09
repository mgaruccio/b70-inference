# Qwen 3.8 B70 GDN-locality Phase0

**Development harness only.** This directory queues and measures attribution; it
contains no GDN kernel candidate and makes no speed claim. The lead owns all
inference-host execution. Nothing here launches Docker or touches the remote
host during CPU-only validation.

## Scope

Phase0 is deliberately finite:

1. run the pinned native MTP4 K4 golden configuration uninstrumented at 512 and
   65,536 input tokens (one warmup plus six measured requests per point);
2. run one separate 65,536-token graph-enabled native profile after the baseline;
3. classify the native trace as far as the graph boundary permits; and
4. run the explicitly labelled eager diagnostic only if graph replay hides the
   GDN operator needed for attribution.

The full 512/8K/32K/64K three-arm campaign is **not** run by this harness. It is
queued only after a candidate is warranted and parity/correctness gates pass.
There is no whole-engine port, production promotion, model/dtype/acceptance
change, or Muse/DSpark work.

## Pinned contract

- one idle `inference-host` B70 at `275000000` microwatts (275 W);
- no running containers, stopped Glimmer, and unchanged production launcher
  SHA256 `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`;
- image digest `f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM `ac7509e2b`);
- native MTP K4, GPTQ symmetric G128 target, FP16 compute, FP8 KV, C1,
  `max-model-len=212992`, `max-num-batched-tokens=8192`, utilization `0.95`,
  graph capture sizes `[1,2,4,8]`, prefix cache off, thinking off;
- the existing `/tokenize` and streaming `/v1/completions` journey, greedy
  seed 42, temperature 0, EOS ignored, 128 forced output tokens;
- all archived model/patch/runtime mounts and campaign source mounts are
  read-only; only the new `/output` mount is writable.

`run-step-profile.py` remains the lifecycle owner: it performs host ownership
checks, the real HTTP gates, metrics/counter capture, runtime identity and
cleanup of only its observed container. The wrapper refuses an existing output
directory and assigns one mode-specific container name.

## Remote staging and execution

Run these from the repository checkout. `--stage-only` requires that the remote
campaign directory does not already exist and stages only the new scripts plus
the canonical timing overlay/patch. It does not run Docker.

```bash
CAMPAIGN=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality
python3 results/20261009-qwen38-gdn-locality/run-phase0.py \
  --stage-only --remote-host inference-host --remote-dir "$CAMPAIGN"

ssh inference-host python3 -u "$CAMPAIGN/run-phase0.py" \
  --mode baseline --out "$CAMPAIGN/baseline-01"

ssh inference-host python3 -u "$CAMPAIGN/run-phase0.py" \
  --mode profile --out "$CAMPAIGN/diagnostic-graph-01"
```

The staging wrapper can combine the first two steps for a fresh campaign:

```bash
python3 results/20261009-qwen38-gdn-locality/run-phase0.py \
  --stage --remote-host inference-host --remote-dir "$CAMPAIGN" \
  --mode baseline --out-name baseline-01
```

If `diagnostic-graph-01` reports `insufficient_gdn_visibility`, run the separate
explicitly non-baseline diagnostic:

```bash
ssh inference-host python3 -u "$CAMPAIGN/run-phase0.py" \
  --mode eager-profile --out "$CAMPAIGN/diagnostic-eager-01"
```

Summarize a retained native trace without importing vLLM:

```bash
ssh inference-host python3 -u "$CAMPAIGN/summarize-gdn.py" \
  "$CAMPAIGN/diagnostic-graph-01/profile" \
  --out "$CAMPAIGN/diagnostic-graph-01/gdn-summary.json"
```

The exact commands, environment, HTTP payloads, raw SSE, metrics counters,
launch argv, runtime identity, profiler trace and cleanup/host checks remain in
the remote output directory. Do not fabricate or copy results into this
checkout.

## Attribution interpretation

`gdn-annotations.py` is active only after the native `/start_profile` boundary
and uses `torch.profiler.record_function` plus deferred `torch.xpu.Event`
pairs. It records bounded call metadata (including tensor shapes and GDN
metadata fields) in the ordinary server log. `summarize-gdn.py` uses the native
trace's CPU `External id` correlations and operator `Input Dims` to distinguish
GDN, matmul, full attention, draft, and other device work.

Graph replay can hide inner kernels. Graph replay timing is therefore
attribution evidence, never a throughput number. If the GDN operator is not
visible, the eager profile is the required diagnostic fallback; its extra
instrumentation and `--enforce-eager` are never baseline settings.
