# Qwen 3.8 B70 GDN-locality Phase0

**Development harness only.** This directory queues graph-aware attribution; it
contains no GDN kernel candidate and makes no speed claim. The lead owns all
inference-host execution. Nothing here launches Docker or touches the remote
host during CPU-only validation.

**Execution paused:** the native baseline passed, but graph attribution remains
inconclusive. The second tracer startup coincided with a fatal AMD CPU execution-unit
MCA/data-fabric reset signature, also recorded before this experiment. The trigger
is unresolved. Do not repeat GPU launches until the host-safety blocker is
resolved. See [`execution.md`](execution.md) for results and partial artifacts.

## Scope and hard boundary

The accepted question is whether Lithos-metal's locality ideas justify a narrow
Intel-native conv-to-GDN workgroup fusion for Qwen3.8 MTP4 on the Arc Pro B70.
The native five-row MTP4 kernel already retains the recurrent state and
convolution window on chip; only the conv-to-rule intermediate and one dispatch
remain candidates. Rollback checkpoints and the existing XPU ABI must remain.

The only candidate gate is **at least 5% of complete, reconciled target
graph-replay device time**. Eager-only timing, a partial graph trace, a missing
trace, an inferred eager-to-graph ratio, or a visible draft-only trace can
neither select nor reject fusion. Such evidence is reported as
`inconclusive`. No whole-engine port, production promotion, model/precision/
acceptance change, Muse/DSpark expansion, or new persistent service/store is in
scope.

## Pinned serving contract

- one idle `inference-host` Intel Arc Pro B70 at `275000000` microwatts (275 W);
- no running containers, stopped Glimmer, unchanged production launcher SHA256
  `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`;
- image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`;
- native MTP K4, GPTQ symmetric G128 target, FP16 compute, FP8 KV, C1,
  `max-model-len=212992`, `max-num-batched-tokens=8192`, utilization `0.95`,
  graph capture sizes `[1,2,4,8]`, prefix cache off, thinking off;
- existing `/tokenize` renderer and streaming `/v1/completions` journey,
  greedy temperature 0, seed 42, EOS ignored, 128 forced output tokens.

The normal uninstrumented baseline was already queued by the lead through the
unchanged four-way runner. Do **not** repeat it from this harness. Its fresh
output is the lead-owned sibling:

```text
/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality/baseline-original-01
```

## Stage, build, and run

All commands below are issued by the lead, serially, on `inference-host`. The
staging directory is intentionally separate from the results directory so the
existing `20260911-qwen38-step-profile-64k/run-step-profile.py` sibling lookup
continues to work. Staging creates no runtime container.

```bash
CAMPAIGN=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality
HARNESS=/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality-harness

python3 results/20261009-qwen38-gdn-locality/run-phase0.py \
  --stage-only --remote-host inference-host --remote-dir "$HARNESS"

ssh inference-host bash "$HARNESS/build-unitrace.sh"
```

`build-unitrace.sh` clones Intel PTI source at exact commit
`887bba6e28ce84cc0d3813ef876e24add107c318`, then builds it in the pinned vLLM
image with no GPU device, no package/driver mutation, bounded CPU/memory/time,
and a read-only cached compiler mount. It writes only fresh
`unitrace-src/`, `unitrace-build/`, and `unitrace-install/` children under the
harness directory. The exact source's supported CMake switches disable MPI,
XPTI, and OpenCL; it has no `BUILD_WITH_OMP` or `BUILD_WITH_PERFETTO` switch,
so invented flags are not passed. Level Zero remains mandatory.

After the build succeeds and the host is idle again, run the normal graph-enabled
serving process under unitrace. The remote wrapper supplies the install root by
default; the explicit form is shown for auditability:

```bash
ssh inference-host python3 -u "$HARNESS/run-phase0.py" \
  --mode unitrace-profile \
  --out "$CAMPAIGN/diagnostic-unitrace-01" \
  --unitrace-install "$HARNESS/unitrace-install"
```

The disposable container launches:

```text
unitrace --start-paused --device-timing --chrome-kernel-logging \
  --chrome-device-logging --session b70gdnlocality \
  --output-dir-path /output/unitrace vllm serve ...
```

The original vLLM command remains unchanged beneath the wrapper. The worker's
normal `/start_profile` hook resumes the unitrace session after the first
non-empty streamed output; `/stop_profile` pauses it after the bounded profile
window. Existing deferred XPU event pairs time complete graph replays. No
unsupported in-graph PyTorch event markers are added. If the binary cannot
start, session control fails, no trace is emitted, or graph records are partial,
the run is a blocker/inconclusive result and no candidate phase is authorized.

## Trace review and decision gate

The unitrace artifact is checked after the owned container exits:

```bash
ssh inference-host python3 -u "$HARNESS/summarize-gdn.py" \
  "$CAMPAIGN/diagnostic-unitrace-01/unitrace" \
  --source unitrace \
  --out "$CAMPAIGN/diagnostic-unitrace-01/unitrace-summary.json"
```

The report remains inconclusive: coverage/reconciliation CLI values are caller
assertions, not independent proof. The following optional fields retain review
notes but cannot authorize fusion or establish a measured GDN share:

```bash
ssh inference-host python3 -u "$HARNESS/summarize-gdn.py" \
  "$CAMPAIGN/diagnostic-unitrace-01/unitrace" \
  --source unitrace \
  --graph-coverage complete \
  --whole-replay-ms <measured-target-replay-ms> \
  --replay-reconciled \
  --out "$CAMPAIGN/diagnostic-unitrace-01/unitrace-summary-reconciled.json"
```

`fusion_decision.fusion_selection_allowed` remains false even with those flags.
The lead must separately inspect actual graph coverage, target provenance,
window ownership and whole-replay alignment before considering the >=5% gate.
Unknown hooks remain unknown. Eager `--mode eager-profile` identifies operations
only; neither eager timing nor a visible-kernel sum selects or rejects fusion.

The older graph torch profile remains available for visibility diagnostics:

```bash
ssh inference-host python3 -u "$HARNESS/run-phase0.py" \
  --mode profile --out "$CAMPAIGN/diagnostic-graph-01"
```

Its summary is diagnostic only. The separate eager profile is likewise diagnostic
only and must not be used as a throughput comparison.

## CPU-only checks

```bash
python3 -m py_compile \
  results/20261009-qwen38-gdn-locality/{run-phase0.py,gdn-annotations.py,gdn-patch.py,summarize-gdn.py,test-phase0.py}
python3 results/20261009-qwen38-gdn-locality/test-phase0.py
```

The fixture verifies explicit target-root provenance even when a target module
contains `mtp`, recognizes `aten::mm`, rejects eager/partial fusion decisions,
and validates the pinned unitrace build contract. It imports no torch/vLLM and
uses no GPU.

## Fresh research record

- <https://raw.githubusercontent.com/lithos-ai/lithos-metal/main/docs/design/mixers.md> — locality/task DAG ideas are useful, but Lithos's fixed row recipes do not transfer unchanged to five-row MTP4.
- <https://docs.pytorch.org/docs/2.13/profiler.html> — CPU scopes and device timing are distinct; graph replay can hide inner work.
- <https://github.com/pytorch/pytorch/blob/v2.13.0/c10/xpu/XPUEvent.h> — XPU events provide boundary timing; they are not supported per-node graph markers.
- <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc> — graph profiling is aggregate and can alter optimization behavior.
- <https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md> — unitrace documents `--start-paused`, session `--resume`/`--pause`, device timing, and Chrome kernel/device logging; the build pins the exact source commit separately.
- <https://intel.github.io/pti-gpu/whatsnew.html> — SYCL graph tracing is evolving/initial proof-of-concept; exact-stack coverage must be validated.

No result in this directory is publishable or a production recommendation until
Benchmarking Standards' paired comparison, long-context, serving, quality,
and environment requirements are separately completed.
