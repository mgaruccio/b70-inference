# B70 GDN locality program — execution

Development-only. Native baseline completed; graph attribution is blocked after an unexplained host reboot. GPU attempts are stopped. No kernel candidate or production change has been made.

## Native baseline (completed)

```sh
ssh inference-host 'timeout --signal=TERM --kill-after=30s 1800s python3 -u /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260914-qwen38-four-way-speed/run-comparison.py --cell mtp4 --out /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality/baseline-original-01'
```

The unchanged reference runner used native MTP4, the pinned `f01e24f6` XPU image, GPTQ G128/FP16 target, FP8 KV, C1, 212992 capacity, batch8192, 275W, graphs enabled, prefix caching/thinking off, greedy seed42 and 128 forced output tokens. The real public journey used `/tokenize` and streaming `/v1/completions`, with API canaries beforehand.

All 24 measured requests plus four warmups passed; no exclusions. Six measured samples per context, one server load. Median post-first streaming decode `(128-1)/(stream_end-first_nonempty)`:

| Input tokens | Decode tok/s | Inclusive Q1–Q3 | TTFT seconds |
|---:|---:|---:|---:|
|512|70.875|69.711–73.362|0.289|
|8192|63.300|62.306–63.314|4.198|
|32768|62.103|61.307–64.009|21.115|
|65536|55.739|54.158–57.466|53.866|

`baseline-summary.json` includes speculative counter deltas, request latency and the exact command. `baseline-original-01.tar.gz` retains ordinary raw requests/SSE/counters, environment, runtime identity, launch arguments and lifecycle output. Recomputing all medians, inclusive quartiles and speculative counter-derived fields from the six measured rows per context reproduced every summary field, including64K. These are baseline results, not a candidate comparison, device-only timing or publishable benchmark.

Host-after invariants passed. Launcher SHA remained `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; no running container or render-device holder remained.

## Attribution prerequisite attempts

All build attempts are CPU-only, pinned-image builds of public PTI source `887bba6e28ce84cc0d3813ef876e24add107c318`. They never mount the GPU or alter the installed runtime/driver. Staging and build directories are fresh siblings of the existing lifecycle inputs.

1. `ssh inference-host 'bash /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality-harness/build-unitrace.sh'` exited1 before compilation: Intel `vars.sh` referenced unset `SETVARS_CALL` under nounset. Log: `unitrace-build-01.log`.
2. The same command under `.../20261009-qwen38-gdn-locality-harness-retry1/` exited1 after configuration: the offline container could not fetch public compute-runtime/Level Zero headers. Log: `unitrace-build-02.log`.
3. The command under `.../20261009-qwen38-gdn-locality-harness-retry2/` compiled and installed the tracer after enabling its public header downloads. The final EXIT trap still returned1 because Docker's CID file lacks a trailing newline; the read was corrected without rebuilding. Log: `unitrace-build-03.log`. A CPU-only check in the pinned image loaded both binary/library and reported unitrace2.3.0, with Level Zero enabled and ITT/XPTI/OpenCL/MPI disabled. The tool's embedded commit string is blank (upstream generation ran outside a Git working directory); the exact source checkout remains pinned above.

The build script now uses a fresh CID file and EXIT cleanup for only its own container. The CPU fixture was updated to check CPU-only/container ownership/compiler read-only isolation rather than incorrectly require an offline build that PTI cannot complete.

## Real graph-capture attempts

The first `unitrace-profile` run used the installed tracer under the same graph-enabled native MTP4 settings, with `--out .../diagnostic-unitrace-01`. The public request completed65536 input/128 output tokens, both HTTP profile controls returned200, session resume/pause returned0, and host cleanup passed. However, all11 unitrace JSON files were zero bytes. The original artifact checker incorrectly counted filenames as visibility; that historical output is retained unchanged in `diagnostic-unitrace-01.tar.gz` and is **not accepted attribution**.

The first run's whole-replay hook retained64 events and dropped143 at its bound; GDN Python hooks saw no inner target calls, as expected under replay. Neither the visible trace nor those truncated boundary records establishes the GDN share. No throughput conclusion is drawn from the heavily instrumented stream.

The fresh `diagnostic-unitrace-02` attempt used explicit session stop/flush after pause and synchronization, a new session name, a bounded512 replay-event budget, and rejection of empty/invalid JSON or missing successful resume/pause/stop controls. Those changes are CPU-tested but **not validated by a completed GPU capture**. The exact attempted command was:

```sh
ssh inference-host 'timeout --signal=TERM --kill-after=30s 1200s python3 -u /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality-harness-retry2/run-phase0.py --mode unitrace-profile --out /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality/diagnostic-unitrace-02 --unitrace-install /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality-harness-retry2/unitrace-install'
```

SSH exited255 with connection reset/broken pipe. The host rebooted: the prior journal ends2026-10-08 22:08:27 EDT and the next boot starts22:09:30 EDT. The partial directory contains launch/preflight assets but zero-byte server/container-observation/trace files; there is no completed public request, summary or clean postflight. Ordinary partial output is retained in `diagnostic-unitrace-02.tar.gz`, the transport error in `diagnostic-unitrace-02-ssh.log`, and bounded read-only journal/host checks in `host-reboot-02.log`.

Available journal entries do not establish the reboot cause. The pstore directory was permission-denied; no privileged investigation or further launch was attempted. Post-reboot checks found no experiment container (running or stopped), inference/tracer process or render-device holder. The launcher SHA still matches the baseline and the power cap is still275000000 microwatts; neither was changed. A host reboot is not a passed host-unchanged gate. More GPU attempts require resolution of the host-safety blocker; no service is restarted.

The selective review also found that unknown hooks defaulted to target and caller flags could make the summary eligible. Both were corrected: unknown ownership stays unknown and the pure summarizer cannot authorize fusion from caller assertions, even with a complete-looking visible trace.

Local stdlib verification: `python3 -m py_compile results/20261009-qwen38-gdn-locality/{run-phase0.py,gdn-annotations.py,gdn-patch.py,summarize-gdn.py,test-phase0.py}`, `python3 results/20261009-qwen38-gdn-locality/test-phase0.py`, and `bash -n results/20261009-qwen38-gdn-locality/build-unitrace.sh` passed. These supplement the baseline's real HTTP journey; they do not validate graph attribution or the stop/flush change on the GPU.

## Selection boundary

Missing kernel records are not missing work. Eager traces identify operations but cannot establish their share of normal target graph replay. The instrumented graph trace is attribution evidence only, never the uninstrumented performance baseline.

Before any candidate is selected, the lead must establish actual graph coverage, explicit target provenance, capture pause/resume correctness, and reconciliation with complete target-replay timing. CLI review flags do not themselves establish these facts. Unknown/partial attribution remains inconclusive, regardless of the largest visible kernel.

Native source audit already found recurrent state and the convolution window resident across all five MTP4 rows. The remaining conditional candidate is only conv-to-rule intermediate traffic/one dispatch, preserving all rollback checkpoints. Fusion is queued behind credible attribution and the predeclared >=5% complete target-replay gate; no negative or positive hotspot conclusion is currently justified.

The source comparison examined `csrc/xpu/gdn_attn/{causal_conv1d.hpp,gated_delta_rule.hpp,gdn_attn_interface.cpp}` at native commit `1796aa8b` (v0.1.12/.1). The installed wheel identifies as0.1.12.3; exact wheel/source parity has not been established. Lithos's [Metal mixer design](https://raw.githubusercontent.com/lithos-ai/lithos-metal/main/docs/design/mixers.md) supplies reusable locality and tuned-multi-dispatch comparison ideas, not portable Metal kernels or an unchanged eight-row recipe. Intel workgroup ownership, graph execution and rollback checkpoints require native implementations. The source audit guides a conditional conv-to-rule hypothesis; it does not prove installed-kernel parity, measured headroom, or benefit.
