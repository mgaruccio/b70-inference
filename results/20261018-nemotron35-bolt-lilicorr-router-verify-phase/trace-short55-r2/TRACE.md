# Trace-probe E2E process

This is a bounded diagnostic adapter for the **same immutable current verify-phase
binary**. It changes no model, kernel, serving source, launcher, public API, or
persistent configuration. It is not a timing run, qualification promotion, or
new trace protocol.

The adapter uses the completed `qualification-r1/RVP-A/create-argv.json` and
`qualification-r1/RVP-W3/create-argv.json` as launch sources. It preserves the
source mount, image, native commands, model/target/drafter arguments, cache and
hardware settings, and all other argv positions. Its only launch delta is a
fresh `C.PREFIX` container name (`TRACE-A` / `TRACE-W3`), a fresh read/write
host bind at `/capture`, the two trace environment entries, and the final
server boundary:

```text
export BOLT_VERIFY_CAPTURE_DIR=/capture
export LLAMA_TRACE=1
export GGML_SYCL_DEBUG=0
```

`BOLT_TRACE_TENSORS=0` and `ONEDNN_VERBOSE=0` remain unchanged. The native
startup shell and native route gates are not rewritten. The actual PID record
is validated against the corresponding untraced RVP role for identical `exe`,
`argv`, and relevant environment, with only `BOLT_VERIFY_CAPTURE_DIR=/capture`
and `LLAMA_TRACE=1` added. It is never passed through the diagnostics-off PID
checker, which would reject the intended diagnostic environment.

## Fresh official research used

- <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
  — `seed`, temperature, and `cache_prompt` affect server behavior. The probe
  therefore replays the frozen payloads unchanged and treats the trace flags as
  diagnostics, not performance settings.
- <https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h>
  — logit rows are indexed by output. Actual emission IDs from
  `choices[0].logprobs.content[*].id` must not be replaced by IDs retokenized
  from output text. Diagnostic comparisons retain the API IDs, pieces, text,
  finish reason, and usage exactly.

The existing framed route analyzer remains the parser. The current route filter
is retained, including `blk.0.*`, `blk.1.*`, `blk.7.*`, router weights 8/21/37/48,
and `output.weight`; the older `blk.0`-only description is not used.

## Remote preconditions and test data

Run exclusively on `inference-host` in the existing B70/xe/oneAPI environment.
Before the live window, the lead confirms:

- the current verify-phase candidate binary and all protected source/binary
  hashes are unchanged, with the existing build/dependency/native/N9 proofs;
- the accepted `nemotron-bolt-probe` has the exact original Config, HostConfig,
  complete mounts, model/image, and is healthy with the original as sole live
  service and DRM owner;
- Docker successfully reserves the unchanged published port after the original
  is stopped; both trace names were absent from `docker ps -a`, and the output
  filesystem retains the inherited 4 GiB reserve plus evidence allowance;
- no trace output directory from a prior attempt is reused.

The public boundary is the existing template/tokenize/chat path. Each arm runs
sequentially, first `TRACE-A`, then `TRACE-W3`, with the same frozen request
payloads and order:

1. 20 quality prompts;
2. the retrieval anchor;
3. `short/ordinary` (streaming, length 55);
4. `short/diagnostic` (non-streaming, 128 tokens, EOS ignored).

That is exactly **23 requests per arm**, not the long ordinary/diagnostic pair.
Each result is compared to the same-role untraced phase qualification
(`RVP-A` or `RVP-W3`). W3 is allowed to differ from A; the trace arm must
reproduce untraced W3. Quality/anchor/ordinary comparisons use exact text,
finish reason, and usage. Diagnostic comparisons use actual API emission IDs and
pieces in addition to text, finish, and usage; no probabilities or retokenized
IDs are substituted.

`BOLT_VERIFY_CAPTURE_DIR=/capture` writes the actual raw CPU sampler rows to
the fresh host bind. The current `common/sampling.cpp` writes one raw F32
vocabulary row per CPU sample, not every layer, so a roughly 1 GiB pair is
expected. The raw `row-*` files are retained byte-for-byte. Immediately before
the final short diagnostic, the adapter snapshots the Docker stdout/stderr
prefix; immediately after it, it saves the full after snapshot and the bounded
suffix returned by the existing `fresh_logs` prefix check. This keeps the window
limited to the real request rather than a guessed log substring.

## Preserved pre-launch failure

`trace-short55-r1` failed before any diagnostic container was created: the
standalone test socket returned `EADDRINUSE` after the original was stopped.
The cause was not conclusively identified from that errno alone. Its original
service restoration passed, including the READY smoke check, with no errors.
The retry uses Docker's existing atomic port reservation and failure cleanup,
not another check/use race. The old output directory is retained unchanged.

The [Linux socket manual](https://man7.org/linux/man-pages/man7/socket.7.html)
distinguishes address reuse from an active listening socket; a bare bind test
is not equivalent to the actual listener's reuse behavior. No port owner is
killed and no port or launcher configuration is changed.

## Commands (lead only)

The CPU fixture suite is also remote-only and must not be run in the interactive
worktree. Use a fresh scratch directory beneath this case's `worker-scratch/`:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
export BOLT_ROUTER_TRACE_PROBE_CPU_SCRATCH="$ROOT/worker-scratch/trace-probe-cpu-r4"
mkdir "$BOLT_ROUTER_TRACE_PROBE_CPU_SCRATCH"
python3 -B "$ROOT/test-trace-probe.py" 2>&1 | tee "$ROOT/trace-probe-cpu-r4.log"
```

After the existing current-phase CPU/native gates and the exclusive-window
preflight pass, run the live capture as a separate command. It creates a fresh
`trace-short55-r2/` directory and leaves trace containers stopped:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
test ! -e "$ROOT/trace-short55-r2"
python3 -B "$ROOT/trace-probe.py" --lead-run-trace 2>&1 | tee "$ROOT/trace-short55-r2.log"
```

Only after that command reports `CAPTURED_RESTORED` run independent restoration
and raw-evidence analysis in a separate process:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
test -d "$ROOT/trace-short55-r2"
python3 -B "$ROOT/trace-probe.py" --lead-after-restore 2>&1 | tee "$ROOT/trace-short55-r2-analysis.log"
```

After both cells reproduce their same-role controls, compare the first real
verification row with the existing strict route and F32 readers:

```bash
python3 -B "$ROOT/analyze-first-frame.py" | tee "$ROOT/trace-first-frame.log"
```

This checks the complete 55-token logical prefix, actual first emission IDs,
first post-prompt batch shapes (A: 1; W3: 4), target backend context, and raw
prefill/verification logit rows. It does not claim equal internal operator inputs
or identify a causal operator solely from a differing route.

No timing, throughput, promotion, or launcher update is allowed in these
diagnostic steps.

## Evidence and cleanup

Retain the ordinary raw artifacts under `trace-short55-r2/TRACE-A/` and
`TRACE-W3/`: source launch argv and positional delta, actual PID and library
records, inspected Config/HostConfig/mounts, raw and timestamped startup/server/
restored logs, native route files, all template/tokenize/HTTP request and
response files, same-role trace reports, short-diagnostic log prefix/after/
fresh-window files, `capture/row-*` F32 files, and the capture/analysis and
restoration JSON files. Keep `job-selection.json` as the exact 23 payload/order
record. Do not add receipts, hashes of trace rows, a protocol, or a state
framework.

The inherited controller's `finally`/signal cleanup is mandatory: on success or
failure, stop each fresh trace container and leave it stopped, restore the
original container and exact original Config/HostConfig/mounts, re-establish
health and the `READY` smoke response, verify sole service/DRM ownership,
hardware, and protected source/binary hashes, and fail closed if independent
restoration fails. No raw result analysis is accepted before that independent
restoration check passes.
