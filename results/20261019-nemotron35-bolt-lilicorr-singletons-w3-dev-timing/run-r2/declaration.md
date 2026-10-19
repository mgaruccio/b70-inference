# Nemotron 3.5 Bolt LiLiCorr singleton W3 timing — development declaration

**Preparation only.** This is a small, development-tier performance measurement
package. It is not a publishable benchmark, a full-suite result, a statistical
significance claim, a deployment change, or a promotion decision. W7 is
explicitly excluded because the qualified W7 run rejected two long cases.

Attempt r1 stopped before native startup or timing because its container validator
incorrectly compared fresh Docker hostnames; the restoration verifier also lacked
`hardware-before.json`. Its raw failure evidence and exact code are retained in
`run-r1/` and `attempt-r1-source/`. The original subsequently passed the existing
qualified live-restoration checks and a fresh READY response. Attempt r2 repairs
those lifecycle checks, uses fresh names/output, and keeps the same benchmark,
qualification gates, and measurement settings. No failed measurement is discarded.

The user-authorized scope is only a controlled baseline-vs-qualified-W3
comparison using the already-qualified r2 source, model, serving arguments,
and prompts. Existing experiments and launchers are read-only.

## Frozen inputs and intentional difference

- Qualified immutable gate: sibling `20261019-nemotron35-bolt-lilicorr-verify-singletons/public-gate.py`.
- Qualified source candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`.
- Original production service: `nemotron-bolt-probe`; it is restored, never
  recreated or replaced.
- `public-gate.py::make_create('VSB-A'/'VSB-W3', h)` supplies the complete
  startup/native/model invocation. The adapter changes only `--name` and uses
  `C.PREFIX + cell.lower()` for fresh cells:
  `VSPF-A1`, `VSPF-W3a`, `VSPF-W3b`, `VSPF-A2`.
- No trace flags, capture bind, model/build/config, power, or persistent
  launcher change is introduced. Existing F32/116-native checks, actual PID1
  argv/environment, Docker records, raw native logs, and source/binary proofs
  remain required.

## Tier, baseline, and schedule

This is a development run. Both arms use the same r2 binaries/model/serving
arguments, the same qualified short-55 and long-8270 prompts, greedy 128-token
output, `cache_prompt=false`, and the existing stream API. The only intentional
comparison is no-draft A versus qualified W3.

The order is fixed ABBA:

```text
A1, W3a, W3b, A2
```

Each cell runs both points. Before timing any cell, the full existing 25-job
qualification contract is run against the frozen original A0 and the current
fresh reference. A1 uses the qualified VSB-A reference; subsequent cells use
fresh A1. W3 must pass all 25 jobs and show a real rejected draft in diagnostics.
No measured request starts before this gate passes.

After prevalidation, each point gets one warmup followed by three measured
requests. This yields six measured samples per arm and point, 24 measured
requests and eight warmups total. Warmups are excluded from summaries. A failed
request remains in raw evidence and aborts the run; it is not retried or
silently dropped.

## Public end-to-end test process (lead only)

Environment and preconditions: `mike@inference-host`, hostname
`inference-host`, the existing qualified original service healthy on its
socket, the qualified A/W3 references and r2 source/binary proofs present, and
at least the inherited reserve/evidence capacity. The lead stages this sibling
at the physical remote root `/home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing`.

The real journey is: check the immutable qualified proof and restored original;
record hardware, source IDs, ownership, and environment; create each fresh
Docker cell; run the existing native/startup gates and actual PID/container
checks; perform public `/apply-template` → `/tokenize` → `/v1/chat/completions`
requests; save every raw HTTP/SSE body, event timestamp, normalized result,
telemetry sample, server log, and failure; stop/preserve every created cell;
restore the original and run READY smoke, health, sole-container, ownership,
hardware, source, and fresh P-proof checks; only then independently reread raw
evidence and calculate descriptive statistics.

Lead commands:

```bash
cd /home/mike/b70-evals/20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing
bash cpu-checks.sh
python3 -B controller.py --lead-run-timing 2>&1 | tee controller-r2.log
# only after the controller reports restored:
python3 -B analyze-timing.py --lead-after-timing-restore 2>&1 | tee analyzer-r2.log
```

`analyze-timing.py` performs its independent live restoration check before
reading timing artifacts. It retains partial/failing rows and refuses a
summary unless cardinality is exactly six per arm/point and all raw/fidelity
checks pass. Raw artifacts are retained under `run-r2/`; they are the source of
truth. No remote runtime, Docker, API, GPU, or SSH work is performed by the
worker or by CPU fixtures.

## Metrics and interpretation

The analyzer reports server `predicted_per_second`, `predicted_ms`, prompt
fields, client TTFC/E2E fields, and any exposed draft counters. Missing draft
counters remain `null`/unavailable, never zero. It reports median and
Hyndman–Fan type-7 IQR per cell and per arm, plus descriptive A→W3 deltas.
`first_content_decode_proxy_tps` is retained only as a proxy and is never
called decode speed or ITL. Cumulative server log statistics are retained as
cumulative; chunks are not summed into per-request timings. Output matching is
reported separately and no raw-F32/vector equality is claimed.

## Fresh external research recorded before implementation

- [llama.cpp server tools and `server-common.cpp/.h`, `server-task.cpp`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/) — server-reported prompt/predicted durations are distinct from client TTFT/E2E; final timing/speculative fields are runtime counters and snapshots are cumulative.
- [llama.cpp PR #12603](https://github.com/ggml-org/llama.cpp/pull/12603) — current server timing/speculative accounting must be interpreted from the pinned runtime, not reconstructed by summing chunks.
- [vLLM serving benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) — maintainer benchmark output distinguishes request, output-token, TTFT, and generation metrics; this development package intentionally does not claim a vLLM/BetterBench result.

## CPU fixture gate and retained concerns

`cpu-checks.sh` runs only synthetic/read-only fixtures. The focused tests cover
exact name-only argv deltas, W7 rejection, ABBA order and 24-row cardinality,
missing counter handling, source/reference guard behavior, raw timing/fidelity
rejection, and restoration-on-error through the inherited lifecycle when the
remote sibling is staged. A green CPU gate is not a remote runtime claim.

No publication checklist, full context sweep, unique-prompt study, confidence
interval, CI gate, service/API change, or persistent launcher change is part of
this package.
