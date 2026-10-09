# Qwen 3.8 B70 GDN locality / megakernel program

**Status: queued Phase0, development-only.** This is an attribution-first
program, not a kernel implementation or a performance result. The bounded
harness lives in `results/20261009-qwen38-gdn-locality/` and uses the existing
step-profile lifecycle through `runpy`.

## Authority and boundaries

The accepted question is whether anything in the Lithos megakernel/mixer work
can inform a B70 build, with native MTP4 GDN locality as the first test. The
only follow-on arms, if evidence justifies them, are native controls versus a
tuned multi-dispatch control versus a narrow workgroup-local fused region.
There is no whole-engine port, production promotion, model/dtype/acceptance
change, Muse/DSpark expansion, custom audit store/protocol, or persistent
launcher change in this program.

The Lithos fixed-eight-row GDN design is not assumed to transfer unchanged to
MTP4's five-row verification shape. A five-row native MTP4 call must be
observed before any locality decision.

## Fresh research record

The lead performed the required fresh primary-source checks before this slice:

- <https://raw.githubusercontent.com/lithos-ai/lithos-metal/main/docs/design/mixers.md>
  describes the Lithos mixer/megakernel locality design and its fixed row-group
  assumptions. **Conclusion:** it is useful motivation for looking at data
  movement and workgroup locality, but its fixed eight-row GDN shape cannot be
  treated as an unchanged MTP4 five-row implementation.
- <https://docs.pytorch.org/docs/2.13/profiler.html> documents
  `record_function`, CPU/XPU profiler activities, shape recording, and the
  distinction between CPU scope time and device timing. **Conclusion:** use
  native profiler traces and public deferred `torch.xpu.Event` pairs; do not add
  CPU scope durations to graph/device timings or present a graph replay as an
  inner-kernel speed claim.

The existing pinned vLLM/XPU source and prior campaign evidence are local
orientation, not substitutes for that research gate. They motivate the
specific `torch.ops._xpu_C.gdn_attention` boundary and the explicit eager
fallback but do not preselect a candidate. A prior source-confirmed comparison
also found that tuned multi-dispatch essentially ties the fused control; that
result does not justify skipping the native five-row attribution gate.

## Predeclared real end-to-end process

All runtime, ML/compiler, Docker, XPU and GPU execution stays on the lead's
single `inference-host`; the worker does not launch it. Preconditions:

- one idle Intel Arc Pro B70, power cap `275000000` microwatts (275 W);
- `docker ps` empty, Glimmer stopped, and production launcher SHA256
  `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`;
- the pinned image digest
  `f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
  (vLLM `ac7509e2b`) and the existing canonical model/patch siblings are
  present;
- the new remote campaign directory and each output directory do not exist.

The golden server contract is native MTP4 K4, GPTQ symmetric G128 target with
FP16 compute, FP8 KV, C1, `max-model-len=212992`,
`max-num-batched-tokens=8192`, utilization `0.95`, graph capture sizes
`[1,2,4,8]`, prefix cache off, and thinking off. The public journey is the
existing `/tokenize` renderer followed by streaming `/v1/completions`, greedy
temperature 0, seed 42, EOS ignored, and exactly 128 forced output tokens.
`results/20260914-qwen38-four-way-speed/mtp4-reference-launch.json` is the
local canonical launch reference; the wrapper validates its effective MTP4
values rather than editing that file. The queued execution is:
   directory with `run-phase0.py --stage-only`. The staging CLI fails if the
   directory exists and stages no runtime or model.
2. Run `--mode baseline` with a fresh `baseline-01` output. The wrapper runs
   only 512 and 65,536 input tokens, one warmup and six measured requests per
   point. It does not pass profiler flags or diagnostic mounts. The existing
   client retains rendered token counts, request JSON, raw SSE, metrics before
   and after each request, and machine-readable summaries.
3. Only after the baseline completes, run `--mode profile` with fresh
   `diagnostic-graph-01`. It profiles one streamed 65,536/128 request through
   the real API. The finite native window is delay 3, maximum 5 worker
   iterations, and 24 non-empty SSE events; actual captured counts, not an
   assumed count, are authoritative.
4. Run `summarize-gdn.py` on the retained native trace. It correlates CPU
   `External id` records to kernel rows and reports actual GDN operator names,
   input shapes, GDN/matmul/full-attention/draft categories, and finite timing
   evidence. It consumes the ordinary trace and server log only.
5. If graph replay hides the GDN operator or inner kernels, run the separate
   `--mode eager-profile` diagnostic. It explicitly disables graph capture and
   enables shape recording; it is an attribution control, never the baseline
   and never a throughput comparison. A missing GDN operator after both
   diagnostics is a blocker to a credible locality decision; ask the lead
   before implementing a candidate.
6. Retain the raw trace, SSE/counter files, commands, launch metadata,
   environment/runtime identity, and cleanup/host-after artifacts on
   `inference-host`. The lifecycle removes only the observed mode-specific
   container and must leave the launcher, power cap, running-container set and
   Glimmer state unchanged. Return the host stopped.

The exact executable commands are in the campaign README. CPU-only checks are
`python3 -m py_compile` for the five new scripts and
`python3 results/20261009-qwen38-gdn-locality/test-phase0.py`; neither imports
PyTorch/vLLM nor exercises a GPU.

## Evidence and gates

Phase0 is a development run under `BENCHMARKING_STANDARDS.md`, not a
publishable benchmark. The Phase0 decision gate is:

- **Configuration gate:** effective launch metadata matches the golden contract;
  only the eager diagnostic may differ in graph flags.
- **Ownership gate:** host preflight and postflight invariants pass; output and
  container names are unique and the new output is fail-if-existing.
- **Public correctness gate:** `/health`, `/v1/models`, `/server_info`, finite
  metrics/counters, `/tokenize`, and complete streaming responses pass with
  exact rendered input and output counts.
- **Attribution gate:** the trace has enough actual call/operator/shape and
  deferred XPU timing evidence to separate GDN from matmul, draft, and full
  attention. Graph-hidden results require the eager diagnostic; no proxy or
  historical timing is substituted.
- **Decision gate:** no speedup is claimed from profile traces. A later
  candidate must be selected from the observed dominant path, keep native MTP4
  K4 and all quality/capacity controls fixed, and pass output parity/correctness
  before a serving A/B.

Only after those gates are met is the next program phase queued:

| Phase | Arm / question | Entry gate | Exit evidence |
| --- | --- | --- | --- |
| P0 | Native MTP4 attribution at 512 + 64K, then graph/eager diagnostic if needed | This document and the harness contract | Credible GDN-vs-matmul/attention/draft attribution or an explicit blocker |
| P1 | Native control versus one bounded GDN locality candidate | P0 identifies a recoverable GDN boundary; no quality/capacity change | Real HTTP correctness, raw trace, finite unprofiled measurements, and unchanged host |
| P2 | Narrow comparison: native MTP4, tuned multi-dispatch control, or narrow workgroup-local fused region | P1 candidate clears parity and shows a meaningful signal | Paired/interleaved evidence at the same 64K contract; no production promotion |
| P3 | Full 512/8K/32K/64K three-arm campaign | Candidate warranted and parity established; lead approval | Development comparison only, with retained raw SSE/counters/environment |

The full campaign is intentionally absent from this commit. No candidate
implementation is selected here.

## Interpretation guardrails

A graph replay event can represent the scheduled region while hiding its inner
GDN/matmul/attention kernels. CPU `record_function` ranges are inclusive host
annotations; `torch.xpu.Event` intervals are deferred current-stream timing.
They are separate evidence sources and must not be summed into a critical-path
budget. Eager diagnostic timing includes its diagnostic configuration and is
not a baseline or a speed claim.
