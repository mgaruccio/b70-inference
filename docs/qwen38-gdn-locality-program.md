# Qwen 3.8 B70 GDN locality / megakernel program

**Status: P0 complete; pinned64K/MTP4 GDN kernel share2.925989% <5%, so conditional P1/P2/P3 are not admitted. Development-only; see [final execution evidence](../results/20261009-qwen38-gdn-locality/execution.md).**
The second tracer attempt coincided with a fatal AMD execution-unit MCA/data-fabric
reset signature. Its trigger is unresolved; GPU work remains paused. See
`results/20261009-qwen38-gdn-locality/execution.md` for ordinary
results and failure evidence. No kernel candidate or gain has been established.
The bounded harness lives in `results/20261009-qwen38-gdn-locality/` and uses the existing
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
- <https://github.com/pytorch/pytorch/blob/v2.13.0/c10/xpu/XPUEvent.h> documents the
  XPU event implementation used for deferred stream-boundary timing. **Conclusion:**
  use it for complete replay boundaries, not as a per-node graph marker.
- <https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc>
  defines aggregate graph profiling and its event/profiling limitations. **Conclusion:**
  graph profiling does not provide a supported per-GDN node timestamp and can alter
  optimization behavior.
- <https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md>
  documents unitrace device timing, Chrome kernel/device logging, and paused-session
  controls. **Conclusion:** the build pins the exact source commit separately; attempt
  the normal graph replay with `--start-paused`, then resume/pause at the disposable
  worker's profile hooks.
- <https://intel.github.io/pti-gpu/whatsnew.html> describes SYCL graph tracing as
  evolving/initial proof-of-concept support. **Conclusion:** missing or partial
  graph records are a tool-coverage blocker, never evidence that GDN is absent.

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

1. Stage the harness into the fresh sibling
   `/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20261009-qwen38-gdn-locality-harness`
   with `run-phase0.py --stage-only`. The staging CLI fails if the directory
   exists and stages no runtime or model.
2. Do **not** rerun the completed uninstrumented four-way baseline. The
   lead-owned `baseline-original-01` output is the unchanged normal-serving
   reference; its raw artifacts and reproduced statistics are retained.
3. After the baseline is idle, run the bounded `build-unitrace.sh` from the
   fresh harness. It uses the pinned PTI source/image/compiler contract below;
   it must not compile concurrently with the baseline and it has no GPU.
4. Run `--mode unitrace-profile` with fresh `diagnostic-unitrace-01`. This is
   the normal graph-enabled MTP4 server under unitrace, with `/start_profile`
   resuming and `/stop_profile` pausing, synchronizing and stopping/flushing a
   finite capture around the real streamed request. The original vLLM launch
   remains unchanged. The revised flush path still lacks a completed GPU test.
5. Review the unitrace trace together with the external whole-replay timing and
   run `summarize-gdn.py`. Missing, partial, draft-only, or unreconciled graph
   coverage is an explicit attribution blocker, not a fusion result.
6. Only as a diagnostic operation-identification control, run
   `--mode eager-profile` if graph replay hides individual operators. Eager
   timing can never establish a GDN share or select/reject a fusion candidate.
7. Retain the raw trace, SSE/counter files, commands, launch metadata,
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
  only the explicitly diagnostic eager mode may differ in graph flags.
- **Ownership gate:** host preflight and postflight invariants pass; output and
  container names are unique and the new output is fail-if-existing.
- **Public correctness gate:** `/health`, `/v1/models`, `/server_info`, finite
  metrics/counters, `/tokenize`, and complete streaming responses pass with
  exact rendered input and output counts.
- **Attribution gate:** only a normal graph-enabled MTP4 unitrace capture with
  complete target graph kernel coverage, explicit target-root provenance,
  external whole-target replay timing, and reconciled coverage can establish a GDN
  device-time share. The unitrace rows must be reconciled to the complete replay,
  not just to a visible subset. Eager-only, partial, missing, draft-only, failed,
  stage-unknown, or unreconciled traces are **inconclusive** and cannot select or
  reject fusion. XPU event boundaries are whole-replay evidence, not unsupported
  in-graph per-node markers.
- **Decision gate:** a fusion candidate is eligible only when explicit-target GDN
  is at least 5% of that reconciled complete graph-enabled target device work.
  Below 5%, or when any attribution prerequisite is missing, the result remains
  **inconclusive**; no candidate is selected or rejected. Any later candidate
  must preserve native MTP4 K4, all quality/capacity controls, and pass
  native-relative output parity/correctness. Unrelated training teacher-fidelity
  gaps are not this kernel-locality gate.

Only after those gates are met is the next program phase queued:

| Phase | Arm / question | Entry gate | Exit evidence |
| --- | --- | --- | --- |
| P0 | Native MTP4 graph-aware attribution at 512 + 64K, with bounded unitrace and eager diagnostic only if needed | This document, the harness contract, and a complete normal graph replay | Reconciled GDN share or an explicit tool-coverage blocker; eager/partial evidence remains inconclusive |
| P1 | Native control versus one bounded GDN locality candidate | P0 reaches the >=5% complete graph-enabled target-work gate; no quality/capacity change | Real HTTP correctness, native-relative parity, raw trace, finite unprofiled measurements, and unchanged host |
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
