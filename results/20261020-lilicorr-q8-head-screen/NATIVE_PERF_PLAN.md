# Q8 output.weight head cost screen

## Status and scope

This is a bounded diagnostic/test slice, not the requested dramatically faster
drafter and not a serving benchmark. It changes no application kernel, math or
precision behavior, SDK/driver, queue/event profiling, graph, launcher,
full8/7x8/max3, rejected-KV, or W7 behavior. It runs one already-qualified
Q8 target library as a **single-library cost screen**; there is no second
baseline bind and no A/B speed claim.

The lifecycle, source/build proof, adapter checks, protected configuration checks,
exclusive ownership, and finally-restoration path are inherited from
`../20261020-lilicorr-q8-target-batch4/run-performance.py`. The package-specific
wrapper only changes the output root, container prefix, one read-only package
mount, and the final native command for one W3 cell:

- remote root: `/home/mike/b70-evals/20261020-lilicorr-q8-head-screen`;
- output: `native-perf-r1` (created with `exist_ok=False`, never overwritten);
- cell: `VSPF-W3a` (one qualified W3 native container);
- package mount: this directory at `/q8-head-screen:ro`;
- existing qualified target package remains the read-only `/iq2-batch4` mount,
  including its `native-batch4.py` checker.

The lead owns remote execution. This worktree performs only CPU contract tests;
no SSH, Docker, GPU, SDK, installation, or push is performed here.

## Pinned ABI and case data

The lead-only completed build metadata was read read-only from
`/home/mike/code/b70-inference/results/20261020-lilicorr-q8-target-batch4/` when
those uncommitted files were not present in this worktree. Relevant pins are:

- `build-attempt-r1/candidate-binaries.json`: candidate `libggml-sycl.so.0`
  SHA-256 `f6f6aa0ac06b8547e5e57981ea95f07c6448bd4f3c3867070efa13fc4ee965ab`;
- `build-attempt-r1/image.json`: the existing image digest
  `intel/deep-learning-essentials@sha256:aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0`;
- pinned `ggml/include/ggml.h` SHA-256
  `475beaa7e959be8c383279d9ce9d5191f37426e6b338270cfb12b26a798129d3`;
- pinned `tests/test-backend-ops.cpp` SHA-256
  `3a2167a0b4d9e26feea23ef299e11fe6468baf19be6e0304ebde57ea240e1c6d`.

The pinned header gives `GGML_OP_MUL_MAT=29`, `GGML_TYPE_F32=0`, and
`GGML_TYPE_Q8_0=8`. The pinned `make_test_cases_from_file` parser consumes:
`op type output-ne[4] param-count params source-count`, then each source as
`type ne[4] nb[4]`, and finally one name token. Therefore the final token below
is an operation/test name, **not a weight name**; the generic fixture does not
set source names.

`native-perf-cases.txt` contains exactly three rows, in n1/n4/n8 order:

| name | output `[ne0,ne1,ne2,ne3]` | source 0 | source 1 |
|---|---|---|---|
| `q8-head-n1` | `[131072,1,1,1]` | Q8_0 `[4096,131072,1,1]`, bytes `[34,4352,570425344,570425344]` | F32 `[4096,1,1,1]`, bytes `[4,16384,16384,16384]` |
| `q8-head-n4` | `[131072,4,1,1]` | same | F32 `[4096,4,1,1]`, bytes `[4,16384,65536,65536]` |
| `q8-head-n8` | `[131072,8,1,1]` | same | F32 `[4096,8,1,1]`, bytes `[4,16384,131072,131072]` |

All rows use two sources, 16 op parameters, and sixteen neutral zero
parameters. No private F32 tag is inserted. The case is the actual large target
`output.weight` geometry, not the drafter projection. Fulltrace source evidence
for this geometry observed `MMVQ_REORDERED` with `accum_prec=0`,
`src1_prec=0`, and `reorder=1`; source-supported mapping is kept distinct from
this generic test-file fixture because the parser does not set source names or
model reorder metadata. Q8 dot is globally nonshared-weights, but this plan
makes no claim about physical rereads: compiler CSE may remove loads.

## Native body and assertions

The container runs the unchanged `test-backend-ops` executable from
`/src/build-sycl/bin`:

1. With the normal native capture directory and `GGML_SYCL_DEBUG=1`, run the
   inherited singleton gates for `q8` and `iq2`. The existing strict checker
   must report Q8 **82 cases / 54 byte-exact cases** and IQ2 **34 cases / 16
   byte-exact cases**, retaining all numeric, negative, padding, alias, canary,
   and route assertions.
2. Separately run the exact diagnostic command:
   `test-backend-ops test -b SYCL0 -o MUL_MAT --test-file
   /q8-head-screen/native-perf-cases.txt -j 1`, with
   `LD_DEBUG=libs`, `SYCL_UR_TRACE=2`, and `GGML_SYCL_DEBUG=1`. The checker
   requires the exact `/src/build-sycl/bin/libggml-sycl.so` load, three named
   operation comparisons (`3/3 tests passed`), no unsupported/large-tensor
   skips, and no `UR_QUEUE_FLAG_PROFILING_ENABLE`. Raw UR launch lines and any
   handles/names are retained. A UR launch proves neither a C++ kernel name nor
   a physical reload, so no such inference is made.
3. Disable diagnostics and run the same immutable executable in three separate
   `perf` processes, in file order n1, n4, n8. Each process uses the complete
   case file, so the strict parser requires three finite positive `us/run` rows
   per process and an aggregate of exactly **9 rows**. No profiling or timing
   knob is added; the built-in perf warmup and at-least-one-second steady graph
   measurement remain in force.

The measured value is synchronous whole `ggml_backend_graph_compute` operator
wall time, including Q8_1 quantization, dispatch, math, and amortized
synchronization. It is not a GPU event/kernel time or serving latency. Inputs
are the test executable's random cache-hot tensors; no serving speedup or net
drafter conclusion can be inferred.

Every command writes an exit-code file. The native container stdout/stderr,
container records, raw copied `/tmp/lilicorr-q8-head-screen` tree, checker JSON,
and all failed outputs are retained before success assertions.

## Real execution journey (lead-only)

**Preconditions:** use `inference-host` with exclusive GPU ownership; the
qualified target package/source/build metadata and existing target remote root
must be present; the new remote root must be staged by the lead and must not
already contain `native-perf-r1`; use the existing cached image digest above.
Do not modify persistent launchers or configuration.

From the staged remote root, the lead runs:

```sh
python3 -B test-native-perf.py -v
if python3 -B run-native-perf.py --lead-run-native-perf \
    >native-perf-lead-run.stdout.log 2>native-perf-lead-run.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-run.exit-code
```

The run invokes the inherited qualified preflight and source/build/adapter/
protected-config guards, stops the original service only through the inherited
exclusive lifecycle, creates exactly one W3 native container, and retains the
raw artifacts above. The inherited `capture` `finally` path stops the
experimental container and restores the original service on success or failure.

Regardless of the first command's exit status, the lead opens a **new process**
and runs:

```sh
if python3 -B run-native-perf.py --lead-verify-restored \
    >native-perf-lead-restore.stdout.log 2>native-perf-lead-restore.stderr.log; then rc=0; else rc=$?; fi
printf '%s\n' "$rc" >native-perf-lead-restore.exit-code
```

That independent check must reverify original identity/source/health, sole GPU
ownership, stopped experimental containers, and restoration state. The lead
then synchronizes `native-perf-r1/` and both process logs/exit codes back to
this result directory without deleting failed or prior artifacts.

Expected output includes `declaration.md`, one `VSPF-W3a/` lifecycle directory,
`create-argv.json`, before/after container metadata, native stdout/stderr,
`raw/verify-singletons-{q8,iq2}.log`, diagnostic raw/checker output,
`perf-{0,1,2}.log` plus checker JSON, `perf-summary.json`, and independent
restoration evidence. The expected result is a qualified **cost screen only**;
no candidate-kernel change, baseline comparison, promotion, or speed claim is
made here.

## Fresh primary research

Fetched 2026-10-20 before implementation:

- [LLVM Alias Analysis](https://llvm.org/docs/AliasAnalysis.html) — alias and
  Mod/Ref results are conservative and enable memory-dependence optimizations;
  therefore compiler CSE may already eliminate repeated loads, so source intent
  is not a physical-reread measurement.
- [Intel oneAPI: Registers and Performance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/registers-and-performance.html)
  — registers are fast but limited; reuse can help, while high register
  pressure can reduce SIMD/occupancy or spill and hurt performance.
- [Intel oneAPI: Finding Kernels with Register Spills](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/finding-kernels-with-register-spills.html)
  — compiler diagnostics and ISA/spill inspection can identify spill behavior
  and kernel timing, but this screen deliberately does **not** enable unitrace
  or profiling because the earlier profiling mode failed native fidelity.

No GUI/Labsocket preview tool is required for this CPU-only slice; the lead's
final review uses the narrow patch/status checks and the retained real-run
artifacts rather than creating a new evidence or benchmark framework.
