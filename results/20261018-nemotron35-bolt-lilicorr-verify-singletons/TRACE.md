# Verify-singletons trace-short55-r1 — preparation, not a result

**Development diagnostic only.** This adapter targets the same new candidate as the
public qualification. It is not a parent run, native-only proxy, timing run, promotion,
or proof that DFlash verification divergence is fixed. The lead must complete the new
public qualification, native/N9 checks, trace capture, independent restoration, and
first-frame analysis on `inference-host`.

## Fixed integration and runtime contract

- Remote evidence root: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons`.
- New private candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r1`.
- The future `public-gate.py` keeps the parent `P`, `C`, `K`, and helper interface.
  Its public cells are `VSA-A`, `VSA-W3`, and `VSA-W7`, with `OUT=ROOT/'qualification-r1'`.
- This adapter uses trace cells `VST-A` and `VST-W3`, physical names derived from
  `C['PREFIX'] + cell.lower()`, and `OUT=ROOT/'trace-short55-r1'`.
- `trace-probe.py` derives each trace launch from the completed new `VSA-A`/`VSA-W3`
  create argv. The only differences are the fresh name, one read/write capture bind at
  `/capture`, `BOLT_VERIFY_CAPTURE_DIR=/capture`, `LLAMA_TRACE=1`, and the matching final
  server environment boundary. The source mount, image, model, target/drafter,
  sampler, hardware/caps, native commands, and every serving argument remain inherited.
- Actual PID1 records are compared directly with the untraced same-role records. Complete
  Config, HostConfig, mounts, bind ordering/coverage, source paths, and container identity
  are checked; Docker may report only `None`/`False` for `OomKillDisable`, never `True`.
  No evidence is normalized or rewritten.

The public boundary remains `/apply-template`, `/tokenize`, and
`/v1/chat/completions`. Run exactly **46 real generation requests**, 23 per trace arm,
in the frozen order: 20 quality requests, anchor, short ordinary, short diagnostic
(128 tokens, EOS ignored). Every trace role must reproduce its same-role public control,
including actual IDs/pieces, text, finish, and usage. W3 may differ from A; that is not a
trace failure. Independent raw analysis is allowed only after the original service has
been independently restored.

## Lead-only execution

Do not build or execute the model locally. Use the existing B70/xe/Intel environment on
`inference-host`, with the explicit remote shell procedure from `qualification.md` and
strict capacity, source, binary, DRM, hardware, health, and restoration checks. Keep
stopped experiments and all raw evidence.

Supplementary CPU checks (not runtime proof) use a fresh scratch directory under the new
root and no service, Docker, GPU, network, or HTTP work:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
attempt=$(mktemp -d "$ROOT/worker-scratch/trace-cpu-XXXXXXXX")
export BOLT_VERIFY_SINGLETONS_TRACE_CPU_SCRATCH="$attempt"
export BOLT_VERIFY_SINGLETONS_CPU_SCRATCH="$attempt"
export PYTHONDONTWRITEBYTECODE=1
{
    date -u +%FT%TZ; hostname; uname -srmo; python3 --version
    python3 -B "$ROOT/test-trace-probe.py"
    python3 -B "$ROOT/test-first-frame.py"
} > "$attempt/tests.log" 2>&1
printf 'TRACE_CPU_PASSED=%s\n' "$attempt"
```

The lead's real journey is separate and must retain its commands/logs:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/trace-short55-r1"
test ! -e "$ROOT/trace-short55-r1.log"
python3 -B "$ROOT/trace-probe.py" --lead-run-trace 2>&1 | tee "$ROOT/trace-short55-r1.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/trace-short55-r1-analysis.log"
python3 -B "$ROOT/trace-probe.py" --lead-after-restore 2>&1 | tee "$ROOT/trace-short55-r1-analysis.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-verify-singletons
test ! -e "$ROOT/trace-first-frame.log"
python3 -B "$ROOT/analyze-first-frame.py" 2>&1 | tee "$ROOT/trace-first-frame.log"
```

## Bounded first-frame route contract

`analyze-first-frame.py` retains the parent strong checks:

- The complete 55-token prompt prefix is required, split exactly into n=5 at positions
  0..4, n=46 at 5..50, and n=4 at 51..54. Their output masks are respectively all
  zero, all zero, and `[0,0,0,1]`. Every target router in those partial-output frames
  remains unmarked (`accum_prec=0`) and matches its exact parent-A route sequence and count.
- A first post-prefix A frame is n=1 at position 55, all outputs, with the exact parent-A
  unmarked sequence `DMMV_ESIMD`, `MOE_SINGLE_MMVQ_PLAIN_Q8_1`,
  `MOE_SINGLE_MMVQ_PLAIN_Q8_1`, `DMMV_ESIMD`, and no intervention mark. W3 is n=4 at
  positions 55..58, all outputs.
- The target backend is anchored by exactly the first-block SSM projections. The five
  target F32 routers are still blocks 1, 8, 21, 37, and 48, with the existing ordered
  route-pair, context, destination, shape, stride, precision, and singleton-view checks.
- For the four exact eligible first-block weights observed in parent A, the unmarked
  prefix route sequences are pinned by both order and count: n=5 has six rows
  (`MMVQ_REORDERED`, grouped attempt/result for each IQ2 weight, then `MMVQ_REORDERED`),
  n=46 has eight rows (the Q8 `MUL_MAT_SYCL`/`LIBRARY_DNNL_F16` pair at each end plus
  the two grouped IQ2 attempt/result pairs), and n=4 has the same six-row sequence as
  n=5. Changed routes, duplicates, removals, or reorderings fail closed.
- W3's intended singleton intervention is limited to **these four exact eligible
  first-block weights**: `blk.1.ffn_latent_down.weight` (Q8_0, `[4096,1024]`),
  `blk.1.ffn_latent_up.weight` (Q8_0, `[1024,4096]`),
  `blk.1.ffn_up_exps.weight` (IQ2_XXS, `[1024,2048,240]`), and
  `blk.1.ffn_down_exps.weight` (IQ2_XXS, `[2048,1024,240]`). The instrumentation and
  validation do not claim to trace all model Q8/IQ2 weights. Exact backing strides,
  types, op IDs, device, reorder, IDs, and original outer strides are checked.
- The intended Q8 route is `DMMV_ESIMD`, and the intended IQ2 route is
  `MOE_SINGLE_MMVQ_PLAIN_Q8_1`; both carry the existing precision mark
  `accum_prec=10` only for the marked n=3..8 family. A marked Q8 callback emits exactly
  one coherent outer n-column record. The IQ2 `MUL_MAT_ID` callback emits exactly n
  coherent singleton rows, each with `src1.ne2 == dst.ne2 == ids.ne1 == 1` and the
  original n-column backing strides. One outer IQ2 row, one singleton row, mixed shapes,
  and duplicate outer representations are rejected; no inner view entries are fabricated.
- Missing weights, wrong context, wrong type, wrong precision/mark, wrong route, wrong
  shape/stride, duplicate or misordered eligible records, native fixture names, and
  drafter-context records fail closed. The unchanged parent W3 route set is an explicit
  negative fixture for the new intervention checks. Synthetic positive fixtures are
  confined to CPU tests and are not runtime evidence.

The actual sampler row 0 is bound to API emission 2. Prefill and verification capture
rows are retained and compared as raw F32 bytes. Any verification differences are
reported honestly; no probability, retokenized-ID, operator-input-bitwise-equality, or
operator-causality claim is made. Public fidelity gates remain authoritative, and this
adapter must not be used to claim parity from trace selection alone.

## Retained evidence

Keep `trace-short55-r1/{VST-A,VST-W3}/` with create argv/argv diff, actual PID and
library records, inspected containers, raw startup/server/restored streams, native copies,
all 23 public/template/tokenize/request/result artifacts per role, trace reports, the
short-diagnostic prefix/after/fresh windows, and untouched `capture/row-*` files. Keep
`summary.json`, `independent-restoration.json`, `analysis.json`,
`first-frame-comparison.json`, top-level logs, and CPU scratch including failures. No new
receipt, trace schema, protocol, timing, or promotion artifact is introduced.
