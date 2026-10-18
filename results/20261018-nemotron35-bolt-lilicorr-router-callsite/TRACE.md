# Callsite candidate trace-short55-r1 — preparation, not a result

**Development diagnostic, no timing/promotion/launcher changes.** This retargets the
successful phase trace-r3 adapter to the **same NEW candidate** used by the public
qualification, not to the parent or a native/test-only proxy. Follow
[qualification.md](qualification.md) first; build, fresh native/N9, CPU freeze and
new public capture/independent restoration must be complete.

## Runtime and request contract

`trace-probe.py` derives its launch from `qualification-r2/RCA2-A/create-argv.json`
and `RCA2-W3/create-argv.json`. Fresh roles are **RCT-A / RCT-W3**, physical names
`C.PREFIX + cell.lower()`, output **`trace-short55-r1/`**. Only name, the fresh
read/write `/capture` bind, `BOLT_VERIFY_CAPTURE_DIR=/capture`, `LLAMA_TRACE=1`, and
the corresponding final server environment boundary differ. `GGML_SYCL_DEBUG=0`,
`BOLT_TRACE_TENSORS=0`, `ONEDNN_VERBOSE=0`, native shell, source mount, model/target/
drafter, sampler, image, hardware/caps and every other serving argv remain unchanged.
Actual PID1 evidence is compared directly to the untraced same-role PID; it is never
rewritten or sent through the diagnostics-off checker. Complete Config, HostConfig
and mounts are checked, allowing **only None/False** for OomKillDisable, never True.

Both inherited fixes are preserved: no pre-start bind probe (Docker reserves the
unchanged published port atomically), and no rewrite of Docker null/false records.
The official Docker/Moby research and conclusions are recorded in qualification.md.
The lead's upstream Nemotron research also establishes why actual callsite routes,
not just native fixture marking, must be checked.

Exactly **46 real generation requests**, **23 per arm**, in frozen order:
20 QA, anchor, short ordinary, short diagnostic (128 tokens, EOS ignored). This is
not a two-request shortcut: preceding history is part of reproduction. The public
boundary remains `/apply-template`, `/tokenize`, `/v1/chat/completions` with unchanged
payloads. Trace A must exactly reproduce **new RCA2-A** and trace W3 **new RCA2-W3**,
including actual API IDs/pieces, text, finish and usage. W3 may differ from A; that is
separate from trace reproduction. Independent analysis rereads both roles' raw
public/template/tokenize evidence only after successful restoration.

## Remote commands and preconditions (lead only)

Use inference-host's existing B70/xe/Intel 2026 environment, not interactive Pi.
Submit these Bash blocks with explicit `bash -lc` as described in qualification.md;
the host login shell is Fish. Do not build, change services outside the inherited
exclusive window, or reuse output/names. Require exact original service, health,
sole DRM ownership, unchanged source/binaries/hardware, and the inherited 4 GiB
reserve plus 1 GiB evidence allowance; check capacity throughout capture.

After the new public independent analysis, run trace CPU fixtures in fresh scratch.
This reads actual **new** baseline controls but does not execute Docker, GPU or HTTP:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
attempt=$(mktemp -d "$ROOT/worker-scratch/trace-cpu-XXXXXXXX")
export BOLT_ROUTER_CALLSITE_TRACE_CPU_SCRATCH="$attempt" PYTHONDONTWRITEBYTECODE=1
{
    date -u +%FT%TZ; hostname; uname -srmo; python3 --version
    printf 'command=python3 -B - %s (trace fixtures and generated bash -n below)\n' "$ROOT"
    python3 -B - "$ROOT" <<'PY'
import runpy, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
t = runpy.run_path(str(root / 'trace-probe.py'))
h = t['C']['helpers']()
before = t['P']['check_tested'](h)
for cell in t['CELLS']:
    subprocess.run(['bash', '-n'], input=t['make_create'](cell)[-1], text=True, check=True)
sys.argv = [str(root / 'test-trace-probe.py')]
runpy.run_path(sys.argv[0], run_name='__main__')
assert before == t['P']['check_tested'](h)
PY
} > "$attempt/tests.log" 2>&1
printf 'TRACE_CPU_PASSED=%s\n' "$attempt"
```

Capture, independent analysis, and first-frame analysis are separate invocations:

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
test ! -e "$ROOT/trace-short55-r1"
test ! -e "$ROOT/trace-short55-r1.log"
python3 -B "$ROOT/trace-probe.py" --lead-run-trace 2>&1 | tee "$ROOT/trace-short55-r1.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
test ! -e "$ROOT/trace-short55-r1-analysis.log"
python3 -B "$ROOT/trace-probe.py" --lead-after-restore 2>&1 | tee "$ROOT/trace-short55-r1-analysis.log"
```

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
test ! -e "$ROOT/trace-first-frame.log"
python3 -B "$ROOT/analyze-first-frame.py" 2>&1 | tee "$ROOT/trace-first-frame.log"
```

## Focused actual-route assertions and expected results

`analyze-first-frame.py` requires both trace roles to have reproduced their same-role
controls after independent restoration. It uses the existing framed route and raw
F32 readers, with the unchanged native parser's strict router-operation semantics:

- Complete **55-token prefix**, exact prompt IDs and initial emitted API ID; target
  prompt chunks n=5 (positions 0..4), n=46 (5..50), n=4 (51..54), respectively output
  flags all-zero, all-zero, `[0,0,0,1]`. Every observed target router is unmarked
  (`accum_prec=0`) for those chunks.
- A first post-prompt batch: **n=1 at position 55**, all outputs, unmarked.
- W3 first post-prompt batch: **n=4 at positions 55..58**, all outputs, each target
  router **`accum_prec=10`**, four ordered singleton `MUL_MAT_SYCL` /
  `LIBRARY_MKL_F32` pairs. The enclosing batch supplies n=4; the inner column views
  correctly have `src1_ne1=dst_ne1=1` while preserving original strides. Do not assert
  every flattened route has ne1=4 or accept a mark without the intended routing.
- All five observed target routers: **blocks 1, 8, 21, 37, 48**, target backend anchored
  by first-block SSM projections. Missing routers, native fixture destination names,
  wrong/drafter context, malformed ordering, changed precision or shapes fail closed.
- Actual sampler row 0 is bound to API emission 2. Raw F32 prefill and verification
  rows are compared bytewise, retaining changed-word counts and max finite delta.
  Prefill bytes must remain identical. Verification differences are reported, not
  hidden or converted into probability/retokenized-ID comparisons. Emission 2 can
  differ across roles; the common logical prefix requires the full prompt and
  common emission 1, not equal outputs from the row being diagnosed. This does not
  establish equality of internal operator operands.

The old phase r3 reproduced same-role outputs; its prefill matched **524,288 bytes**
while all **131,072** verification words differed (max about **0.9920769**). Its router
accumulator marks were zero because the generic guard was bypassed. Those are prior
observations, **not** this candidate's expected result or pass evidence. If the new
actual marker fails, preserve evidence and investigate strict Hadamard/LoRA operand
identity; do not weaken the predicate or substitute native/test-only marks. Route
selection does not prove internal operand equality or identify a causal operator.

## Evidence and restoration

Retain `trace-short55-r1/{RCT-A,RCT-W3}/`: argv/PID/libraries, complete inspected
records, raw startup/server/restored logs, ready/restored native files, exact 23-job
selection, all HTTP/template/tokenize evidence, trace reports, bounded short-diagnostic
prefix/after/fresh log windows, and untouched `capture/row-*` F32 rows. Keep
`independent-restoration.json`, `analysis.json`, `first-frame-comparison.json`, top-level
logs, and CPU scratch including failures. No new receipt, trace schema or protocol.

Inherited finally/signals stop/preserve experimental containers and restore the exact
original ID/config/mounts/source/binaries/DRM/hardware/health/READY smoke, including on
failure. A separate live restoration check precedes raw-result analysis. Never
normalize actual records or delete a failed attempt. No timing or promotion claim is
permitted. The lead must execute this new-candidate journey; preparation is not proof
that wide B70 speculative verification is fixed.
