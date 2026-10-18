# Router columns 2..8 — experimental preparation only

**Development tier, quality-sensitive, not a benchmark result or a proven fix.**
Goal: keep working toward faster speculation **without weakening fidelity**. No
launcher/service change, timing run, promotion, new observer, or public controller
is included. CPU tests, build, native tests and the public API journey are **not
executed by this worker**. The lead owns all remote execution and restoration.

## Motivation and declared difference

The supplied live evidence is narrower than a router diagnosis: prior active W3
and W7 passed 20 quality checks plus the retrieval anchor, but actual emission IDs
diverged at short token 116 / long token 14; timing was blocked. The same-width
suppression control (max3/min4/pmin0) retained `n_rs_seq=3`, four outputs and the
full block of eight. Its loaded drafter executed 504 generation calls / 11930.648
ms, proposing/accepting zero tokens. All four short/long ordinary + actual-ID
requests matched frozen AND fresh A exactly. The service was fully restored and
independently checked. This implicates active verification/rollback; it does **not**
prove router numerics causal. This patch is an isolated hypothesis test.

- Parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`.
- Candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1`.
- Accepted remains `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`.
- Runtime delta: eligible F32 router `MUL_MAT` input columns **2..8**, rather than
  exactly two. Draft max3 means target columns **up to 4**; max7 means **up to 8**.
  M=240/304/368/432/496 are router output dimensions, not draft depths.
- Existing helper names are intentionally retained. K=4096, exact weight suffix,
  op/flags/types, device/buffer bounds, layout, no-view, no-alias and no-reorder
  conditions remain unchanged. Op/precision metadata and original strides are
  copied unchanged. Two stack arrays have eight slots; loops visit only actual n.
  All columns are preflighted before any launch, with no queue/pool/reorder work;
  n=1 ends recursion, n>8 falls through, and there is no fallback after column 0.
  All unrelated dispatch and default-off observers are byte-identical to parent.

`source-hashes.json` records the two **freshly supplied parent files**, the exact
patched bytes and patch SHA-256; these are input pins, not execution evidence.
No full runtime copies are retained. Build and CPU fixtures check these pins,
not a reconstructed historical patch chain.

## Native coverage and its limits

The old `^router_single2=1,` fixture class, initialization and all 30 registrations
are unchanged. The derived fixture prefixes its vars with `router_columns8=1,`,
so the old **anchored** filter still selects exactly 30. The new anchored filter
selects 62: five M values times n=2..8 (35 positives), n=1/n=9 (10 controls),
unrelated name/offset view/padded view at n=8 (15 controls), and valid unsupported
M=256 at n=4/n=8 (2 controls). Views have valid backing allocations; no invalid
GGML metadata or alias operations are constructed. Both inherit the existing
`1e-7` NMSE CPU-reference tolerance and discriminating per-column initialization.

The printed native command enables existing `GGML_SYCL_DEBUG` **and**
`BOLT_VERIFY_CAPTURE_DIR` (the latter actually enables `BOLT_ROUTE`). The analyzer
requires all successful cases **and** actual outer `MUL_MAT_SYCL` / inner FP32
library pairs, including original strides, types, precision tags, device, context,
and exact operation counts. New positives require n singleton pairs; controls
require one unsplit pair. M=496,n=9 intentionally retains the existing oneDNN F32
threshold path; the other native cases use MKL F32. Expected totals are 404 route
records for the new 62 cases and 90 for the old 30 cases on this candidate.

**Integration detail:** old n=3 fixtures now take three singleton operations.
Keep the old 30-case prefix/count, but use this case's analyzer for their changed
route expectation; the parent's n=3-unsplit checker is no longer appropriate.
Do not drop any other inherited native gates. Raw logs are neither timestamp-
stripped nor normalized. Capture directly with `tee`, without Docker timestamp
prefixes. ANSI status coloring is recognized in place. CPU fixtures generate
only in-memory, explicitly synthetic parser inputs, never runtime evidence.

The existing native mechanism compares the same graph across SYCL and CPU with
NMSE and bytewise sentinels; it does not directly compare a device batched output
to a separately executed device singleton output. No new comparison framework
is added. Route records also do not expose column data pointers. Discriminating
inputs plus preserved bounds/offset code help detect wiring errors, but **neither
native tolerance nor route identity proves bitwise equality or public fidelity**.
The complete public gate below remains mandatory.

## Lead-only remote CPU/build commands

Stage only this case at
`/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8`.
Run the following **on inference-host via the lead's managed remote job**, not in
Pi's interactive environment. No dependency installation is needed for CPU tests.

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8
cd "$ROOT"
test ! -e "$ROOT/cpu-r1.log"
python3 -B "$ROOT/test-trace.py" 2>&1 | tee "$ROOT/cpu-r1.log"
bash "$ROOT/build-candidate.sh"
# PRINT ONLY; this does not execute native tests or start any container:
python3 -B "$ROOT/run-trace.py" --print-native-command
```

The eight CPU tests cover exact parent/patch/candidate hashes and patch scope,
old fixture preservation, all positive route counts, negative routing, metadata,
malformed/missing records, numerical failures and the native command's syntax.
A nonzero CPU result blocks build/live work; preserve its log, do not overwrite it.

Build mechanics match `20261017-...-router-single2/build-candidate.sh`: `cp -a
--reflink=auto` of the parent **including build cache**, identical `/src` paths in
every container, fuzz-zero patch, Intel 2026 image and `/opt/onednn`, disposable
apt bootstrap only, and `ggml-sycl test-backend-ops -j6`. No GPU device mount or
host install. The image must already exist. Source/candidate and container-name
collisions fail; existing/failed attempts are never reused. Free-space checks
reserve 4 GiB before copying, before building and after building. Require the
parent `/src` build cache rather than reconfiguring a fresh tree.

The build snapshots and rechecks accepted AND parent full non-build source maps
and five-binary manifests, including after a failed attempt. Candidate full
source comparison must differ only at `SOURCE_SCOPE`; the five-binary comparison
must differ only at `libggml-sycl.so.0` and `test-backend-ops`. All five binaries
receive an `ldd` precheck in a read-only, network-disabled, device-free container.
Retain `build-attempt-r1/{build.log,image.json,exit-code.txt}`, manifests,
`runtime-precheck.log`, `source-diff.json`, and `post-build-verification.log`.

## Lead-only native execution (never automatic)

After the CPU/build gates pass, the lead snapshots the accepted service/hardware,
stops the original, verifies exclusive DRM access, and uses the existing pinned
native-container mechanism. Mount candidate `/src:ro`, oneDNN `/opt/onednn:ro`,
this case `/router-columns8:ro`, and a **new** `ROOT/native-r1/` as `/capture` RW.
Execute the printed **inner Bash command** only inside that controlled window;
keep command/container/PID/environment/startup logs, errors and the unmodified
`native-r1/{router_single2,router_columns8}.log` files. Do not append to old logs.
The helper never creates a container, stops a service, touches DRM, or runs a
shell command itself. It has no run-hardware option.

After restoration, the two pure analyzer commands are:

```bash
python3 -B "$ROOT/analyze-trace.py" --tag router_single2 "$ROOT/native-r1/router_single2.log"
python3 -B "$ROOT/analyze-trace.py" --tag router_columns8 "$ROOT/native-r1/router_columns8.log"
```

Keep the 4 GiB reserve, all failures and candidate artifacts. Do not promote or
reuse a failed build. Cleanup/restoration belongs to the existing lead controller,
not to these preparation helpers.

## Predeclared public-boundary qualification (lead integration still required)

Reuse the existing `20261018-nemotron35-bolt-lilicorr-width-sweep/` controller and
frozen width-1 A0 payloads/references read-only; adapt only the case/source/binary
pins and new router native checker. Do not run the old controller unmodified
against its old candidate and call that validation of this patch.

1. Sequential fresh **A (no draft), W3 (max3/min0), W7 (max7/min0)** on this
   candidate. Preserve target/drafter, CPU threads, hardware caps, context,
   cache dtypes, sampler and all other arguments from the existing sweep.
   `cache_prompt=false`, temperature 0, seed 42. W3/W7 keep pmin0 and full block8.
2. Each arm gets **25 checks**: 20 frozen quality prompts (max256, natural stop),
   retrieval anchor (historical prompt8238, natural stop), short55/long8270
   ordinary requests (128 tokens, ignore EOS), plus the corresponding two
   untimed actual-emission-ID diagnostics. Exercise the real public template,
   tokenize and streaming completion interfaces, not a simulated controller.
3. Require exact prompt IDs, text, finish and usage, and actual emitted IDs plus
   pieces against frozen A **and fresh A**. No retokenized-output ID or probability
   equality substitute. Keep proposal/acceptance/rejection telemetry; use the
   existing real rejected-draft coverage rule, not synthetic acceptance.
4. Preserve raw requests/responses, actual PID1 argv/env, startup logs, telemetry,
   first divergence and failures. A valid fidelity mismatch rejects that width
   but continues bounded capture; infrastructure/corrupt-input failure aborts.
5. On every exit restore the exact accepted ID, Config, HostConfig and complete
   canonical Mounts, health and natural-stop smoke, sole DRM owner, hardware,
   full protected source maps and five binaries. Perform a **fresh independent
   restoration check before analysis**. Preserve failed candidates and evidence.
6. Only a width passing all 25 checks may become eligible for **separately
   declared** timing. Native success alone is not completion. No promotion.

## Research supplied before implementation

Fresh primary research was completed by the lead before this slice. Conclusions
used here (not causal proof):

- <https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/domains/blas/gemm> — FP32 GEMM is defined, but not bitwise identity across shapes.
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html> — strict FP32 does not establish accumulation-order identity.
- <https://uxlfoundation.github.io/oneDNN/dev_guide_attributes_deterministic.html> — repeatability is platform/implementation-scoped.

## Worker static close gate

Observed: linked worktree verified; `bash -n build-candidate.sh` passed; AST-only
parsing passed for all three Python files and all four inline build Python
blocks; `bash -n` passed on the **AST-extracted native command string**; JSON
parsing passed. No helper module was imported. `git apply --stat` parsed the
patch: two files, 49 insertions / 8 deletions. SHA-256 generation from supplied
and edited bytes matched the written patch hash. Staged whitespace checks passed
(normal files: `git diff --cached --check`; patch: the same with
`core.whitespace=-blank-at-eol` for unified-diff context blank lines). Temporary
source copies were removed; only these seven case files are staged. These are
**static checks**, not CPU fixture, C++ compile, native, public-API or performance
results. No local SSH, Docker, GPU, API, build or test execution occurred.
Publication checklist is incomplete; nothing here is standard-compliant
performance evidence.
