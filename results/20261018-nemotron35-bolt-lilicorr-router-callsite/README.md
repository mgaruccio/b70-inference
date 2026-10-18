# Nemotron target-router callsite correction — preparation only

**Development preparation, not a benchmark or qualified fidelity fix.** No build,
GPU/model execution, service change, timing, or promotion was performed here.
The phase intervention was inactive on the observed target path; this package
only connects its existing strict predicate to that path.

- Parent: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-verify-phase-r1`
- NEW candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-callsite-r1`
- NEW artifact root: `/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite`

## Only source delta

`router-callsite.patch` adds 11 lines in **`src/models/nemotron-h.cpp`**, immediately
between `build_lora_mm(model.layers[il].ffn_gate_inp, cur)` and its router callback.
It requests F32 accumulation only for the same Nemotron MoE architecture, F32
weight/input/output types, direct MUL_MAT operand identities, **3 <= columns <= 8**,
columns equal to graph `n_tokens`, and `n_outputs == n_tokens`.

N1/N9 and zero/one-output checkpoint prefill remain unmarked by this insertion.
This is not a universal prompt/verification classifier: a small full-output
prompt can also satisfy the inherited condition. The old generic guard stays.
Backend dispatch/native tests, MTP (`nemotron-h-moe.cpp`), precision policy,
Q8/IQ2/SSM math and launch configuration are unchanged.

### Source evidence and identity limitation

The lead's fresh external fetch on 2026-10-18:
<https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/nemotron-h.cpp>
confirms direct router construction followed by passing `router_logits` into
`build_moe_ffn`. Actual private-parent reads confirm this at lines 311/326;
`src/llama-graph.cpp:2171` guards only the `probs_in == nullptr` branch.
Upstream research locates the callsite; the private parent governs the patch.

Actual `build_lora_mm` (`llama-graph.cpp:1615`) first calls
`build_hadamard_input`. With no rotation registered for this weight it preserves
`cur`, then constructs a direct MUL_MAT. This call supplies no scale (`w_s`
defaults to nullptr); the F32 weight does not enter the NVFP4-only branch.
A registered Hadamard transform changes input identity; an applicable LoRA wraps
the result in ADD. Those cases intentionally fail the strict guard. Source reads
do **not** prove the loaded model's rotation/adapter membership. Do not widen the
predicate to hide that ambiguity: prove the effective mark and route at runtime.
The caller uses `cur`, not `inp_latent`; latent-down construction does not replace
`cur`. The graph calls this target FFN before its last-layer output-row cropping.

## Lead-only preparation/build commands (not run by this worker)

Copy only this package into the fresh artifact root. Use Bash on inference-host
(the login shell is Fish); SSH options used for read-only inspection were
`-o BatchMode=yes -o ConnectTimeout=20 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no`.
Do not execute model/benchmark runtimes in interactive Pi.

```bash
set -euo pipefail
ROOT=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-callsite
PHASE=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-verify-phase
cd "$ROOT"
python3 -B check-source.py | tee source-check-r1.log
# Run inherited CPU tests in NEW scratch, not in the old case's writable CASE dir.
mkdir cpu-inherited-r1
for f in test-trace.py analyze-trace.py run-trace.py build-candidate.sh router-verify-phase.patch; do
    cp "$PHASE/$f" "cpu-inherited-r1/$f"
done
python3 -B cpu-inherited-r1/test-trace.py 2>&1 | tee cpu-inherited-r1/tests.log
# Review first. This is the actual new private candidate build, never the parent.
bash build-candidate.sh
python3 -B "$PHASE/run-trace.py" --print-native-command > native-command-r1.sh
```

The builder is narrowly derived from the phase builder and reuses its read-only
protection helpers/manifests. It checks accepted, reference-parent (`widths-single2`),
columns8, **and verify-phase-r1** source maps and all six binaries before/after,
including on failure. It refuses existing candidate/attempt paths, applies with
fuzz=0, verifies the entire non-build source map differs only at the target file,
and retains the inherited **4 GiB reserve plus 1 GiB evidence allowance**.
Recheck capacity before live tracing; this is not an unlimited evidence budget.

The Intel 2026 image is pinned to the digest observed in the phase build's
`build-attempt-r1/image.json`. The inherited disposable-container apt bootstrap
still uses networking; the dependency precheck is offline (`--network none`).
No image pull, GPU mount, host install, reconfiguration, or service operation is
added. `/src` and read-only oneDNN mounts, six-artifact `ldd` checks, and
`cmake --build /src/build-sycl --config Release -j6 --target llama-server test-backend-ops`
are retained. This is a repeatable recipe, not a claim of bit-reproducible apt.

Evidence stays in the NEW root: `build-attempt-r1/{build.log,image.json,exit-code.txt}`,
`source-check-{parent,candidate}.log`, `source-diff.json`, `runtime-precheck.log`,
`post-build-verification.log`, and `{parent,candidate}-binaries.json`. Actual six
binary hashes are saved **before** requiring that only `libllama.so.0.7.0` changed.
Unexpected relinks fail the gate; no candidate hash is guessed or copied forward.

## Required lead E2E after integration

Use the inherited [native/N9 process](../20261018-nemotron35-bolt-lilicorr-router-verify-phase/README.md)
and [public journey/settings](../20261018-nemotron35-bolt-lilicorr-router-verify-phase/qualification.md),
with a fresh attempt and the NEW candidate mounted at `/src:ro`. The phase public
adapter has old hardcoded paths: **do not run it unchanged and call that a new
candidate test**. Lead owns the bounded runtime adaptation and live orchestration.

1. Require the new build/dependency/source gates and inherited CPU gates. In the
   inherited exclusive-device window, run the printed native command with the old
   analyzer case mounted read-only at `/router-verify-phase` and NEW `/capture`.
   Expected case/record/split counts remain 30/70/5, 61/402/35, 2/4/0. These native
   tests construct their own marks; they do not prove this target callsite works.
2. Retain the separate seeded N9 unchanged-path control against the existing
   reference and new candidate, with fresh output paths. Previously the two
   device outputs matched **17,856 bytes**; rerun rather than inherit that pass.
   The inherited N9 CPU-tolerance error remains an error, not a relabeled pass.
3. Exercise unchanged real template/tokenize/chat boundaries, frozen requests and
   settings, fresh A/W3/W7, actual emissions and real rejections. Capture actual
   target routes: explicitly show F32 marking **and per-column singleton routing**
   on W3 verification, and unmarked N1/N9 plus zero/one-output checkpoint prefill.
   Check unchanged prefill bytes and same-role traced/untraced reproduction,
   including the ordered short55 history; compare wider outputs to fresh/frozen A.
4. Preserve raw requests/responses, traces, logs, maps, failures and stopped
   experiments in the new root. Independently verify original config/mounts,
   hardware, health and exact accepted-service restoration after each window,
   including failure. Do not delete old evidence or change persistent launchers.
   No timing or promotion is authorized.

## Worker close gate observed

- `bash -n build-candidate.sh` passed; both embedded container shell bodies also
  passed `bash -n` without execution. Python `compile(..., 'exec')` passed for
  `check-source.py` and all three builder heredocs (no runtime imports locally).
- On inference-host (Python 3.14.7), streamed read-only CPU/source checks passed:
  exact target insertion before callback, full predicate equivalence to the
  actual parent's generic guard, precomputed-router caller, and nine rejecting
  mutations (width1/9, graph/output count, architecture, type, both identities,
  MTP path). `check-source.py` reproduces these checks without writing source.
- `git -C "$PARENT" apply --check --verbose -` and
  `patch --directory="$PARENT" --fuzz=0 --dry-run --batch --forward -p1`, both fed
  the patch on stdin, passed without offsets/fuzz. No private source was written.
  The builder's exact read-only `check_protected` heredoc also passed remotely
  (`BOLT_VERIFY_CALLSITE_PROTECTED_PASSED`), covering all four source trees/24 binaries.
- Patched source SHA256 computed in memory:
  `a3858c2c00cac43665f1fb22824a39ec7f94190bb10f77fe51cf8e3630d6bd38`.
  This is **not** a built library hash or runtime-route/fidelity result.

Full C++ build, inherited CPU/native/N9 reruns, candidate artifact hashes and public
E2E remain pending lead execution. No claim that wider verification is fixed yet.
