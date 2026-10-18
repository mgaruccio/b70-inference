# Bolt router-control operand observer

## Predeclaration — development diagnostic, observer only

Prepare ONE bounded observer experiment against the completed `router-single2` computational parent. Continue Bolt/LiLiCorr correctness/fidelity diagnosis before timing or promotion; the accepted runtime is immutable. This is not another arithmetic control, a performance comparison, a fidelity improvement claim, or a standard-compliant publishable benchmark. Repository preparation base: `420baa6e`.

The computational parent activated two actual singleton F32 router calls in all 67 C0 width-2 target batches, passed its six router native cases and inherited gates, and preserved S0's 128 IQ2-parent vectors. Nevertheless its first S0/C0 raw-score difference remained output 2 before rollback; the first emitted difference moved from 40 to 41 (S0 ID 3367 / C0 ID 1032), with 60/67 proposals accepted. Dispatch activation does not establish router-output equality. This observer directly measures that question without presupposing the answer or claiming a sole root cause.

### Paths and immediate-parent mapping

- Parent source: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1`.
- Candidate (lead creates): `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-control-operands-r1`.
- Parent result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2`.
- New result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-control-operands`.
- Container prefix: `bolt-lilicorr-router-control-operands-r1-`.
- Worker writes/execution: new result root's `worker-scratch/` only. Shared sources, weights, references and parent results are read-only.

`analyze-trace.py:parent_reference` deliberately maps:

- **S0** requests/responses/raw captures: `parentRoot/run-r1/S0`; validated alignment: **`parentRoot/reanalysis-S0.json`**.
- **C0** requests/responses/raw captures and alignment: **`parentRoot/continuation-c0/run-r1/C0`** (`alignment.json` within that cell).

The original failed S0 cell has no rewritten alignment and must stay that way. Binary manifests, protected-source data and source-diff data remain at the parent result root. IQ2 is historical context, NOT this observer's immediate parent.

Pinned parent artifacts (SHA256):

```text
llama-server              22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d
libllama-server-impl.so    43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede
libggml-sycl.so.0          b2b3b7950c471bb5658f8474eb1866157fd3311a183e07d03a665842a8cad3d9
test-backend-ops           5f43858b1cbd1083ced719a8700f93c30a0e68e94f307541f0bee9e43f5e8389
libllama-common.so         934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812
```

## Research and bounded implementation

Fresh official-source research supplied by the lead for this experiment:

- <https://github.khronos.org/SYCL_Reference/iface/queue.html>: commands on the same `in_order` queue are ordered as predecessor dependencies; this is not a cross-queue guarantee. `memcpy` returns an event; `wait_and_throw` waits for completion and delivers synchronous/asynchronous errors through prescribed handling. These guarantees do **not** establish floating-point invariance of instrumentation. Host storage lifetime is implemented explicitly, not inferred from the documentation.
- <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html>: GEMM describes queue, strides and computation, not a cross-shape bitwise-equality guarantee.

Reuse the proven six-capture observer from `../20261017-nemotron35-bolt-lilicorr-moe-operands/moe-operands.patch`. Only `ggml/src/ggml-sycl/ggml-sycl.cpp` differs from the NEW parent. The patch is identical to that observer except for two hunk offsets (+80 parent lines), so it applies with `--fuzz=0` without backups. Do not replace or remove the parent's router-single2 helper/dispatch. Native test source/binary and the other three non-SYCL binaries must remain byte-identical.

Capture **only** the first post-prompt target frame, output ordinal 2 / position 55:

- `blk.1.ffn_gate_inp.weight`: F32 `[4096,240]`, weight reorder 0; F32 `src1[4096,n]`, `dst[240,n]`.
- `blk.1.ffn_latent_down.weight`: reordered Q8_0 `[4096,1024]`, weight reorder 1; F32 `src1[4096,n]`, `dst[1024,n]`.

S0 captures full n=1 buffers; C0 captures full n=2 buffers. Copy full weight and src1 before the original `GGML_OP_MUL_MAT` computation and full dst after it, on the existing in-order consumer queue. Preserve checked byte spans, exclusive files, once-per-projection arming, and host vectors alive through `wait_and_throw` (and completion on the exception path). No tensor callbacks, oneDNN verbosity, alternate matmul, arithmetic changes, extra operators/blocks, new queue, or new runtime knob/protocol. The existing default-off `BOLT_MOE_OPERAND_DIR=/moe-operands` and diagnostic-only arm marker are reused, not generalized. F32 tensor `extra` is nullable.

The operand `reorder` field describes the **weight layout for every role**: router=0; latent-down=1 even for its F32 src1/dst. Each arm must produce exactly six files under its private `moe-operands/`: `{router,latent_down}.{weight,src1,dst}.bin`. Full weight spans are 3,932,160 and 4,456,448 bytes; full src1/dst spans follow n. Files stay remote.

### Route correlation and observer invariance

Preserve the corrected router-single2 independent-FIFO parser verbatim. `BOLT_VERIFY` and operand records share the FIFO common logger; direct-stderr `BOLT_ROUTE` has a separate FIFO. Docker can interleave these streams arbitrarily. Do **not** require route text between batch/decoded or operand-before/after text. Independently anchor target route groups to the unique block-0 SSM backend, then match exactly one-to-one with VERIFY frames by order and retained-stride width. Keep strict prefills `5@[0..4]`, `46@[5..50]`, `4@[51..54]`, no target prefix, no missing/extra/reordered/unknown leading frames or leftover target records. Keep foreign-backend drafter routes separate; backend context is not llama context.

For C0 the capture still brackets the FULL original n=2 operation (`ne1=2`). Its router executes two actual n=1 outer `MUL_MAT_SYCL` / inner `LIBRARY_MKL_F32` pairs, retaining original n=2 src1/dst `nb2/nb3`. Correlate exactly those two pairs: only their activation/destination `ne1` differs from the full capture; all other shapes/strides/types must match. This is not generic shape relaxation. Require router activation in every target frame, not just the observed one.

**BOTH S0 and C0 must preserve ALL 128 immediate-parent raw vectors byte-for-byte, prompt IDs, emitted IDs, ordinary/diagnostic text, batch inputs, positions and rows.** Any drift aborts attribution, including a last-vector C0 drift. `reference_equivalence_required=True` now deliberately records observer policy; the computational parent's C0 report-only policy is not inherited. Operand comparisons occur only after both full-arm invariants pass. Cross-arm raw-logit comparison still stops at and includes the first different emitted output, so all compared vectors have common prior output history. Compare full weights and token-0 activation/output columns at position 55, with bounded equality/difference reporting and no predetermined localization.

## Defined real E2E — lead only, after review

Environment/preconditions: `mike@inference-host`; original accepted `nemotron-bolt-probe` API port 18098, container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, sole running service. Hardware remains 16 CPU caps at 3,801,000, boost 0, GPU 275 W. Preserve original Config, HostConfig, all mounts, five binaries and protected sources. Require fresh candidate/result paths and disk reserve. No accepted/parent rebuild, launcher change, package/host/service/hardware reconfiguration or promotion.

The lead uses explicit local Bash and remote Bash (remote login is Fish), with:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host 'bash --noprofile --norc -s'
```

Supply the remote command through a here-document; use stdin from `/dev/null` for SSH commands that do not consume a script. Stage these six reviewed assets at the new remote result root, not in any parent. Run long steps using the lead's existing managed background runner, not an unbounded foreground tool call. In that root, the executable sequence is:

```bash
bash build-candidate.sh
python3 -B test-trace.py > lead-cpu.log 2>&1
python3 -B run-trace.py > lead-run.log 2>&1
python3 -B analyze-trace.py > lead-analysis.log 2>&1
```

Run each only after the preceding step succeeds. The builder clones the new parent into the candidate, pins upstream/source hashes, applies the observer with fuzz 0, and builds only `ggml-sycl` with six jobs in disposable `intel/deep-learning-essentials:2026.0.0-devel-ubuntu24.04` / IntelLLVM 2026, oneDNN 3.14 mounted read-only. Reuse the parent build cache/bootstrap; apt operations are inside that disposable image only, not a claim of zero container package changes. Dependency checks, source-scope checks, all five accepted/parent/candidate artifacts and post-build protected-source checks are mandatory; only candidate SYCL may differ.

The real system journey per arm:

1. Fresh native numerical/dispatch/state/canary gates: native 385, OUT_PROD 190 numerical / 9 support, IQ2 six, latent seven, attention thirteen, attention-view six, router six. Startup gates verify actual routes and frozen runtime settings before requests.
2. API quality **20/20** via the inherited `probe-api.py` public HTTP client.
3. Frozen 55-token prompt; forced 128 outputs, temperature 0, seed 42, cache false, ignore EOS. S0 max=1/min=2; C0 max=1/min=0; LiLiCorr `n_rs_seq=1`, K=2, capacity=2 and identical reservation. Q8 KV, F32 recurrent state, ctx 65,536, batch/ubatch 1,024, parallel 1, graphs 0.
4. Warmup ordinary stream, then ordinary stream; separate diagnostic HTTP request using real token IDs and full raw 131,072-F32 logit vectors. Arm only this diagnostic; retain exactly six operand files. Preserve failures without retries for agreement. Stop the temporary arm to flush logs before analysis; require within-arm text/IDs and BOTH-arm immediate-parent all-128 invariants before interpreting cross-arm operands.
5. Analyze route activation, raw-history cutoff and direct operand equality/mismatch at output 2 / position 55. No timing or promotion regardless of result.

Artifacts: `build-attempt-r1/{build.log,exit-code.txt}`, dependency/post-build logs, three five-artifact manifests and protected-source/source-diff metadata; `run-r1/{S0,C0}` requests/responses, ordinary events, native/startup/server logs, raw `capture/`, six-file `moe-operands/`, alignment and summaries; root `run-r1/analysis.json` plus original/restored container and hardware snapshots. Preserve error output and original failed evidence. Raw tensors, weights and raw logits remain remote; any later Git evidence is scoped metadata/lossless gzip logs only.

Cleanup is unconditional in the inherited controller: stop temporary containers, retain their logs/inspect evidence, restart the **same** original container, and verify unchanged identity/Config/HostConfig/all mounts, all five protected binaries/sources, hardware, health, READY and sole-running-original status. `analyze-trace.py` rechecks restoration. Do not create a continuation preemptively; only an actual future failure could justify one.

## Worker preparation results (not real observer E2E)

Executed CPU/parser/source/patch checks only in `newRoot/worker-scratch`, through the SSH/Bash boundary above. No Docker build, GPU/model inference, API calls, service/hardware changes or accepted/parent mutation. Tested local assets are byte-identical copies of remotely generated/tested files. Environment: Linux `7.2.0-1-cachyos` x86_64, Python 3.14.7, GCC 16.2.1, GNU patch 2.8, Bash 5.3.15.

- **82/82 final CPU tests pass**, in four bounded invocations of `python3 -B test-trace.py Trace.<name> ...`, each with a 75-second cap. Exact cwd, complete commands, test names, outputs and exit 0 are retained at `worker-scratch/checks/cpu-final-{00-16,16-38,38-60,60-82}.log` (16/22/22/22 tests). The four logs cover every final test exactly once. `source-environment.log` verifies coverage, syntax, environment and final asset hashes.
- Uses REAL saved S0 position-70 shifted routes and REAL C0's 67 two-singleton-pair frames with retained width-2 strides; checks missing/extra/reordered records and arbitrary FIFO-preserving stream interleaving. Full analyzer tests use real saved requests/text/IDs/raw vectors/routes; **only the six new operand records and buffers in those CPU fixtures are synthetic**. They do not establish live observer invariance or output equality.
- Source guards, inherited native parser gates, all-frame router activation checks and parent-mapping/full128 negative regressions pass. Actual parent C++ router preflight extraction compiles/runs on CPU: 85 exclusions, distinct columns, metadata, canaries and late-preflight/no-partial-launch checks pass. This is not a SYCL/MKL execution test.
- `patch-r1.log`: original observer reapplied with fuzz 0 at +80-line offsets, but GNU patch created `.orig`. `patch-r2.log`: only the two hunk positions updated; dry-run/apply now have no offsets/backups, and resulting source is byte-identical to the first application. Only `ggml/src/ggml-sycl/ggml-sycl.cpp` differs. Candidate scratch source SHA256: `45171817aba1360fdec8fc0460eef203f327533a5fdfc7df410e6f01644d410f`. Reversing the patch restores the complete parent source; router helpers/arithmetic/native source and corrected route parser remain unchanged.
- `source-environment.log`: accepted protected sources and both accepted/parent five-binary manifests match saved evidence; new parent source differs from its IQ2 ancestor only in its two pinned computational-control files. No parent evidence was rewritten.

Retained preparation failures: `cpu-r1.log` reached its 90-second cap after 33 passes; no semantic failure was reported before timeout. `cpu-r2-iq2-native.log` exposed an inherited whole-source-tail comparison incompatible with observer additions; replaced by the established observer's unchanged-IQ2-function assertions. `cpu-r3-observer.log` exposed the inherited C++ extraction helper's bool/void-only return-type pattern and an old expected container prefix; reused the prior observer's helper and updated the prefix. Corrected targeted reruns and the complete final partitioned suite are green. Earlier logs/exit files remain remote; nothing was retried to obtain model agreement.

**Pending with lead:** review/integration, real candidate build, fresh native/API gates, real S0/C0 captures, empirical both-arm observer parity, direct operand result and live restoration verification. No live instrumentation-invariance, router-equality, fidelity, quality or performance conclusion is made here.

## Lead-observed result

Executed on `mike@inference-host` with the declared SSH/local+remote Bash boundary: `bash build-candidate.sh`, then `python3 -B test-trace.py`, then `python3 -B run-trace.py` and `python3 -B analyze-trace.py`. The real SYCL build/dependency/protected-source/five-binary gates passed. All **82 lead CPU tests passed** in one complete invocation (`lead-cpu-r1.log.gz`). Both real arms completed on the first paired attempt; no continuation or inference rerun was needed.

Both arms passed native385 + OUT_PROD190/9 + IQ2six + latent7 + attention13 + view6 + router6, followed by API20/20, warmup, ordinary and diagnostic requests. **BOTH arms preserve all128 immediate-parent raw vectors byte-for-byte, plus prompt/IDs/text/batch inputs/selected positions/rows.** Observer attribution and full-pair restoration checks pass.

At first post-prompt position55/output2, token0:

- Router: full F32 weight3,932,160 bytes, input column16,384 bytes, and output column960 bytes are ALL byte-identical between S0 and C0. The previous213/240 unequal router outputs are now eliminated at this observed boundary.
- Latent-down: full reordered Q8 weight4,456,448 bytes, input column16,384 bytes and output column4,096 bytes are also ALL byte-identical.

This directly confirms that the router singleton control removed the observed router discrepancy. It does NOT establish whole-model fidelity: final raw logits still first differ at output2 before rollback, and emitted tokens first differ at41. Only these observed projections at position55/token0 are excluded as the remaining differing boundary; other operations/state are not thereby proved equal.

Candidate SYCL SHA256: `0a3d7ce7131e808bfaf8e8739416516d8a473fe447fa4bb0a7123af7e53e919e`. Other four binaries, including native test `5f43858b1cbd1083ced719a8700f93c30a0e68e94f307541f0bee9e43f5e8389`, are identical to the router-single2 parent. The accepted source/archive, original container identity/config/HostConfig/mounts/five hashes, hardware, health, READY and sole-running-original checks passed and were rechecked by the analyzer.

Canonical remote evidence: `run-r1/analysis.json`, both `alignment.json` files, and each cell's raw `capture/` and six `moe-operands/` files. Local evidence retains metadata and lossless gzip (mtime0) logs; JSON reports larger than1MB are also losslessly compressed (`run-r1/analysis.json.gz`, per-arm `alignment.json.gz`). Full raw tensors/weights/logits remain remote. Worker failures and final CPU checks are retained under `worker-checks/`. No timing, performance, quality or promotion claim.

Next investigation: identify safe complete-block-output capture points in the early model, including residuals and fusion paths, to localize the remaining divergence without changing arithmetic or using tensor callbacks. Computational parent remains router-single2, not this observer.
