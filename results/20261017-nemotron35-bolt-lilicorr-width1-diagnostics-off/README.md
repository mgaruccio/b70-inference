# Diagnostics-off width-one serving-fidelity preparation

**Development validation only; real diagnostics-off A0/C0 pair PENDING.** No timing comparison, standard-publication claim, promotion, higher depth, new build, source patch, persistent launcher change, or optional inference probe. This is preparation for the ongoing bounded Bolt LiLiCorr experiment, not a new runtime or evidence service. Completed cases and helpers remain read-only. This README is frozen with tested code; put preparation outcomes in `preparation-results.md` and real outcomes in `RESULTS.md`, not here.

## Immutable runtime and intentional difference

Remote root R is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off`. Only R and R/worker-scratch are writable remotely. Local changes are only this results directory. The worker prepares code and runs remote CPU fixtures only; **the lead runs the real pair**.

Reuse `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1` unchanged. `run-a0.py` in the completed no-draft case supplies all five binary pins and complete candidate/accepted 3805-file source maps. Five candidate SHA256s:

- llama-server: `22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d`
- libllama-server-impl.so: `43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede`
- libggml-sycl.so.0: `64d6a58f17cb92ea9fe6354d9c553f74724565a58fc018d2f4bbd013d090a72e`
- test-backend-ops: `bdd86349fac5ab43fa32d465aad27cec7f31dcad0651f7611403d8a783f10c7d`
- libllama-common.so: `934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812`

A0 is true no-draft: natural target K1, n_rs_seq=0, output capacity 1, no drafter/features. C0 retains the exact active draft-dflash max1/min0 tail, natural target K2, n_rs_seq=1, output capacity 2 and features, separate drafter context. No metadata equality or shape forcing. Context65536, batch/ubatch1024, parallel1, q8 KV, default F32 recurrent state, graphs0, SSM legacy, OUT_PROD, deterministic oneDNN, head MMVQ, IQ2 singleton pairs, Q8 latent/attention/views, and five F32 router-width controls are untouched.

Both arms preserve the accepted original `nemotron-bolt-probe`, ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, localhost:18098. Exact Config/HostConfig/mounts/source/binaries are protected. Hardware remains boost0, 16x3801000 CPU, GPU275W; runtime checks compare the existing exact hardware snapshot. No hardware mutation.

## Source-verified diagnostics boundary

Actual immutable source inspected remotely, not patched:

- `common/sampling.cpp:24-59,651-705`: nonempty BOLT_VERIFY_CAPTURE_DIR gates raw writes/logging; off passes nullptr to set_logits' raw-logit output. **Value `0` is ON**; unset is required.
- `tools/server/server-context.cpp:47-75` and event sites 3539/3891/4084/4431/4615-4628/4697-4809 use the same gate.
- `ggml/src/ggml-sycl/ggml-sycl.cpp:149-257`: BOLT_ROUTE uses the same capture gate and returns before formatting when absent. Generic GGML_SYCL_DEBUG is independent; its default at line560 is 0.
- `server-context.cpp:972-1052,1336-1341`: callback/readbacks require BOLT_TRACE_TENSORS=1; retained value is 0. Exact forbidden callback markers include BOLT_TENSOR_TRACE_ENABLED, BOLT_PROJECTION_REPEAT and common_debug_cb_eval.
- `server-context.cpp:1585-1589`: LLAMA_TRACE generic logging must be unset for serving. Lines4801-4805 set speculative probability to 1.0f with TODO probabilities; these numbers are not fidelity evidence.

The pure argv transform removes Docker's `-e BOLT_VERIFY_CAPTURE_DIR=/capture` and host `/capture` mount. It prepends `export BOLT_VERIFY_CAPTURE_DIR=/tmp;` to the **unchanged native shell commands**, whose route parsers need BOLT_ROUTE. Before the unchanged server marker/exec it replaces only the old diagnostic boundary with `unset BOLT_VERIFY_CAPTURE_DIR LLAMA_TRACE; export GGML_SYCL_DEBUG=0;`. BOLT_TRACE_TENSORS=0 remains. No native command, numerical environment value, server argv, or spec flag is changed. Backend native tests do not invoke the llama sampler, so /tmp is just a nonempty enabling flag, not a serving raw-capture destination.

All native gates must still pass before EACH server: native385 + OUTPROD190/support9 + IQ2six + latent7 + attention13/view6 + router30. Preserve all native evidence. Native BOLT logs before the marker are expected; after it reject BOLT_VERIFY/BOLT_ROUTE and active callback markers, while allowing normal INFO. Probe **actual PID1** before first public request and after last inference: argv must match the exec-ed llama-server, `/proc/1/environ` must lack capture/LLAMA_TRACE and contain tensor0/debug0. Filtering occurs inside the container; only those four diagnostic keys/statuses, PID/argv/exe and scoped `/tmp/row-??????` filenames leave the probe. Container Config.Env is not substituted for PID environment. No raw folder, watcher, or cleanup is created.

Residual getenv/branch instructions remain. This is a **diagnostics-off behavioral path**, not a zero-overhead clean binary. No fresh raw-vector or route-equivalence claim is possible or made.

Ongoing-task official research remains retained in the no-draft/long preparation: https://github.khronos.org/SYCL_Reference/iface/queue.html and https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/general-matrix-matrix-multiplication.html . This validation preserves the same runtime and arithmetic, with no new SYCL/oneMKL design or broad research task.

## Fixed public-boundary journey (lead only)

Preconditions: CPU gate green and its exact helper/reference hashes unchanged; fresh `run-r1` and names `bolt-lilicorr-width1-diagnostics-off-r1-a0` / `...-c0`; original healthy/sole-live/exact identity; source/five binary/hardware checks; at least 5GiB free. The controller freezes the payloads once, runs A0 then C0, and refuses reuse/overwrite.

Exactly **27 generation requests per arm**, 6016 total maximum output tokens per arm:

1. Same saved 20 quality requests from five-width S0/C0, unchanged max256 and functional criteria.
2. Same decode-profile `run-r1/baseline/long/anchor.request.json`: actual expected8238; exact stripped BOLT-7319-CEDAR, finish stop, actual usage prompt>=8192.
3. Short: same no-draft/five-width frozen 55-token payload for one streaming warmup and ordinary128; then existing diagnostic_payload variant (nonstream, verbose, return_tokens+logprobs/top5, backend_sampling=false).
4. Long: same `measured-00.request.json` used by the passed 8270-token long pair for one identical warmup and ordinary128; then its existing diagnostic variant. Never regenerate a nonce or use historical8269 warmup.

Before each arm's inference, save actual `/apply-template` and `/tokenize` exchanges for all27 payloads. Use full chat kwargs enable_thinking=false and explicit add_generation_prompt=true; tokenize rendered prompt with add_special=true, parse_special=true, with_pieces=false. Record IDs/counts, compare them post-restoration to usage, same-arm traced prompt IDs and cross-arm IDs. Short reference IDs are from saved alignment, NOT plain-user tokenization. Long IDs use the already validated public endpoints plus saved alignment. Historical55/8270/8238 are checks, not inferred counts.

Every HTTP call retains exact serialized request body, URL/method/headers, raw response body, status/headers and partial errors. No HTTP or generation retries. Response bodies have a conservative 4MiB safety cap; exceeding it fails closed with received evidence preserved. No fidelity normalization/comparison of A0 can block C0. Genuine native, environment, source, capacity or HTTP/public-contract failure stops capture and restores.

After both arms, or any exception/signal: stop and preserve candidates, restart original, require READY/stop, live health, sole-live identity, exact Config/HostConfig/mounts, all sources/five binaries and unchanged hardware. No deletion, cleanup, launcher editing, or promotion. Only THEN run the separate analyzer; it independently queries live original inspect/health and hardware, not merely stored snapshots. READY restoration is the existing original-server smoke, not another arm probe.

Analyzer gates: all quality20/anchor functional criteria; exact requests/text/finish/usage against saved same-arm long references (also five-width quality references) and cross-arm; ordinary and warmup saved-reference text/finish/usage; within-arm warmup/ordinary/diagnostic agreement. Short diagnostic compares to no-draft A0 or five-width C0; long diagnostic to long-fidelity A0/C0. Compare all128 primary emitted IDs to same-arm traced reference and cross-arm, preserving both sequences and first difference. Require primary token pieces/text and verbose IDs consistent. Ordinary flags are unmodified and no ordinary/quality/anchor actual-ID claim is invented. Probability/top-logprob values are ignored. API proposal/acceptance counts are retained only as diagnostics, not performance or route claims.

Artifacts: remote `run-r1/{A0,C0}/` payloads, public contracts, raw HTTP, native/full server logs, before/after serving-PID records, exact argv/container evidence; capacity/source/binary/hardware/restoration records; exclusive `run-r1/analysis-r1/analysis.json` and independent live-restoration check. Failure artifacts remain. Parser fixes never justify repeated inference; a failed exclusive analysis directory is not overwritten. No full source, binaries, weights or old raw logits are copied into Git.

## Space and execution

Keep 4GiB reserve **plus 1GiB evidence allowance** before each arm and every public/inference request, and record remaining free. Serving has no raw writes. Per arm, 81 HTTP responses (54 public +27 generation) at 4MiB cap are at most324MiB. Remaining allowance covers serialized requests/public IDs, diagnostic JSON, duplicate native/startup/full logs, analysis and filesystem overhead. CPU evidence records actual previous native/startup log sizes; prior traced logs include more instrumentation than this path. No prior captures are removed. Original lead observation was11016155136 free bytes; preparation records a fresh CPU filesystem measurement, not a hardware probe.

All execution is remote with explicit local/remote Bash and fixed SSH isolation options. Use the lead's background tool, not a foreground multi-minute wait. CPU command (worker has no bg_run; request lead execution before commit):

```bash
/usr/bin/bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host /usr/bin/bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off/cpu-checks.sh'
```

The runner has a180-second CPU timeout +10-second kill grace, unique retained `worker-scratch/cpu-*`, before/after code/reference hashes, read-only source/five-binary checks, synthetic public/HTTP/native/PID/environment/negative cases, and subprocess/network denial during tests. These CPU fixtures are **not real off-path E2E**. Successful small evidence goes to `preparation/`; all attempts/failures stay in scratch.

Lead real commands, only after preparation integration and approval to launch this pair (run each through background execution; analyzer only after controller exit/restoration):

```bash
/usr/bin/bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "/usr/bin/bash -lc '\''cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off && set -o noclobber && python3 -B run-off.py --lead-run-one-pair > controller-r1.log 2>&1'\''"'
/usr/bin/bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "/usr/bin/bash -lc '\''cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off && set -o noclobber && python3 -B analyze-off.py --lead-after-restore > analyzer-r1.log 2>&1'\''"'
```

No publication checklist is claimed complete: this bounded fidelity-only preparation is not a standard-publishable benchmark.
