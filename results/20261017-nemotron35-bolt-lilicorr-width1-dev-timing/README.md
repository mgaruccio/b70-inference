# Three-arm width-1 development timing — pre-run declaration

**Preparation only; real benchmark pending.** This frozen declaration is not a results claim. User authorization: **“Run development benchmark”**, limited to the balanced six-cell experiment below. Prior fidelity artifacts remain historically **no timing**; no retrospective use of their timing fields. No deployment, promotion, publication, runtime rebuild/patch, hardware change, installation, or new service is authorized.

## Tier, baseline and intentional differences

Development component probe, not community-comparable or section-7 compliant. No full context sweep, unique-per-run prompts, BetterBench, vLLM benchmark, peak memory, formal confidence interval or significance claim.

- **O**: restart the exact protected `nemotron-bolt-probe`, ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, accepted `llama.cpp-bolt-b6334e37-nvfp4-ssm-prefill-r1`. Never recreate or alter its Config, HostConfig, mounts, source, command, environment or launcher. Only its existing native shell gates run, derived from actual saved/live command and the accepted runner; not candidate router30 tests.
- **A**: existing `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`, true no-draft, natural K=1.
- **C**: same five-width runtime, active LiLiCorr/DFlash max1/min0, natural K=2.
- **O→A is a runtime/source/compiled-hook bundle**, A→C narrows drafting cost, O→C is the overall bundle. Do not attribute all gains to LiLiCorr. Accepted oneDNN repeatability limitations remain: O greedy text identity is observed, not required or promised.

Exactly **O1,A1,C1,C2,A2,O2**. Each cell runs **both** short55 and long8270, in that order. Each point gets one warmup then three measurements: six measurements per arm/point, 36 measured requests total, two fresh loads per arm. Not twelve separate cells. Four new stopped-preserved containers use prefix `bolt-lilicorr-width1-dev-timing-r1-` and suffixes `a1,c1,c2,a2`.

A/C creation calls diagnostics-off `contracts.make_create` and changes **only the name**. Existing native385 + OUTPROD190/support9 + IQ2six + latent7 + attention13/view6 + router30 gates remain. Native `/tmp` flag is unset before serving, as is LLAMA_TRACE; tensor0/debug0; no capture mount. All numerical flags, legacy SSM, OUTPROD, deterministic oneDNN, IQ2 pairs, Q8/head/router controls stay unchanged. Compiled getenv/branches are still present; this is not a zero-overhead binary.

## Public-boundary test process

Environment: remote `mike@inference-host`, existing dependencies, model/drafter, runtime sources and artifacts. No workload/dependency/runtime executes in interactive Pi. Remote root is `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing`; exclusive fresh `run-r1`, no resume or replacement. CPU scratch stays under `worker-scratch/`.

Preconditions: green frozen CPU fixtures; historical diagnostics-off `analysis-r1/analysis.json` actually passed; full source maps and five accepted/candidate binary pins unchanged; exact original healthy and sole live container; boot/boost0/16×3801000 CPU caps/275W match prior verified state. Pre-cell/request free space must retain **4 GiB reserve +1 GiB conservative evidence allowance**, without cleanup. Source/weight metadata is read-only.

Every cell, before measurements:

1. Record pre-cell hardware, telemetry mapping/temperature and ownership. O1/O2 are exact restarts, never recreation. Stop and read the original's stdout/stderr log streams twice while stopped; demand stable/drained per-stream prefixes. Only their appended suffixes count as fresh restart evidence; old accumulated native passes are not proof. Startup/native gates/readiness are excluded from timing.
2. Record native/startup, immutable argv/container, allowlisted actual PID1 diagnostic environment and source-appropriate argv, loaded-library paths, and available static model/KV/recurrent buffer lines. Candidate PID checks are never applied to O with candidate argv assumptions.
3. Use the unchanged quality20 payloads (`max_tokens=256`) and 8238-token retrieval anchor. Each own `/apply-template` and `/tokenize` request preserves exact helper fields and confirms saved prompt IDs/usage. All quality checks and exact `BOLT-7319-CEDAR` retrieval must pass, including O. Record O quality text variation without forcing candidate identity.
4. A/C only: two diagnostic requests, short and long, outside timing; check actual emitted `choices[0].logprobs.content[*].id` against respective diagnostics-off references, verbose IDs, pieces, text, finish and usage. No raw-vector equality or probability-fidelity claim; no ordinary-output retokenization passed off as actual IDs.
5. Send **exact saved ordinary** short55/128 and long8270/128 payloads: temperature0, seed42, cache_prompt=false, ignore_eos=true, streaming with final usage. No nonce, new prompts, response_format or logprobs changes. One warmup per point, then three measured requests. Every response must finish length, output128, cache_n0, predicted_n128, exact prompt count. A/C ordinary text/finish/usage must match known diagnostics-off reference (and hence diagnostic); O may differ if its own quality passes.
6. Record post-cell hardware/ownership/temperature/source hashes/PID environment/logs; stop candidates and preserve them. Partial/failed responses and failed command output remain. No inference retries, replacement measurements, outlier removal, or retry-until-match. Any gate or measured-response failure aborts and restores, never selects only good samples.

Repeating frozen prompts intentionally fixes prompt mix; cache=false plus response cache_n0 is verified. This does **not** claim unique-content cold long-context performance or the section-7 sweep.

## Timing and research

Fresh external research supplied before implementation:

- https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events — include_usage final chunk, potentially missing on interruption; incomplete streams fail while preserving bytes/events.
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html — streamed chunks can contain multiple tokens; token-increment normalization is needed for true ITL.
- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/bench/speed-bench/README.md — server prompt/decode and client end-to-end metrics are distinct.

Pinned b633 source, not new upstream, controls interpretation. Read remotely: `tools/server/server-common.cpp:84–102` serializes prompt_ms/per_second, predicted_ms/per_second and draft proposal/accepted counts. `server-common.h:395–470` uses absolute microsecond timestamps, `t_gen_last - t_prompt_last`, `/1000` milliseconds, and **n_gen-1** decode steps. `server-context.cpp` invokes drafting/acceptance in the generation loop (candidate lines3645/4758); `source-contracts.json` records both pinned sources, hashes and exact line excerpts. Server prediction time includes drafter+verification wall, **not GPU-kernel-only** or draft-only time. Preserve every returned timing field.

`stream_chat` follows `probe-api.py::chat` monotonic timing semantics but accepts the exact frozen body and retains partial SSE (the old helper rebuilds payloads and loses partial responses on exceptions). Elapsed is from HTTP start through complete stream consumption; request/evidence persistence is outside client timing. **TTFC = time to first nonempty delta.content** (legacy TTFT), not necessarily first actual token. `(N-1)/(elapsed-TTFC)` is **client first-content decode proxy**, not ITL/token arrival timing. `N/elapsed` is separately labeled end-to-end output rate. Primary server prediction metrics, prompt metrics, TTFC, elapsed, both client rates and proposals/acceptances are retained.

Analysis is a separate process **after restoration and a fresh independent live original health/hardware/source/five-binary check**. Report raw rows, per-cell and per-arm medians/IQR, absolute and percentage deltas O→A/A→C/O→C. Quartiles: Hyndman–Fan type7, linear interpolation at `(n-1)*p`. Six samples per arm/point arise from only two process loads and are correlated; descriptive only, no CI/significance. Invalid/incomplete runs retain all rows and suppress aggregate comparisons rather than dropping failures.

## Telemetry, isolation and restoration

Reuse existing `/home/mike/inference/launchers/glimmer-phase0-instrument.py::GpuSampler`, hash-frozen. Read-only 0.2s Xe energy1/2, GT0/1 frequency and temp2. Validate card0 PCI identity/vendor/xe driver and hwmon2 ancestry/name against B70 at real preflight; do not guess, repair or install. Missing counters/frequencies are unavailable. Missing thermal observation or inability to verify B70 mapping/isolation is a bounded blocker, not permission for infrastructure changes. `sudo -n fuser -v` snapshots **all** DRM card/render nodes with root visibility; only expected model PID is allowed, or none while unloaded. Docker-only state is not proof of no host GPU clients. These are snapshots, not continuous isolation proof; no new monitor service.

Sampling start/stop is outside the client interval and stop runs in finally. Raw samples, sampling-window duration, actual sample span, energy J, mean watts over sample span, both frequencies and temperatures remain. Window includes response persistence overhead, unlike client elapsed. Any negative counter delta is labeled unavailable reset/wrap; unknown modulus is never guessed. Insufficient/missing samples are unavailable; undetected whole-counter wraps remain a caveat. No peak VRAM, occupancy or bandwidth measurement is established. Static startup buffer lines are not peak VRAM.

No enforced thermal equilibrium, temperature waiting loops or clock changes. Balanced order plus pre/post observations and measured-window temperature/power reveal drift; no constant-temperature claim. Record configured caps each cell and run, model/drafter revisions/historical verification, exact argv/env, kernel/driver, source IDs/maps and five binaries. No secrets/full PID environment dump.

Finally, on success/error/SIGINT/SIGTERM/SIGHUP: stop/preserve candidates; start original only if not running; always verify same ID/Config/HostConfig/mounts, READY/stop, health, sole-live, source/binaries and hardware. Healthy O2 is not needlessly restarted. Every cleanup action is attempted even if another fails. SIGKILL/host failure cannot be caught; no invented recovery service. No artifacts or containers are removed by the harness.

## Commands and retained evidence

Use explicit local and remote Bash, with SSH options exactly as below. The lead runs long commands through its existing background tool, not foreground Pi Bash. This worker has no `bg_run`; it prepares the bounded CPU runner and asks the lead to execute it **before commit**.

```bash
# CPU-only (<=180s + kill grace); does not use Docker/GPU/native/API/hardware.
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host bash -s' <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing
bash "$R/cpu-checks.sh"
REMOTE
```

CPU fixtures cover actual frozen references, candidate-only rename, distinct native contracts, fresh restart boundaries, unchanged originals, exact order/counts, warmup exclusion, SSE/usage/TTFC/units and interruption, counter missing/reset labels, quality/fidelity failure, type7 quartiles, signal/error restoration and stopping all candidates despite individual cleanup failure. Synthetic clocks/API/samplers/commands; source/binary hashing is read-only. An audit hook forbids subprocess/native execution, sockets, sysfs/proc/device reads in tests. Failures remain in unique scratch attempts. Only green exact code/helper/reference hashes freeze into `preparation/tested-code.json`. README is hashed and must not be edited after green; write actual preparation results to unhashed `preparation-results.md` and future real results to `RESULTS.md`.

Lead-only real run, same explicit SSH/Bash transport (no worker execution):

```bash
set -euo pipefail
cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing
(set -o noclobber; : > controller-r1.log)
python3 -B run-timing.py --lead-run-six-cells 2>&1 | tee -a controller-r1.log
# Only after the above restores; independent live checks precede statistics:
(set -o noclobber; : > analyzer-r1.log)
python3 -B analyze-timing.py --lead-after-restore 2>&1 | tee -a analyzer-r1.log
```

Retain commands, code/reference hashes, CPU attempts, controller/analyzer failures, environment and model provenance, original/candidate configurations, per-stream native/server logs and original prefix hashes/lengths, public requests/responses, ordinary/diagnostic HTTP bytes/events/results, raw telemetry, capacities, restoration and `run-r1/analysis-r1/analysis.json`. No binaries, runtime-source copies, weights or raw logits in Git. Publication checklist deliberately incomplete. No deployment/promotion/publication result is implied by green CPU tests.
