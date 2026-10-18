# Nemotron 3.5 Bolt LiLiCorr width 3/7 sweep — preparation declaration

**Preparation only.** This directory defines a lead-only, development-tier experiment. It contains no live result and makes no speed, fidelity, deployment, or promotion claim. The worker performed no SSH, Docker, native, GPU, API, benchmark, helper-test, or hardware execution.

The user goal remains verbatim: **after width1 was slow, user asked “why only width 1”, “then why did you stop”, then “... why are you still sitting here the”. We must continue toward faster speculation, not declare done at the first failed width.**

## Scope and frozen inputs

Only `results/20261018-nemotron35-bolt-lilicorr-width-sweep/` is changed. The prior width-1 case `20261017-nemotron35-bolt-lilicorr-width1-dev-timing/` is frozen and passed (`5344dbd0`); its diagnostics-off helpers and references are reused read-only. No prior artifact, runtime source, launcher, model weight, hardware setting, or global configuration is edited.

The candidate source is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`. Current source facts are pinned before any launch: `common/speculative.cpp:1296-1350` sets `n_draft=params.n_max` while `n_block_tokens=is_lilicorr ? block_size : ...`; the actual C1 startup has `dflash.block_size=8`, `lilicorr.candidate_topk=8`, `n_max=1/min=0/p_min=0`. Therefore a full eight-position drafter block is already performed at width 1 and the usable draft limit is 7. Widths **3** and **7** are bounded screening sizes (nominal verification 4/8); they are not claims that the native wider output dimensions are draft depth.

The native router’s five widths are output dimensions 240/304/368/432/496, not draft depths. The single2 workaround applies only to n2; n3 remains ineligible. Native coverage is not treated as wider-depth public-fidelity evidence. Historical max4/max1 token-116 failures are not conflated with this current fixed-width-1 source.

The exact frozen requests and outputs are read from the accepted diagnostics-off A0 references:

- all 20 quality prompts, `max_tokens=256`, natural stop;
- retrieval anchor, historical prompt length 8238, natural stop;
- short ordinary prompt length 55 and long ordinary prompt length 8270, each `max_tokens=128`, `ignore_eos=true`, streaming usage, finish `length`;
- untimed actual-ID diagnostics for both short and long ordinary requests;
- public `/apply-template` then `/tokenize` requests and exact frozen prompt IDs.

The current flags are `--spec-draft-n-max` and `--spec-draft-n-min`; the removed legacy `--draft-max` is forbidden. Candidate launch argv is derived from the immutable diagnostics-off helper and changes only the case-local name plus the intentional max value 3 or 7. Diagnostic flags remain off (`BOLT_TRACE_TENSORS=0`, `GGML_SYCL_DEBUG=0`, no capture mount, no `LLAMA_TRACE`, no `BOLT_VERIFY_CAPTURE_DIR`).

## Fresh external research declaration

Fresh external research was completed before implementation. Pinned source/code facts above take precedence over mutable documentation.

- <https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md> — documents real DFlash verification/acceptance statistics; this case records exposed proposal/acceptance counts but does not compare them for output equality.
- <https://github.com/ggml-org/llama.cpp/pull/22105> — trained-block behavior clamps the effective block/verification work; this is why width 7 is bounded by the observed block size rather than an assumed native width.
- <https://github.com/ggml-org/llama.cpp/pull/30506> — LiLiCorr candidate top-k is distinct from draft depth and includes a max-7 example; it supports screening 3/7 without treating router output dimensions as depth.
- <https://github.com/ggml-org/llama.cpp/pull/22397> — speculative flag refactor; use the current `--spec-draft-n-max/min` interface and reject removed aliases.

## Phase 1 — QUALIFY (lead only)

`run-qualify.py` creates the exclusive `qualification-r1/` directory (`exist_ok=False`) and screens exactly fresh `A`, `W3`, `W7`. It never resumes, overwrites, retries, or silently drops a request. A is the same candidate runtime with no draft. W3/W7 use only the intentional max flag change.

Every cell preserves native/startup/static-buffer logs, command output/errors, actual PID1 argv and selected PID1 environment, five-binary/source hashes, hardware/capability snapshots, root DRM ownership, thermal/frequency/energy mapping, and the 4 GiB reserve plus 1 GiB evidence allowance. Docker mounts are compared as complete canonicalized objects: order is normalized, but all fields and multiplicity remain.

A fresh A must reproduce frozen A. Each width must pass functional behavior **and** exact text/finish/usage against matched no-draft A for every quality, anchor, and ordinary case. Diagnostics compare actual emitted IDs from `choices[0].logprobs.content[*].id`, pieces, text, finish, and usage against frozen A; proposal/acceptance counters and probabilities are explicitly not equality gates. No raw F32 equality or retokenization of output is claimed. Full divergent API output and first differing actual token ordinal are retained.

Natural-stop checks come from all 20 quality prompts plus the anchor. Fixed-length checks come from both ordinary `128`/`ignore_eos` requests. Candidate qualification also requires diagnostics to exercise a real rejected draft (`draft_n_accepted < draft_n`); no synthetic acceptance is invented. If a behavior is not exercised it is reported and the width is not eligible. A valid fidelity mismatch disqualifies only that width and screening continues. Native/source/binary, lifecycle, isolation, hardware, transport, or API-corruption failures abort the phase and restore.

After every success or error, including interruption, all created candidates are stopped and preserved, and the exact accepted ORIGINAL is restored. Restoration requires identical `Id`, `Config`, `HostConfig`, and complete mount objects, READY smoke, health, sole live container, sole DRM ownership, unchanged hardware/caps, and unchanged source/five-binary hashes. `analyze-qualify.py` performs a fresh live restoration check before writing `qualification.json` and the qualified-width list.

If neither width passes, phase 2 is blocked and no timing request is launched.

## Phase 2 — TIME-ELIGIBLE (lead only, only after independent qualification restore)

`run-timing.py` first verifies the restored ORIGINAL live again, then creates the separate exclusive `timing-r1/` directory. It filters **only** by the qualified widths while preserving the predeclared balanced order:

```text
A1, W3a, W7a, W7b, W3b, A2
```

A1/A2 are fresh no-draft A loads. Each selected cell runs both frozen short55 and long8270 ordinary requests, one warmup plus three measured requests per point: six measured samples per arm/point over two fresh loads. Warmups are never entered into statistics. Before each timed cell, the full quality/anchor/ordinary/diagnostic gates are repeated. Any failed timed response aborts and restores; it is never rerun or dropped.

The stream timing boundary and telemetry semantics reuse the prior accepted helper: timing starts at request open, records first nonempty content, requires final usage/finish, retains raw server timings and raw energy/frequency/temperature samples, and always stops the sampler. No raw native captures, profile hooks, or invented fields are enabled. Existing speculative duration fields are included only when actually present in preserved server logs. `analyze-timing.py` runs after restoration and independently reports per-cell and per-arm median/IQR and deltas versus A; it computes no confidence interval or significance and permits no promotion.

## Remote CPU fixture gate (lead must stage and execute)

The CPU fixtures are recorded/synthetic only and are intentionally not executed in Pi. They cover launch-diff isolation, per-width PID/startup validation, frozen payload/reference preservation, quality and actual-ID mismatch rejection with first difference, width filtering/order/counts, complete-object mount comparison, warmup exclusion, stream/usage/TTFC/error evidence, source/binary preservation, interruption cleanup, and all cleanup attempts after an individual stop failure. An audit hook rejects subprocess, socket, `/proc`, `/sys`, and DRM access during the tests.

After staging this directory at the exact remote path, the lead must execute this exact command; a nonzero result blocks live work and its attempt remains preserved:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no \
    -o ControlPath=none -o ForwardAgent=no mike@inference-host bash -s <<'REMOTE'
set -euo pipefail
R=/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width-sweep
cd "$R"
bash "$R/cpu-checks.sh"
REMOTE
```

The lead then owns the real public-boundary sequence, only if the CPU gate is green:

```bash
cd /home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-width-sweep
python3 -B run-qualify.py --lead-run-qualification 2>&1 | tee -a qualification-controller.log
python3 -B analyze-qualify.py --lead-after-qualification-restore 2>&1 | tee -a qualification-analysis.log
python3 -B run-timing.py --lead-run-time-eligible 2>&1 | tee -a timing-controller.log
# only after timing restores:
python3 -B analyze-timing.py --lead-after-timing-restore 2>&1 | tee -a timing-analysis.log
```

The result remains development tier and is not a full publication sweep: no BetterBench, vLLM benchmark, long-context sweep, CI/significance claim, raw-vector claim, or deployment change is authorized.
