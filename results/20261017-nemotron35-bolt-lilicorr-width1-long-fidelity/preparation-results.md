# Preparation results — 2026-10-18

**Harness prepared; real fresh 8K A0/C0 pair NOT RUN.** Development-only synthetic/fixture CPU evidence, not fidelity, timing, or promotion evidence. The lead owns integration and the real public-boundary process declared in the frozen README.

## Changed surface

Only this new results case, mirrored at `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity`:

- `contracts.py`: exact existing fixture/argv reuse, immutable runtime/source/native helpers, public chat-render/tokenize contracts, source-derived capture budget and refusal.
- `run-long.py`: lead-only, exclusive `run-r1`, one fresh A0/C0 capture pair, all requests retained, independent finally-restoration, no fidelity postprocessing between arms.
- `analyze-long.py`: independent post-restoration analysis, actual prompt N, strict large-prefill FIFO route gates, real width-one rollback history, full raw-vector comparison only through shared history inclusive of first differing emission.
- `test-long.py`, `cpu-checks.sh`: 15 bounded remote CPU regressions; no native/GPU/API/service execution. Synthetic 8K transformations are explicitly labelled.
- `README.md`: development declaration, source/research conclusions, fixed real test journey, exact lead commands, evidence retention, restoration and non-goals.
- `preparation/`: small raw CPU results, environment, tested-code hashes, source/binary checks and budget; first failed attempt retained under `failed-cpu-0cNCuUyF/`.
- This unhashed `preparation-results.md`: final observed preparation report. No hashed controller, helper, test, runner or README was edited after the successful freeze.

## Exact CPU execution and observed results

The lead ran the same command via its background tool (worker had no `bg_run`):

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '\''bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity/cpu-checks.sh'\''"'
```

The runner executes `timeout --signal=TERM --kill-after=10s 180s python3 -B /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity/test-long.py`, with all output preserved and `PYTHONDONTWRITEBYTECODE=1`. All executable Python/testing ran remotely, never in the interactive Pi environment.

Successful attempt: `worker-scratch/cpu-YoNLTPQa`, lead background task `bbdc77e0d`, exit **0**. Environment recorded **2026-10-18T07:49:47Z**, `inference-host`, Linux `7.2.0-1-cachyos x86_64`, Python **3.14.7**. Observed unittest summary: **Ran 15 tests in 15.124s — OK**. Runtime is only test-process duration, not an inference performance result.

Coverage actually passed:

1. Real frozen fixtures/exact A0 and C0 argv and all unchanged native/startup parsers, including synthetic timestamp-prefixed startup and actual saved timestamped full server logs for both arms.
2. Both complete source maps and all five binary pins, plus all seven endpoint/sampling/dispatch source pins.
3. Pinned `/apply-template`/chat processing source contract, mocked public render then tokenize with add_special/parse_special true, strict response types.
4. Sampling/lattice source-based disk bound, exact threshold acceptance, insufficient-capacity refusal, unchanged reserve and API caps.
5. Source-bound large-prefill dispatch and width64/65 transition, Q8/NVFP4 precision, five router thresholds.
6. Synthetic larger contiguous prefill partitions, either FIFO stream printed first, separate target/drafter backends, real saved C0 width2 head layouts.
7. Missing/extra/reordered routes, backend/stride/type/precision/cardinality defects, grouped fallback and unsupported fused dispatch rejected.
8. Synthetic 8270-token prompt alignment with all128 raw samples and real saved accept/reject/rollback histories; alternate legal prefill partition.
9. Wrong prompt IDs/positions/output flags, row mapping, rollback/removal, checkpoints and replay fail closed.
10. Full raw-file path/type/symlink/byte-count guards and actual nonfinite F32 rejection.
11. All128 full vectors, inclusive first divergence at ordinals1/116/128, forbidden post-divergence reads, exact first score difference, labels and no parent-shape invariance requirement.
12. Existing quality20 criteria verified against saved results and actual legacy main with mocked HTTP; original retrieval criterion retained, including whitespace and >=8192 count behavior.
13. Exact original Config/HostConfig/mount/ID checks and independent restoration attempts after a synthetic log-collection failure.
14. Exclusive controller run and structural absence of fidelity analysis before restoration or container removal.
15. Failed postprocessing retains HTTP body and still processes later valid responses.

First attempt: `worker-scratch/cpu-0cNCuUyF`, lead task `b5170d4ab`, exit **1**. Observed **Ran 15 tests in 13.076s — FAILED (errors=1)**. Exactly test11 errored with `KeyError: 'label'`: the saved legacy C0 alignment fixture lacks a label. The new production analyzer explicitly sets `label` in `align()` and `compare()` strictly requires A0/C0. Fix was **fixture-only** explicit honest labels, plus wrong-label rejection tests. Production label guard was not weakened or changed. Timestamp-format coverage was added before the successful rerun. Both failed logs and scratch artifacts remain; no cleanup occurred.

## Frozen hashes and storage

`preparation/tested-code.json` freezes exact tested code/helpers; a subsequent read-only remote `check_tested_code()` returned `EXACT_TESTED_CODE_UNCHANGED`. It also observed `real_run_exists=False`.

New tested controller SHA256: `810f8a07f73ec8b85d28c79cc2779f66f3bacb9bebcbaddeca4d8080dc03b6e2`.
New tested analyzer SHA256: `e04b3db229d6b484cdeef5ef4f0f0188bb1a34fa369633e31915817df9feb9d0`.
Tested contracts SHA256: `53ebc412129abb68beb0e9a8e9d407a1d6660b1c1043ea76d15a66b7dd0d5fa3`.
Remaining controller/test/README/helper hashes are in the raw manifest.

`preparation/hash-checks.json`: candidate **3805** source files and accepted **3805** source files all unchanged; both five-binary maps match. Candidate binary pins:

- llama-server: `22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d`
- libllama-server-impl.so: `43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede`
- libggml-sycl.so.0: `64d6a58f17cb92ea9fe6354d9c553f74724565a58fc018d2f4bbd013d090a72e`
- test-backend-ops: `bdd86349fac5ab43fa32d465aad27cec7f31dcad0651f7611403d8a783f10c7d`
- libllama-common.so: `934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812`

Seven additional source SHA256 pins, verified against the live files without executing the runtime, are in `contracts.py:SOURCE_PINS`: sampling, speculative lattice, server context, server common/chat parsing, SYCL dispatch, fused GEMM implementation and fused GEMM eligibility. No sources or binaries were copied into this case/Git.

Observed `preparation/disk-budget.json`: 24 requests/arm, caps total **5632**, source-bound raw rows **5680**, row bytes **524288**, raw bound **2977955840** bytes, logs/evidence allowance **1073741824** bytes, reserve **4294967296** bytes, required free **8346664960** bytes, actual free at successful test **11561844736** bytes, `fits=true`. This is a preparation snapshot only; the controller must recheck each arm and every request, and may safely refuse C0 after retained A0 usage. No storage cleanup, changed caps/reserve, capture toggle or storage infrastructure.

## Limits and handoff

No Docker/native/build/GPU/API/service/hardware or real inference execution by the worker. No runtime/source/model/launcher changes. No higher depth or optional prompt case. No inference retry. Local activity was read/edit/file transfer/Git only. Only new-case code, documentation and small CPU evidence are delivered. All raw vectors/weights/binaries remain remote and outside Git.

The fresh A0/C0 pair, true public endpoint tokenization, native gates in the new candidate containers, anchor/long outputs, new full-vector fidelity, and live original restoration are **pending lead execution after integration**. Prior logs and synthetic tests do not establish these results. Commit is for preparation only; no push from the worker.
