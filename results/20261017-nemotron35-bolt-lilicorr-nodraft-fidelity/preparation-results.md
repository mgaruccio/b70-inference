# Preparation result — true no-draft A0 remains pending

## Observed CPU close gate

The lead launched background task `ba9be2773` on the prepared remote snapshot:

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-nodraft-fidelity/worker-scratch/prepare-r1/cpu-checks.sh prepare-r1"'
```

Observed: **16 tests passed in 7.020s; `CPU_SUITE_EXIT=0`.** Environment: `inference-host`, Linux `7.2.0-1-cachyos`, Python `3.14.7`, `PYTHONDONTWRITEBYTECODE=1`. The runner checked 11,549,812 KiB free against the 4-GiB reserve. This duration describes CPU test execution only, not inference performance.

Prior remote syntax checks: `ast.parse` for `run-a0.py`, `analyze-a0.py`, and `test-a0.py`, plus `bash -n cpu-checks.sh`, passed. No Python/AST/tests were executed locally. No real A0, Docker, build/compiler/native execution, GPU/API/service/hardware operations were performed by this worker.

Artifacts copied verbatim from the new remote result's `worker-scratch/`:

- `preparation/prepare-r1.log`: complete CPU command/environment/hash/test output.
- `preparation/prepare-r1-syntax.log`: remote Python AST success output.
- `preparation/hash-checks.json`: full five-artifact candidate/accepted binary hashes, two changed candidate source hashes, and exact full-source-map check results (**3,805 files in each tree**).

The remote `worker-scratch/prepare-r1` snapshot and logs are retained. No CPU preparation failure occurred. An initial transfer shell consumed its remaining stdin after creating the empty attempt directory, so it did not transfer files or execute syntax/tests. The transfer was then completed into that empty directory; no test log, completed fixture, or runtime was overwritten or discarded.

## Exact tested file hashes (SHA-256)

```text
dd7b6a278deafbb27dc32f40700386e392432c3ea20fcdc74031d05c434e0229  run-a0.py
24333f6ba2c3b4aaabd667f23c5a9ef6fc4c2740828deb7590d9ee76f4710e28  analyze-a0.py
4d2aa2f0c03d49d8285f85b566f79ec36fd0a4a3bf960c58a8bf109e1de34ace  test-a0.py
0bde127cd65cb61970079d9f3ac423b6d97acf940d71a78a27dc6a89a04353ec  cpu-checks.sh
```

The implementation/test/runner files were unchanged after this successful suite. Finalization only retained evidence and updated documentation.

## What passed, and what it does not prove

Passed: exact four-change S0→A0 argv transformation; no draft tail/mount and unchanged native30 command/read-only helper mounts; prior real no-draft K1 startup; immutable source and five binary hashes; saved actual native-gate parser reuse; frozen requests; original exact Config/HostConfig/mount restoration checks on saved evidence; no imported main/analysis before restoration; independent FIFO association; exact route/cardinality/metadata and per-width MKL/DNNL boundaries; valid variable contiguous prompt chunk coverage; single-only128 alignment/common raw guards; inclusive shared-history cutoffs at ordinals1/116/128; truthful A0/S0/C0 labels; batch-shape differences not falsely treated as parent invariance failures; within-arm text consistency; and saved20-case quality comparison mechanics.

**Transformed route/frame fixtures are synthetic.** Saved S0/C0 raw captures and API cases exercise common mechanics; relabeling them for CPU tests does not establish fresh A0 dispatch, vectors, outputs, or serving behavior. Saved native logs prove parser reuse, not new native execution. No raw vectors, weights, binaries, or full runtime source are included in this commit.

## Remaining lead-owned real E2E

Install the tested scripts and current README at the new remote result root, then execute the single fresh A0 capture process in `README.md`: native gates, real API20, frozen warmup128/ordinary128/diagnostic128, stop/preserve A0, exact original restoration/READY/health/source/binary/hardware checks. Only afterward execute `analyze-a0.py` for actual A0/S0 and A0/C0 comparisons. Preserve the fresh capture and failed summary if a parser blocker appears; do not weaken checks or retry inference for agreement.

Status: **preparation complete, real A0 fidelity unknown, timing/promotion blocked.** This is development-only evidence, not standard-compliant publication.
