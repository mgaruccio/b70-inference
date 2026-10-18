# Preparation results — 2026-10-18

**CPU preparation passed; real development benchmark pending and lead-only.** No inference timing result, deployment, promotion, or publication claim. `README.md` remains the unchanged, hash-frozen pre-run declaration.

## Actual verification

The lead executed `cpu-checks.sh` remotely through its background tool (task `b389b6b23`). The attempt exited **0**, with **13 tests passed in 2.776 seconds**. This duration is CPU test-suite duration, not inference performance.

Environment: `inference-host`, Linux `7.2.0-1-cachyos` x86_64, Python `3.14.7`; recorded start `2026-10-18T12:42:42Z`. Attempt retained at:

```text
/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing/worker-scratch/cpu-iD7rCilD
```

Exact entry point:

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host bash -s' <<'REMOTE'
set -euo pipefail
bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing/cpu-checks.sh
REMOTE
```

The script runs Bash syntax validation, Python AST parsing, then `timeout --signal=TERM --kill-after=10s 180s python3 -B` with its retained inline wrapper. Tests use recorded/synthetic fixtures and fake clocks, transport, samplers and lifecycle commands; an audit hook prohibits subprocess/native/API/socket/sysfs/proc/device execution in the tests. All source/binary checks were read-only.

Passed coverage: exact O-A-C-C-A-O order and 36 measured requests; two loads/arm, six samples/arm/point, warmup exclusion; candidate rename-only transformation; distinct accepted/candidate native contracts and original stopped-log prefix handling; unchanged original configuration; exact historical payload/public IDs and diagnostic emission IDs; SSE final usage, first nonempty content, units, partial/interrupted/error evidence; quality/fidelity/cache/count failures and permitted O text variation; telemetry cleanup, missing/reset/wrap handling; type7 quartiles and invalid-run suppression; every controller-phase fault restoration, and continued cleanup after an individual candidate stop failure. See `preparation/tests.log` for all 13 named tests.

`preparation/hash-checks.json` records matching **all five accepted and candidate binaries** and **3,805 files in each full source map**. No source/runtime changed. Candidate five-binary pins match the completed no-draft case, including SYCL `64d6a58f…` and server implementation `43dc7092…`; accepted SYCL remains `076b1647…` and server implementation `67ae7c86…`.

`preparation/tested-code.json` freezes the six new files, imported helpers, existing sampler and reference hashes before/after the successful tests. No frozen code or README was edited after green. `preparation/source-contracts.json` records both pinned source timing contracts: server-common.cpp94/96 serialization, server-common.h409–444 absolute timestamp subtraction, and original/candidate generation-loop drafting, acceptance and timestamp update locations. These are source evidence only, not retrospective interpretation of old fidelity timings.

## Lead read-only prerequisite check

`lead-preflight-r1.log` records card0 `0000:0b:00.0`, Intel Battlemage G31 **Arc Pro B70 [8086:e223]**, xe driver, hwmon2 ancestry and xe name, readable energy1/2 and temp2 (`60000` millidegrees C at that observation). Noninteractive root fuser exited0 and found only PID608788 on renderD128; the exact original container was running with that PID. This establishes available prerequisites at that snapshot, not thermal equilibrium or continuous isolation. The real controller repeats its live gates. No gate relaxation or installation was needed.

The worker performed no Docker/GPU/native/API/service/hardware/timing execution. The lead prerequisite check was read-only and ran no GPU inference or service change. No cleanup deleted prior evidence.

## Delivery and remaining limits

Prepared files: `contracts.py`, `run-timing.py`, `analyze-timing.py`, `test-timing.py`, `cpu-checks.sh`, `README.md`, this unhashed report, six `preparation/` evidence files and the lead preflight log. All are confined to this new result directory; the matching remote copies are retained.

The lead still owns controller/analyzer review and the authorized real six-cell run through the public API, restoration and independent post-restoration analysis. CPU fixtures are not real E2E proof. The prior diagnostics-off fidelity PASS remains historical and contains no benchmark measurement to substitute for this fresh run. No formal significance, raw-vector equality with diagnostics off, O greedy identity, peak VRAM/occupancy/bandwidth, or standard-publication compliance is claimed.
