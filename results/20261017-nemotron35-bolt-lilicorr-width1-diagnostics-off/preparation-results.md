# Preparation results — diagnostics-off serving fidelity

**Preparation complete; real diagnostics-off A0/C0 pair still PENDING.** Development-only CPU evidence, not live serving fidelity, fresh route/raw-vector equivalence, a timing result, or promotion. Frozen README and executable code were not edited after the successful test/hash freeze.

## Delivered

- `contracts.py`: immutable source/five-binary/reference guards; minimal native-on/serving-off argv transform; actual PID1 allowlisted environment/exec validation; post-marker trace absence; fixed27 requests; public prompt and actual diagnostic emitted-ID contracts; 4GiB+1GiB capacity guard.
- `run-off.py`: lead-only exclusive A0/C0 controller, all unchanged native gates, exact raw HTTP/partial-error retention, serving PID probes before/after, restoration before fidelity analysis.
- `analyze-off.py`: independent post-restoration live original check, both-arm normalization, quality/anchor/public prompt/saved-reference comparisons, short and long128-ID comparisons without probability or raw/route claims; exclusive failed analysis retention.
- `test-off.py`, `cpu-checks.sh`, frozen `README.md`: remote bounded CPU regression process and defined lead-run real public-boundary journey.
- `preparation/`: actual successful CPU evidence, frozen executed/helper/reference hashes, source/binary checks and capacity; `preparation/failed-cpu-zH5kOWMd/` preserves the earlier failure's raw log/environment/exit status.

## Exact remote CPU command and outcomes

The lead ran both attempts through background execution; the worker had no background tool. No worker Docker, native, GPU, API, service, hardware-control or build execution occurred. All executable Python/tests ran on inference-host. SSH command:

```bash
/usr/bin/bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host /usr/bin/bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off/cpu-checks.sh'
```

The runner executes its retained inline Python wrapper with `timeout --signal=TERM --kill-after=10s 180s python3 -B`, preserving unique attempts. Host: inference-host; Linux7.2.0-1-cachyos x86_64 GNU/Linux; Python3.14.7.

- **First attempt:** 2026-10-18T08:20:59Z, remote `worker-scratch/cpu-zH5kOWMd`, lead task `b1d2a9128`; exit1. Eleven tests ran in **7.742s**, nine passed and two errored. Both errors exposed the same **real controller HTTP metadata bug**, not a fixture-only problem: `urllib.request.Request.method` was absent when POST was inferred from request data. Fixed `run-off.py` to use `Request.get_method()` while preserving request data and inferred POST. The failed attempt was not erased or reclassified.
- **Successful retest:** 2026-10-18T08:23:19Z, remote `worker-scratch/cpu-bjn9nwOP`, lead task `baf5b3784`; exit0. **All11 tests passed in9.499s.** See `preparation/tests.log`. The before/after code and reference hashes matched; `preparation/tested-code.json` records the exact successful files/imported helpers/reference fixtures. Final remote SHA256 readback matched the six frozen local-case entries.

Coverage includes unchanged full numerical argv/native commands, native-on/serving-off boundary, actual PID probe program redaction and negative environment/exec cases (including capture=0), post-marker rejection/native gate preservation, actual saved short55/long8270/anchor8238 prompt IDs, fixed27 payloads, primary emitted IDs/token pieces with probability placeholders ignored, partial/non200 HTTP retention, both-arm synthetic request/normalization flow, early failure retention, prompt misalignment refusal, source/binary drift refusals, capacity edge refusals, restoration aggregation and independent-live-check failures. Native and HTTP fixtures are recorded/synthetic only; passing these does not establish real diagnostics-off E2E.

## Runtime pins and capacity

CPU checks read, but did not execute or mutate, the immutable runtime `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1`. All five candidate and accepted binary hashes and **both3805-file source maps** passed. Exact pins are in frozen README and `preparation/hash-checks.json`; actual source-gate proof is `preparation/source-contracts.json`. No old helper, completed result, runtime source, binary, raw-logit capture or launcher changed.

`preparation/disk-budget.json` measured **11016069120 bytes free** during CPU checks. Required before each arm/public/inference request is **5368709120 bytes** (4294967296 reserve +1073741824 evidence allowance). The fixed27 requests cap each arm at6016 output tokens; no serving raw capture is permitted. Recorded previous native/startup logs were10609961 bytes(A0) and10653964 bytes(C0), comfortably inside the conservative evidence allowance alongside bounded HTTP.

Final read-only `df -B1 --output=avail R` at **2026-10-18T08:25:17Z** reported **11015565312 bytes free**. `run-r1` was absent, and both CPU scratch attempts remained. No cleanup was performed.

## Remaining lead work / limits

Integrate the scoped preparation commit, then run the README's one real pair and independent post-restoration analyzer. Native checks, real PID environment, real public tokenization/HTTP output, same-arm/cross-arm fidelity, actual absence of serving traces/raw files, and final live restoration are **still pending for this diagnostics-off pair**. No inference was repeated for the transport fix. Existing getenv/branch instructions remain; only behavioral diagnostics-off is claimed. Preserve stopped containers and all failures; no persistent configuration promotion.
