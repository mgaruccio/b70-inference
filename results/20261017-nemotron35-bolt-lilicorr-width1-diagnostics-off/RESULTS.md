# Lead-observed diagnostics-off serving fidelity

**Passed, development validation only. No timing or promotion.** `README.md` remains the unchanged tested pre-run declaration; its pending status is historical. No rebuild, runtime/source patch, numerical-setting change or persistent launcher change occurred.

## Real execution

After integrating preparation as `de946cf8`, the lead checked frozen code/reference hashes, source contracts, all five candidate/accepted binary pins and both full source maps. Preflight free space was11,016,892,416 bytes versus5,368,709,120 required (4GiB reserve +1GiB evidence). Per-arm, public-request and inference-request capacity checks passed.

From `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-diagnostics-off`:

```bash
python3 -B run-off.py --lead-run-one-pair 2>&1 | tee -a controller-r1.log
# Only after the controller completed and restored the original:
python3 -B analyze-off.py --lead-after-restore 2>&1 | tee -a analyzer-r1.log
```

Both ran through explicit local `bash -c`, remote `bash -s`, `set -euo pipefail`, and SSH `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host`. Log files were created exclusively. Both exited0. Exactly one fresh A0/C0 pair, no inference retry or parser continuation. The earlier CPU transport-metadata failure remains retained; it was corrected before launch.

## Diagnostic separation actually verified

Both arms passed unchanged native385 + OUTPROD190/support9 + IQ2six + latent7 + attention13/view6 + router30 gates. Native route logging remained enabled by the shell-only `/tmp` capture flag. Before server exec, the flag and `LLAMA_TRACE` were unset; generic SYCL debug was set0 and tensor callbacks remained0. The capture Docker environment entry and mount were absent.

Actual `/proc/1/environ` checks before and after each arm's27 requests found only these diagnostic keys:

```text
GGML_SYCL_DEBUG=0
BOLT_TRACE_TENSORS=0
```

`BOLT_VERIFY_CAPTURE_DIR` and `LLAMA_TRACE` were absent, not set to the still-enabling string `0`. PID1's executable and argv matched the intended server. Scoped `/tmp/row-??????` inventories were empty before and after both arms. No host capture directory was created. Native BOLT route evidence existed before the server marker; forbidden serving trace/readback markers were absent afterward.

This is a diagnostics-off **behavioral** path. The compiled getenv/branch checks remain; this is not a zero-overhead binary.

## Fidelity results

- All27 cases per arm passed, and all27 cross-arm comparisons passed: identical serialized request bytes, public prompt IDs, response text, finish reason and usage.
- The unchanged quality20 and8238-token retrieval anchor passed. Both returned the same saved-reference outputs, including exact `BOLT-7319-CEDAR` retrieval.
- Short diagnostic:55 prompt/128 completion tokens, zero cached tokens. All128 actual emitted IDs and token pieces match each arm's traced reference and the other arm; no first difference.
- Long diagnostic:8270 prompt/128 completion tokens, zero cached tokens. All128 actual emitted IDs and token pieces likewise match both references/cross-arm; no first difference.
- Warmup, ordinary and diagnostic text/finish/usage and prompt IDs agree within each arm for both lengths. The ordinary streaming payloads were not modified to request IDs; actual completion-ID claims are restricted to the diagnostics.
- C0's diagnostic API reported57 accepted/69 proposed for short,57/70 for long. These are diagnostic counts, not speed measurements or fresh route attribution.

**No raw-vector, fresh-route-equivalence or probability-fidelity claim is made.** Raw capture was disabled, and speculative probability fields remain placeholders. Actual IDs came from primary `choices[0].logprobs.content[*].id`, cross-checked with verbose IDs, token pieces and returned text.

## Restoration and retained evidence

The controller stopped/preserved both experimental containers and restored the original exact container identity, Config/HostConfig/mounts, READY/stop, live health, sole-running status, source/five-binary hashes and hardware. The separate analyzer independently re-queried the live original container, health and hardware. Accepted runtime, weights and persistent launcher remain unchanged.

Post-analysis free space was10,999,832,576 bytes. No prior captures or other data were removed. Canonical remote evidence is `run-r1/analysis-r1/{analysis.json,live-restoration.json}`, per-arm before/after PID records, complete HTTP/public-tokenization evidence and native/server logs; concise summary is `lead-inspection-r1.json`. Local logs and JSON larger than1MB use lossless gzip with mtime0. Tested code and README hashes remain unchanged.

The true no-draft,8K and diagnostics-off gates now support planning a matched **development** performance comparison. No timing run has started, no standard-publication checklist is complete, and no promotion is authorized. A future comparison must distinguish the accepted baseline from the modified-runtime no-draft control and active LiLiCorr bundle rather than attributing all bundled changes to drafting.
