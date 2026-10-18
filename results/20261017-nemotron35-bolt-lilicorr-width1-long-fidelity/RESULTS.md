# Lead-observed 8K width1 fidelity result

**Development fidelity passes; no timing or promotion.** `README.md` remains the unchanged, CPU-tested pre-run declaration (its pending status is historical). No runtime/source/build changes were made.

## Execution and protection

After integration (`6a0e9cfb`), the lead verified the frozen tested-code hashes, seven endpoint/sampling/dispatch source pins, five candidate/accepted binary hashes and both complete source maps. The conservative per-arm budget required8,346,664,960 bytes including1GiB evidence allowance and4GiB reserve;11,560,722,432 bytes were available immediately before launch. Every arm/request rechecked remaining capacity.

From `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity`:

```bash
python3 -B run-long.py --lead-run-one-pair 2>&1 | tee -a controller-r1.log
# Only after capture/restoration completed:
python3 -B analyze-long.py --lead-after-restore 2>&1 | tee -a analyzer-r1.log
```

Both used explicit local `bash -c`, remote `bash -s`, `set -euo pipefail`, and SSH `-o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host`; log files were created exclusively. Both exited0. Exactly one fresh A0/C0 pair ran, with no inference retry, parser fix or continuation. The earlier CPU fixture failure remains preserved and is not an inference failure.

Both arms passed unchanged native385 + OUTPROD190/support9 + IQ2six + latent7 + attention13/view6 + router30 startup gates. True A0 retained no drafter, K1 and no target feature extraction; C0 retained active draft-dflash max1/min0, K2 and target features. Same immutable five-width runtime, q8 KV/F32 recurrent state, context65536, batch/ubatch1024, parallel1, graphs0 and unchanged hardware.

## Observed results

- Both retrieval anchors returned exactly `BOLT-7319-CEDAR`, finish `stop`, actual prompt8238/completion11 tokens, zero cached prompt tokens.
- Public `/apply-template` plus `/tokenize` IDs matched actual diagnostic prefill IDs and usage. The long prompt contained8270 tokens in both arms. Actual chunks were `[5,1024,1024,1024,1024,1024,1024,1024,73,1020,4]`; these were observed rather than imposed.
- **All128 complete131072-element F32 raw-logit vectors are byte-identical A0/C0. All128 emitted token IDs match.** No first score or output difference; comparison covers all128 outputs. Actual batch-input and selected-row metadata differ, as expected for active drafting.
- A0 used127 singleton post-prompt frames. C0 used70 width2 post-prompt frames, proposing70 and accepting57 tokens, with13 rejections. These are actual history counts, not performance measurements.
- All24 paired public requests (quality20, anchor, warmup, ordinary, diagnostic) have identical request payloads, public prompt IDs, output texts, finish reasons and usage counts. Both functional quality20 and retrieval gates pass. Within-arm ordinary/diagnostic text, finish, counts and diagnostic ID consistency pass. Only the diagnostic establishes actual returned-ID/full-raw-vector equality; no such claim is made for the other requests.
- All selected all-frame dispatch/shape/stride/precision/cardinality/ordering checks passed, including larger-prefill shared-IQ2 F16 library routes and successful grouped expert dispatch. Coverage remains the declared representative weights, not every intermediate activation.

A0 retained495 raw rows/259,522,560 bytes; C0 retained503/263,716,864 bytes, including quality, warmup and ordinary sampling. Post-analysis free space was11,016,802,304 bytes, above the untouched4GiB reserve. No captures were deleted.

## Restoration and evidence

The controller stopped/preserved both experimental containers and restored the original `nemotron-bolt-probe` with exact identity/configuration/HostConfig/mounts, READY/stop, sole-running status, unchanged hardware and accepted/candidate source/binary hashes. The independent analyzer separately checked the live original container and `/health` (`analysis-live-restoration.json`). No persistent launcher or accepted runtime was changed.

Remote canonical evidence: `run-r1/analysis.json`, per-arm `alignment.json`, full logs, exact HTTP/public-tokenization captures, capacity records and `capture/`; concise summary: `lead-inspection-r1.json`. Local logs and JSON larger than1MB are losslessly gzip-compressed with mtime0. Raw vectors stay remote. The tested declaration and code hashes remain unchanged.

This extends the passed55-token frozen probe to the existing8K retrieval/developer workload at draft width1. It does not validate all64K contexts, longer completions, higher draft depths or production quality. Next is verification of the diagnostic-off serving path without altering the numerical controls. Timing and promotion remain blocked.
