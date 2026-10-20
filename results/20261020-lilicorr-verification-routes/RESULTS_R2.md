# Full verification-route capture R2 — passed diagnostics, not clean timing

The corrected adapter completed on `inference-host`. The separate after-capture invocation independently verified restoration and analyzed the original raw files: `status=analyzed`, `errors=[]`, `independently_restored=true`. All four arms passed 25 public checks each; all 24 measured responses matched the required outputs. W3 retained short `(129 proposed, 84 accepted)` and long `(133, 82)` in every measured sample. Original launchers were unchanged.

## Measured-request route census

`census-r2.json` uses the existing route-frame parser. The final eight processing tasks per W3 arm match the unchanged controller's short warmup + three measured, then long warmup + three measured order. Full prompt positions, 128 target sampler events, saved SSE creation epochs, and measured artifact/usage fields were cross-checked.

| Point | Measured responses | Target verification batches per response | Singleton records per selected projection |
|---|---:|---|---:|
| Short, 55 prompt tokens | 6 | 43 × 4 columns | 172 |
| Long, 8,270 prompt tokens | 6 | 44 × 4 columns + 1 × 2 columns | 178 |

These counts cover the representative `blk.1.ffn_up_exps.weight` and `blk.1.ffn_down_exps.weight`, not every layer. Their verification records use the actual `MOE_SINGLE_MMVQ_PLAIN_Q8_1` route, 240 experts, IDs row stride 960 bytes, and no reorder. Per-frame singleton count and preserved parent stride-derived width matched the actual target input width. Four-column verification is therefore the directly observed case selected by the separate IQ2 batch4 experiment; two-column tails can stay unchanged.

This is a call/shape census, **not GPU event timing, a dominant-cost proof, or evidence of an optimization speedup**. Serving intentionally enabled `LLAMA_TRACE=1` and `BOLT_VERIFY_CAPTURE_DIR`; resulting timings are diagnostic only. No whole-model raw-F32 equality is claimed.

## Retention and exact reproduction

- Remote originals: `/home/mike/b70-evals/20261019-lilicorr-q8-drafter/verification-routes-20261020-r2/`.
- Local complete copy: `capture-r2/`. All **9,084 files / 4,725,510,945 bytes** were individually SHA-256 matched against the remote originals; see `retention-r2.json`.
- All **6,600 raw sampler rows**, each 524,288 bytes, remain unmodified locally and remotely: 3,460,300,800 bytes (3.22 GiB). They are not staged as Git blobs. The inventory retains every path, size, and SHA-256.
- Large non-sampler text files use lossless `.gz` sidecars for Git retention; `compression-r2.json` records round-trip checks. Raw originals remain intact. Materialize a sidecar's uncompressed original when replaying scripts that expect the raw filename.
- The checked-in `run-fulltrace-r2.sh` invokes the existing capture and, separately, the independent after-capture restoration/analysis. `analysis-r1/analysis.json`, native files, HTTP exchanges, source/binary/process evidence, and restoration reports are retained under the copy above.

The native IQ2 batch4 build and clean performance experiment are separate from this diagnostic capture. Nothing here promotes a candidate or changes the operational service.
