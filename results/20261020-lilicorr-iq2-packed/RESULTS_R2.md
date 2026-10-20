# Packed IQ2 n4 R2 — fidelity passed, speed regressed

**Rejected. Not promoted. The requested dramatic net speedup remains unmet.** Development-tier evidence only; no significance or publication-standard claim.

## Real-system gates

- Frozen SDK rebuilt all105 SYCL objects. Only `libggml-sycl.so.0` changed versus both original R2 and the completed clean batch4 seed; all other five checked artifacts stayed byte-identical. Source/seed/graph before/after checks passed. The candidate `mmvq.cpp` remained `0f05037c825033930b2b663ac73b87898c9b798563568877ee869f62bd9b73ba`.
- All **100 public output checks / 24 measured responses** passed, with diagnostics OFF during serving.
- Short W3 counters stayed **129 proposed / 84 accepted**; long stayed **133 / 82**. Full eight-position drafter, 7×8 noncausal lattice, max3, rejected-KV handling and W7 exclusion were unchanged.
- Native Q8 **82 cases / 54 byte-exact**, IQ2 **34 / 16**, passed in every arm. Retained ready/restored logs are independently rechecked using frozen A or private W3 route handling; no case, byte, negative, canary or alias assertion was relaxed.
- The separate original identity/source/binary/health/sole-observed-GPU-owner check passed with no errors: container `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, PID1093111, health `ok`, sole observed owner `nemotron-bolt-probe`. See `candidate-r2/analysis-live-restoration.json`.

## Actual clean serving performance

Geometric means of six measured responses per arm/point; every sample is retained in `clean-summary-r2.json`.

| Prompt | NoDraft decode tok/s | W3 decode tok/s | W3 / NoDraft |
|---|---:|---:|---:|
| Short,55 tokens | 44.8626 | 38.1556 | 0.8505 |
| Long,8270 tokens | 43.1668 | 35.8864 | 0.8313 |

| Prompt | NoDraft end-to-end tok/s | W3 end-to-end tok/s | W3 / NoDraft |
|---|---:|---:|---:|
| Short | 38.6667 | 32.7032 | 0.8458 |
| Long | 11.1081 | 9.9814 | 0.8986 |

The candidate is **14.95% / 16.87% slower than NoDraft in decode**, not a net gain. Its W3 decode rates are about6.2% below the previous passed clean batch4 capture; that comparison is cross-run development context, not a fresh paired candidate-to-candidate experiment. The fresh NoDraft controls above are the decisive comparison. The complete range, sample standard deviation and all outliers remain in the summary; no sample was removed.

Rate accounting is unchanged: server decode uses127 generation steps for a128-token output because the first token comes from prefill. End-to-end rate uses128 outputs divided by the retained HTTP elapsed time. No proxy metric replaces either rate.

## Decision and limits

The integer packing is exact but slower on this frozen stack. CPU arithmetic proof and the `dpct::dp4a` function name did not establish efficient hardware lowering. The source-level helper itself is vector extraction plus scalar additions. No GPU kernel-duration or instruction-lowering claim is made from this run.

Keep the prior qualified source/service as-is. Do not promote or combine this candidate into a future performance baseline. Further work must verify actual frozen compiler lowering before selecting another integer-dot substitution, or choose a different source-grounded optimization. Unsupported `__spirv_*` declarations are not treated as a documented SYCL API.

## Commands and retained evidence

Remote package: `/home/mike/b70-evals/20261020-lilicorr-iq2-packed-r2`. Exact lead commands:

```sh
python3 -B build-candidate.py
# After the build/EXIT trap completed:
python3 -B source-proof.py --built
python3 -B run-performance.py --lead-run-timing
# Separate process/SSH connection:
python3 -B run-performance.py --lead-after-timing-restore
```

Capture and after-process exited0. The original was restored automatically and independently verified afterward. The runtime used the declared A1/W3a/W3b/A2 order,25 public checks/cell, short55/long8270, temperature0, seed42, cache_prompt=false, ignoreEOS, max128, and one warmup plus three measured requests per point/cell. No persistent launcher changed.

Local raw results: `candidate-r2/`; remote originals: the same subdirectory of the package above. Completed code/declarations are not rewritten after capture. `retention-candidate-r2.json` records the remote/local size/SHA comparison; large text uses lossless gzip sidecars for Git while local/remote originals remain intact. Decompress a sidecar to its original filename when a reader requires the raw representation.

The earlier R1 patch-delivery failure remains separately retained in `failed-build-r1/` and `FAILED_BUILD_R1.md`. R2 only repaired unified-diff context, with exact GNU fuzz=0/Git forward/reverse/re-forward regression tests. Its intended inference source and arithmetic did not change.


Retention verification: **2,547 files / 1,015,899,570 bytes** matched remote size/SHA-256; all16 role-correct native-log rechecks passed. Forty-nine large text artifacts have lossless gzip sidecars whose decompressed bytes independently match their raw hashes. Raw local/remote originals remain untouched.
