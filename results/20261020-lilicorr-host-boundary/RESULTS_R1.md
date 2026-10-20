# Original-queue host boundary R1 — passed diagnostic, not a speedup

**Development diagnostic only. Not promoted.** This is not a clean performance comparison, a GPU-duration measurement, or the requested dramatically faster drafter.

## Real-system result

The frozen SDK rebuilt all 105 SYCL objects. The candidate differs from the completed device-timing seed only by restoring the qualified original `dpct/helper.hpp`; only the SYCL library changed versus that seed. Original R2 and the seed remained unchanged. The rejected profiling-enabled results remain preserved separately.

The actual A1/W3a/W3b/A2 capture and separate after-process passed:

- **100 public checks / 24 measured responses**, all exact output checks passed.
- Every W3 short response retained **129 proposed / 84 accepted**; long retained **133 / 82**.
- All native gates passed. Sixteen retained ready/restored Q8/IQ2 logs were independently rechecked with the role-correct frozen A checker or private W3 batch4 checker: Q8 **82 cases / 54 byte-exact**, IQ2 **34 / 16** per check. Native checks alone do not claim public fidelity; the public checks above are separate.
- Full eight-position drafter, 7×8 lattice, maximum3 and noncausal rejected-KV dependencies/cleanup remained unchanged.
- All selected GPU timestamps remained **unavailable**. Event metadata reports unsupported profiling; no device-duration value was inferred.
- Independent original identity/source/binary/health/sole-GPU restoration passed with no errors: container `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, health `ok`, sole live owner `nemotron-bolt-probe`. See `profile-r1/analysis-live-restoration.json`.

The reviewed analyzer correction counts 24 measured output rows independently of the unchanged context analyzer's 16 W3 phase records. The host report selects 12 measured W3 requests; four W3 warmups are not included in its means.

## Host attribution

Arithmetic means of six measured W3 requests per point, **milliseconds per response**. Raw host clocks split prefill from decode at the first draft-batch begin; chunk counts/rows agree with the existing full-context phase records. All samples and ranges are retained in `host-summary-r1.json`.

| Existing host interval | Short prefill | Long prefill | Short decode | Long decode |
|---|---:|---:|---:|---:|
| Target getters | 2.350 | 65.373 | 0.157 | 0.160 |
| Feature CPU memcpy/index loops | 1.646 | 236.185 | 5.092 | 5.229 |
| Injection/process submission | 4.380 | 324.007 | 31.857 | 33.105 |
| Projection getter | 2.183 | 37.581 | 0.519 | 0.568 |
| Projected-ring CPU loops | 0.018 | 0.126 | 0.128 | 0.128 |
| Separately measured log cost | 0.005 | 0.026 | 0.074 | 0.080 |

Existing full-context decoder-phase intervals, which are **not additive with their nested components**:

| Interval | Short decode | Long decode |
|---|---:|---:|
| `process_batch` | 37.944 | 39.390 |
| `draft_batch` | 315.121 | 344.430 |
| `draft_select` | 0.632 | 0.673 |

Target getters can include target queue drain/output reorder. Injection submission can include internal waits; the projection getter can include injection/shared-queue drain. These intervals must not be relabeled pure CPU work, pure GPU compute, or all drafter overhead. Instrumented `draft_batch` is not the previous clean `t_draft_us` measurement, and no cross-run subtraction is used here.

The actual short-prefill width46 `fc.weight` marker is `LIBRARY_DNNL_F16`, **host-only**. No corresponding width46 observation is claimed for the long prompt, and no oneDNN primitive GPU duration is claimed.

## Optimization decision

The large feature-copy and injection costs are predominantly **cold prefill**. Decode feature-copy loops are only about 5ms per response, and the complete decode process interval is about38–39ms. Another tiny copy/host-submission tweak is not a credible path to the requested dramatic net gain.

The next bounded experiment instead targets the four-token IQ2 verification route: preserve the native per-output floating scale, K accumulation and XOR reduction, and test an exact packed-integer replacement for IQ2_XXS's scalar integer dot work. It does not assume cross-token expert overlap, add four-way staging, or change geometry. The bundled `dpct::dp4a` is source-level vector/scalar code; neither a hardware dot instruction nor a speedup is assumed. Actual native/public clean serving results must decide it.

## Commands, transport, retention

Executed on `inference-host` in separate processes/SSH connections:

```sh
cd /home/mike/b70-evals/20261020-lilicorr-host-boundary
python3 -B run-profile.py --lead-run-profile
python3 -B run-profile.py --lead-after-profile-restore
```

Both exited0. The unchanged frozen workloads used short55 / long8270, temperature0, seed42, cache_prompt=false, ignoreEOS, max128, 25public jobs/cell, one warmup plus three measured requests per point/cell. The original service was restored automatically and independently checked afterward. No persistent launcher changed.

An initial IPv4 SSH connection reset during the pre-run build proof, before capture started. A subsequent direct check confirmed no capture directory and the original service healthy. The successful retry used Tailscale IPv6 with `HostKeyAlias=inference-host` and strict checking of the same known host key, without any host/network configuration change. The failed transport output is retained separately.

Remote originals: `/home/mike/b70-evals/20261020-lilicorr-host-boundary/`. Local originals: this directory. **2,557 files / 2,111,315,408 bytes** matched remote SHA-256 and size; see `retention-profile-r1.json`. All raw originals remain intact. Fifty-seven large text files have lossless gzip sidecars for Git; decompress a sidecar to its original filename when a reader requires the raw representation. Completed code and declaration files were not rewritten after capture.
