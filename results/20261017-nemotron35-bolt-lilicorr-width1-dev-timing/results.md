# Width-1 development benchmark results

**Development run passed; active width-1 drafting was slower. No promotion or publication claim.**

The predeclared O1,A1,C1,C2,A2,O2 sequence completed on `inference-host`, with both frozen prompt lengths in every cell. All 36 measurements and 12 warmups passed. There were no inference retries, replacement measurements, dropped samples, or outlier removal. Each arm/point has six correlated measurements from only two process loads; medians and Hyndman–Fan type-7 IQRs are descriptive, not confidence intervals or significance claims.

- **O:** exact accepted original container/runtime, no drafting.
- **A:** existing five-width candidate runtime, no drafting.
- **C:** same candidate runtime with active LiLiCorr/DFlash max1/min0 drafting.

The frozen [pre-run declaration](README.md) remains unchanged, including its preparation-only wording. This file records the subsequent outcome. The source of final numerical results is [`run-r1/analysis-r2/analysis.json`](run-r1/analysis-r2/analysis.json).

## Primary results

128 output tokens per request. Values below are **median [IQR width]**. Server prediction throughput includes generation-loop wall time (drafting and verification where active), not GPU-kernel-only throughput. TTFC is time to first nonempty streamed content, not necessarily first actual token.

| Prompt tokens | Arm | Server prediction tok/s | Server prediction ms | Client elapsed s | TTFC s |
|---:|:---:|---:|---:|---:|---:|
| 55 | O | 40.363 [0.096] | 3146.460 [7.443] | 3.695 [0.021] | 0.554 [0.021] |
| 55 | A | 44.861 [0.037] | 2830.955 [2.317] | 3.291 [0.046] | 0.459 [0.045] |
| 55 | C | 30.018 [0.042] | 4230.750 [5.951] | 4.710 [0.051] | 0.477 [0.046] |
| 8270 | O | 39.057 [0.067] | 3251.682 [5.607] | 11.499 [0.014] | 8.244 [0.021] |
| 8270 | A | 43.167 [0.026] | 2942.066 [1.791] | 11.514 [0.006] | 8.572 [0.004] |
| 8270 | C | 29.177 [0.015] | 4352.718 [2.232] | 13.618 [0.012] | 9.264 [0.009] |

Changes are ratios of the corresponding arm medians, not a median of paired ratios:

| Comparison | 55-token prediction rate | 8270-token prediction rate | 55-token elapsed | 8270-token elapsed |
|:---|---:|---:|---:|---:|
| O→A | +11.14% | +10.52% | −10.94% | +0.13% |
| A→C | **−33.09%** | **−32.41%** | **+43.13%** | **+18.28%** |
| O→C | −25.63% | −25.30% | +27.47% | +18.43% |

O→A is a runtime/source/compiled-hook bundle comparison, not a LiLiCorr attribution. A→C is the matched candidate drafting comparison. A's faster generation did not improve long-request elapsed time: its median prompt processing increased from O's 8239.211 ms to 8566.556 ms; C took 9259.422 ms. Short prompt-processing medians were O 469.858, A 457.155, C 475.007 ms.

C reported 69 proposed / 57 accepted draft tokens for every short measurement, and 70 / 57 for every long measurement. Those acceptance counts did not translate into a speedup. O/A draft counters were absent, not measured zeros. The benchmark does not identify a kernel-level bottleneck or isolate draft-only time.

### Load-to-load checks

Per-cell median server prediction tok/s, in the declared run order:

| Prompt | O1 | A1 | C1 | C2 | A2 | O2 |
|:---|---:|---:|---:|---:|---:|---:|
| Short | 40.371 | 44.843 | 30.011 | 30.028 | 44.888 | 40.355 |
| Long | 39.031 | 43.145 | 29.173 | 29.187 | 43.169 | 39.104 |

Every raw measurement, per-cell/arm quartile, server timing field, client end-to-end output rate and first-content decode proxy is retained in the analysis and request artifacts. The proxy is not true inter-token latency.

## Quality, identity, and scope limits

- Every cell passed all 20 quality cases and the 8238-token retrieval anchor: 21 checks per cell, 126 total.
- Every timed response and warmup had the declared prompt count, 128 output tokens, length termination, predicted_n=128, and cache_n=0.
- A/C ordinary text, finish and usage matched their frozen diagnostics-off references in all measurements. Their separate, untimed short/long diagnostics matched actual emitted IDs, pieces, text, finish and usage.
- O passed its own functional checks but was not output-identical: its six short measurements had **five text variants**, with one matching the candidate reference; its long measurements had **four variants**, with three matching. A and C each had one variant per point, all six matching. O comparisons therefore are not identical-output microbenchmarks and retain the accepted oneDNN repeatability limitation.
- No actual emission-ID claim is made for ordinary timed responses; no ordinary-output retokenization substitutes for IDs. No raw-vector or probability-fidelity equality claim is made here.
- Frozen repeated prompts and cache-disabled checks do not establish unique-content cold long-context performance, a full context sweep, or community-comparable/section-7 results.

## Telemetry and isolation

All measured samples reported GT0 2800 MHz and GT1 400 MHz. There were no missing temperature/frequency samples or unavailable sampled energy deltas. Counter results retain the caveat that undetected full wraps cannot be excluded.

| Prompt | Arm | Observed temp2 sensor range °C | Median sampled card power W | Median observed card energy delta J |
|:---|:---:|---:|---:|---:|
| Short | O | 54–68 | 223.494 | 806.424 |
| Short | A | 66–69 | 247.649 | 794.216 |
| Short | C | 65–71 | 263.332 | 1213.865 |
| Long | O | 64–71 | 257.280 | 2939.021 |
| Long | A | 66–71 | 263.598 | 3011.292 |
| Long | C | 63–71 | 258.017 | 3465.267 |

These power/energy values cover each telemetry **sample span**, not exactly the client request interval. Median spans were 3.608/3.207/4.610 seconds for short O/A/C and 11.423/11.424/13.429 seconds for long O/A/C. The encompassing sampling window also includes persistence overhead. They are not exact whole-request energy measurements. Raw package/card counters, spans, windows, frequencies and temperatures are retained per measured request.

No thermal equilibrium was enforced; observed temperatures differed across arms. Balanced order and recorded clocks/temperatures do not eliminate thermal or time-order confounding. Root-visible DRM ownership snapshots passed at declared boundaries, but do not prove continuous exclusive access. Hardware caps remained unchanged. Peak VRAM, occupancy and bandwidth were not measured; startup buffer sizes are not peak memory.

## Analyzer correction: preserved failure, unchanged measurements

The controller completed and restored the original service. The first post-restoration analyzer independently verified live restoration, but rejected O1/O2 because it compared Docker's `Mounts` array positionally. Inspection established that every complete mount object was unchanged; only array ordering differed. Container ID, Config and HostConfig were identical.

- Original frozen code, [`analyzer-r1.log`](analyzer-r1.log), and failed [`analysis-r1/analysis.json`](run-r1/analysis-r1/analysis.json) are preserved.
- [`mount-inspection-r1.json`](mount-inspection-r1.json) records the complete-object equality checks. Its note also retains the initial inspection mistake of applying a live-Running assertion to the intentionally stopped O1 after-snapshot; that inspection changed nothing.
- [`analyze-timing-r2.py`](analyze-timing-r2.py) differs only by comparing sorted JSON representations of complete mount objects, preserving all fields and multiplicity, and writing into a new exclusive `analysis-r2` directory. It retains the original frozen-code/reference checks and fresh live-restoration verification before analysis.
- Six CPU-only [regression tests](test-analysis-r2.py) passed in 27.838 seconds before creating `analysis-r2`. They reproduce the old failure, accept ordering differences, reject every changed mount field, missing/duplicated mounts and extra fields, protect other original configuration fields, and verify **all 36 measured rows are exactly unchanged from the failed analysis**. Subprocess launches and socket connect/bind operations were denied during these tests. See [`analysis-r2-tests.log`](analysis-r2-tests.log) for hashes, the exact source diff and test output.
- Final analysis passed with zero errors. No inference was repeated, and no sample was replaced or filtered.

## Execution and restoration evidence

The lead ran these commands on `inference-host`, from `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-dev-timing`, via explicit noninteractive SSH/Bash transport. New logs/directories were protected against overwrite:

```bash
python3 -B run-timing.py --lead-run-six-cells        # exit 0: all six cells captured/restored
python3 -B analyze-timing.py --lead-after-restore   # exit 1: Mounts ordering only; preserved
python3 -B test-analysis-r2.py                     # exit 0: six CPU regressions, before analysis-r2 existed
python3 -B analyze-timing-r2.py --lead-after-restore # exit 0: fresh live verification and final analysis
```

The original preparation also passed 13 CPU tests; their log and frozen hashes remain under `preparation/`. The real public-boundary process, payloads, environment, preconditions, expected results and cleanup were defined before launch in the declaration. Native gates, quality requests, prompt/template/tokenization checks, warmups and measured streaming requests all ran through that declared path.

The exact original `nemotron-bolt-probe` container (`7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`) is restored and healthy. READY/stop smoke, same identity/configuration/mounts, sole-live-container and DRM ownership checks, hardware state, both protected source maps and both five-binary sets passed. All four candidates were stopped and preserved. Healthy O2 was not needlessly restarted. No runtime source, model weight, binary, cap or deployment was changed.

Key artifacts:

- `run-r1/summary.json`, `restoration.json`, `restoration-live.json`
- `run-r1/analysis-r2/live-restoration.json`, `analysis.json`
- `run-r1/{O1,A1,C1,C2,A2,O2}/`: native/server logs, configurations, PID environment checks, quality/diagnostic/public-boundary requests and responses, warmups, all measured SSE and telemetry
- `controller-r1.log`, both analyzer logs, correction test log and mount-inspection record

**Decision:** retain the accepted original deployment. This bounded result provides no performance case for enabling width-1 drafting, and does not authorize promoting the no-draft candidate bundle either.
