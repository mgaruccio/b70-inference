# Exact-value device embedding: small W3 gain, no net drafting win

**Fidelity passed; not promoted.** Keeping LiliCorr's shared embedding values on
B70 improved measured W3 throughput by about 1% over the fresh original Q8 path.
It still loses to no drafting by 10.56% short / 12.66% long and requires a 2 GiB
device table. The user's net-speed objective remains open.

## Clean end-to-end result

Six measured responses per arm/point, same qualified ABBA controller, greedy
seed42, cache_prompt=false, ignore EOS, max128, short55 / long8270 prompts:

| Point | Original Q8 W3 tok/s | Exact F32-cache W3 tok/s | Gain vs Q8 | Same-run no draft tok/s | W3 vs no draft |
|---|---:|---:|---:|---:|---:|
| Short | 39.724 | 40.145 | +1.06% | 44.886 | −10.56% |
| Long | 37.341 | 37.715 | +1.00% | 43.180 | −12.66% |

W3 median generation time improved 3197.0875→3163.5065 ms short and
3401.063→3367.3355 ms long: about 33.6/33.7 ms per response. These are real public
server `predicted_per_second` / `predicted_ms` results, not isolated-kernel or
first-content/prefill proxies. Diagnostic profiling was off.

Throughput Q1–Q3 (tok/s), n=6 each: original W3 short39.689–39.757 /
long37.299–37.373; exact-value W3 short40.116–40.174 / long37.617–37.749.
No-draft medians drifted only +0.067% short / +0.103% long between the two clean
captures. Both captures retain all warmup/measured rows and ABBA replicas; no
outliers were removed. This is a development comparison, not a randomized paired
study, significance claim, or publishable benchmark.

## Fidelity, implementation and capacity

- All native qualification and 100 public output checks passed. All 24 measured
  rows passed, with no errors. W3 counters restored exactly to short129/84 and
  long133/82 proposed/accepted. Full block8, 7x8 noncausal lattice/KV cleanup,
  target IQ2/Q8 singleton arithmetic and W7 exclusion remain unchanged.
- The rejected raw-Q8 cache is preserved in `RESULTS_Q8_REJECTED.md` and
  `candidate-r1/`. Its six short acceptance failures were not rewritten or waived.
  The fixed cache invokes the same CPU Q8 `to_float` used by original GET_ROWS,
  then uploads F32 chunks. Existing GPU F32 GET_ROWS copies those values rather
  than rounding through the backend's half2 Q8 dequantizer.
- The native numerical tests cover all 65,536 half bit patterns, int8 extremes,
  signed zero, 32/4096-wide rows and chunk boundaries. Ten source/numerical and
  seven module tests passed locally and on inference-host. ASan/UBSan remained
  fatal except the documented pre-existing C callback-type metadata exception in
  the extracted native C test TU; it does not change production compilation.
- Frozen SDK R3 compiled all 44 module objects, including 11 Unity units, with
  existing generated commands. Only `libllama.so.0.7.0` changed among the six
  pinned artifacts. GPU libraries/kernels, server, test executable, common library,
  protected source and model bytes remained unchanged. No Ninja log/timestamp
  repair or configuration regeneration was used.
- Both actual W3 startup records show SYCL0→SYCL0, Q8 source570,425,344 bytes,
  F32 tensor/buffer2,147,483,648 bytes. The full short/long journey succeeded with
  this allocation. Scratch is bounded to16 MiB and fenced before every reuse;
  publication follows the final upload. **Peak total board VRAM was not measured**;
  allocated table size and successful capacity are not a peak-usage measurement.

No persistent launcher or model file was changed. All candidate containers were
stopped and the original service restored automatically. A separate process then
verified original identity, source, health and sole ownership before analysis.
A later original-container inspect and bounded log capture are retained too.

## Reproduction and artifact paths

Package on inference-host:
`/home/mike/b70-evals/20261020-lilicorr-device-embedding`.
Source is the isolated `...-lilicorr-device-embedding-r3`, based on qualified
`b6334e37e6ef286fc5091a41b6561b2113d915fb`. Exact commands:

```bash
CXX=clang++ python3 -B test-device-embedding.py
CXX=clang++ python3 -B test-build-r2.py
bash build-candidate-r3.sh
python3 -B source-proof.py --built > lead-built-proof-r3.json
python3 -B run-performance.py --mode candidate --lead-run-timing
# A new independent process, after capture/restoration:
python3 -B run-performance.py --mode candidate --lead-after-timing-restore
```

- `device-embedding-f32.patch`, `source-hashes-f32.json`: complete four-private-file
  implementation and exact original/candidate pins; no new public knob/API.
- `cpu-r3-host.log`, `cpu-checks-f32.log`: observed CPU/numerical results.
- `build-attempt-r3/`: SDK image, full source/seed checks, 44-object generated
  compile/link evidence, six binary hashes, dependency and exit-status proof.
- `lead-built-proof-r3.json`: actual completed whole-tree/six-artifact proof.
- `candidate-r2/*/startup-placement.json`: actual 2 GiB allocation/activation.
- `candidate-r2/analysis-r1/analysis.json`: all native/public/measured validation,
  acceptance, per-cell/arm samples, medians, quartiles and unchanged gate status.
- `candidate-r2/analysis-live-restoration.json`: separate-process original proof.
- `final-original-inspect.json`, `final-original-tail.*.log`: later current state.
- Fresh baseline: sibling `../20261019-lilicorr-q8-drafter/` contains
  `FRESH_BASELINE_20261020.md` and `timing-20261020-baseline-r1/` raw/derived data.
- Earlier full-context critical-path evidence: sibling
  `../20261020-lilicorr-context-profile/RESULTS.md`; its host-wall durations must
  not be presented as device-event time.

The next optimization needs a new measured critical-path hypothesis, not an
assumption that the remaining residual is pure CPU cost or that a 1% win justifies
2 GiB. No-drafting remains the unchanged operational choice. Further kernel work
versus broader verification-cost investigation is a scope/effort decision; no
additional source or launcher change is implied by this checkpoint.

## Lossless retention

All 2,533 transferred candidate/build/current-state files matched remote SHA256.
Git uses 100 verified lossless log copies (926,287,843 raw bytes → 54,711,707 gzip
bytes); every raw original remains locally and remotely. Decompress for original
filenames. Failed Q8 and earlier stopped-build data remain separate and unchanged.
