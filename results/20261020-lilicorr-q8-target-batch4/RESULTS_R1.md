# Exact target-Q8 n4 batching — fidelity passed, no net win

**Development only; NOT promoted. The dramatic-speedup goal remains unmet.**

This private candidate coalesces only eligible n4 reordered Q8_0/F32 ESIMD target
launches. Native arithmetic, local geometry/SLM/barrier/reduction, eligibility,
other widths, full8-position drafter / 7x8 / max3 and rejected-KV behavior remain
unchanged. It is based on passed IQ2 batch4, not rejected packed IQ2.

## Build and real qualification

- Frozen SDK, complete **105-object** SYCL module closure; exact GNU/Git patch.
- Whole source, original/seed, generated graph and protected-artifact checks passed.
  Only `libggml-sycl.so.0` changed. Three files differ from passed seed, five from
  original; no host install, math, queue, graph, driver or launcher change.
- Eight source/preflight CPU tests and sixteen binding/parser tests passed;
  selective read-only native arithmetic/SLM/stride review passed.
- Native Q8 **82 cases / 54 byte-exact** and IQ2 **34 / 16** gates passed in both W3
  arms. Each W3 native Q8 log additionally records **seven actual n4 launches** with
  `physical_launches=1`; every frozen numeric/negative/canary check is unchanged.
- Existing exclusive clean **A1/W3a/W3b/A2** controller: **100 public jobs**, eight
  warmups and **24 measured samples** completed. All public and measured outputs
  matched under existing validators. W3 counts stayed **129/84 short, 133/82 long**.

## Measured result

Six measured samples per arm/point, median headline; no outlier removal or
significance claim. Same short55/long8270 prompts, temp0/seed42, cache disabled,
ignore EOS and output128. Existing server metric rates 127 native generation steps.

| Point | NoDraft decode tok/s | W3 decode tok/s | W3/NoDraft | NoDraft end-to-end tok/s | W3 end-to-end tok/s |
|---|---:|---:|---:|---:|---:|
| short | 44.8151 | 41.2037 | 0.9194x | 38.8080 | 35.8938 |
| long | 43.1532 | 38.8172 | 0.8995x | 11.1071 | 10.1757 |

W3 remains **8.1% / 10.0% slower in decode**, and **7.5% / 8.4% slower end to
end**, than fresh NoDraft. This candidate is not the requested outcome. Do not
promote it or present small cross-run movement versus older W3 runs as a paired
speedup. `clean-summary-r1.json` and the unchanged `candidate-r1/analysis-r1/analysis.json`
retain the actual observations and all individual samples.

## Restoration, process failures and artifacts

The capture restored the original, and a **separate SSH/Python process** repeated
original identity/source/health/sole GPU ownership and stopped-experiment checks,
then the unchanged analysis. Both exited zero, without restoration errors. Original
Id remained `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`,
health `ok`, sole observed GPU owner PID **1124074**. No persistent configuration
or service promotion occurred.

One remote **CPU fixture** preflight failed before compilation because local and
remote packed-result directory names differ. Its output is retained in
`remote-bindings-cpu-attempt-r1.txt`; explicit fixture-path binding fixed it, all
sixteen remote checks passed, and the still-fresh build then ran. This was not a
failed compiler, native byte, serving or GPU attempt. The initial inherited test
mock assumption failure is also retained in `bindings-cpu-attempt-r1.txt`.

Exact build/capture commands and raw outputs remain in this package and its
remote counterpart `/home/mike/b70-evals/20261020-lilicorr-q8-target-batch4`.
Cross-host SHA verification and lossless gzip sidecars are recorded in
`retention-r1.json`; raw originals are retained, not deleted. Large cumulative
original-service stdout may remain outside Git with its checksum. Native/public
requests, outputs, metadata, build closure and separate restoration evidence are
the authority, not this summary. Fresh primary research and the defined real
journey are in `README.md`.
