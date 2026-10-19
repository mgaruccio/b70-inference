# Q8 batched candidate: valid experiment, insufficient improvement

**Continue optimization; do not promote this candidate.** The original service
was restored and independently verified before raw analysis. No persistent
launcher, model, qualified ancestor, arithmetic tolerance, or output gate changed.

## Observed result

Frozen-SDK build passed; the sole changed binary is `libggml-sycl.so.0`, SHA256
`954eece0e4a7e14df0962bb81cbebf89a11153ed504b2f87b821cf8c3948a87a`.
The full source delta is exactly the reviewed three files. Protected ancestor
source/binaries passed before/after build and execution. Kernel and binding
reviews found no blockers. All six binding tests passed on inference-host.

Each of the four cold cells passed the unchanged native startup gates (including
the 116 existing Q8/IQ2 cases and inherited F32/router checks). All **100 public
prevalidation jobs** passed: 25 per cell, with frozen and fresh A comparisons.
Independent raw analysis accepted all **24 measured responses**, six per arm and
prompt point, with no errors and matching no-draft outputs.

Median server `predicted_per_second` (tok/s):

- Short55: A **44.784265**, W3 **33.154786**; W3 is **25.97% slower**.
- Long8270: A **43.192291**, W3 **31.082218**; W3 is **28.04% slower**.

Previous W3 medians were 32.775433 / 30.655266. The new W3 values are 1.16% /
1.39% higher across runs. That is not a controlled old-versus-new comparison or
a significance claim. The within-run comparison still clearly fails the goal.
Q8 launch batching alone did not remove the major cost.

Measured W3 draft counts are 127 proposed / 84 accepted (short) and 133 / 82
(long). Existing server per-position acceptance is `(0.791, 0.674, 0.488)` and
`(0.800, 0.667, 0.356)`. The third proposal is not invariably rejected; do not
infer that simply reducing the window removes only wasted work.

## Reproduction and evidence

Environment: inference-host, B70, original frozen SDK/model/arguments, isolated
`llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-batched-q8-r1` mounted read-only.
Commands in this package on that host:

```bash
bash build-candidate.sh
python3 -B test-performance.py -v
python3 -B run-performance.py --lead-run-timing
python3 -B run-performance.py --lead-after-timing-restore
```

- `build-attempt-r1/`, `post-build-verification.log.gz`, binary/source JSON:
  build, exact delta and ancestor checks.
- `timing-r1/*/ready-*.log.gz`: unchanged actual native gate logs.
- `timing-r1/*/prevalidation.json`: all 100 qualification outcomes.
- `timing-r1/*/{short,long}/`: original HTTP/SSE, timings and telemetry.
- `timing-r1/summary.json`: capture completed and restoration succeeded.
- `timing-r1/analysis-live-restoration.json`: independent live restoration.
- `timing-r1/analysis-r1/analysis.json`: independent raw checks/statistics.

Raw logs are retained remotely and locally. All 83 lossless `.log.gz` archives
passed decompressed SHA256 equality checks: 842,842,187 raw bytes became
47,958,840 archive bytes. The two preparation CPU logs remain in their original
committed form. No observed records are rewritten.

This remains development-tier/output-matching evidence. It is not a claim of
end-to-end raw-F32 equality, statistical significance, or production eligibility.

## Next bounded step

Measure remaining GPU/host costs with an external profiler in the sibling
`20261019-nemotron35-bolt-lilicorr-w3-kernel-profile` case. Source inspection shows
IQ2 token-serial dispatch and NVFP4 MoE per-expert fallback/ID synchronization,
but those route facts alone do not establish their runtime costs. No lower
precision, weakened gate, or new application benchmark framework is proposed.
