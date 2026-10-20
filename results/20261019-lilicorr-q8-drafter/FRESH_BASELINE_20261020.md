# Fresh clean Q8 baseline — 2026-10-20

The original qualified Q8 W3 path was repeated, with diagnostic profiling **off**,
while the device-embedding candidate was being prepared. No inference-host build
ran during timing. This is a development comparison, not a publication benchmark
or promotion.

## Actual path and results

The existing `run-performance.py` private `bind()` / `capture()` path was reused.
Only fresh `OUT` and container `PREFIX` were rebound:

```text
OUT=/home/mike/b70-evals/20261019-lilicorr-q8-drafter/timing-20261020-baseline-r1
PREFIX=bolt-lilicorr-q8-clean-baseline-20261020-r1-
```

Source, six qualified artifacts, target/drafter model bytes, environment, argv,
public requests and native gates were unchanged. Qualified source ancestry was
explicitly asserted. No diagnostic source, profiling flags or new controller was
used. The same ABBA A1/W3a/W3b/A2 journey ran 25 public jobs/cell and short55 /
long8270 prompts, one warmup + three measured responses/point/cell, greedy seed42,
cache_prompt=false, ignore EOS, max128. W7 stayed excluded.

All native qualification, 100 public outputs and 24 measured outputs passed.
Measured W3 counters are exactly 129/84 proposed/accepted short and 133/82 long.
Automatic restoration passed, followed by a **new independent process** checking
original identity, source, health and sole ownership before analysis. Analysis
status is `analyzed`, `independently_restored=true`, 24 rows, no errors.

Medians of six measured responses per arm/point:

| Point | No draft tok/s | Original Q8 W3 tok/s | W3 throughput vs no draft |
|---|---:|---:|---:|
| Short | 44.856 | 39.724 | −11.4% |
| Long | 43.136 | 37.341 | −13.4% |

Server `predicted_ms` medians: short A=2831.296 / W3=3197.0875; long
A=2944.1885 / W3=3401.063. The headline metric is server `predicted_per_second`,
not a first-content/prefill proxy. No outliers were removed or significance claimed.
The original Q8 drafter is **still slower than no drafting** on this path.

## Invocation and retention

On inference-host, Python loaded the original runner with `runpy.run_path`,
updated **`m['bind'].__globals__`** (not the returned namespace copy) for OUT/PREFIX,
then used the actual inherited `load_dependencies`, `Runtime`, and `capture`.
After capture, a new Python process repeated the identical binding, called
`Runtime.verify_restored`, and called the original analyzer. It additionally
asserted all measured W3 proposal/acceptance counters. Neither process rebound
qualified `p['CANDIDATE']`. Local shell wrappers used explicit `bash` with SSH
`BatchMode=yes`, `ConnectTimeout=30`, `ControlMaster=no`, `ControlPath=none`, and
`ForwardAgent=no`, passing `bash -s` on inference-host.

The initial background invocation was rejected by local fish heredoc syntax
(exit127, before SSH/GPU/service activity); its original local task log remains.
The identical invocation under explicit bash completed successfully. No failed
application observation was rewritten or silently repaired.

Evidence is in `timing-20261020-baseline-r1/`, including declaration, all native /
public / timing files, `analysis-live-restoration.json`, and
`analysis-r1/analysis.json`. All 2,480 copied raw files matched remote SHA256.
Git uses 74 verified lossless log copies (909,239,014 raw bytes → 53,642,362 gzip
bytes); every raw original remains locally and remotely. Decompress for original
filenames. Model and protected source files were not copied into Git or changed.
