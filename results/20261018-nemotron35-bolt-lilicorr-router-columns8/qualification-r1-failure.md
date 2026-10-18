# Qualification r1 stopped at a parser error

Development attempt r1 failed before model startup and before any public qualification request. The accepted service was restored, then freshly and independently verified in `qualification-r1/failure-independent-restoration.json` (`passed: true`). No timing or promotion occurred.

The old 30-case selector reported `30/30 tests passed` and a successful SYCL backend. The new analyzer then rejected the raw ANSI-colored backend status:

```text
Backend SYCL0: \x1b[1;32mOK\x1b[0m
```

At `analyze-trace.py:115`, its backend-status assertion accepted only an uncolored `OK`. The same analyzer already supported ANSI coloring for individual case results. The prepared fixtures colored case results but left the backend status uncolored, so the 12 public-gate and eight native/source CPU fixtures did not catch this inconsistency.

The 62-case selector did not run. The 30-case numerical success is not a claim that its route records passed the new analyzer: execution stopped before that validation. The runtime patch and binaries are not being changed to address this parser error. The correction must match the status coloring in place, without stripping or rewriting raw evidence, while still rejecting missing, duplicate or unsuccessful backend statuses. The saved raw 30-case log should be replayed through the corrected parser before another GPU attempt.

Retained evidence includes `public-capture-r1.log`, `qualification-r1/summary.json`, the original CPU freeze and declaration, the stopped `bolt-lilicorr-width-sweep-r1-r8-a` container, its complete failure logs and `restored-router-single2-native.log`, plus controller and independent restoration checks. The cumulative original stdout prefix is stored losslessly as `.log.gz` in Git; the uncompressed remote original is preserved.

The next attempt must use a new output directory and new container names after rerunning CPU checks. This failed attempt must not be overwritten or represented as a fidelity rejection or an inference retry.
