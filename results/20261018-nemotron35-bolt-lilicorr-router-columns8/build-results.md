# Isolated candidate built; native/API results pending

**Development preparation only. No fidelity fix, performance gain or promotion is claimed.**

On `inference-host`, all eight remote CPU fixtures passed. The isolated candidate then built successfully from the parent cache mounted at the same `/src` path, using the existing Intel 2026 image and oneDNN installation. Only `ggml-sycl` and `test-backend-ops` were requested as build targets; the linker also relinked their dependency targets as recorded in the full log.

The patch applied with fuzz zero and passed forward/reverse patch checks. Full source-map checks found exactly the two declared changed source files. Accepted and parent source maps and all five pinned binary hashes remained unchanged. Among the candidate's five pinned binaries, only `libggml-sycl.so.0` and `test-backend-ops` changed. Dependency checks found no missing libraries. More than the required 4 GiB disk reserve remained.

The compiler emitted an unused `gate` parameter warning in unchanged code. It did not prevent the build. Build prerequisites and curl dependency updates occurred inside the disposable container, not on the host. Neither build nor dependency-check container had GPU device mounts; the running inference service was not stopped or reconfigured.

## Commands and retained failure

Remote working directory:
`/home/mike/b70-evals/20261018-nemotron35-bolt-lilicorr-router-columns8`

```bash
python3 -B test-trace.py 2>&1 | tee -a cpu-r1.log
bash build-candidate.sh
```

The first orchestration launch failed **locally before SSH** because the background shell was Fish and rejected Bash heredoc syntax. `build-launch-r1-shell-error.log` retains that failure. The corrected launch explicitly invoked `bash -c`. It ran the sole remote CPU/build attempt; no failed remote build tree was reused or removed.

Evidence: `cpu-r1.log`, `build-attempt-r1/{build.log,exit-code.txt,image.json}`, `runtime-precheck.log`, `post-build-verification.log`, source-difference/protected-source records, and accepted/parent/candidate five-binary manifests.

The candidate is `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-columns8-r1`. Actual native execution and all 75 declared public checks remain pending. In particular, the old 30-case selector's n=3 routes must now be checked by the new analyzer, not the old n=2-only expectation. Device batched-versus-singleton byte equality remains unestablished; CPU fixtures and compilation are not substitutes for the native or public gates.
