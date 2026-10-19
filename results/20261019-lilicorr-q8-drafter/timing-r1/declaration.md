# LiLiCorr Q8 drafter — development adapter

**Preparation only.** This package is a lead-only, development-tier experiment adapter. It
has no backend/source change, no persistent launcher change, no DFlash configuration
investigation, and no GPU result. The worker only adds this package and runs CPU fixtures.

## Frozen inputs and intentional difference

- Qualified baseline/runtime: the existing `20261019-nemotron35-bolt-lilicorr-verify-singletons`
  package and its qualified source/binary ancestry.
- Reused timing machinery: `20261019-nemotron35-bolt-lilicorr-singletons-w3-dev-timing`
  `controller.py`, `contracts.py`, and `analyze-timing.py` through a fresh private
  `runpy` namespace, following the completed drafter-adapter binding pattern.
- Candidate file on `inference-host`:
  `/home/mike/b70-evals/20261019-lilicorr-q8-drafter/lilicorr-q8-25.gguf`
  (exactly 1,437,595,904 bytes, SHA256
  `d420b2bdcfbbe937795b9dc94d2659305eb01244309172019976dabefad363d7`).
  The original drafter SHA256 is
  `080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194`.
- A is unchanged and remains no-draft. Only fresh W3 cells replace the read-only
  `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr/draft:/draft:ro` bind with
  the pinned candidate file at the existing server-visible filename
  `/draft/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf`.
  The server command/PID argv, model/runtime flags, IQ2/Q8 native gates, W7 exclusion,
  qualified ancestry, W3 block/acceptance, and restoration proof are untouched. The
  adapter checks the real Docker bind and re-pins the file before/after the lifecycle.
- This is a model-file-only Q8 candidate. It does not use or alter the previous
  batched-Q8 backend detour.

The conversion was already performed with the existing quantizer. Exactly 25 original NVFP4
tensor types were converted directly via F32 to Q8_0; 128 F32/BF16 payloads were byte-identical,
and only `general.file_type` changed from 39 to 7. No FP16 intermediate was used. This is a
fidelity hypothesis, not a speed claim.

## Tier, baseline, and schedule

This is development tier only. The primary comparison is no-draft A versus Q8-drafter
W3 on the same host, with the existing qualified short-55 and long-8270 ordinary requests:
greedy temperature 0, seed 42, `max_tokens=128`, EOS ignored, `cache_prompt=false`,
streaming with usage. Each fresh cell runs all 25 existing prevalidation jobs before any
timing. The inherited ABBA order is `A1, W3a, W3b, A2`; each point gets one warmup and
three measured requests, giving six measured samples per arm and point. Raw failures are
retained and restoration is mandatory. Results are descriptive medians/IQRs, not a
community-comparable claim; no Q8 speedup is guaranteed.

The completed FP16 development result is context only: `37.4092/33.7999` versus
`44.8599/43.152` for the corresponding W3 versus no-draft measurements, which was not a
speedup. Historical comparisons were not controlled. This package has no proven Q8 speedup.

## Executable end-to-end process (lead on inference-host)

Preconditions: use the exclusive `mike@inference-host` checkout at
`/home/mike/b70-evals/20261019-lilicorr-q8-drafter`; no other GPU workload/container
may own the device or port; the qualified original service must be healthy and sole;
the qualified A/W3/W7 artifacts, r2 source/binary proofs, and the candidate file must
be present; verify at least the 4 GiB runtime reserve plus 1 GiB evidence allowance.
The lead must not reuse an old output directory or container name.

```bash
cd /home/mike/b70-evals/20261019-lilicorr-q8-drafter
set -euo pipefail
python3 -B test-performance.py -v 2>&1 | tee cpu-checks.log
python3 -B run-performance.py --lead-run-timing 2>&1 | tee performance-capture-r1.log
# Run only after the capture command reports restored:
python3 -B run-performance.py --lead-after-timing-restore 2>&1 | tee performance-analysis-r1.log
```

The capture command records the candidate file pin, hardware/source/ancestry proofs,
actual create argv and PID records, Docker `HostConfig.Binds`/`Mounts`, native startup
and IQ2/Q8 native logs, all 25 public prevalidation requests per cell, six raw timing
rows per arm/point, telemetry, and failures under fresh `timing-r1/`. It stops every
created cell and independently restores the original service. The analysis command
performs a fresh restoration/sole-container/ownership/source check before rereading raw
HTTP/SSE, timing, fidelity, and bind evidence. Expected results are unchanged A/frozen
outputs, passing W3 fidelity with a real rejected draft, and measured performance only if
the actual run supports it. A CPU-green fixture is not native, public, or performance
proof.

## External research recorded before implementation

- [Intel/OpenVINO GPU device guidance](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html): existing Intel-supported format paths are documented for the relevant GPU inference workflow; this does not establish this adapter's Q8 end-to-end speed.
- [Intel Arc Pro B70 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf): records the target B70 hardware context; it provides no guarantee that this converted Q8 drafter will improve throughput.

## CPU scope and unresolved concerns

`test-performance.py` exercises only namespace isolation, exact A/W3 argv deltas,
existing container/PID command preservation, candidate file size/hash/regular-file pins,
Docker bind/mount inspection, rejection of wrong permissions/identity, and qualified
ancestry non-rebinding. It never starts Docker, accesses GPU/DRM, calls a socket/API, or
mutates a remote service. Native equivalence, public 25-job fidelity, memory/capacity,
thermal behavior, and performance remain lead-owned and unresolved until the real
journey completes. The full benchmarking publication checklist is intentionally out of
scope.
