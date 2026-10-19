# LiliCorr B70 decoder-backend coverage continuation

**Tier: development diagnostic only.** This package is a mechanical coverage
continuation, not a performance result, device-time measurement, clean speed
comparison, or optimization claim. It adds one exact backend marker so the lead
can profile the first decoder-backend segment rather than only the final lattice
head. No persistent launcher is changed or promoted.

User request:

> ok go ahead and get going on the drafter optimizations. the goal is to get the
drafter performant enough that we start seeing the speed benefits we should be.
profile performance and think about how to opimize the drafter for intels
architecture

## Continuation contract

The completed package
`results/20261019-lilicorr-b70-sycl-profile` remains the analyzer and build
seed. Its exact schema, shared `ggml_time_us()` clock, phase correlation,
multiple-segment handling, partial-node accounting, restoration checks, and
public/native gates are reused unchanged. The new analyzer is only a wrapper
that accepts the exact additional marker `inp_noise_embd`; `lilicorr_lattice`
and the `lilicorr.` prefix remain valid, while target-side lookalikes remain
invalid. The source verification in the qualified tree found
`cb(inpL,"inp_noise_embd",-1)` in the draft decoder; target Nemotron does not
use this marker.

The backend source change is exactly one additional conditional:

```cpp
|| std::strcmp(tensor->name, "inp_noise_embd") == 0
```

It changes no math, kernel, precision, ordering, waits, replay, or source
behavior. The same `LLAMA_LILICORR_PROFILE=1` opt-in is required. Host wall
intervals include submission/wait behavior and are not device timings; logging
wall is reported separately from core wall. Synthetic fixture records test
parser behavior and do not prove live producer emission.

The qualified immutable parent is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-verify-singletons-r2`
(upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`). The lead-only candidate
is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-decoder-profile-r1`.
The completed SYCL seed is
`/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-b70-sycl-profile-r1`.
The full `lilicorr-profile.patch` and its pins prove the three-source diff
against the qualified parent. `seed-incremental.patch` is kept separately and
is the only patch applied to the reflinked seed, avoiding a fresh full target
rebuild.

Only these source paths differ from the qualified parent:

- `common/speculative.cpp`
- `src/llama-context.cpp`
- `ggml/src/ggml-sycl/ggml-sycl.cpp`

The only expected binary deltas remain `libggml-sycl.so.0`,
`libllama-common.so`, and `libllama.so.0.7.0`; all six artifacts and complete
source maps are checked fail-closed. The Q8 drafter adapter and its exact
model pin are loaded unchanged from the existing adapter, not transcribed or
re-pinned here.

## Current official research retained

- Intel [Kernel Launch optimization guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/kernel-launch.html): host submission and device completion are distinct, so this package makes no device-time claim.
- Intel [SYCL Graph Usage Guide](https://intel.github.io/llvm/syclgraph/SYCLGraphUsageGuide.html): graph pointer lifetimes and update synchronization constrain replay; this continuation does not enable or replay graphs.

## Local preparation checks

These checks use only committed fixtures and inherited local code; they do not
build, use a GPU, access a remote host, or establish performance:

```bash
python3 -B results/20261019-lilicorr-b70-decoder-profile/test-profile.py
bash -n results/20261019-lilicorr-b70-decoder-profile/build-candidate.sh
```

The test runs the completed 18-test SYCL suite unchanged, then tests combined
and incremental patch application/reversal/hash pins, unchanged operation
ordering, the exact marker contract, private namespace/source-bind rebinding,
and source/binary/model repin behavior. It retains the distinction between
synthetic parser fixtures and live emission.

## Lead-only build and end-to-end journey

On `inference-host`, with the frozen SDK/image, at least 5 GiB free, and the
exclusive healthy B70 `nemotron-bolt-probe` setup, run:

```bash
bash results/20261019-lilicorr-b70-decoder-profile/build-candidate.sh
python3 -B results/20261019-lilicorr-b70-decoder-profile/source-proof.py --built
python3 -B results/20261019-lilicorr-b70-decoder-profile/run-profile.py --lead-run-profile
# independently verify live restoration/identity/source/ownership, then:
python3 -B results/20261019-lilicorr-b70-decoder-profile/run-profile.py --lead-after-profile-restore
```

Use the unchanged ABBAA1 sequence `VSPF-A1,W3a,W3b,A2`, all native/IQ2/Q8
gates, 25 public jobs per cell, and the established 8 timing points per cell
(55-token short and 8270-token long prompts; greedy `temperature=0`, seed 42,
`cache_prompt=false`, `ignoreEOS=true`, `max_tokens=128`; one warmup and three
measured requests). Retain the same 100 public and 24 measured outputs and
proposal/acceptance counts (short 129/84, long 133/82). Verify final and signal
restoration plus an independent live-after-restore health, identity, source,
and ownership check. Do not truncate the 8-block/full 7x8 lattice, use rejected
KV, enable NVFP4, or tune DFlash configuration.

Expected retained evidence is under the new `profile-r1/` and build-attempt
paths: raw server logs, phase and SYCL analysis, restoration/ownership records,
source snapshots, seed proof before/after, dependency checks, and the exact
three-library delta. Results remain development-only until the lead performs
and reviews this journey.
