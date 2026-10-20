# LiliCorr host-boundary timing — development only

**Bounded source/CPU slice.** This is not a speed, numerical, GPU-timing, full
critical-path, or promotion claim. It abandons GPU event profiling and retains
only the host target-to-drafter boundary timings from the completed device
increment. The profiling-enabled native mode is explicitly **REJECTED**: the
native W3 `LLAMA_LILICORR_PROFILE=1` route failed the unfused
`BOLT_ROUTER` 77/82 gate before the public journey. Do not describe this mode
as a whole-GPU measurement or as the requested dramatically faster drafter.

## Frozen composition and ownership

Apply these files only to a private clone, never to an original inference
checkout:

1. Qualified r2 source, upstream `b6334e37e6ef286fc5091a41b6561b2113d915fb`,
   inspected at `/tmp/lilicorr-verify-r2-source-inspect-20261020`.
2. `../20261020-lilicorr-context-profile/lilicorr-profile.patch`, SHA-256
   `6691e17bc02ced6af489aad4dff3795eb1341c846acc8cbf0f292a02dee18056`.
3. `../20261020-lilicorr-iq2-batch4/iq2-batch4.patch`, SHA-256
   `2688d72ca13d1c7b03683ce399ec0af38e6aaf554d43aaafbeac06a66e5324e1`.
4. This directory's `host-boundary.patch`, SHA-256
   `d690f88d099109cde176e3f223b678316f47c2b1ca1f7b8a020819154aa06f00`.

`host-boundary.patch` is exactly the committed device-timing patch
(`84af63e37259303c3758cf765d935f6fd08fa2e5aec135b6ddd5b7fc932ad0b2`) with
its complete `ggml/src/ggml-sycl/dpct/helper.hpp` hunk removed. Therefore the
qualified r2 helper is byte-for-byte restored. All other host components,
actual library branch/shape/host-submission metadata, and existing strict
full-context profile bytes are unchanged from the device increment.

`disable-device-profiling.patch` is a single-file incremental rollback from
the completed device-r1 composition to this host composition, SHA-256
`3984d186c0fcaf8f0c95bdc124dc76d1917e1766a33e11db68c827502eb7f295`.
`source-pins.json` retains parent/context/composed/host diagnostic maps, the
old device diagnostic map, both patch hashes, and all reused fixture pins.
The two supported routes are tested: composed seed + full host patch, and
composed seed + full device patch + this one-file rollback.

Lead-owned remote paths are:

- remote root: `/home/mike/b70-evals/20261020-lilicorr-host-boundary`
- candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-host-boundary-r1`
- SDK seed: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1`

All non-SYCL source and five non-SYCL artifacts must remain identical to the
SDK seed. Only the SYCL 105-file closure is rebuilt; common/llama sources were
already byte-identical in the verified 204-object build. No SDK, build,
remote, GPU, install, spawn, push, affinity, math, wait, geometry, public API,
queue, or flag workaround is introduced here. The frozen exact-8 / 7x8 lattice,
maximum-3 retention, noncausal rejected-KV handling, arithmetic, precision,
geometry, NoDraft, NVFP4, and DFlash constraints remain unchanged.

## Host-only contract

Restoring the original queue initializer means it does not request
`sycl::property::queue::enable_profiling`; `DPCT_PROFILING_ENABLED` is not a
frozen build flag. The existing diagnostic header remains byte-identical to the
device increment, but an event record on this host route is
`status=unsupported_profiling` with `device_start_ns`, `device_end_ns`, and
`device_elapsed_ns` unavailable. Library markers remain host-only
(`device_unavailable_library`). There is no event timestamp query or retained
profiling event on the host route. The existing queue wait is still the only
synchronization; no extra wait, barrier, queue, or API is added.

The retained host boundary fields mean:

- target getter time may include target queue drain and output reorder;
- feature CPU-loop time is measured only around the existing loops and reports
  actual calls and bytes;
- injection/process time includes host submission and internal waits;
- the projection getter may include injection/shared-queue drain;
- projected-ring CPU-loop time reports only actual copied rows/bytes.

These are host intervals with explicit attribution, not GPU timestamps. No
`device_elapsed_ns` value is converted into a duration, and no clean-speed or
whole-drafter/whole-GPU claim is permitted.

## Fresh official research and decision

Fresh official research was performed before this slice:

- [Intel oneAPI DPC++/C++ Compiler 2026 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html): the Level Zero v2 adapter documents that only in-order command lists are supported and out-of-order command lists have undefined behavior. It does not identify a precise correctness fix for the observed profiling-enabled route. The final research decision is **NO-GO** for GPU event profiling; `UR_L0_USE_DRIVER_INORDER_LISTS=0` is a documented overhead/behavior knob, not an authorized correctness workaround.
- [Khronos SYCL queue reference](https://github.khronos.org/SYCL_Reference/iface/queue.html): queue properties distinguish `in_order` and `enable_profiling`; queue waits are blocking completion operations. The host variant leaves the original queue construction intact.
- [Khronos SYCL event reference](https://github.khronos.org/SYCL_Reference/iface/event.html): `get_profiling_info` may block until a command completes and is invalid when the submitting queue lacks `enable_profiling`. Queries therefore remain outside the host route; existing device-side query code is not exercised by this composition.

## CPU/source close gate

The bounded check is:

```sh
python3 -B results/20261020-lilicorr-host-boundary/test-host-source.py
```

`cpu-checks.txt` is the raw output from that command. It uses the original
read-only r2 snapshot, exact context/IQ2 fixtures, the existing event and
process C++ mock extraction drivers, and small CPU-only `g++` mocks. It checks
forward/reverse ancestry, exact full-host versus device-plus-rollback
composition, helper byte identity, host queue/profiling behavior, unchanged
kernel launch arithmetic/geometry, old schemas, actual host component
attribution, library host-only markers, and original wait-error behavior. It
never builds llama/SYCL, accesses a GPU, installs an SDK, or contacts the
remote candidate.

The lead must still run the real public journey in the authorized environment:
use the same all-native 100-public / 24-measured request and counter gates,
restoration proof, and the unchanged exact-8 / 7x8 full lattice. The host mode
must pass the same native controls, counters, and restoration checks before any
further consideration. A host-boundary CPU pass is not native or public E2E
proof; the profiling-enabled failure stays rejected.
