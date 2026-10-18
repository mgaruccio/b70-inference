# Seeded router N=9 control preparation

**Test-only development preparation; not a benchmark result and not a qualification.**

This case prepares a paired parent/candidate `test-backend-ops` executable for the
single known failure:

```text
router_columns8=1,router_single2=1,weight=blk.48.ffn_gate_inp.weight,
layout=0,type_a=f32,type_b=f32,m=496,n=9
```

The only source change is in `init_tensor_uniform`: the per-chunk engine seed is
predeclared as `0x5A17u + start`. No tolerance, shape, fixture, runtime, or
backend source is changed. The fixed seed makes parent and candidate operands
replayable; it does not reproduce the original `std::random_device` operands.
Do not choose or retry another seed after observing results.

## Lead CPU build

Run only on `inference-host`, using the existing Intel 2026.0.0 image and oneDNN
mount, with no host installs or GPU mounts:

```text
bash /home/mike/code/.pi-worktrees/b70-inference/bolt-router-n9-control/results/20261018-nemotron35-bolt-lilicorr-router-n9-control/build-test.sh
```

The script privately copies the router-columns8 parent with reflink, applies
`seeded-test.patch` at fuzz 0, builds only `test-backend-ops -j6`, performs the
five-binary/three-source protected checks, and runs an `ldd` dependency precheck.
Failed attempts are retained and destination/container name collisions abort.

## Lead-owned native comparison

The lead must run two fresh native-only containers, parent then candidate, with
identical fixed-seed binary and selected filter, no model load/API, and the same
DNN/OPT/VMM/ESIMD/FUSION/GRAPH settings. Mount the selected backend tree at
`/src:ro`, plus the candidate `build-sycl/bin/test-backend-ops` overlay at the
same path. Retain actual route, error/summary, raw logs, mounts, and dependency
output. A numerical failure is retained as data; infrastructure aborts are not
rerolled. Parent and candidate both failing does not qualify the runtime.

No native execution, SSH, Docker, build, API, or test was performed during this
preparation. This is not standard-compliant performance evidence.
