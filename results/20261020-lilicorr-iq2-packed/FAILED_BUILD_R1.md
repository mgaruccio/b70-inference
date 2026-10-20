# R1 failed before compilation — preserved

The lead invoked `python3 -B build-candidate.py` in remote `/home/mike/b70-evals/20261020-lilicorr-iq2-packed` using inputs from commit `fed9fe60f20d4bd1d486547961f3b1e7732a967e`. The generated build exited1 at GNU patch application, before the SDK container/compiler or any GPU/public journey ran.

The first increment hunk had three leading context lines and one trailing context line. GNU `patch --fuzz=0 --batch --forward -p1` rejected it on the pristine qualified seed; Git accepted it. The second hunk applied, leaving an unusable partial source tree. That partial source is not compiled, repaired in place or reused.

Retained verbatim under `failed-build-r1/`:

- the complete original12 input files, checked against `input-sha256.txt`;
- the generated shell driver, raw build output and failure exit code;
- parent/seed source and binary records;
- partial `mmvq.cpp`, `.orig` and `.rej` files.

The seed source and all six binary maps matched their before/after records. Its `mmvq.cpp` hash remained `aabdc95aac76ec817c46c5d3fe27bd0f95907cffdd96434007fc6cc6f47b3a42`. Original `nemotron-bolt-probe` was independently observed running, not OOM-killed, health `ok`, with unchanged container ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e` and PID1072495. No service stop or GPU mount occurred in this attempt.

The old remote package and source `llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-packed-r1` remain untouched. R2 uses a new remote package/source and capture name. Adding standard trailing context changes only patch packaging; GNU fuzz=0 and Git now produce the same intended `mmvq.cpp` hash `0f05037c825033930b2b663ac73b87898c9b798563568877ee869f62bd9b73ba` on independent pristine trees. Both forward/reverse/re-forward paths are now part of the CPU test, in addition to the unchanged integer/F32/UBSan checks. No fuzz relaxation or arithmetic change was made.

Disk preflight observed 30,412,116KiB available. The protected SDK seed uses 464,984KiB; the existing build checks require seed-size plus 5,242,880KiB free before cloning. Those checks remain enabled for R2. This failure is not a device-fidelity or performance result; those gates are still pending.
