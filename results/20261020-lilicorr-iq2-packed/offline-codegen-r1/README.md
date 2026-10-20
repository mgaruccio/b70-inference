# Actual n4 object offline lowering — not runtime timing

Inputs are the read-only `mmvq.cpp.o` files from completed passed batch4 R1 and
rejected packed R2. `objects-before.sha256` identifies them; before/after checks
passed. The selected kernel is the same full mangled n4 symbol in both objects.
No application source, build flags, graph/queue policy or library was changed.

Frozen oneAPI 2026 tools unbundled each real object's SPIR64 LLVM bitcode, then
`sycl-post-link -split=kernel -symbols` selected `modules_300.bc/.sym`.
`llvm-spirv --spirv-ext=+SPV_INTEL_subgroups` translated the existing subgroup
operations. This is an offline translator option matching operations already in
the object, not an application compiler flag or correctness workaround.

Existing host `/usr/bin/ocloc` **26.31.39395** (hash in
`host-compiler-and-object-check.txt`) used:

```
ocloc compile -file selected-sg.spv -spirv_input -device 0xe223 -out_dir native
ocloc disasm -file native/<generated.bin> -device 0xe223 -dump disasm
```

Device ID `0xe223` resolves to `bmg-g31`. This host offline compiler output is
**not proof of the serving runtime's final JIT code**. Retained intermediate
bitcode/SPIR-V, compiler logs, native outputs, disassembly and selected-symbol
summary are the evidence. The compiler emits an extra void symbol-table kernel;
analysis selects the exact real n4 symbol and excludes that extra assembly.

Static instruction-line counts for the selected kernel:

| | Passed batch4 | Rejected packed |
|---|---:|---:|
| dp4a | 0 | 8 |
| mul | 62 | 38 |
| mad | 2 | 2 |
| send | 27 | 26 |
| load | 25 | 24 |
| store | 1 | 1 |
| assembly bytes | 49,449 | 34,459 |
| GRF allocation metadata | 128 | 128 |
| SIMD metadata | 16 | 16 |

These are **static lines, not dynamic instructions, latency, occupancy or spill
measurements**. Absent spill metadata is not a no-spill proof. Packed output has
an integer accumulator dependency chain through destinations
`r96 -> r19 -> r35 -> r40 -> r48 -> r51 -> r56 -> r57`. This is a hypothesis for
further investigation, not a demonstrated bottleneck. Packed R2 remains rejected
because its real serving comparison regressed despite exact fidelity. No
accumulator-split candidate exists at this point.

Failed extraction/analysis attempts were tooling failures (required subgroup
extension, root-owned temporary output subdirectories, extra compiler-generated
assembly), not application or native-fidelity failures. Their original task logs
and probe scripts are retained separately. A final host package lookup also
failed because `dpkg-query` is unavailable; no installation or system change was
attempted, and `ocloc --version` plus its executable hash were captured.
