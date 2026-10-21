# EAGLE4 generic engine hooks on frozen B70 vLLM

**Engine overlay only. Full EAGLE4 has not been served or validated by this slice.**
No model/GPU run, benchmark, launcher promotion, version spoof, native kernel
implementation, or legacy-chain substitution is included.

## Exact sources and license

- Base: the actual exported Python package from
  `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
  `vllm==0.27.2rc1.dev77+gac7509e2b`, `torch==2.13.0+xpu`.
- Generic hook reference: Eagle-patched vLLM
  `3ba23801e80d2a0d0a9af21ce94484aa2adb8795` (`eagle/v0.1.0`).
- Consumer contract checked against official Eagle
  `75436712e7415d352cf9f17d070019d07410786f` (`v0.1.0`), specifically
  `eagle_harness/vllm/adapter.py`, `config.py`, `speculator.py`,
  `gdn_tree_verify.py`, and `round_commit.py`.
- Current proposer consumer: read-only `/home/mike/code/EAGLE-4-b70` snapshot at
  `3bf84f13cfa3d761fb3ed455ee0e468858a508d5` plus the lead's uncommitted keyword-count
  change. Tested `speculator.py` SHA-256:
  `a327e217f77b8f3402105186c79db25241283cfa663591df40b3b6b34c65778f`.
  Its import/virtual ABI test runs in a separate interpreter from release tests.

`engine.patch` preserves the vLLM Apache-2.0 SPDX/copyright headers. Its new
`registry.py` and `verify_extension.py` are exact copies from the hook reference;
other existing engine files receive focused hunks, not whole-file replacements.
`LICENSE` is the upstream vLLM Apache license. The new XPU dispatch helper and
backport edits are identified here and in `build_overlay.py`.
No Eagle source-available algorithm/kernel source is vendored in this overlay.

The lead/researcher completed fresh official architecture/hooks research before
this assigned slice; local source inspection does not replace that gate. The
lead owns integrated native full-algorithm verification.

## Contract mapping

- `registry.SpeculativeMethodSpec`: official configure, lookahead, hidden-state,
  async-scheduling and linear-attention extension contracts; lazy factory lookup.
- `SpeculativeConfig`: opaque `method_config`, registered-name validation,
  configure callback, graph hash, hidden-state classification. Built-in names
  cannot be replaced by the registry.
- `VllmConfig`: registered methods require V2; explicitly disabled V2 fails.
  Scheduler/warmup both already consume its lookahead property; that property
  now calls the registered hook. Official K=4/depth=14/verify-width=16 reserves
  **56**, not 15 or 16, lookahead slots. Existing batch-size width schedules are
  not changed. Registered methods with PP/CP fail closed on this backport.
- V2 factory: constructs the registered speculator before built-in selection.
- `BaseSpeculator`: official `prepare_verify_inputs`, `make_rejection_sampler`,
  `post_accept`, plus default varlen-graph/extra-query declarations. Native
  defaults remain no-ops and the ordinary chain rejection sampler is retained.
- Rejection sampler transport: the release `invalid_drafts` argument and `_verify`
  query-bound/verification-token metadata are passed through real chunking.
  Only the verifier sees invalid token markers; native logits processors keep
  original token IDs and their existing call API. The official tree-rejection
  subclass is exercised on CPU through this base transport with device kernels
  and logits processing doubled, not substituted for native algorithm evidence.
- V2 runner: verify preparation follows slot mappings, attention metadata and
  model positions, including dummy/profile and full-graph paths. Custom sampler
  selection is at sampler construction. `post_accept` receives the unchanged
  sampler output/counts **before** request-state advance and `propose`.
  The frozen `post_forward(finished_req_ids)` lifecycle is retained; no newer
  `defer_finalize` API or second finalization is introduced.
- GDN builder: creates the registered extension and attaches it to metadata.
  Common GDN dispatch bypasses chain convolution/recurrence for extension rows;
  fused chain MTP is disabled only when such an extension is present.
- XPU GDN: the fused SYCL entry point now reaches the same `GDNSpecRows` contract.
  Pure speculative batches go straight to the extension. Mixed batches gather
  speculative projections, invoke the extension, and scatter the result; only
  non-speculative rows go through the existing SYCL operator. Extension-inactive
  dispatch is unchanged. This is layout/dispatch, **not** an Intel tree kernel.
- XPU platform: explicit registered `CUSTOM` returns its registered path before
  causal-backend heuristics. Unregistered `CUSTOM` raises, never falls back.
- Varlen graphs: existing V2 varlen support is wired to the speculator flag, with
  official bounded-query builder declarations/support checking. Incompatible
  builders fail instead of pretending uniform support is varlen support. Native
  adaptive-verification behavior remains intact. This does not claim a dynamic
  per-round verify width: official physical width remains batch-size scheduled.

## Opt-in calibration state control

`_dummy_run(..., valid_dummy_state_slots=True)` now reaches `execute_model`
and `prepare_dummy_attn`. The default remains `False`; native no-spec/MTP
warmup and capture keep their existing metadata behavior. No profile-time
Mamba filtering was added: the frozen runner already builds those groups.

The official first-column assignment alone is insufficient here:
`set_dummy_context` rewrites it. The installed regression test reproduces
C2/q16/context64 roots changing from `[1, 2]` to `[0, 5]`, with later verify
columns still NULL. The correction assigns distinct nonzero Mamba slots
**after** context setup, covering the base and speculative columns (and the
position-selected columns in align mode). It rejects insufficient pools rather
than wrapping/aliasing state slots. Ordinary attention context tables and padded
rows stay unchanged. The opt-in also supplies the existing per-request draft
counts so frozen eager q>1 metadata selects GDN verification, not prefill.
No model-state/builder redesign or invented cost data is involved.

The native caller must snapshot/restore these real state slots, as required by
the official calibrator; this is not a safe flag for concurrent live requests.
The calibration change touches only the generated runner implementation and adds four
unchanged metadata/table dependencies to the existing hash checks.
Fresh external reference: [vLLM null-block fix discussion](https://github.com/vllm-project/vllm/pull/33937)
identifies block 0 as reserved/unallocated; that supports requiring non-NULL
calibration states. Exact frozen source and installed tests govern this port.
This control is a prerequisite to native calibration, not measured-cost or
serving qualification.

## Per-call autoregressive parent control

Only frozen `v1/worker/gpu/spec_decode/autoregressive/speculator.py` changes in
this follow-up. The semantic reference is the pinned official
`target_dependent_ar/speculator.py:propose` and its decode helpers; the newer
AR implementation, graph specialization and DP protocol are not transplanted.

- `num_speculative_tokens=None` retains the configured maximum. Integer overrides
  in `[0, maximum]` control `draft_max_seq_len`, both decode loops, and the final
  input-update boundary without mutating `self.num_speculative_steps`.
- Zero and one still execute the real parent prefill, then **no decode**. Zero
  returns invalid proposals; unused columns of partial calls are set to `-1`.
- Frozen decode graphs bake in the configured count, including the final-step
  input update. Partial decodes therefore use the existing eager dispatcher.
  Default/max decode graphs and count-independent prefill graphs remain enabled.
- Auxiliary concatenation accepts native EAGLE3 or the existing registered
  `uses_aux_hidden_states` contract. These pinned sources have neither
  `is_eagle3()` nor `use_eagle3_aux_hidden_state`; no method identity is spoofed.
- The actual current Harness virtual `_prefill` and inherited `_run_model` are
  exercised: combine three auxiliary tensors, run prefill, select top-k seed
  rows, pass them to the drafter, and reach the released tree-growth boundary.
  Native-chain sampling/decode is a tripwire. Growth itself, native arithmetic,
  measured costs and serving are not claimed by this CPU boundary test.

The audited `_prefill`/`_run_model` virtual signatures match. Current Harness
calls the parent with `num_speculative_tokens=steps`. Its optional DP argument
is named `dp_sync_state`, while the frozen parent uses `num_tokens_across_dp`;
the tested B70 single-DP path passes `None`. This is not a DPSyncState backport.
Fresh external reference: [vLLM speculative configuration](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
documents the configured token budget. Keep that budget unchanged when making
a per-call seed request; exact per-call semantics come from the pinned source.

## Installation and replay

High-bit address correction (modified 2026-10-21 for `b70-inference`):
`v1/worker/mamba_utils.py` preserves all 64 opaque address bits when populating
signed int64 metadata tensors. High-bit XPU pointers are represented by the
corresponding signed two's-complement integer; existing low-bit addresses,
metadata dtype, tensor identity and Triton pointer casts remain unchanged.
Both state-cache and block-table pointers are covered. This fixes the observed
`Overflow when unpacking long long` without changing state-copy arithmetic.

Stop the inference process. Apply to a disposable/test runtime first, **not** the
immutable export or a persistent launcher. All 31 pinned files (17 changed,
14 unchanged dependencies) are checked before any source write. Every output is
replayed and syntax-compiled in memory; source drift, conflicting new files,
symlinks, and wrong output hashes are rejected. All replacement files are staged
before replacement. Known installed files are accepted without rewriting.

```sh
python /overlay/install.py --package-root /opt/venv/lib/python3.12/site-packages/vllm --check
python /overlay/install.py --package-root /opt/venv/lib/python3.12/site-packages/vllm --apply
python /overlay/install.py --package-root /opt/venv/lib/python3.12/site-packages/vllm --check
python /overlay/install.py --package-root /opt/venv/lib/python3.12/site-packages/vllm --apply
```

Observed upgrade from the calibration overlay (`9c99066`): 15 already installed,
2 pending → 2 written (AR count control and Mamba pointer packing). `vllm.__version__` remains unchanged.
The image's default `/workspace/vllm` cwd shadows
the installed package: run checks with `docker exec --workdir /tmp`, and verify
`vllm.__file__` points at the installed package. Do not mistake source-tree
imports for testing this overlay.

The preceding calibration revision is accepted unchanged: only the original
frozen AR parent and Mamba metadata initializer need patching. All previous output hashes are unchanged.
Older pre-calibration runner revisions still fail the hash guard; use a fresh
pinned runtime rather than bypassing validation. Never patch a running worker.

Reproduce the patch, without changing either reference:

```sh
python3 patches/eagle4-vllm-ac7509e2b/build_overlay.py \
  --base /home/mike/code/vllm-b70-ac750-python/vllm \
  --official /home/mike/code/vllm-eagle4-b70/vllm
```

The generator checks the official HEAD and each consumed file against that
commit. The shipped manifest pins the actual base and installed file hashes.
Rebuilding the overlay is not authorization to qualify another runtime.

## Defined and executed engine test process

Environment: CPU-only throwaway `full-eagle4-proposer-hooks` on `inference-host`,
exact pinned image above; **no `--device`, no `/dev/dri`, network disabled**.
Own scratch: `/home/mike/b70-evals/20261021-full-eagle4/engine-proposer-hooks`.
Overlay/tests/official reference source mounted read-only at `/check`.
Original pinned files copied inside the container to `/tmp/e4-base/vllm` before
installation. The parent export and all host Python packages remain unchanged.

1. Run the public installer CLI check/apply/check/apply above; require exact
   source validation, expected write counts and unchanged package version.
2. Assert actual imports resolve to the installed package and XPU device count
   is zero. Do not load model weights or execute device kernels.
3. Run the installed CPU contract tests with real vLLM classes/methods and CPU
   tensors. Native device-op boundaries and model forward are test doubles;
   the runner, factory, config, metadata builder, platform selector, common GDN
   dispatch and XPU GDN dispatch are the actual patched implementations.
4. Verify non-prefix accepted KV/GDN-path test data is compacted before count
   advance/proposal, and connector finalization occurs once. Verify tree
   positions are staged on eager and capture/full-graph paths. Verify extra
   query reservation and the unchanged no-spec/native defaults.
5. Verify official `missing_hooks()` returns `[]`, official config gives 56
   lookahead slots, a resolvable registered backend is selected through the
   actual XPU platform, and unknown/unregistered methods/backends fail closed.
6. Call the real installed `_dummy_run` → `execute_model` → `prepare_dummy_attn`
   → hybrid/common/GDN metadata builders with C1/C2, q1/q16, context64. Cover
   none/all/align modes, eager/profile and FULL graph-metadata paths. Require
   disjoint in-range non-NULL state indices after context setup, q16 GDN spec
   rows, unchanged attention context, padded NULL rows, persistent tensor
   addresses and identical omitted/false flag results. Check native no-spec/MTP
   defaults and insufficient-pool rejection. UVA/pinned transfers, model forward,
   graph execution and device operations are doubled; metadata is not.
7. Run the installed AR parent with 0/1/intermediate/max/default counts, fused
   and unfused loops, and eager/graph dispatch. Check exact prefill/decode counts,
   final-step input updates, cleared tails, auxiliary values, invalid bounds,
   immutable configured count/method and native MTP/EAGLE3 defaults. Only Triton
   launches, model/sampling and graph execution are CPU doubles; preparation
   wrappers, request copying, parent control, virtual prefill/model plumbing,
   metadata and graph selection execute their actual implementations.
8. Import the current Harness from `/check/current` in a fresh interpreter.
   Check explicit count forwarding, its real virtual prefill, seed tensors and
   no-chain tripwires. Stop at tree growth, not at a fake parent success.
9. Remove only the owned throwaway container; retain own raw logs.

Exact test command (remote shell invoked via `ssh inference-host bash -s`):

```sh
docker exec --workdir /tmp \
  -e EAGLE4_CPU_CONTRACTS=1 \
  -e EAGLE4_FROZEN_PACKAGE=/tmp/e4-base/vllm \
  -e EAGLE4_ENGINE_OVERLAY=/check/overlay \
  -e EAGLE4_OFFICIAL_ROOT=/check/official \
  -e EAGLE4_CURRENT_ROOT=/check/current \
  full-eagle4-proposer-hooks /opt/venv/bin/python \
  /check/tests/test_eagle4_full_engine_overlay.py -fv
```

Local stdlib-only replay checks:

```sh
python3 tests/test_eagle4_full_engine_overlay.py -v
```

Lead verification after address correction: **25 installed CPU tests passed,
zero skips**, 24.400 seconds, exit 0, in `full-eagle4-pointer-cpu`. The actual
initializer is exercised at signed/unsigned pointer boundaries with persistent
int64 buffers. The reference tree and current port must be distinct mounts:
`EAGLE4_OFFICIAL_ROOT=/check/official` is the unmodified base;
`EAGLE4_CURRENT_ROOT=/check/current` is the integrated port.
Raw log: `inference-host:/home/mike/b70-evals/20261021-full-eagle4/integration/cpu-contracts-pointer-r2.log`.
An earlier retry's two fixture-path failures are retained in `cpu-contracts-pointer-r1.log`.
Paired native state verification used the real initializer throughout and passed
**81 tests, zero skips/XFAILs** (15.72 seconds); see the Eagle state-commit notice
and `integration/native-state-pointer-r2.log`. These are component/metadata
checks, not full-model serving or measured-cost qualification.

Proposer-slice observation before the pointer correction: **24 tests passed, 0 skipped** (including the current-Harness
subprocess test) in the pinned container (`cpu-contracts-03.log`, 15.519 seconds,
exit 0). The isolated consumer also reports 1 test passed, 0 skipped. Local replay:
6 passed, 18 installed-engine tests skipped. No device kernel execution occurred.

Local execution skips the installed-engine class intentionally. The container
run executes all tests, with no skips. Raw outputs are retained in own remote
scratch as `cpu-contracts-*.log`, `installer-upgrade.log`, `environment.log` and
`consumer-source.log`. The first two proposer CPU runs caught fixture assumptions
about configuration construction and missing DP attributes; only fixtures were
corrected. Earlier engine/calibration evidence remains in sibling `engine-hooks`
and `engine-calibration-hooks`. Imports explicitly use `/tmp` and assert the
installed package/AR module paths.

## Integration limits (do not turn hook presence into a completion claim)

- Intel tree-attention, GDN tree kernels, accepted-state/KV compaction kernels,
  exact cost/depth-control lowering and newer adapter API adaptations belong to
  the other assigned workers. The backport calls those contracts but does not
  implement or validate their native arithmetic.
- The untouched official `tree_attention.py` imports
  `fa_utils.is_quantized_kv_cache`, which the frozen package does not export at
  that location. This is an adapter import difference, **not** permission to
  disable checks or fall back to causal attention. The adapter slice must adapt
  it along with the renamed target-dependent AR imports/API.
- Adapter sampler differences still require explicit porting: the frozen
  `Sampler` has `_requires_logits_processing` and individual penalty/bad-word/
  bias states, not the newer `needs_logits_processing`/`logits_processors` API.
  Its `apply_sampling_params` eighth argument is `skip_top_k_top_p`, **not** the
  newer query-bound array; the adapter must not accidentally bind that array
  as a truthy skip flag. Frozen `gumbel_sample` also lacks `is_drafting`.
  This overlay preserves native sampler semantics; it does not import the
  newer logits-processor refactor or pretend untouched adapter calls work.
- No Mamba2 target dispatch backport is claimed; the B70 target is GDN. The
  official generic extension module includes its Mamba2 type definitions, but
  only the GDN execution paths are wired here.
- Exact frozen hook qualification is not a general vLLM version compatibility
  whitelist. Keep the official out-of-range warning and package version intact.
- Full completion requires integrated real-B70 branching/cost decisions,
  tree-attention and GDN fork/commit tests, then public HTTP target-vs-full-E4
  greedy/sampled/code/long-context parity. CPU contract tests cannot replace it.
