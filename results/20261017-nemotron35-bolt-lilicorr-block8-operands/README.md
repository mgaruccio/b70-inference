# Bolt block8 MoE operand observer

## Predeclaration — bounded development diagnostic

Prepare one default-off, unchanged-arithmetic observer against the completed
`router-single2` computational parent. This is a fidelity-localization capture,
not a timing run, arithmetic control, promotion, or publishable benchmark. The
accepted runtime, model, launchers, hardware, and parent evidence remain
immutable.

The first complete boundary is S0/C0 output ordinal 2 at position 55, token 0.
Both arms must first pass the inherited all-128 parent raw-vector, prompt/text,
ID, batch-input, position, row, and within-arm checks. Cross-arm score
comparison is limited to the common history through the first different output.
No operand equality is an execution gate: all six actual operands are compared
only after both arm gates, and a mismatch is evidence of a different boundary,
not proof of causality.

## Paths and frozen parent

- Parent source: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-single2-r1`
- Candidate: `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-block8-operands-r1`
- Parent result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-router-single2`
- New result root: `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-block8-operands`
- Temporary container prefix: `bolt-lilicorr-block8-operands-r1-`
- Worker scratch: new result root `worker-scratch/` only

Pinned parent artifacts:

```text
llama-server              22073e7fb3ae3a68f694cef109873a94860834996bdf30c597a8ea99da97184d
libllama-server-impl.so    43dc709205a1f2557b3872ff3789aeefb3d4c8a88b92b69bfa72caceb2468ede
libggml-sycl.so.0         b2b3b7950c471bb5658f8474eb1866157fd3311a183e07d03a665842a8cad3d9
test-backend-ops           5f43858b1cbd1083ced719a8700f93c30a0e68e94f307541f0bee9e43f5e8389
libllama-common.so        934b90be6129c55ba040218f2a895670761a6d9c371da8a1afae568791bbd812
```

The parent has no block8 route records. CPU route/operand fixtures add only
synthetic block8 records to real parent frames and batches; they are not live
runtime evidence.

## Exact observer scope

Only `ggml/src/ggml-sycl/ggml-sycl.cpp` changes. The selector adds exactly
`blk.8.ffn_gate_inp.weight` and `blk.8.ffn_latent_down.weight`; it does not
select blanket `blk.8.*`. Existing block0/block1/block7/output route selection,
router-single2 arithmetic and guards, IQ2 helpers, queues, and synchronization
remain unchanged.

At the first target frame, the observer brackets the original
`GGML_OP_MUL_MAT` on the existing in-order consumer queue:

- `blk.8.ffn_gate_inp.weight`: F32 `[4096,304]`, full weight `4,980,736`
  bytes; F32 `src1[4096,n]`, `dst[304,n]`.
- `blk.8.ffn_latent_down.weight`: reordered Q8_0 `[4096,1024]`, full weight
  `4,456,448` bytes; F32 `src1[4096,n]`, `dst[1024,n]`.
- S0 uses `n=1`; C0 uses `n=2`.
- Files are exactly `{router,latent_down}.{weight,src1,dst}.bin`, created
  exclusively once per projection under the private `/moe-operands` mount.
- `reorder` describes the projection weight for every role, including F32
  activation/output roles. F32 `extra` is nullable.

The existing span, type, shape, stride, contiguous/non-view, owner/device,
buffer-bound, queue-order, completion, error, and host-lifetime guards are
retained. There are no callbacks, new queues, numeric changes, alternate
matmuls, extra block captures, or new runtime knobs.

## Route correlation

The common FIFO contains `BOLT_VERIFY` and `BOLT_MOE_OPERAND`; direct-stderr
`BOLT_ROUTE` remains a separate FIFO. Textual line order across these streams is
never used. The parser keeps the unique block0 SSM backend anchor, one-to-one
route-group/VERIFY pairing, foreign backend routes, strict prefill batches
`5@[0..4]`, `46@[5..50]`, `4@[51..54]`, and all target-frame cardinality,
order, and retained-stride width checks.

Old block1 C0 routes retain the two singleton outer/inner router pairs and
retained width-2 strides. Block8 is validated separately: its C0 router is one
full `n=2` `MUL_MAT_SYCL` + `LIBRARY_MKL_F32` pair at width 304, never the
block1 singleton relaxation. Width 46 uses `LIBRARY_DNNL_F32` for the router;
other widths use `LIBRARY_MKL_F32`. Only the selected block8 router and
latent-down routes are checked by actual names, types, shapes, cardinality, order,
and stride-derived width on every target frame; unselected expert/shared routes
remain outside this narrow route extension.

## Research

Fresh official-source checks were made through the remote research boundary:

- <https://github.khronos.org/SYCL_Reference/iface/queue.html> — an in-order
  queue orders predecessor commands; `memcpy` returns an event and
  `wait_and_throw` reports completion/errors. This does not guarantee
  cross-queue or floating-point bitwise invariance.
- <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html>
  — GEMM documents queue/stride computation behavior, not a cross-shape
  bitwise-equality guarantee.

Empirical equality therefore remains mandatory and is not inferred from either
API contract.

## Defined real E2E — lead only, pending

Run only on `mike@inference-host` through explicit local Bash plus:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host 'bash --noprofile --norc -s'
```

After source review and the disposable six-job IntelLLVM 2026/oneDNN 3.14
CPU build, run the full CPU/native/API gates for each S0 and C0 arm, then the
real warmup, ordinary, and one diagnostic request. Use the frozen 55-token
prompt, 128 greedy outputs, temperature 0, seed 42, cache false, ignore EOS;
S0 `max=1/min=2`, C0 `max=1/min=0`; `n_rs_seq=1`, K=2, capacity=2, q8 KV,
F32 recurrent state, context 65536, batch/ubatch 1024, parallel 1, graphs 0,
callbacks/verbose off. Preserve failed outputs in `worker-checks/`; do not
retry agreement, change the runtime baseline, clean disk, or alter the sole
original service. No timing or promotion is permitted.

## Preparation result

Preparation checks ran remotely on `inference-host`, without Docker/build/GPU/native/API
execution:

- Final `worker-checks/cpu-full-r7.log`: `Ran 85 tests in 169.738s`, `OK`; the
  suite also reports the router preflight exclusion/canary gate as passed.
- Remote targeted route preflight passed for S0 (130 frames) and C0 (70 frames):
  every target frame had only the two selected block8 routes, with the 304-wide
  router and selected latent-down cardinality/shape checks passing.
- `python3 -B -m py_compile analyze-trace.py run-trace.py test-trace.py` and
  `bash -n build-candidate.sh` passed.
- `worker-checks/patch-r10.log` passed fuzz-free dry-run/apply/reverse; the
  generated source hunk headers are retained in `generated-source.diff`.
- Earlier failed preparation attempts remain contemporaneously retained in
  `worker-checks/` (`cpu-full-r1` through `cpu-full-r6`, patch drafts, and the
  first-error summary); they were not overwritten.

`build-candidate.sh` and real S0/C0 capture remain lead-only and pending. Real
E2E, empirical both-arm parity, operand equality, restoration, timing, and
promotion conclusions are not claimed here.

## Lead-observed result — block8 router differs with identical operands

Lead verification ran entirely on `mike@inference-host`, via explicit local Bash and remote `bash -s` with the declared SSH flags, from this remote result root. Commands were `python3 -B test-trace.py 2>&1 | tee lead-cpu-r1.log` (**85 passed in169.708s**), then `bash build-candidate.sh`; after read-only review found no blockers, `python3 -B run-trace.py 2>&1 | tee controller-r1.log` and `python3 -B analyze-trace.py 2>&1 | tee analysis-r1.log`. The latter two were fail-fast chained. Preparation route checks above use synthetic records, not runtime evidence; the following results come from the actual compiled candidate and API captures.

Real build/dependency/source/five-binary gates passed. Both S0/C0 passed native385 + OUT_PROD190/support9 + IQ2six + latent7 + attention13 + view6 + router6, fresh API20/20, warmup, ordinary stream and separate diagnostic. Both arms preserve **all128 immediate-parent raw vectors byte-for-byte**, plus prompt/IDs/text/batch inputs/selected positions/rows. No continuation or inference rerun was needed.

At output2/position55/token0, the block8 router's full4,980,736-byte F32 weight and16,384-byte input match exactly across S0/C0. Its1,216-byte output differs: **274/304 float bit patterns differ; maximum absolute difference4.291534423828125e-6**, starting at byte0. The latent-down full4,456,448-byte reordered Q8 weight,16,384-byte input and4,096-byte output all match exactly. These are two projections and six files per arm, not six model weights; expert/shared operands were not captured.

Live route gates passed for all target frames: block8's router uses one full-width outer/inner MKL F32 pair at n1/n2 (DNNL F32 at prefill46); the existing block1 C0 router still uses two singleton pairs. The measured localization is `block8_router_projection`, not a sole-root-cause or complete-fidelity claim. Final raw scores still first differ at output2 before rollback, and emitted tokens at41.

Candidate SYCL SHA256 `fdc36e2a590dd3ebf0825d656ef19a866e0c8ad8d2c8fd5b19d37869dda2ffef`; the other four artifacts remain identical to router-single2. Controller restoration and independent analyzer recheck passed for the original container identity/config/HostConfig/all mounts, five hashes, protected sources, hardware, live health, READY and sole original process. The older router-control `test-trace.py` also matches committed bytes on the remote host: SHA256 `20f69c5b2357bc44e0b6f6246db8273260bf1a3643e3e9dc2642f677180d1a70`. No older baseline fixture change remains.

Canonical remote evidence: `run-r1/analysis.json`, per-arm alignments, raw `capture/` and `moe-operands/`, `router-difference.json`, and `router-width-inventory.json`. Local logs, worker failure logs/excerpts and JSON larger than1MB are lossless gzip with mtime0; remote originals remain intact. Valid unified-diff blank context markers are retained in patch/diff assets rather than treated as source whitespace. Lead removed one test-only trailing space before its85-test run. An initial archive command had a shell-quoting SyntaxError before producing any archive; corrected retrieval used the same saved artifacts, with no inference retry.

## Next bounded computational control — not yet implemented or verified

Fresh GGUF header inventory finds40 F32 routers with K4096 and exactly five output widths:240(12 routers),304(6),368(9),432(10),496(3). The current240-only guard covers only12 shapes-instances. Identical operands/different router outputs have now been measured at240 and304; other widths are not independently proven divergent.

The next development-only control will reuse the existing two-singleton path for the closed width set `{240,304,368,432,496}`, require matching source/destination dimensions, and preserve all other eligibility/range/alias/queue guards. Native coverage will exercise each width and unsupported/mismatched-width exclusions. Live route checks will cover representatives blk1(240),blk8(304),blk21(432),blk37(368),blk48(496); this is representative coverage, not a claim that all40 activations were logged. No operand observer is carried into that computational runtime. S0 must preserve all128 parent vectors; C0 may change and must follow its own newly generated proposal history, with comparisons limited to common prior history. Accepted runtime remains immutable. Timing and promotion stay blocked pending actual fidelity results.
