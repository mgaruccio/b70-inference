# Lead review and observed attribution failure

Computational preparation commit `e94d94cf4dd8c5c6d64d36ce5627c7515435ea66` was integrated as `f49f4ee2`. Read-only review found no blocking defect in the two-column preflight, bounded recursion, allocation/alias checks, F32 null-extra handling, column offsets, precision preservation, existing MKL submission path, or restoration checks. This is a source-review result, not a numerical or fidelity pass.

The lead then completed the real SYCL build and 66 remote CPU tests. The S0 run passed inherited native gates, the six new router numerical/actual-route cases, API20/20, warmup and ordinary/diagnostic requests. The original all-frame analyzer failed at position70 with `missing or ambiguous router target backend`; the original failed summary and captures are retained. The controller restored the original service. C0 had not run.

The lead separately revalidated all128 saved S0 raw vectors, text, IDs, batch inputs, selected positions and rows against the IQ2 computational parent. This does not bypass the failed attribution gate or mark the original run complete.

## Recovery-design review

The saved direct route stream is intact:130 target router operations, with widths5,46,4 followed by127 singleton operations. At position70, six direct route records precede the delayed BOLT_VERIFY batch marker; textual batch containment is therefore invalid. The earlier review missed this cross-logger framing assumption.

The follow-up read-only review confirmed the two-stream association for this parallel1 / GGML_SYCL_ENABLE_GRAPH=0 runtime:

- `bolt_verify_batch` enqueues LOG_INF before `llama_process`; the async common logger preserves FIFO order, but Docker merges its output with direct stderr without a cross-stream ordering guarantee.
- `yield_to_queue` executes decode on the current thread and cannot nest. The next batch cannot be enqueued before the prior `llama_process` returns.
- With graphs disabled, SYCL dispatch walks graph nodes in order. Router BOLT_ROUTE records are synchronous stderr writes during dispatch. Both singleton recursive calls occur inside the same target decode.
- Identify one target backend using both block0 SSM projection names; do not equate a llama context pointer with a SYCL backend pointer. Retain other-backend/drafter routes separately.
- Associate ordered target router operations with ordered target batch events using exact cardinality, width/stride/type/precision checks and no leftovers. Require the prefills5/46/4 (width46 uses LIBRARY_DNNL_F32), one pair for singleton batches, and two width-one pairs with original width-two strides for width-two batches.
- Preserve the existing first-selected unrelated route gates. Reanalyze saved S0; run only missing C0 in a separate namespace. No computational-source rebuild or S0 inference rerun is justified by this parser failure.

At review time, parser correction and C0 execution were still pending. Their subsequent observed results are recorded in README.md and continuation-comparison.json. No timing, promotion or overall fidelity pass is asserted here.
