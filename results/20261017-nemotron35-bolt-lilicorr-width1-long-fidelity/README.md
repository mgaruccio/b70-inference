# Bounded 8K width-one fidelity preparation

**Development fidelity only. Real fresh A0/C0 pair PENDING. No timing, promotion, higher depth, new build, runtime/source change, or observer callback change.** This is not a standard-publishable performance comparison under `BENCHMARKING_STANDARDS.md`.

## Fixed scope and prerequisites

Remote root (R): `/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity`. Only this case and its `worker-scratch/` are writable. Completed cases and all helpers are read-only. Worker's CPU tests are synthetic/regression checks, not new inference evidence.

Reuse `/home/mike/inference/src/llama.cpp-bolt-b6334e37-nvfp4-lilicorr-router-widths-single2-r1` exactly. `contracts.py` reuses `run-a0.py`'s five binary pins and complete 3805-file candidate/accepted source guards, plus pins the endpoint, sampling, and dispatch sources inspected here. No rebuild or clone. Tested local controllers and legacy helpers are frozen by `preparation/tested-code.json` before launch.

- Baseline A0: **true no-draft**, entire speculative tail and draft mount removed. Natural target K1, `n_rs_seq=0`, output capacity 1, no feature extraction.
- Candidate C0: saved active `draft-dflash`, max 1/min 0, exact old speculative argv. Natural target K2, `n_rs_seq=1`, output capacity 2, DFlash features; separate drafter context.
- No parent batch/row invariance requirement or shape forcing. No claim of raw completion IDs for quality20/anchor.
- Both: q8 KV, default F32 recurrent state, context 65536, batch/ubatch 1024, parallel 1, graphs 0, callbacks/verbose off. Existing native385 + OUTPROD190/support9 + IQ2 six + latent seven + attention13/view6 + router30 gates unchanged, through the immutable read-only mounts.
- Original sole-live `nemotron-bolt-probe`, ID `7239883bad607d3be561d1511a4a24fb78a0ddc96c847f752761a0ad242fc54e`, localhost:18098; exact Config, HostConfig, mounts, source and five accepted binaries protected. Hardware must equal saved 275W/boost0/16x3801000 state. Worker does not query/change hardware or services.

Existing official research for this ongoing task is retained in the no-draft case: https://github.khronos.org/SYCL_Reference/iface/queue.html and https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-2/gemm.html . Queue/stride semantics do not imply cross-shape bitwise identity. Exact pinned source, not current upstream assumptions, governs the new endpoint/dispatch contracts.

## Frozen input and real public-boundary process (lead only)

Preconditions: CPU gate green with unchanged controller/helper hashes; same model/runtime/source pins; original healthy and sole-live; names `bolt-lilicorr-width1-long-fidelity-r1-a0` and `...-c0` absent; `run-r1` absent; fresh capacity checks pass. No optional 256-token code case and no 55-token rerun.

The controller freezes and reuses the saved requests from `20261017-nemotron35-bolt-decode-profile/run-r1/baseline/long/`:

1. After unchanged native gates, the same API20 in saved order and with **all original max_tokens=256**.
2. `anchor.request.json`: historical **8238** prompt tokens, max 128, nonstream, EOS respected. Original criterion from `probe-api.py`: stripped text exactly `BOLT-7319-CEDAR`, finish `stop`, actual usage prompt count >=8192. No new completion-count criterion.
3. `measured-00.request.json`: historical **8270** prompt tokens, max 128, temperature 0, seed 42, cache off, ignore EOS true, streaming/include_usage. Freeze once and send **identical full payload/messages** for warmup and ordinary in both arms. Do not use historical 8269-token warmup or regenerate a nonce.
4. One separate 128-token diagnostic, derived only by existing `diagnostic_payload`: nonstream, verbose, return_tokens, logprobs/top5, backend_sampling false; remove stream_options. No other payload changes.

Before each arm's FIRST inference, save the actual `/apply-template` request/response and `/tokenize` request/response for every frozen request under `public/00..23.*`. Source `tools/server/server-context.cpp:5880+` uses the same `oaicompat_chat_params_parse` as chat. `server-common.cpp:1360+` defaults add_generation_prompt to true and merges boolean `chat_template_kwargs.enable_thinking=false`. Then `server-context.cpp:5103` calls tokenization with **add_special=true, parse_special=true**. Therefore render the full messages with explicit add_generation_prompt=true, then tokenize the returned prompt with both true, with_pieces=false. Raw message content alone is NOT the chat prompt. Historical counts are expectations, not substitutes for actual IDs/counts.

For each generation retain exact JSON payload and unmodified HTTP body (SSE for streams), HTTP status/headers, including partial/error bodies. The controller performs no fidelity or quality postprocessing between arms. Only native/startup, public tokenization, HTTP transport, and capacity safety gates can stop capture. All quality/anchor responses in both completed arms are normalized independently after restoration; a failed response/parser does not erase later valid evidence.

The diagnostic window is delimited by the Docker log prefix immediately before its HTTP request and the complete log after stopping the candidate (draining async logging). Save prefix byte offset/hash, through offset, full log and exact window. Invalid boundaries are post-restoration failures, not permission to recapture. A single fresh A0 then C0; no inference retries, even on parser failure.

After capture (or exception/signal), stop and **preserve** candidates, restart the exact original, require READY/stop and live /health, sole-live identity, unchanged Config/HostConfig/mounts/source/binaries/hardware. Independent analysis runs only after restoration and makes a separate live inspect/health check; stored snapshots are not presented as live health. Nothing is deleted. No persistent launcher changes; all model weights untouched.

Lead commands, each through explicit local AND remote Bash and the fixed SSH isolation options (launch via the lead's background tool, never a foreground multi-minute wait):

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '\''cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity && python3 -B run-long.py --lead-run-one-pair > controller-r1.log 2>&1'\''"'
# ONLY after the controller has exited and original restoration succeeded:
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '\''cd /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity && python3 -B analyze-long.py --lead-after-restore > analyzer-r1.log 2>&1'\''"'
```

Expected: unchanged native gates; functional API20 and anchor; actual public tokenization equals diagnostic BOLT_VERIFY prompt IDs, usage counts, and cross-arm IDs; within-arm ordinary/diagnostic text/finish/count and diagnostic verbose/logprob IDs agree. Actual contiguous prefill chunks <=1024 are checked, never a forced partition. A0 must be singleton-only. C0 acceptance/rejection/rollback counts come from the capture, not old expected counts.

Independent direct BOLT_ROUTE FIFO and asynchronous BOLT_VERIFY FIFO are paired using the unique block0 SSM backend anchor, not relative print positions. Every target frame requires exact weight order/cardinality, shapes/strides/types/precision/backend/results: block0, block1 experts/latent/shared, block7 attention, all five routers 1/8/21/37/48 (before last layer87), and the pruned head. Foreign drafter routes are retained separately. NVFP4 retains F32 library operands/BF16 accumulation hint; Q8 large prefill uses F16 library; shared IQ2 is fused only through width64, then F16 library. `supports_mmq()` is false in the pinned runtime. Grouped IQ2 requires `9*n <= active_experts*64`; only successful result=1 is accepted. An actual fallback or unsupported shape preserves the complete capture and fails after restoration; no parser relaxation or arithmetic patch.

Compare the full 131072-element F32 vectors only under shared prior emitted history, **including the first differing emission**, or all128 if IDs match. Report first score/ID difference and actual batch/selected-row metadata, without claiming that different K1/K2 metadata is an error. Existing raw basename, directory/symlink, regular-file, byte-count, vocabulary and finite-value guards remain.

Artifacts: `run-r1/{A0,C0}/` exact requests/public renders/IDs/HTTP bodies/native logs/full server log/diagnostic window/remote-only captures; per-request capacity records; source/binary/hardware/restoration records; independent `analysis.json`, response checks and alignments. No raw vectors, runtime sources, weights, or binaries in Git. Keep failures. No publication checklist or performance claim is satisfied by this bounded pair.

## Storage bound

Observed before preparation: free **11562459136 bytes** (lead earlier 11562463232). Reserve stays **4 GiB**. Each arm requests at most **5632** output tokens: 20*256 + anchor128 + 3*128. Expected ~495 capture rows is NOT the budget.

Pinned source bound: `common/sampling.cpp:651-790` writes exactly once per sampler call, before the grammar retry (retry reuses the file), and each greedy sample is appended to the accepted list. `common/speculative.cpp:1335-1369` takes the LiLiCorr lattice branch and continues without sampler calls. `server-context.cpp:556+` clamps draft width to remaining output budget minus one; single and accepted tokens increment n_gen before stop handling. With max draft1 and rollback capacity1, no checkpoint replay is needed. Thus budget **cap+2 raw rows per request**, including an entire extra terminal width-one verification even if an accepted bonus is not emitted after EOS/stop. This is more conservative than the source's cap/one-discarded-bonus bound; it does not assume one raw row per API-reported output token. Request 128 diagnostic aligns exactly128 real emissions separately.

Maximum raw rows/arm = **5680**, each **524288** bytes: **2977955840** raw bytes. Add **1 GiB** for startup, selected-route/Docker logs, diagnostic duplication, public requests/responses, JSON, and filesystem overhead; retain **4 GiB** free: initial required **8346664960 bytes**. The log amount is a conservative allowance, not a measured performance value or a generic bound for other workloads. No debug/callback/all-layer verbose logging is enabled. All remaining requests are budgeted anew before EVERY request and before C0, using actual free bytes and capture file counts/logical/allocated bytes. A0's retained evidence consumes free space; C0 can be refused safely, with A0 and original restoration preserved. Never shrink API caps/reserve, toggle capture, clean up, or add storage infrastructure.

## CPU preparation only

Worker has no `bg_run`: hand the following bounded runner to the lead before commit. It imports (does not launch) controllers, reads real existing fixtures/argv/logs/captures, checks the actual five binaries/full source maps read-only, and exercises synthetic larger prefill/FIFO/rollback/cardinality/metadata/dispatch/tokenizer/budget/refusal/restoration failures. Public HTTP and restoration commands in these tests are mocks. No Docker, native tests, GPU, API, service or hardware execution occurs.

```bash
bash -lc 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=no -o ControlPath=none -o ForwardAgent=no mike@inference-host "bash -lc '\''bash /home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-width1-long-fidelity/cpu-checks.sh'\''"'
```

Runner: 180-second CPU timeout (+10-second kill grace), unique preserved `worker-scratch/cpu-*` per attempt; success copies small results to `preparation/` and freezes tested code. Failures and logs are retained. No cleanup step. The lead owns the real journey above after integration; passing synthetic tests is not new 8K fidelity evidence.
