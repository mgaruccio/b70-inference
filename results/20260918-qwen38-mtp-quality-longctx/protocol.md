# Frozen executable-quality and cold long-context measurements

User selected **Both**: executable coding-quality checks, then cold long-context performance/capacity. Development measurements only, not the full repository standard-publishable suite. No training, checkpoint reselection, production changes or promotion.

## Frozen comparison

Stock native MTP versus the existing LR 1e-6 decay final-step-652 head, SHA256 `1af9142095c1d387847c83f27bdc330df8d34f81d0d73683c593ef5768babde5`. Same Qwen GPTQ model revision `9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e` and vLLM XPU image `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.

B70 32GB, 275W cap, native MTP4/C1, target GPTQ INT4 g128, unchanged draft RTN INT4 g128, FP16 compute, FP8 KV, configured context 212,992. Existing temporary reference launcher: balanced mode, graph sizes `[1,2,4,8]`, uniform-prefill guard and loopback API. Persistent launcher hash must remain `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`.

## Functional coding-quality phase

Use the **full 164-task HumanEval+ dataset**, not private traces or the consumed engineering holdouts. Public benchmark overlap with model pretraining is possible; this is a regression comparison, not proof of unseen-task generalization. MBPP+, GSM8K and IFEval are not part of this bounded round.

The official EvalPlus image is frozen as `ganler/evalplus@sha256:26b118098bef281fe8dfe999bf05f1d5b45374b4e6c00161ec0f30592aef4740`, installed package **0.4.0.dev2**, full HumanEvalPlus **v0.1.10**. Both official `latest` and `v0.3.1` tags resolve to the same image; its installed version, not its tag, is authoritative. This identity was inspected and recorded before generation. Freeze the downloaded dataset hash too. Use the full dataset (no Mini or no-extreme subset), official sanitizer with each task's entry point, and official evaluator.

**Pre-generation canary amendment:** the image's full canonical set scores163/164 because its `find_zero` special oracle skips updating success/progress before `continue`. The official upstream implementation fixes those two accounting lines. Build a temporary evaluator from the above base image with the complete official EvalPlus source pinned at `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` (`pip install --no-deps --force-reinstall`), record its immutable local image ID, and require164/164 canonical base+extra passes before generation. No benchmark tests, model outputs, scoring expectations or time limits are weakened. Preserve both earlier canary failures: the initial one-task probe rejected incomplete coverage, and the complete set exposed the upstream bug. Source: https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/eval/__init__.py .

Frozen dataset SHA256: `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`; prompt-only records SHA256: `87095ecb1cbbca5b344eeb895c2d74992bd0a9e717df6b5efb6b4a6dfda8b052`. These files stay unchanged across evaluator repair and all generation cells.

Use EvalPlus's chat instruction prefix (self-contained Python script in a Markdown code block) with the original task prompt; do not include canonical solutions or test inputs in model messages. Existing native corpus client → `/tokenize` and `/v1/chat/completions`; one non-thinking greedy completion/task, seed42, temperature0, top-p1/top-k−1, natural termination, **2,048 output-token cap**, complete-sequence limit8,192. No prompt truncation or skipped tasks. Preserve raw API responses and token IDs, then sanitize separately for executable scoring. Thinking is explicitly disabled per request. The output budget differs from EvalPlus's default768 and is identical across heads.

Run **stock/candidate/candidate/stock (ABBA)**, all164tasks in deterministic numeric-ID order per cell, four synthetic warmups excluded. Prefix cache remains enabled with unique salts for cold requests. Official base+extra pass@1 is primary; report original-base pass@1 separately. Report counts per cell and task-level stock/candidate regressions/improvements and within-head variability. Do not turn repeated completions into pass@2: each cell remains one sample/task. A task passed by both stock repetitions but failed by a candidate repetition is a regression flag, not automatically proof of a systematic model regression; report which failures are repeatable.

### Generated-code isolation

Separate generation and evaluation. Evaluation containers are CPU-only, network-disabled, read-only root filesystem, unprivileged UID/GID, all capabilities dropped, no-new-privileges, bounded CPU/memory/process count and outer timeout. Mount only public benchmark data, generated solutions, and the trusted evaluator wrapper read-only; no model weights, credentials, private traces, host Docker socket, or writable host directory. Work/cache/results live on container tmpfs. After the evaluator subprocess exits, its trusted wrapper emits results and captured logs to stdout before the container exits (tmpfs does not survive container shutdown); remove owned evaluation containers afterward. Use official evaluator timeouts and bounded parallelism consistently across cells. Run a trusted canonical-solution canary through this exact evaluator path before grading generated samples.

No generated engineering tool calls are executed. HumanEval source is executed only within the isolated evaluator. Passing this one coding benchmark does not establish tool-use quality, broad reasoning quality or long-context answer correctness.

## Cold long-context phase

After quality generation/evaluation, reuse the existing `qwen38_long_context_bench.py` public-boundary harness on stock then candidate. Lengths: **512, 8,192, 16,384, 32,768, 65,536, 120,000, 160,000, and212,000 actual prompt tokens**. Per head/length: one excluded warmup and **six measured runs**, each temperature0/seed42, **128 output tokens with EOS ignored**. Use the existing deterministic engineering-content renderer, actual `/tokenize` framing, and streaming `/v1/completions`. Compare identical tokenized prompts by length/trial across heads. No silent truncation.

Disable prefix caching in **both temporary long-context launchers**, record the modified temporary launcher/config hashes and verify `/server_info`; do not change the persistent launcher or add a new inference path. This intentional cold-cache profile differs from the earlier salted-prefix engineering tests. Thinking is disabled in the harness's explicit rendered template.

**Pre-measurement endpoint fallback:** the first stock attempt aborted with zero points because `/server_info` returns404. Its temporary launcher hash, explicit `--no-enable-prefix-caching`, and vLLM initialization log all confirm cache disablement. Preserve `long-A{,-output}` unchanged. Resume stock/candidate in fresh `long-verified-A/B` directories using the harness's existing `--confirm-prefix-cache-disabled` fallback, guarded by those same launcher/hash/runtime-log checks for each cell. Report this as operator-asserted with launch/log evidence, not API-verified. Workload lengths, repeats, output budget and settings are unchanged.

**Candidate staging interruption:** stock completed all eight lengths successfully. Before any candidate measurement, preparation failed with `IsADirectoryError`: `/tmp/qwen-b70-patch-uniform-decode-prefill.py` was a directory rather than the original file. Use the already-retained stock guard from the experiment directory, verified against launch metadata SHA256 `baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad`, and resume only the missing candidate cell (`run.sh long-candidate`). Preserve stock results unchanged. This adds an inter-cell gap; it is not a retry or exclusion of failed performance samples.

Report per-length median/IQR TTFT, prefill/decode/end-to-end rates, actual token counts, errors/timeouts/OOMs and speculative counters where valid. Retain runtime memory/KV-capacity logging; do not mislabel reserved cache or initialization allocation as a measured transient GPU-memory peak. Maximum successfully tested length is not a proven hard capacity ceiling. The212k point is a predeclared near-limit test, not an adaptive search. If a point/server fails, retain evidence and stop/classify through the existing harness; no cherry-picked retry or unreported fallback length.

This is a sequential two-cell cold sweep, not a time-interleaved server ABBA; six within-server repetitions estimate local dispersion but not all order/thermal variability. No long-context semantic-quality claim from the synthetic performance input.

## E2E, evidence and cleanup

Preflight observed: host idle,81GiB free,275W, correct frozen head and launcher hashes. An initial SSH command timed out; the bounded retry succeeded. A later `find | head` returned141/SIGPIPE after valid preflight output, not a GPU/readiness failure.

Execution path: pinned runtime/data staging → trusted evaluator canary → native stock warmup/full HumanEval generation → isolated official tests → matched candidate repeats and stock repeat → cold-prefix-disabled native stock/candidate sweeps → aggregate paired quality and per-length performance analysis → stop/remove all owned containers and verify persistent launcher, power and free space. Exact raw prompts/responses, failures, evaluator output, generation counters, SSE, token counts, launcher configs and commands remain under `/home/mike/b70-evals/20260918-mtp-quality-longctx/` on inference-host. No ML packages go into Pi. Git receives measurement helpers and aggregate reports only.

Quality regressions do not trigger tuning/reselection. Long-context measurements may still document behavior, but any quality/capacity regression must accompany performance results. No promotion is authorized regardless of scores.

## Fresh primary sources

- https://raw.githubusercontent.com/evalplus/evalplus/master/README.md and `docs/cli.md`, `docs/execution.md`: official OpenAI-compatible generation, executable base+extra tests and Docker warning. Defaults are not assumed; installed runtime is inspected.
- https://raw.githubusercontent.com/evalplus/evalplus/v0.3.1/evalplus/codegen.py and `evalplus/provider/openai.py`: official chat instruction and sanitizer/entry-point semantics; greedy means one completion/task.
- https://raw.githubusercontent.com/evalplus/evalplus/v0.3.1/evalplus/data/humaneval.py: full HumanEvalPlus v0.1.10 and local dataset override, allowing offline evaluation without changing test content.
- https://raw.githubusercontent.com/vllm-project/vllm/ac7509e2b/docs/features/automatic_prefix_caching.md: prefix reuse bypasses prefill work, hence explicit cache disablement for the cold sweep.
