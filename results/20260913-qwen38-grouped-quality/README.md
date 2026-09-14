# Qwen3.8 grouped quality harness

**Experiment-only, bounded, and not a production launcher.** This directory is
for the quality-sensitive comparison requested for the immutable
`../20260913-qwen38-native-grouped-verify` custom `build-06` campaign. It does
not build a model, change vLLM, install anything on the inference host, or
modify `scripts/start-qwen38.sh`. The lead owns the separate `run-quality.py`
launcher.

The three arms are:

1. `target`: target model with no speculation;
2. `native`: the same target model with lossless native MTP4;
3. `candidate`: the same target model with the grouped `build-06` path.

Every arm receives the same frozen JSONL row and the same HTTP request body.
The endpoint/serving configuration is the intentional arm difference. This is
a regression detector for this decoding change, not a general Qwen quality
benchmark and not a broad quality-equivalence claim.

## Files and commands

`quality.py` is stdlib-only at import time and has four subcommands:

```text
prepare   download pinned sources and freeze prompts
generate  POST one request per row to /v1/chat/completions
score     score saved responses offline with the pinned official tooling
compare   compare scores, identities, token divergence, repeats, and truncation
```

Invoke all subcommands without a leading space.

### Prepare once in the remote evaluator image

`prepare` is the only step that needs dataset-download network access. It
refuses source hash/count drift and writes raw downloaded files below
`sources/`, `prepared.jsonl`, and `manifest.json`:

```bash
R=/absolute/path/to/results/20260913-qwen38-grouped-quality
RUN=/absolute/path/to/quality-run
IMAGE=qwen38-quality:20260913
mkdir -p "$RUN"
docker run --rm --network bridge --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --user 65532:65532 \
  -v "$RUN:/input:rw" \
  "$IMAGE" prepare --out /input/prepared
```

The image is built from this directory, with no inference runtime:

```bash
docker build --pull=false -t "$IMAGE" "$R"
```

A full prepared set contains 2,402 primary rows plus 64 additional repeat
requests (2,466 HTTP requests per arm):

| Task | Primary rows | Scoring |
|---|---:|---|
| IFEval `google/IFEval`, train | 541 | official lm-eval strict and loose |
| GSM8K `openai/gsm8k`, main/test | 1,319 | official task-format strict and flexible exact |
| HumanEval+ | 164 | EvalPlus base and extra tests, pass@1 |
| MBPP+ | 378 | EvalPlus base and extra tests, pass@1 |

The first 500 numerically sorted IFEval IDs are marked
`divergence_sample=true`. The first 32 of those IDs are each repeated twice
as `<base-id>/repeat-1` and `<base-id>/repeat-2`. Repeats are never counted as
additional benchmark items by `score`; they are retained for nondeterminism
diagnostics. Long-context requests are **not** fabricated here: the lead may
pass a separately supplied long-request JSONL (with an `id` and `prompt` or
`messages`) through `generate`.
For an explicit one-request smoke run (not a full result), use `--limit 1`:

```bash
docker run --rm --network host --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --user 65532:65532 \
  -v "$RUN/prepared:/input/prepared:ro" -v "$RUN/smoke:/output:rw" \
  "$IMAGE" generate --arm target --data /input/prepared \
    --out /output/target.jsonl --base-url http://127.0.0.1:8000/v1 --limit 1
```

Repeat that command for `native` and `candidate`, changing only `--arm` and
output filename. `--limit` is deliberately explicit; generation does not
silently downsample, retry, repair, or filter.

### Generate full arms

After the lead starts each already-qualified server/configuration, run exactly
the same command shape once for each arm, changing only the endpoint/arm
lifecycle and output path:

```bash
mkdir -p "$RUN/generation"
# Run once with --arm target, once native, and once candidate.
ARM=target
docker run --rm --network host --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \
  --user 65532:65532 \
  -v "$RUN/prepared:/input/prepared:ro" -v "$RUN/generation:/output:rw" \
  "$IMAGE" generate --arm "$ARM" --data /input/prepared \
    --out "/output/$ARM.jsonl" --base-url http://127.0.0.1:8000/v1
```

`generate` makes one non-streaming stdlib `urllib` POST per row and saves the
exact canonical request body, parsed response, raw response text and base64
bytes/hash, output and prompt token IDs, finish/stop reasons,
usage, truncation, errors, and monotonic/UTC timing. `return_token_ids=true`
is mandatory for both ID lists: IDs are never inferred from text. A missing or
malformed output ID, finish reason, content, usage count, or HTTP response is
recorded and makes that row failed; the command exits nonzero after retaining
all rows. There is no silent retry or repair. `*.meta.json` records arm,
selected IDs, completeness, hashes, and the request contract.

The request contract is fixed for all arms and all scored tasks:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "seed": 42,
  "stream": false,
  "return_token_ids": true,
  "chat_template_kwargs": {"enable_thinking": false},
  "cache_salt": "C1",
  "max_tokens": 4096,
  "eos": "normal (no ignore_eos)"
}
```

GSM8K additionally carries the official YAML stop strings
`Question:`, `</s>`, and `<|im_end|>`. All other tasks leave stop strings
empty. The pinned model is already compatible with
`enable_thinking:false`; no response text is rewritten before scoring.

### Offline scoring

All scoring is offline.  The single `score --task all` output keeps the four
task families together for `compare`; it requires the sandbox flag because the
same run also invokes EvalPlus:

```bash
mkdir -p "$RUN/scores"
for ARM in target native candidate; do
  docker run --rm --network none --read-only \\
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \\
    --cpus 4 --memory 8g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=2g \\
    --user 65532:65532 \\
    -v "$RUN/prepared:/input/prepared:ro" \\
    -v "$RUN/generation:/input/generation:ro" \\
    -v "$RUN/scores:/output:rw" "$IMAGE" \\
    score --task all --allow-code-execution \\
      --data /input/prepared --generation "/input/generation/$ARM.jsonl" \\
      --out "/output/$ARM.scores.json"
done
```

For a no-code smoke check, run the same command with `--task ifeval` and/or
`--task gsm8k` and omit `--allow-code-execution`.

EvalPlus is the only code-execution path.  The all-task command above is valid
only in the sandbox image: `--allow-code-execution` is explicit, the process is
the non-root image user inside Docker, `QUALITY_EVAL_SANDBOX=1` is present, and
generated inputs are under the read-only `/input` mount.  For a code-only run,
use `score --task evalplus --allow-code-execution` with the same network-none,
read-only, cap-drop, non-root, CPU/memory/PID limits and only `/output` writable.

The evaluator calls `lm-eval==0.4.13`'s IFEval implementation and task-format
regexes, and `evalplus==0.3.1`'s own base/extra test evaluator and pass@1
estimator. Prepared source rows are supplied through EvalPlus's documented
override mechanism, so scoring cannot silently select a newer dataset. Missing
or failed generation rows abort scoring; no fake zero is emitted. Official
EvalPlus result JSON is preserved beside each score file.

### Compare

```bash
mkdir -p "$RUN/compare"
docker run --rm --network none --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 \
  --cpus 4 --memory 4g --tmpfs /tmp:rw,noexec,nosuid,nodev,size=512m \
  --user 65532:65532 \
  -v "$RUN:/input:ro" -v "$RUN/compare:/output:rw" "$IMAGE" \
  compare \
    --target /input/generation/target.jsonl \
    --native /input/generation/native.jsonl \
    --candidate /input/generation/candidate.jsonl \
    --target-score /input/scores/target.scores.json \
    --native-score /input/scores/native.scores.json \
    --candidate-score /input/scores/candidate.scores.json \
    --out /output/compare.json
```

Run `compare` once per score family if separate score files are used, or pass
score outputs produced by one `score --task all` invocation. The report keeps
absolute arm scores and native/candidate deltas, request/prompt/token identity,
first-divergence position distributions, repeatability, truncation, missing
IDs, and descriptive paired Boolean differences with a normal 95% interval.
Intervals are descriptive only and do not establish quality equivalence or
non-inferiority.

## Exact pinned inputs

`prepare` verifies these bytes before parsing and records every value in
`manifest.json`:

| Input | Immutable revision/release | Expected bytes/hash |
|---|---|---|
| `google/IFEval` train JSONL | HF `966cd89545d6b6acfd7638bc708b98261ca58e84` | 541; `6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2` |
| `openai/gsm8k` main/test parquet | HF `740312add88f781978c0658806c59bc2815b9866` | 1,319; `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| `openai/gsm8k` main/train parquet | same HF revision | 7,473; `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` |
| HumanEvalPlus release | EvalPlus `v0.1.10`, package `0.3.1` | 164; gzip `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101` |
| MbppPlus release | EvalPlus `v0.2.0`, package `0.3.1` | 378; gzip `af43697e8791c4c149bdfd6b489d8b5412507551ac20e28a439f650b8225db63` |

## Research record

Fresh primary-source inspection informed the implementation:

- [lm-eval IFEval task](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.13/lm_eval/tasks/ifeval): task uses `google/IFEval` train, prompt field, zero shots, and four strict/loose prompt/instruction metrics.
- [lm-eval IFEval scorer](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.13/lm_eval/tasks/ifeval/utils.py): strict checks each registered instruction; loose scoring tests bounded newline/asterisk variants.
- [lm-eval GSM8K YAML](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.13/lm_eval/tasks/gsm8k/gsm8k.yaml): main/test, train few-shot split, `Question:`/`Answer:` format, five shots, official strict and flexible regex filters, and normal EOS stop strings.
- [EvalPlus CLI](https://github.com/evalplus/evalplus/blob/v0.3.1/docs/cli.md): official generation/sample JSONL shape and `evalplus.evaluate` entry point.
- [EvalPlus execution guidance](https://github.com/evalplus/evalplus/blob/v0.3.1/docs/execution.md): generated code is untrusted and requires isolation; the Docker invocation above adds network-none, read-only, cap-drop, non-root, CPU/memory/PID limits.
- [vLLM OpenAI-compatible serving](https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/): `/v1/chat/completions` is the public boundary; request bodies remain OpenAI-compatible with server extensions retained for token IDs.
- [vLLM reproducibility guidance](https://docs.vllm.ai/en/stable/usage/reproducibility.html): deterministic settings do not justify assuming batch/XPU invariance, so repeats and token-level divergence are retained.
- [Official Python image](https://hub.docker.com/_/python): the Dockerfile pins the available `python:3.11.11-slim-bookworm` manifest digest.

## Lead integration contract and defined E2E process

The lead must keep the existing custom campaign and launcher immutable. The
launcher should build the remote CPU image, run the commands above with the
same prepared directory for all arms, save raw generation/score/compare files,
and retain container logs and image inspection. The public journey is:

1. On the remote evaluator host, verify the inference host is the existing
   Qwen3.8 C1 service and record the pre-run launcher hash and the existing
   275 W setting. Build this directory's image and record `docker image inspect`.
2. Run `prepare` once. Check `manifest.json`, all counts, source hashes, the
   prepared JSONL hash, five fixed GSM shots, 500 divergence marks, and 32×2
   repeat rows.
3. Start each real arm at the public HTTP boundary. Run `generate --limit 1`
   for each arm, then offline IFEval/GSM8K smoke scoring (and explicit sandbox
   EvalPlus smoke scoring if desired). Keep the JSONL, meta, score, logs, and
   exit code; a missing ID or HTTP failure is a failed smoke gate.
4. With the same prepared data and unchanged scored request contract, run the
   full 2,466 requests for target, native, and candidate. Score all three arms
   offline in the sandbox and run `compare`.
5. Save `prepared/`, `generation/`, `scores/`, `compare/`, image metadata,
   commands, environment, raw stdout/stderr, and exit-code files as the
   experiment evidence. Do not replace failed rows with retries or omit them.
6. Stop/remove only the temporary evaluator and arm containers, verify no
   quality container remains, verify the launcher file hash is unchanged, and
   re-record the existing 275 W setting. Do not modify or rebuild the
   immutable custom `build-06` campaign from this harness.

The worker ran only local stdlib syntax/contract checks; no dataset download,
package install, inference request, GPU/ML runtime, or remote execution was
performed in Pi.


## Executed validation (2026-09-13–14)

**Verdict: quality neutrality is NOT established.** All three full arms completed and scored, but the candidate lost three HumanEval+ passes relative to native MTP4. Do not promote this kernel on the strength of throughput alone. This is a quality-sensitive development evaluation, not the complete standard-publishable performance package or a statistical equivalence test.

| Quality metric (%) | Target-only | Native MTP4 | Grouped candidate | Candidate − native (pp) |
|---|---:|---:|---:|---:|
| IFEval prompt strict (541) |83.36|82.26|82.07|−0.18|
| GSM8K strict (1319) |96.66|96.29|96.36|+0.08|
| HumanEval+ pass@1 (164) |85.37|84.76|82.93|−1.83|
| MBPP+ pass@1 (378) |71.16|70.90|71.16|+0.26|

HumanEval/1, /19 and /130 changed from native pass to candidate fail, with no offsetting passes. IFEval had7 regressions/6 improvements, GSM8K strict1/2, MBPP+0/1. The descriptive paired normal95% interval for the HumanEval+ delta is −3.89 to +0.23pp: an observed regression, not proof of a deterministic causal quality loss. Only one full sample per task/arm was taken. No non-inferiority margin was prespecified. Code prompts explicitly request unfenced complete scripts; official EvalPlus scoring is used without sanitizing generated text. GSM8K uses a fixed first-five-training-example prompt rather than randomized few-shot sampling. These are paired-regression settings, not a claim of canonical leaderboard comparability.

### Speed comparison (separate completed ABBA experiment)

| Context | Native tok/s | Candidate tok/s | Change |
|---:|---:|---:|---:|
|512|70.8663|70.1855|−0.96%|
|8192|63.2092|65.5345|+3.68%|
|32768|61.2679|63.2420|+3.22%|
|65536|54.4637|59.2835|+8.85%|

Source: `../20260913-qwen38-native-grouped-verify/analysis-confirmation.json`, 12 samples/arm/length. The64K conditional prompt-cluster bootstrap interval was +1.63 to +11.68%; TTFT remained approximately53.9s. These are NOT throughput estimates from the quality corpus, and target-only has no matched ABBA speed cell.

### Fidelity and limitations

All2,466 request payload hashes and rendered prompt-token sequences match across arms; zero failed requests. Among the500 preselected IFEval prompts, exact target-output token identity was54.4% for native and56.2% for candidate. Candidate/native identity was55.6% (278/500), with median first divergence111.5 (one-based). The target control itself is nonrepeatable: of32 repeated prompts, base/repeat1 match30, base/repeat2 match29, and repeat1/repeat2 match28. Native matches30/32 in each comparison; candidate29/29/30. Thus native does not reproduce target-only exactly, and these results do not isolate kernel-induced divergence from ordinary runtime nondeterminism. No batching-invariance or precision change was introduced to hide this discrepancy.

Each full arm also completed two repeats of the same six real65536-token prompts, generating128 tokens with EOS ignored and retaining actual prompt/output IDs. Target/native matched4/12; target/candidate6/12; native/candidate5/12. Within-arm repeat identity was target4/6, native2/6, candidate4/6. This is a long-input fidelity diagnostic, not a long-context task-accuracy benchmark.

Scored output budget4096, EOS normal. Truncated requests were retained: target2, native3, candidate4. Inspect `comparison.json` for their IDs, strict/loose IFEval, GSM8K strict/flexible, base/extra code scores, repeatability and full divergence histograms. `analysis-native-candidate.json` adds direct task transitions and64K identity. No samples were discarded and no quality score was substituted for an error.

### Execution, corrections and artifacts

All model/evaluator execution occurred on `inference-host`. The pinned server image, target/draft weights, K4, FP8 KV,212992 configured capacity,8192 batch,C1 and275W remained unchanged; target-only removes only `--speculative-config`. Candidate build06 SHA is `e0c6f2a78a1a50eef9dcc11b9c378c2e94799a3f5ffa0c8971849f03b3c1ddec`; `full-candidate-01/candidate-execution-evidence.json` records eligible observed152-page interleaved dispatch plus FULL capture/run evidence, with no unsupported-q5 logs. Production launcher SHA stayed `63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4`; all full summaries report successful cleanup/host unchanged, and final `docker ps` was empty.

Two initial shell commands failed127 before Docker because Fish rejected Bash syntax; saved Bash scripts fixed the launch boundary. Build01 succeeded with CPU Torch2.6.0+cpu and pinned evaluators. Preparation01 failed on official MBPP/404 IEEE infinity. The fix preserves complete native EvalPlus source JSON in each row's `source_json` string and hashes those bytes; `source` retains ordinary metadata, while the scorer restores original test-input JSON verbatim. No infinity was coerced and no example removed. Build02/preparation02 succeeded. The first target smoke generated/scored inputs successfully but scorer import failed on missing `langdetect`; build03 installs the official `lm-eval[ifeval]==0.4.13` extra and downloads `punkt_tab` at build time for offline use. Review also caught/fixed GSM8K flexible extraction (last number, per pinned YAML `group_select:-1`), with a multi-number regression test.

The final executed continuation command was:

```bash
ssh inference-host bash /home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4/20260913-qwen38-grouped-quality/continue-quality.sh
```

It rebuilt evaluator03, scored the retained target smoke, generated/scored native and candidate smokes, then generated/scored all three full arms serially and ran comparison. Exit0, elapsed8h40m. `continue-quality.sh`, `run-quality.py`, `score-cell.sh` retain exact steps; full per-arm generation timeout86400s. Every code scorer container had no network, read-only root, no capabilities, no-new-privileges, non-root UID, CPU/memory/PID limits, tmpfs, and only dedicated input/output mounts—no host home, Docker socket or GPU. Build metadata identifies evaluator image; each score cell retains its image ID. Raw results are in `full-{target,native,candidate}-01/`; preparation02 holds the frozen corpus/manifest/source hashes. Original source download URLs/checksums and the remote archives allow reacquisition of excluded parquet/gzip downloads.

Offline verification:

```bash
python3 results/20260913-qwen38-grouped-quality/check-local.py
python3 results/20260913-qwen38-grouped-quality/analyze-native-candidate.py > results/20260913-qwen38-grouped-quality/analysis-native-candidate.json
```

No model/runtime was installed in Pi. Native Lab preview/patch UI was unavailable (`ENOENT lab.sock`); the previously authorized CLI fallback was used. Production remains unchanged.
