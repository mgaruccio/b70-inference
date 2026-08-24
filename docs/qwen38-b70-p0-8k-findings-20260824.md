# P0 ~8k greedy divergence findings — 2026-08-24

Live BF16 was restored after this run. This is a quality-gate result, not a speed claim.

## Artifacts

`~/b70-evals/qwen38-b70-gptq-int4-mtp4/20260824T010528Z-p0-8k/`

- Prompt: 365 repeats, **8459** tokens, `temperature=0`, `seed=12345`, `max_tokens=128`
- BF16: 36 verify events, 7.16 s, finish `length`
- Draft-INT4 / cap64 / MTP-4: 37 verify events, 7.03 s, finish `length`
- Public contents still differ (same mismatch class as 2026-08-23)

## First divergence

Join by `(step, context_len)`, not `request_id` (each server mints its own chatcmpl id).

| Step | ctx | BF16 draft → target | INT4 draft → target | Accept |
| --- | ---: | --- | --- | --- |
| 2 | 8459 | `need to respond to` → `need answer respond to` | identical | prefix 1 |
| **3** | **8461** | `user's request.` → `user's request:` | `user's request:` → `user's request:` | **3 vs 4** |

Decoded with the pinned model tokenizer:

- `13` = `.`
- `25` = `:`

At step 3 the **target IDs match**: `[1156, 579, 1622, 25]` (`user` + `'s` + `request` + `:`).

Only the last **draft** token differs (`.` vs `:`). BF16 rejects it and writes the target `:`. INT4 accepts all four drafts and also takes a **bonus** token.

Committed after step 3:

- BF16: 4 tokens, no bonus, next `context_len=8465`
- INT4: 5 tokens, bonus used, next `context_len=8466`

Every later join fails because the prefixes are different lengths. The visible 8k text split (`repeated` vs `repeated identical sentence:`) is downstream of that bonus.

## Classification

**Draft-only proposal difference that becomes accept/bonus bookkeeping.**

Not a first-step target-numerical error: both targets already wanted `:`. Draft-INT4 proposed the target token and therefore received the bonus; BF16 proposed `.` and did not.

## Keep / kill

**Freeze draft-INT4 as a deployable / agent-quality platform.**

The mismatch is explained, but it **does change committed tokens** (bonus). That violates the keep rule that a draft-only difference must be unable to change the target stream.

It may remain a short-decode curiosity. Do not use it for Harbor/Pi claims or as the live service.

## Comparator note

`compare_qwen38_b70_p0_traces.py` currently joins on `request_id`, which is unique per server. Use `(step, context_len)` for cross-server greedy pairs.
