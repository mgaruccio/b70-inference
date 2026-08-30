# Qwen B70 — next session

Current measured context work:
[`qwen38-b70-200k-context-20260830.md`](qwen38-b70-200k-context-20260830.md)

Live `start-qwen38.sh` is still **131k / 0.88 / 64 seqs**. Do not raise
`max-model-len` at 0.88.

Optional next cuts (explicit, not implied):

1. Persist `power/control=on` for the B70 (udev). Headless
   `multi-user.target` is not a VRAM win.
2. Decide whether live should become C1 200k @ 0.95 (drops concurrency).
3. Unblock `reasoning_effort` if we want model xhigh thinking on this
   cell (`supports_reasoning_effort = false` in the eval toml).

Older speed/quality handoff remains
[`qwen38-b70-next-session-handoff-20260823.md`](qwen38-b70-next-session-handoff-20260823.md)
(draft-INT4 still parked).
