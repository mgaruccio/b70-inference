"""Prompt set for DFlash acceptance characterization.

Two suites, tagged so we do not mix paper-comparable τ with this cell's
real workload:

* eval — Spec-Bench / DFlash-paper families (GSM8K, HumanEval, MT-Bench).
* work — local C1 use: coding-agent, failing tests, code review, tool calls,
  inference-ops triage.

The frozen gate prompt stays in vllm-dflash-instrument.py (sky). Do not
reuse it here.
"""

PROMPTS = (
    {
        "id": "gsm8k-janet",
        "suite": "eval",
        "kind": "math",
        "source": "GSM8K test[0]",
        "prompt": (
            "Solve this grade-school math problem. Show every arithmetic step.\n\n"
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
            "morning and bakes muffins for her friends every day with four. She sells "
            "the remainder at the farmers' market daily for $2 per fresh duck egg. How "
            "much in dollars does she make every day at the farmers' market?"
        ),
    },
    {
        "id": "humaneval-0",
        "suite": "eval",
        "kind": "code-completion",
        "source": "HumanEval/0",
        "prompt": (
            "Complete the following Python function. Return only the function "
            "implementation, including the docstring.\n\n"
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
    },
    {
        "id": "mtbench-81",
        "suite": "eval",
        "kind": "chat-writing",
        "source": "MT-Bench question 81 turn 1",
        "prompt": (
            "Compose an engaging travel blog post about a recent trip to Hawaii, "
            "highlighting cultural experiences and must-see attractions."
        ),
    },
    {
        "id": "fail-test-fix",
        "suite": "work",
        "kind": "coding-agent",
        "source": "local C1 / Harbor-like failing test",
        "prompt": (
            "A pytest run failed. Fix `parse_duration` so the tests pass. Do not "
            "rewrite unrelated code.\n\n"
            "```python\n"
            "def parse_duration(text: str) -> int:\n"
            "    # '90s', '2m', '1h30m' -> seconds\n"
            "    return int(text)\n"
            "```\n\n"
            "Failures:\n"
            "  assert parse_duration('90s') == 90  # got ValueError\n"
            "  assert parse_duration('2m') == 120\n"
            "  assert parse_duration('1h30m') == 5400\n"
            "  assert parse_duration('1h30m15s') == 5415\n\n"
            "Write the corrected function and a brief note of what was wrong."
        ),
    },
    {
        "id": "review-race-patch",
        "suite": "work",
        "kind": "code-review",
        "source": "local C1 code review",
        "prompt": (
            "Review this Python patch for a local inference helper. Name concrete "
            "bugs, missing tests, and a minimal fix. Quote lines.\n\n"
            "```diff\n"
            " def record_metric(name, value, store={}):\n"
            "-    store[name] = value\n"
            "+    store[name] = store.get(name, 0) + value\n"
            "+    return store\n"
            "\n"
            " async def fetch_all(urls, session):\n"
            "-    return [await session.get(u) for u in urls]\n"
            "+    return await asyncio.gather(*[session.get(u) for u in urls], return_exceptions=True)\n"
            "```"
        ),
    },
    {
        "id": "harbor-debug",
        "suite": "work",
        "kind": "coding-agent",
        "source": "local C1 agent debug loop",
        "prompt": (
            "You are debugging a Harbor/terminal-bench style task on a local repo. "
            "`pytest tests/test_cli.py::test_retry` fails with:\n\n"
            "  AssertionError: expected 3 attempts, got 1\n"
            "  cli.py:41: in run\n"
            "      result = client.call(payload)\n"
            "  # retry loop never runs because `attempts` is assigned after the call\n\n"
            "Inspect the likely control-flow bug, propose a patch to `run()`, and "
            "add a regression test. Keep the change small."
        ),
    },
    {
        "id": "tool-pytest",
        "suite": "work",
        "kind": "tool-use",
        "source": "local C1 tool-calling",
        "prompt": (
            "You have one tool, `run_pytest`, with schema "
            '{"path": "string", "expr": "string?"}. '
            "A user says: run the duration parser tests under tests/test_duration.py "
            "and summarize failures. First emit a single JSON tool call in the form "
            '{"tool":"run_pytest","args":{...}} then, assuming the tool returned '
            "two failures (ValueError on '90s' and '1h30m'), explain the failures "
            "and the code change to make."
        ),
    },
    {
        "id": "serve-oom-triage",
        "suite": "work",
        "kind": "inference-ops",
        "source": "local C1 inference debugging",
        "prompt": (
            "A vLLM-XPU container on one Arc Pro B70 dies 40s after start. Logs end "
            "with: max_num_batched_tokens (2048) is smaller than max_model_len (8192) "
            "after --no-enable-chunked-prefill. KV cache is fp8, --max-num-seqs 1, "
            "speculative DFlash n=20. Explain the scheduler constraint, the safe "
            "flag change that preserves the 8k window, and what not to change on "
            "the resting GPTQ DFlash cell."
        ),
    },
)
