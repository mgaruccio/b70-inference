#!/usr/bin/env python3
"""Apply the validated community prefill-classification guard.

The pinned legacy ``GPUModelRunner`` classified a uniform decode from batch
shape alone.  A prompt chunk can have that same shape under MTP, so this
maintained patch supplies the runner's CPU-counter prefill state and excludes
prefill batches from the shape-based classifier.  The intentional
``force_uniform_decode`` override remains authoritative.

Source behavior was validated on B70 in the 20260908 prefill-guard trial.
Source:
https://raw.githubusercontent.com/AnnoyingTechnology/intel-arc-b70-llm-inference/main/docker/patches/patch_uniform_decode_prefill.py
Related upstream fix: https://github.com/vllm-project/vllm/pull/53059
Validated 20260908 trial source SHA-256:
baa4647398874c19175ea74fe6f5d8dd6c2d83fc4bd0e5f2a68558afd983f5ad
"""

import argparse
import importlib.util
from pathlib import Path


MARKER = "B70_FIX_UNIFORM_DECODE_PREFILL"

# These anchors and replacements intentionally preserve the validated runner
# transformation.  Refuse drift rather than patching a different vLLM source.
OLD_CLASSIFIER = '''    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
'''
NEW_CLASSIFIER = '''    @staticmethod
    def _is_uniform_decode(
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
        has_prefill: bool = False,
        force_uniform_decode: bool | None = None,
    ) -> bool:
        """
        Checks if it's a decode batch with same amount scheduled tokens
        across all requests.
        """
        return (
            (
                not has_prefill  # B70_FIX_UNIFORM_DECODE_PREFILL
                and (max_num_scheduled_tokens == uniform_decode_query_len)
                and (num_tokens == max_num_scheduled_tokens * num_reqs)
            )
            if force_uniform_decode is None
            else force_uniform_decode
        )
'''
OLD_CALL = '''            num_tokens=num_tokens,
            num_reqs=num_reqs,
            force_uniform_decode=force_uniform_decode,
'''
NEW_CALL = '''            num_tokens=num_tokens,
            num_reqs=num_reqs,
            has_prefill=bool(
                (self.input_batch.num_computed_tokens_cpu[:num_reqs]
                 < self.input_batch.num_prompt_tokens[:num_reqs]).any()
            ),
            force_uniform_decode=force_uniform_decode,
'''


def patch_text(source: str) -> str:
    """Apply once or validate an exact replay; refuse drift before writing."""
    if MARKER in source:
        if (
            source.count(MARKER) != 1
            or source.count(NEW_CLASSIFIER) != 1
            or source.count(NEW_CALL) != 1
        ):
            raise RuntimeError("partial/changed uniform-decode prefill patch")
        compile(source, "gpu_model_runner.py", "exec")
        return source

    if source.count(OLD_CLASSIFIER) != 1:
        raise RuntimeError("uniform-decode classifier anchor changed")
    if source.count(OLD_CALL) != 1:
        raise RuntimeError("uniform-decode call anchor changed")

    result = source.replace(OLD_CLASSIFIER, NEW_CLASSIFIER, 1)
    result = result.replace(OLD_CALL, NEW_CALL, 1)
    compile(result, "gpu_model_runner.py", "exec")
    return result


def _runner_path(root: Path | None) -> Path:
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise RuntimeError("vllm package not found")
        root = Path(spec.origin).parent
    return root / "v1/worker/gpu_model_runner.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, help="vllm package root; omit inside serving container"
    )
    args = parser.parse_args()

    path = _runner_path(args.root)
    before = path.read_text(encoding="utf-8")
    after = patch_text(before)
    if after != before:
        path.write_text(after, encoding="utf-8")
        state = "patched"
    else:
        state = "already patched"
    print(f"{MARKER}: {state} {path}", flush=True)


if __name__ == "__main__":
    main()
