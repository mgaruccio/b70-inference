# Qwen3.8 B70 GDN metadata overlay v1

Versioned overlay for the exact Qwen B70 XPU stack:

- image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`
- vLLM source: `ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9` (`0.27.2rc1.dev77+gac7509e2b`)
- target: `/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/gdn_attn.py` (the runtime module the image actually loads; `/workspace/vllm/...` is a build-tree path, not the patched target)

`patch_gdn_metadata.py` is the metadata-only half of vLLM PR #43955. It
preallocates/reuses the speculative token arange, avoids copying an empty
non-spec index, skips the redundant copy when the static arange is already
the source, and fails fast instead of silently truncating an oversized batch.
It does not change `gpu_model_runner.py`, model/eval configs,
`_xpu_C.gdn_attention`, or any kernel artifact.

## Ordering and isolated application

Apply the existing Qwen patches first, then this overlay:

1. `patch_mtp_nightly.py`
2. `patch_mtp_boundary.py`
3. `patch_gdn_metadata.py` (this directory)

The following is an isolated, non-serving application command. Substitute the
host paths for the two existing patches; it mounts only patch scripts and does
not mount `/dev/dri`, model weights, or the live inference container:

```bash
IMAGE='vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f'
MTP_PATCH=/path/to/patch_mtp_nightly.py
BOUNDARY_PATCH=/path/to/patch_mtp_boundary.py
METADATA_PATCH=/path/to/patch_gdn_metadata.py

docker run --rm --entrypoint bash \
  -v "$MTP_PATCH:/patch_mtp_nightly.py:ro" \
  -v "$BOUNDARY_PATCH:/patch_mtp_boundary.py:ro" \
  -v "$METADATA_PATCH:/patch_gdn_metadata.py:ro" \
  "$IMAGE" -lc \
  'set -eu; python /patch_mtp_nightly.py; python /patch_mtp_boundary.py; python /patch_gdn_metadata.py --path "$(python -c "import vllm.v1.attention.backends.gdn_attn as m; print(m.__file__)")"'
```

For the runtime module that already has the first two patches, inspect without writing with `python /patch_gdn_metadata.py --dry-run --path "$(python -c "import vllm.v1.attention.backends.gdn_attn as m; print(m.__file__)")"`;
rollback this overlay (leaving the boundary patch in place) with the same command plus `--reverse`.
The patcher is idempotent and fails closed unless the exact post-boundary
anchors are present. It prints whether it patched, would patch, or was already
patched.

## Verification process (offline fixture boundary)

The executable verification uses Python 3.12+ and the extracted post-boundary fixture at `tests/fixtures/qwen38_b70_gdn_metadata/gdn_attn_after_mtp_boundary.py`. The test invokes this patcher's CLI in a subprocess against a temporary copied target, then checks the public CLI journey: dry-run leaves the copy unchanged, apply changes it, a second apply reports `already patched`, and `--reverse` restores the post-boundary bytes. Run:

```bash
python -m py_compile patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/patch_gdn_metadata.py
pytest -q tests/test_qwen38_b70_gdn_metadata_patch.py
```

Expected result: `4 passed`; pytest owns and removes its temporary target. The test never imports vLLM, mounts a container, opens a service endpoint, or changes the inference host.

## Research record

Fresh primary-source checks on 2026-08-23 informed this narrow implementation:

- [vLLM PR #43955](https://github.com/vllm-project/vllm/pull/43955) is closed
  and unmerged; its `gdn_attn.py` hunk contains the static metadata changes,
  while its `gpu_model_runner.py` hunk is intentionally excluded here.
- [Pinned `gdn_attn.py` source](https://github.com/vllm-project/vllm/blob/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9/vllm/v1/attention/backends/gdn_attn.py)
  confirms the versioned source anchors used by the fixture and patcher.
- [Python `pathlib` documentation](https://docs.python.org/3/library/pathlib.html)
  and [`argparse` documentation](https://docs.python.org/3/library/argparse.html)
  support the explicit target-path and dry-run/reverse CLI used by this
  standalone patcher.

No live service was mounted or modified by this artifact or its tests.
