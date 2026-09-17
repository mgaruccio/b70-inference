"""Merge private native captures without counting identical trajectories twice.

Run in the pinned ML container, never Pi. The allowlist contains train-only
prompt IDs and source families; held-out data must not enter this dataset.
Counts are supervised positions in distinct full trajectories, not distinct
prefixes and not epoch exposures. Original captures are never removed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import qwen38_train_mtp as trainer


def native_runtime():
    path = Path(__file__).resolve().parents[2] / "patches/qwen38-b70-vllm-0.27.2rc1-gac7509e2b/b70_mtp_native_capture.py"
    spec = importlib.util.spec_from_file_location("b70_mtp_merge_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trajectory_key(record):
    digest = hashlib.sha256(record["input_ids"].contiguous().numpy().tobytes())
    digest.update(record["loss_mask"].contiguous().numpy().tobytes())
    return digest.hexdigest()


def merge(sources, output, allowlist, config, max_length=2048):
    rt = native_runtime()
    output = rt.private_directory(output, create=not Path(output).exists())
    if not allowlist or any(not isinstance(k, str) or not isinstance(v, str) for k, v in allowlist.items()):
        raise ValueError("train-only prompt/family allowlist required")
    counts = dict(sequences=0, useful_positions=0, input_tokens=0, added_sequences=0, duplicate_records=0)
    seen, families = set(), set()

    def read(path):
        with rt._private_open(path, "rb") as handle:
            record = rt.torch.load(handle, map_location="cpu", weights_only=True)
        trainer.validate_record(record, config, max_length)
        family = record.get("metadata", {}).get("source_group")
        if record["prompt_id"] not in allowlist or family != allowlist[record["prompt_id"]]:
            raise ValueError("capture is outside the train-only family allowlist")
        return record

    def count(record, key):
        seen.add(key)
        families.add(record["metadata"]["source_group"])
        counts["sequences"] += 1
        counts["useful_positions"] += int(record["loss_mask"][2:].sum())
        counts["input_tokens"] += len(record["input_ids"])

    for path in sorted(output.glob("*.pt")):
        record = read(path)
        key = trajectory_key(record)
        if path.stem != key or key in seen:
            raise ValueError("existing merged dataset identity mismatch")
        count(record, key)
    for source in sources:
        source = rt.private_directory(source)
        if source == output:
            raise ValueError("source and merged dataset must differ")
        paths = sorted(source.glob("*.pt"))
        if not paths:
            raise ValueError("source contains no completed captures")
        for path in paths:
            record = read(path)
            key = trajectory_key(record)
            if key in seen:
                counts["duplicate_records"] += 1
                continue
            rt.save_tensor(output / (key + ".pt"), record)
            count(record, key)
            counts["added_sequences"] += 1
    return {**counts, "source_families": len(families)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    try:
        rt = native_runtime()
        rt.torch.set_num_threads(4)
        allowed = rt.load_json(args.allowlist)
        data = json.loads(args.model_config.read_text())
        config = SimpleNamespace(**data.get("text_config", data))
        counts = merge(args.source, args.output, allowed, config)
        print(json.dumps(counts, sort_keys=True))
    except Exception as exc:
        print("Capture merge failed (" + type(exc).__name__ + "); private content suppressed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
