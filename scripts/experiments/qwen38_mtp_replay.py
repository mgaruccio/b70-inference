"""Replay exact on-policy token IDs through pinned vLLM; no retokenization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import qwen38_mtp_tune_probe as tune


def sequences(sources):
    seen = set()
    for source in sources:
        summary = json.loads((source / "summary.json").read_text())
        if summary.get("status") != "completed" or not summary.get("generate"):
            raise ValueError("replay source must be a completed natural-EOS generation cell")
        for line in (source / "prompts.jsonl").read_text().splitlines():
            record = json.loads(line)
            name = record["id"]
            if name in seen or not tune.re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError("duplicate or unsafe prompt ID")
            seen.add(name)
            row = json.loads((source / (name + "-result.json")).read_text())
            if not row["transport_pass"]:
                raise ValueError("invalid source transport")
            prompt, output = row["prompt_token_ids"], row["token_ids"]
            if not prompt or len(output) < 2 or len(prompt) + len(output) > 2048:
                raise ValueError("tiny replay requires complete sequences <=2048 and >=2 output tokens")
            yield {"name": name, "input_ids": prompt + output,
                   "loss_mask": [False] * len(prompt) + [True] * len(output), "split": record["split"]}


def finish_dataset(capture, output):
    # ML imports are deliberately confined to the pinned container on inference-host.
    import torch
    manifest = json.loads((capture / "replay-inputs.json").read_text())
    if json.loads((capture / "summary.json").read_text()).get("status") != "completed":
        raise ValueError("capture cell did not complete")
    output.mkdir(parents=True, exist_ok=False)
    total = 0
    for record in manifest:
        if record["split"] not in ("train", "heldout"):
            raise ValueError("invalid prompt split")
        raw = torch.load(capture / "features" / (record["name"] + ".pt"), weights_only=True, map_location="cpu")
        if not torch.equal(raw["input_ids"], torch.tensor(record["input_ids"])):
            raise ValueError("captured token sequence mismatch")
        if not torch.equal(raw["loss_mask"], torch.tensor(record["loss_mask"])):
            raise ValueError("captured mask mismatch")
        raw["prompt_id"] = record["name"]
        split = output / record["split"]
        split.mkdir(exist_ok=True)
        torch.save(raw, split / (record["name"] + ".pt"))
        total += int(raw["loss_mask"][2:].sum())
    tune.probe.save(output / "summary.json", {"sequences": len(manifest), "useful_positions": total,
                    "source": str(capture), "semantics": "target-only prefill replay; no retokenization"})
    print("DATASET=" + str(output), flush=True)


def compare_replays(left, right):
    a = json.loads((left / "replay-outputs.json").read_text())
    b = json.loads((right / "replay-outputs.json").read_text())
    if a.keys() != b.keys() or not a:
        raise ValueError("replay prompt sets differ or are empty")
    mismatches = [name for name in a if a[name] != b[name]]
    result = {"requests": len(a), "exact_matches": len(a) - len(mismatches), "mismatches": mismatches,
              "scope": "capture on/off same target-only replay; not prefill/speculative-decode hidden-state parity"}
    tune.probe.save(right / "capture-parity.json", result)
    print(json.dumps(result), flush=True)
    if mismatches:
        raise RuntimeError("capture on/off output mismatch; stop before training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-patch", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--finish-dataset", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--guard", type=Path, default=Path("/tmp/qwen-b70-patch-uniform-decode-prefill.py"))
    args = parser.parse_args()
    if args.compare:
        compare_replays(*args.compare)
        return
    if args.finish_dataset:
        if not args.out:
            parser.error("--finish-dataset requires --out")
        finish_dataset(args.finish_dataset, args.out)
        return
    if not args.source or not args.out or not args.runtime_patch:
        parser.error("--source --out --runtime-patch required")
    records = list(sequences(args.source))
    args.weights, args.replay = None, True
    args.head, args.suite, args.cascade_patch, args.alpha = "dense", "mtp-replay", None, 0
    cell = tune.TuneCell(args)
    control = cell.out / "features" / "capture-request.json"
    outputs = {}
    try:
        shutil.copy2(Path(__file__), cell.out / Path(__file__).name)
        tune.probe.save(cell.out / "replay-inputs.json", records)
        cell.start()
        for record in records:
            if args.capture:
                tune.probe.save(control, {k: record[k] for k in ("name", "input_ids", "loss_mask")})
            payload = {"model": "qwen38", "prompt": record["input_ids"], "max_tokens": 1,
                       "temperature": 0, "seed": 42, "return_token_ids": True}
            response = tune.probe.post("/v1/completions", payload)
            tune.probe.save(cell.out / (record["name"] + "-response.json"), response)
            choice = response["choices"][0]
            if response["usage"]["completion_tokens"] != 1 or len(choice.get("token_ids", [])) != 1:
                raise RuntimeError("replay response must contain one generated token ID")
            outputs[record["name"]] = {"token_ids": choice["token_ids"], "text": choice["text"]}
            if args.capture and not (cell.out / "features" / (record["name"] + ".pt")).is_file():
                raise RuntimeError("complete captured feature file missing after response")
            control.unlink(missing_ok=True)
            print("REPLAY=" + record["name"], flush=True)
        tune.probe.save(cell.out / "replay-outputs.json", outputs)
        cell.summary.update(status="completed", sequences=len(outputs))
    except BaseException as error:
        cell.summary.update(status="failed", error=repr(error))
        raise
    finally:
        control.unlink(missing_ok=True)
        cell.close()


if __name__ == "__main__":
    main()
