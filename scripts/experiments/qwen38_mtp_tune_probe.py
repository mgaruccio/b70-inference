"""Development-only native MTP tuning cells; never modify the production launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import urllib.request

import qwen38_lossy_probe as probe
import qwen38_mtp_reference as reference

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"


def make_corpus(out: Path):
    """Use official train/validation splits, not the MBPP test split or solutions."""
    out.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(MBPP_URL, timeout=60) as response:
        raw = response.read()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for split, low, high in (("train", 601, 700), ("heldout", 511, 534)):
        rows = []
        for record in sorted(records, key=lambda r: r["task_id"]):
            if low <= record["task_id"] <= high:
                checks = record["test_list"]
                prompt = ("Implement this Python task. Return only runnable Python code, without markdown. "
                          "Include necessary imports.\n\n" + record["text"]
                          + "\n\nYour code must satisfy:\n" + "\n".join(checks))
                rows.append({"id": f"mbpp-{record['task_id']}", "prompt": prompt,
                             "checks": "\n".join(checks), "split": split})
        if len(rows) != high - low + 1:
            raise RuntimeError(f"MBPP {split} range incomplete")
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    probe.save(out / "source.json", {"url": MBPP_URL, "sha256": hashlib.sha256(raw).hexdigest(),
               "split_reference": "https://github.com/google-research/google-research/tree/master/mbpp",
               "train_ids": [601, 700], "heldout_ids": [511, 534],
               "purpose": "pipeline smoke, not a representative agent-workload benchmark"})


def aggregate(rows):
    totals = {}
    for row in rows:
        for name, value in row["metric_deltas"].items():
            if value < 0:
                raise RuntimeError("metric counter reset during a request")
            totals[name] = totals.get(name, 0.0) + value
    def counter(suffix):
        return sum(v for k, v in totals.items() if k.split("{", 1)[0] == "vllm:" + suffix)
    drafts = counter("spec_decode_num_drafts_total")
    proposed = counter("spec_decode_num_draft_tokens_total")
    accepted = counter("spec_decode_num_accepted_tokens_total")
    positions = {}
    for key, value in totals.items():
        if key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{"):
            match = re.search(r'position="(\d+)"', key)
            if match:
                positions[match[1]] = positions.get(match[1], 0.0) + value
    if not drafts or not proposed or not positions:
        raise RuntimeError("missing live speculative acceptance counters")
    import statistics
    return {"requests": len(rows), "draft_passes": drafts, "proposed": proposed,
            "accepted": accepted, "accepted_per_draft_pass": accepted / drafts,
            "draft_token_acceptance": accepted / proposed,
            "accepted_per_position": positions,
            "position_acceptance_per_draft_pass": {k: v / drafts for k, v in positions.items()},
            "median_decode_tps": statistics.median(r["decode_tps"] for r in rows),
            "median_elapsed_s": statistics.median(r["elapsed_s"] for r in rows),
            "note": "Draft-pass counters are not asserted to count every verifier call. "
                    "Position denominator includes all draft passes, including boundary-truncated passes."}


def check_outputs(directory: Path):
    """Check natural-EOS answers only in the existing isolated CPU sandbox."""
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "completed" or not summary.get("generate"):
        raise ValueError("functional checks require a completed natural-EOS cell")
    records = [json.loads(line) for line in (directory / "prompts.jsonl").read_text().splitlines() if line.strip()]
    results = []
    for record in records:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", record["id"]):
            raise ValueError("unsafe prompt ID")
        row = json.loads((directory / (record["id"] + "-result.json")).read_text())
        source = row["content"].strip()
        fenced = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", source, flags=re.DOTALL)
        if fenced:
            source = fenced.group(1)
        result = probe.sandbox(source, record["checks"])
        results.append({"id": record["id"], "finish_reason": row["finish_reason"], **result})
    report = {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results,
              "scope": "public MBPP example tests; not MBPP+ or a full quality qualification"}
    probe.save(directory / "functional.json", report)
    print(json.dumps({k: report[k] for k in ("passed", "total", "scope")}), flush=True)


class TuneCell(probe.Cell):
    def start(self):
        if self.power.read_text().strip() != "275000000":
            raise RuntimeError("expected fixed 275 W power cap")
        if probe.command("docker", "ps", "--format", "{{.Names}}").strip():
            raise RuntimeError("experiment requires idle host")
        guard = self.args.guard.read_bytes()
        if hashlib.sha256(guard).hexdigest() != probe.GUARD_SHA:
            raise RuntimeError("prefill guard mismatch")
        (self.out / "patch_uniform_decode_prefill.py").write_bytes(guard)
        (self.out / "persistent-launcher.sh").write_bytes(self.original)
        source = self.out / "reference-source" / "patches"
        source.mkdir(parents=True)
        hashes = {}
        for name in reference.PATCHES:
            shutil.copy2(reference.PATCH_ROOT / name, source / name)
            hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        text = reference.launcher_text(self.original, self.out)
        if self.args.runtime_patch:
            patch = self.args.runtime_patch.resolve()
            shutil.copy2(patch, self.out / patch.name)
            # Runtime patch is applied after the deployed patches; its overlay must
            # substitute native tensors before their existing RTN packing runs.
            mounts = " -v " + shlex.quote(str(patch.parent) + ":/mtp-training:ro")
            if self.args.weights:
                mounts += " -v " + shlex.quote(str(self.args.weights.resolve()) + ":/mtp.safetensors:ro")
                mounts += " -e B70_MTP_WEIGHTS=/mtp.safetensors"
            marker = "exec docker run --rm --name qwen38 --ipc=host"
            text = text.replace(marker, marker + mounts, 1)
            text = text.replace("exec vllm serve /model ",
                                f"python /mtp-training/{shlex.quote(patch.name)}; exec vllm serve /model ", 1)
        if getattr(self.args, "replay", False):
            spec_flag = r'--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":4}"'
            if text.count(spec_flag) != 1:
                raise RuntimeError("replay speculative-config anchor changed")
            text = text.replace(spec_flag, "--no-async-scheduling", 1)
            if self.args.capture:
                (self.out / "features").mkdir()
                marker = "exec docker run --rm --name qwen38 --ipc=host"
                text = text.replace(marker, marker + " -e B70_MTP_CAPTURE_DIR=/profile/features", 1)
        launcher = self.out / "launcher.sh"
        launcher.write_text(text)
        probe.command("bash", "-n", str(launcher))
        for module in (Path(__file__), Path(probe.__file__), Path(reference.__file__), Path(reference.dflash.__file__)):
            shutil.copy2(module, self.out / module.name)
        self.summary.update(tier="development", image=reference.IMAGE, context=212992,
                            draft_quantization="existing S+M1 RTN INT4", speculative_tokens=4,
                            prefix_caching=False, thinking=False, temperature=0, seed=42,
                            reference_patch_sha256=hashes,
                            weights=str(self.args.weights) if self.args.weights else "stock",
                            intentional_difference="mtp.* overlay only" if self.args.weights else "none")
        if getattr(self.args, "replay", False):
            self.summary.update(speculative_tokens=0, capture=self.args.capture,
                                draft_quantization="not loaded (target-only replay)",
                                intentional_difference="target-only synchronous replay; capture toggle")
        self.log = (self.out / "server.log").open("w")
        self.proc = subprocess.Popen(["bash", str(launcher)], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server exited; inspect retained server.log")
            try:
                probe.get("/health")
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("server startup exceeded 900s")
        models = json.loads(probe.get("/v1/models"))
        if not any(m["id"] == "qwen38" and m["max_model_len"] == 212992 for m in models["data"]):
            raise RuntimeError("wrong model/context")
        probe.save(self.out / "models.json", models)
        self.summary.update(status="running", models=models)
        print("CELL_READY=" + str(self.out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-corpus", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-outputs", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--guard", type=Path, default=Path("/tmp/qwen-b70-patch-uniform-decode-prefill.py"))
    parser.add_argument("--runtime-patch", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--generate", action="store_true", help="natural EOS; retain on-policy training token IDs")
    args = parser.parse_args()
    if args.make_corpus:
        make_corpus(args.make_corpus)
        return
    if args.check_outputs:
        check_outputs(args.check_outputs)
        return
    if not args.out or not args.prompts or args.tokens < 2:
        parser.error("--out, --prompts and --tokens >=2 are required")
    if args.weights and not args.runtime_patch:
        parser.error("--weights requires --runtime-patch")
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in prompts]
    if not prompts or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids):
        parser.error("prompt IDs must be unique safe filenames")
    args.head, args.suite, args.cascade_patch, args.alpha = "dense", "mtp-tune", None, 0
    cell = TuneCell(args)
    try:
        shutil.copy2(args.prompts, cell.out / "prompts.jsonl")
        cell.start()
        cell.chat("warmup", "Write a Python function that adds two integers. Return only code.", count=64)
        # Do not include warmup counters or timing in the comparison.
        cell.rows.clear()
        for record in prompts:
            cell.chat(record["id"], record["prompt"], count=args.tokens, forced=not args.generate)
        cell.summary["measurements"] = aggregate(cell.rows)
        cell.summary.update(status="completed", generate=args.generate, max_tokens=args.tokens)
    except BaseException as error:
        cell.summary.update(status="failed", error=repr(error))
        raise
    finally:
        cell.close()


if __name__ == "__main__":
    main()
