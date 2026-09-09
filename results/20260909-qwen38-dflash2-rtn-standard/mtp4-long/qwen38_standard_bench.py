#!/usr/bin/env python3
"""Run standard benchmark clients against disposable Qwen DFlash2 or MTP4 cells."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import urllib.error

import qwen38_dflash2_probe as dflash


def run_logged(argv, destination, *, check=True, timeout=7200):
    destination.with_suffix(".command.txt").write_text(shlex.join(argv) + "\n")
    print("COMMAND=" + shlex.join(argv), flush=True)
    with destination.open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {destination}")
    return result.returncode


def snapshot(out, name):
    (out / "spec-metrics" / f"{name}.prom").write_text(dflash.probe.get("/metrics"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--betterbench", type=Path, required=True)
    parser.add_argument("--draft-int4", type=Path)
    parser.add_argument("--reference-mtp", action="store_true")
    parser.add_argument("--long-context-only", action="store_true")
    parser.add_argument("--near-limit", type=int, help="explicit practical prompt length for the cold sweep")
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--prefill-patch", type=Path)
    parser.add_argument("--guard", type=Path, required=True)
    args = parser.parse_args()
    # These are the already-validated configurations, not new tuning knobs.
    args.mode, args.context = ("control", 212992) if args.reference_mtp else ("dflash2", 32768)
    args.graph, args.audit, args.suite = True, False, "standard"
    if args.reference_mtp and args.draft_int4:
        parser.error("the MTP reference does not use the DFlash2 checkpoint")
    if not args.reference_mtp and not (args.patch and args.prefill_patch):
        parser.error("DFlash2 requires --patch and --prefill-patch")
    if args.draft_int4 and args.draft_int4.resolve() in (dflash.TARGET.resolve(), dflash.DRAFT.resolve()):
        parser.error("INT4 checkpoint must be separate from original target/draft weights")
    long_script = Path(__file__).with_name("qwen38_long_context_bench.py")
    if args.long_context_only and not long_script.is_file():
        parser.error("the repository long-context client is missing")
    cell_class = dflash.DFlashCell
    if args.reference_mtp:
        from qwen38_mtp_reference import MTPReferenceCell
        cell_class = MTPReferenceCell
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"benchmark interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    cell = cell_class(args)
    print("ARTIFACT_DIR=" + str(cell.out), flush=True)
    for name in ("betterbench", "vllm-bench", "spec-metrics"):
        (cell.out / name).mkdir()
    if not args.long_context_only:
        shutil.copytree(args.betterbench, cell.out / "betterbench-source", ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
    shutil.copy2(__file__, cell.out / Path(__file__).name)
    cell.summary.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        betterbench_commit="1de941d256ddd633a8c117963ba72aebe4b4d5e4",
                        client_concurrency=[1, 2, 4, 8, 16], server_max_num_seqs=1,
                        comparison_order="sequential, not interleaved", tier="pending")
    try:
        cell.start()
        run_logged(["bash", "-lc", "uname -a; cat /etc/os-release; lscpu; free -b; lspci -nn; lspci -vv 2>/dev/null | sed -n '/VGA compatible controller/,+55p'; cat /sys/class/drm/card0/device/hwmon/hwmon2/power1_cap"], cell.out / "hardware.txt")
        run_logged(["docker", "exec", "qwen38", "python", "-m", "vllm.collect_env"], cell.out / "collect_env.txt", check=False)
        run_logged(["docker", "exec", "qwen38", "python", "-m", "pip", "freeze"], cell.out / "packages.txt")
        run_logged(["docker", "exec", "qwen38", "cat", "/model/config.json"], cell.out / "target-config.json")
        cell.gates()
        cell.quality()
        if args.long_context_only:
            shutil.copy2(long_script, cell.out / long_script.name)
            long_argv = ["python3", str(long_script), "--out", str(cell.out / "long-context"),
                         "--confirm-prefix-cache-disabled"]
            if args.near_limit is not None:
                long_argv += ["--near-limit", str(args.near_limit)]
            run_logged(long_argv, cell.out / "long-context-console.txt", timeout=14400)
            cell.summary["status"] = "long_context_client_completed"
        else:
            # Keep the vLLM benchmark client version fixed even for the older MTP server.
            client = ["docker", "exec", "qwen38", "vllm", "bench", "serve"]
            if args.reference_mtp:
                client = ["docker", "run", "--rm", "--network", "host", "-v", f"{cell.out}:/profile",
                          "-v", f"{dflash.TARGET}:/model:ro", "--entrypoint", "vllm", dflash.IMAGE, "bench", "serve"]
            run_logged(client + ["--help"], cell.out / "vllm-bench-help.txt")
            snapshot(cell.out, "betterbench-before")
            run_logged(["docker", "exec", "-w", "/profile/betterbench-source", "-e", "PYTHONUNBUFFERED=1", "qwen38",
                        "python", "-m", "betterbench.cli", "run", "--endpoint", "http://127.0.0.1:8000/v1",
                        "--model", "qwen38", "--corpus", "/profile/betterbench-source/corpus/v1",
                        "--greedy", "--seed", "42", "--passes", "20", "--warmup", "3",
                        "--note", "server_max_num_seqs=1", "--note", "prefix_cache=off",
                        "--note", "thinking=off", "--note", "comparison=sequential",
                        "--out", "/profile/betterbench/results.json"], cell.out / "betterbench" / "console.txt")
            snapshot(cell.out, "betterbench-after")
            for concurrency in (1, 2, 4, 8):
                name = f"c{concurrency}"
                snapshot(cell.out, "vllm-" + name + "-before")
                run_logged(client + ["--backend", "openai", "--base-url", "http://127.0.0.1:8000",
                                     "--endpoint", "/v1/completions", "--model", "qwen38", "--tokenizer", "/model",
                                     "--dataset-name", "random", "--input-len", "512", "--output-len", "128",
                                     "--num-prompts", "48", "--num-warmups", "3", "--max-concurrency", str(concurrency),
                                     "--request-rate", "inf", "--seed", "42", "--temperature", "0", "--ignore-eos",
                                     "--save-result", "--save-detailed", "--metric-percentiles", "25,50,75,90,99",
                                     "--percentile-metrics", "ttft,tpot,itl,e2el", "--result-dir", "/profile/vllm-bench",
                                     "--result-filename", name + ".json"], cell.out / "vllm-bench" / (name + ".txt"))
                snapshot(cell.out, "vllm-" + name + "-after")
            cell.summary["status"] = "benchmark_clients_completed"
            cell.summary["remaining"] = ["long-context sweep", "artifact analysis and publication checklist"]
    except (Exception, KeyboardInterrupt) as error:
        cell.summary.update(status="failed", error=repr(error))
        if isinstance(error, urllib.error.HTTPError):
            cell.summary["http_error_body"] = error.read().decode(errors="replace")
        raise
    finally:
        cell.summary["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cell.close()


if __name__ == "__main__":
    main()
