#!/usr/bin/env python3
"""Lead-run, offloaded GPTQ/native-block-FP8 acceptance A/B. No throughput claims."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import sys
import traceback
import uuid

from b70_quant_observe import digest, require, target_identity

ROOT = Path(__file__).resolve().parent
DRIVER_SHA = "26f7075a3b8677fd2ff8d210cd71a497b55b86e31ac5504affcc502ac41d90ce"
PACKAGE = "/opt/venv/lib/python3.12/site-packages/vllm"


def load_json(path):
    return json.loads(Path(path).read_text())


def reference_record(path, ns, *, offloaded, canonical_sha):
    path = Path(path).resolve(strict=True)
    summary = load_json(path / "summary.json")
    launch = load_json(path / "launch-metadata.json")
    require(summary["status"] == "passed" and summary["target"] == ns["GPTQ_TARGET"], "invalid GPTQ reference")
    for field, expected in (("image", ns["IMAGE"]), ("context", 8192), ("kv_cache_dtype", "fp8"),
                            ("runner", "v2-eager-C1"), ("speculative_config", ns["SPEC_CONFIG"])):
        require(summary[field] == expected, "reference setting mismatch: " + field)
    serve = launch["serve"]
    if offloaded:
        require(serve[serve.index("--cpu-offload-gb") + 1] == "8", "paired arm needs 8GiB offload")
        require(summary["quant_control"]["arm"] == "gptq"
                and summary["quant_control"]["canonical_sha256"] == canonical_sha,
                "paired arm/canonical overlay mismatch")
    else:
        require("--cpu-offload-gb" not in serve, "historical reference must be no-offload")
    return {"path": str(path), "summary_sha256": digest(path / "summary.json"),
            "launch_sha256": digest(path / "launch-metadata.json"),
            "effective_source_sha256": digest(path / "effective-dspark-source.json"),
            "offload_budget_gb": 8 if offloaded else 0,
            "interpretation": "target quantization plus native kernel bundle" if offloaded else
                "historical same-target no-offload diagnostic; overlay may differ; no isolated offload/performance inference"}


def compare_request(out, reference, label):
    current = load_json(out / (label + "-result.json"))
    other = load_json(reference / (label + "-result.json"))
    for key in ("label", "prompt", "sampling", "cache_salt", "rendered_prompt_token_ids"):
        require(current[key] == other[key], "paired input mismatch: " + label + ":" + key)
    require(load_json(out / (label + "-request.json")) == load_json(reference / (label + "-request.json")),
            "paired request payload mismatch: " + label)
    return {"label": label, "inputs_equal": True,
            "reference_proposals": other["proposals_delta"], "candidate_proposals": current["proposals_delta"],
            "reference_accepts": other["accepts_delta"], "candidate_accepts": current["accepts_delta"],
            "reference_emitted_per_step": other["emitted_per_step"],
            "candidate_emitted_per_step": current["emitted_per_step"],
            "reference_per_position": other["per_position_accepted_counters"],
            "candidate_per_position": current["per_position_accepted_counters"]}


def configure(ns, options):
    require(digest(options.canonical_overlay) == options.canonical_sha256, "canonical overlay SHA256 mismatch")
    ns["ROOT"] = ROOT
    ns["PREVIOUS"] = options.previous_campaign.resolve(strict=True)
    ns["GPTQ_TARGET"] = ns["TARGET"]
    ns["TARGET"] = str((options.target or (Path(ns["TARGET"]) if options.arm == "gptq"
                                         else ROOT / "target-fp8")).resolve(strict=True))
    if options.arm == "gptq":
        require(ns["TARGET"] == ns["GPTQ_TARGET"], "GPTQ arm must use the existing target snapshot")
    identity = target_identity(ns["TARGET"], options.arm, options.target_manifest)
    historical = reference_record(options.no_offload_reference, ns, offloaded=False,
                                  canonical_sha=options.canonical_sha256)
    pair = None
    if options.arm == "fp8":
        require(options.paired_with is not None, "FP8 requires the completed paired GPTQ offload arm")
        pair = reference_record(options.paired_with, ns, offloaded=True, canonical_sha=options.canonical_sha256)
    control = {"arm": options.arm, "cpu_offload_budget_gb": 8, "canonical_sha256": options.canonical_sha256,
               "target_identity": identity, "no_offload_reference": historical, "paired_reference": pair,
               "comparison": "target weight quantization plus native execution bundle; NOT bit precision alone",
               "throughput_comparison_valid": False, "cross_target_output_identity_required": False}
    original_mounts, original_validate = ns["dependency_mounts"], ns["validate_assets"]
    original_source_check, original_matrix, original_save = ns["source_check"], ns["run_matrix"], ns["save"]

    def save(path, obj):
        if path.name == "summary.json":
            obj.update(quant_control=control,
                       target_dtype="FP16 compute; " + ("official FP8 block(128,128)" if options.arm == "fp8"
                                                       else "GPTQ Int4 symmetric G128"))
        original_save(path, obj)

    def validate(draft, *, require_draft):
        manifest = original_validate(draft, require_draft=require_draft)
        manifest["files"]["draft_overlay"] = {"path": str(ROOT / "patch-quant-control.py"),
                                              "sha256": digest(ROOT / "patch-quant-control.py")}
        manifest["quant_control"] = {name: digest(ROOT / name) for name in
            ("run-quant-control.py", "patch-quant-control.py", "b70_quant_observe.py")}
        manifest["canonical_overlay"] = {"path": str(options.canonical_overlay), "sha256": options.canonical_sha256}
        manifest["diagnostic_driver"] = {"path": str(options.driver), "sha256": DRIVER_SHA}
        if pair:
            previous = load_json(Path(pair["path"]) / "dependencies.json")
            require({k: v["sha256"] for k, v in previous["files"].items()} ==
                    {k: v["sha256"] for k, v in manifest["files"].items()}, "paired dependency source mismatch")
            require(previous["quant_control"] == manifest["quant_control"], "paired experiment assets changed")
        return manifest

    def mounts(out, draft, cell, kv_cache_dtype):
        require(cell == "dspark" and kv_cache_dtype == "fp8", "fixed DSpark/FP8-KV control only")
        argv, metadata = original_mounts(out, draft, cell, kv_cache_dtype)
        name = "qwen38-quant-" + options.arm + "-" + uuid.uuid4().hex[:12]
        argv[argv.index("--name") + 1] = metadata["container_name"] = name
        serve = metadata["serve"]
        serve[serve.index("--quantization") + 1] = "fp8" if options.arm == "fp8" else "gptq"
        serve += ["--cpu-offload-gb", "8"]
        # Keep the existing prefill -> DSpark -> boundary order and its source check.
        argv[-1] = argv[-1].rsplit("exec ", 1)[0] + "exec " + shlex.join(serve)
        frozen = {"canonical.py": options.canonical_overlay,
                  "patch-quant-control.py": ROOT / "patch-quant-control.py",
                  "b70_quant_observe.py": ROOT / "b70_quant_observe.py"}
        for name, source in frozen.items():
            shutil.copyfile(source, out / name)
        require(digest(out / "canonical.py") == options.canonical_sha256, "canonical changed during snapshot")
        for name in ("config.json", "model.safetensors.index.json"):
            shutil.copyfile(Path(ns["TARGET"]) / name, out / ("target-" + name))
        if options.arm == "fp8":
            shutil.copyfile(options.target_manifest, out / "target-manifest.json")
        save(out / "target-identity.json", identity)
        insert = argv.index("--entrypoint")
        argv[insert:insert] = [
            "-v", f"{out / 'canonical.py'}:/quant/canonical.py:ro",
            "-v", f"{out / 'patch-quant-control.py'}:/experiment/patch_dspark_bf16.py:ro",
            "-v", f"{out / 'b70_quant_observe.py'}:{PACKAGE}/_b70_quant_observe.py:ro",
            "-e", f"B70_QUANT_CONTROL_ARM={options.arm}",
            "-e", f"B70_QUANT_CANONICAL_SHA={options.canonical_sha256}",
            "-e", "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=0",
            "-e", "VLLM_BATCH_INVARIANT=0"]
        metadata["quant_control"] = control
        return argv, metadata

    def source_check(out, container, cell):
        original_source_check(out, container, cell)
        native = load_json(out / "target-native.json")
        memory = load_json(out / "memory-after-load.json")
        require(native["passed"] is True and native["arm"] == options.arm
                and memory["arm"] == options.arm, "native kernel/offload report missing or failed")
        control["post_load_native"] = {"sha256": digest(out / "target-native.json"), "offload": native["offload"]}
        control["post_load_memory"] = memory

    class PairedClient(ns["DiagnosticClient"]):
        def chat(self, label, *args, **kwargs):
            row = super().chat(label, *args, **kwargs)
            if label.startswith("experiment-"):
                require(set(row["per_position_accepted_counters"]) == {str(i) for i in range(7)},
                        "missing K7 per-position counters: " + label)
                require(row["draftsteps_delta"] > 0 and row["proposals_delta"] > 0,
                        "matrix request did not speculate: " + label)
                if pair:
                    compare_request(self.out, Path(pair["path"]), label)
            return row

    def matrix(client, probe):
        result = original_matrix(client, probe)
        comparisons = {}
        for label, record in (("paired-gptq-offload", pair), ("historical-gptq-no-offload", historical)):
            if record is None or (label.startswith("historical") and options.arm != "gptq"):
                continue
            rows, errors = [], []
            for row in result["rows"]:
                try:
                    rows.append(compare_request(client.out, Path(record["path"]), row["label"]))
                except Exception as exc:
                    errors.append(str(exc))
            comparison = {"reference": record, "rows": rows, "errors": errors,
                          "all_36_inputs_equal": len(rows) == 36 and not errors,
                          "output_token_identity_tested": False, "throughput_compared": False}
            save(client.out / (label + ".json"), comparison)
            if label.startswith("paired"):
                require(comparison["all_36_inputs_equal"], "paired matrix mismatch")
            comparisons[label] = {"all_36_inputs_equal": comparison["all_36_inputs_equal"], "errors": errors}
        result["input_pairing"] = comparisons
        return result

    ns.update(save=save, validate_assets=validate, dependency_mounts=mounts, source_check=source_check,
              DiagnosticClient=PairedClient, run_matrix=matrix)
    return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("gptq", "fp8"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--driver", type=Path, default=ROOT.parent / "20260911-qwen38-dspark-acceptance-diagnostics/run-acceptance-diagnostics.py")
    parser.add_argument("--previous-campaign", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--canonical-overlay", type=Path, required=True)
    parser.add_argument("--canonical-sha256", required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--target-manifest", type=Path, default=ROOT / "download-result.json")
    parser.add_argument("--no-offload-reference", type=Path, required=True)
    parser.add_argument("--paired-with", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--request-timeout", type=int, default=1800)
    options = parser.parse_args()
    require(options.startup_timeout > 0 and options.request_timeout > 0, "timeouts must be positive")
    require(digest(options.driver) == DRIVER_SHA, "existing diagnostic driver changed")
    driver = runpy.run_path(str(options.driver))
    ns = driver["run"].__globals__
    options.cell, options.kv_cache_dtype = "dspark", "fp8"
    out = options.out.resolve()
    require(out.is_relative_to(ROOT) and out != ROOT and not out.exists(), "output must be a NEW child of experiment root")
    try:
        configure(ns, options)
        return ns["run"](options)
    except Exception:
        # The reused driver handles server failures/owned cleanup. Also retain
        # failures in its pre-launch setup (which precedes its try/finally).
        out.mkdir(parents=False, exist_ok=True)
        (out / "preflight-failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
