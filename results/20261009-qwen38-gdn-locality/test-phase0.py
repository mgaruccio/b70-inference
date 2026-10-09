#!/usr/bin/env python3
"""CPU-only fixture for the bounded Phase0 harness and trace classifier."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import runpy
import tempfile


ROOT = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    annotations = load("gdn_annotations_fixture", ROOT / "gdn-annotations.py")
    summary = load("gdn_summary_fixture", ROOT / "summarize-gdn.py")
    patch = load("gdn_patch_fixture", ROOT / "gdn-patch.py")
    harness = runpy.run_path(str(ROOT / "run-phase0.py"))

    assert harness["BASELINE_LENGTHS"] == (512, 65_536)
    assert harness["GOLDEN_CONTRACT"]["num_speculative_tokens"] == 4
    assert harness["GOLDEN_CONTRACT"]["max_model_len"] == 212_992
    assert annotations.is_gdn_component("QwenGDNLinearAttention", "vllm.model_executor.models.qwen3_5_mtp")
    assert annotations.is_gdn_component("GatedDeltaNet", "vllm.v1.attention.backends.gdn_attn")
    assert not annotations.is_gdn_component("FlashAttention", "vllm.attention")
    assert annotations.classify_stage("target", "model.layers.0.linear_attn") == "target"
    assert annotations.classify_stage("draft", "speculator.model.layers.0") == "draft"
    assert annotations.classify_stage("target", "qwen3_5_mtp") == "target"
    assert annotations.classify_operator("aten::gdn_attention") == "gdn"
    assert annotations.classify_operator("_xpu_C::gdn_attention_core_xpu") == "gdn"
    assert annotations.classify_operator("_vllm_fa2_C::varlen_fwd") == "attention"
    assert annotations.classify_operator("aten::mm") == "matmul"
    assert annotations.classify_operator("aten::mm", "b70_gdn/target/module:qwen3_5_mtp") == "matmul"
    assert summary.classify_operator("aten::gdn_attention", "b70_gdn/target/operator:gdn_attention") == "gdn"
    assert summary.classify_operator("aten::mm", "b70_gdn/target/module:qwen3_5_mtp") == "matmul"
    eager_gate = summary.fusion_decision(
        evidence_mode="eager-profile",
        graph_coverage="complete",
        whole_replay_ms=100.0,
        replay_reconciled=True,
        gdn_kernel_ms=20.0,
    )
    assert eager_gate["decision_status"] == "inconclusive"
    assert not eager_gate["fusion_selection_allowed"]
    assert eager_gate["gdn_share_of_whole_replay"] is None
    partial_gate = summary.fusion_decision(
        evidence_mode="unitrace-profile",
        graph_coverage="partial",
        whole_replay_ms=100.0,
        replay_reconciled=True,
        gdn_kernel_ms=20.0,
    )
    assert not partial_gate["fusion_selection_allowed"]
    assert partial_gate["gdn_share_of_whole_replay"] is None
    eligible_gate = summary.fusion_decision(
        evidence_mode="unitrace-profile",
        graph_coverage="complete",
        whole_replay_ms=100.0,
        replay_reconciled=True,
        gdn_kernel_ms=6.0,
        gdn_provenance="target",
    )
    assert eligible_gate["fusion_selection_allowed"]
    assert eligible_gate["gdn_share_of_whole_replay"] == 0.06
    assert "unitrace-profile" in harness["DEFAULT_OUT_NAMES"]
    assert harness["UNITRACE_SOURCE_SHA"] == "887bba6e28ce84cc0d3813ef876e24add107c318"
    source = "import torch\n\nclass XPUWorker:\n    def profile(self, is_start=True, profile_prefix=None):\n        return None\n"
    patched = patch.patch_text(source)
    assert patch.MARKER in patched
    assert patch.patch_text(patched) == patched

    trace = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "b70_gdn/target/operator:gdn_attention",
                "pid": 1,
                "tid": 2,
                "ts": 0,
                "dur": 100,
            },
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "aten::gdn_attention",
                "pid": 1,
                "tid": 2,
                "ts": 10,
                "dur": 20,
                "args": {"External id": 7, "Input Dims": [[5, 24, 256]]},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "gdn_attention_kernel",
                "pid": 3,
                "tid": 4,
                "ts": 12,
                "dur": 40,
                "args": {"External id": 7},
            },
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        trace_path = Path(directory) / "rank0.pt.trace.json.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
            json.dump(trace, stream)
        report = summary.summarize(trace_path)
        assert report["status"] == "ok"
        assert report["gdn_operator_event_count"] == 1
        assert report["gdn_operator_events"][0]["input_dims"] == [[5, 24, 256]]
        assert any(row["category"] == "gdn" for row in report["categories"])
        assert any(row["category"] == "gdn" for row in report["categories"])
        unitrace_path = Path(directory) / "unitrace.json"
        unitrace_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"cat": "kernel", "name": "causal_conv1d_spec_kernel", "dur": 60, "ts": 1},
                        {"cat": "kernel", "name": "other_kernel", "dur": 940, "ts": 2},
                    ]
                }
            ),
            encoding="utf-8",
        )
        unitrace_report = summary.summarize_unitrace(unitrace_path)
        assert unitrace_report["status"] == "ok"
        assert unitrace_report["gdn_kernel_count"] == 1
        assert not unitrace_report["fusion_decision"]["fusion_selection_allowed"]
        assert unitrace_report["target_gdn_kernel_count"] == 0
        assert unitrace_report["fusion_decision"]["gdn_share_of_whole_replay"] is None
        assert unitrace_report["attribution_status"] == "inconclusive"
        target_path = Path(directory) / "unitrace-target.json"
        target_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {"cat": "kernel", "name": "causal_conv1d_spec_kernel", "dur": 6000, "ts": 1, "args": {"stage": "target"}},
                        {"cat": "kernel", "name": "other_kernel", "dur": 94000, "ts": 2},
                    ]
                }
            ),
            encoding="utf-8",
        )
        target_report = summary.summarize_unitrace(
            target_path,
            graph_coverage="complete",
            whole_replay_ms=100.0,
            replay_reconciled=True,
        )
        assert target_report["fusion_decision"]["fusion_selection_allowed"]
        assert target_report["fusion_decision"]["gdn_share_of_whole_replay"] == 0.06
        artifact_out = Path(directory) / "artifact-check"
        (artifact_out / "unitrace").mkdir(parents=True)
        (artifact_out / "unitrace" / "chrome_trace.json").write_text("{}", encoding="utf-8")
        (artifact_out / "server.log").write_text(
            "B70_GDN_LOCALITY_SUMMARY: "
            + json.dumps({"unitrace_control_events": [{"action": "resume", "returncode": 1}]})
            + "\n",
            encoding="utf-8",
        )
        assert harness["_unitrace_artifact_check"](artifact_out) == 1
        artifact_check = json.loads((artifact_out / "unitrace-artifact-check.json").read_text(encoding="utf-8"))
        assert artifact_check["status"] == "blocked"

    build_script = (ROOT / "build-unitrace.sh").read_text(encoding="utf-8")
    assert "887bba6e28ce84cc0d3813ef876e24add107c318" in build_script
    assert "--network=none" in build_script
    assert "-DBUILD_WITH_MPI=0" in build_script
    assert "-DBUILD_WITH_XPTI=0" in build_script
    assert "-DBUILD_WITH_OPENCL=0" in build_script
    assert "-DBUILD_WITH_OMP=" not in build_script
    assert "-DBUILD_WITH_PERFETTO=" not in build_script
    print("phase0 CPU fixtures: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
