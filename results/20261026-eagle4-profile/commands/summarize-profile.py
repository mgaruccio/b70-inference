#!/usr/bin/env python3
"""Bounded summary of retained PyTorch CPU/XPU traces; no benchmark claims."""
import bisect
import collections
import gzip
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = []
for path in sorted((root / "control/trace").glob("rank*.gz")):
    events = json.load(gzip.open(path, "rt"))["traceEvents"]
    spans = collections.defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("name", "").startswith("eagle."):
            spans[event["name"]].append((event["ts"], event["ts"] + event["dur"]))
    for value in spans.values():
        value.sort()
    rounds = len(spans["eagle.propose"])
    assert rounds > 0
    starts = {name: [a for a, _ in values] for name, values in spans.items()}
    def covers(name, timestamp):
        index = bisect.bisect_right(starts[name], timestamp) - 1
        return index >= 0 and spans[name][index][1] >= timestamp
    ops = {e["args"]["External id"]: e for e in events
           if e.get("cat") in ("cpu_op", "user_annotation") and "External id" in e.get("args", {})}
    runtime = {e["args"]["correlation"]: e for e in events
               if e.get("cat") == "xpu_runtime" and "correlation" in e.get("args", {})}
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    stages = {name: {"calls": len(values), "calls_per_round": len(values) / rounds,
                     "host_ms_per_round": sum(b - a for a, b in values) / rounds / 1000,
                     "correlated_xpu_ms_per_round": 0.0} for name, values in spans.items()}
    groups = collections.defaultdict(lambda: {"calls": 0, "xpu_ms": 0.0})
    host_groups = collections.defaultdict(lambda: {"calls": 0, "inclusive_host_ms": 0.0})
    matched = 0
    for event in kernels:
        name = event["name"].split("<")[0].split("(")[0][:160]
        groups[name]["calls"] += 1
        groups[name]["xpu_ms"] += event["dur"] / 1000
        launch = runtime.get(event.get("args", {}).get("correlation"))
        if launch is not None:
            matched += 1
            for stage in stages:
                if covers(stage, launch["ts"]):
                    stages[stage]["correlated_xpu_ms_per_round"] += event["dur"] / rounds / 1000
    for op in ops.values():
        if op.get("cat") != "cpu_op":
            continue
        host_groups[op["name"]]["calls"] += 1
        host_groups[op["name"]]["inclusive_host_ms"] += op.get("dur", 0) / 1000
    walls = [b[0] - a[0] for a, b in zip(spans["eagle.propose"], spans["eagle.propose"][1:])]
    summaries.append({
        "trace": path.name, "decode_rounds": rounds, "event_count": len(events),
        "profiled_round_mean_ms_not_benchmark": sum(walls) / len(walls) / 1000,
        "xpu_duration_sum_ms_per_round_not_union": sum(e["dur"] for e in kernels) / rounds / 1000,
        "correlated_kernel_fraction": matched / len(kernels), "stages_inclusive_do_not_sum": stages,
        "top_kernel_groups": sorted(({"name": k, **v} for k, v in groups.items()), key=lambda v: -v["xpu_ms"])[:25],
        "top_cpu_ops_inclusive_do_not_sum": sorted(({"name": k, **v} for k, v in host_groups.items()),
                                                   key=lambda v: -v["inclusive_host_ms"])[:20],
    })
result = {"caveats": ["Profiler overhead is substantial; these are attribution diagnostics, not unprofiled latency.",
                       "Nested CPU stages and CPU operations are inclusive and cannot be summed.",
                       "XPU durations are correlated to host launch timestamps within stage scopes; sums are not busy-time unions.",
                       "Profiling started after the first HTTP output token; requested max_iterations=12."],
          "profiles": summaries}
(root / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
