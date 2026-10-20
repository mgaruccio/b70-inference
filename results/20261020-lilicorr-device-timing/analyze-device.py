#!/usr/bin/env python3
"""Fail-closed reader for the committed LiliCorr device/host diagnostic.

This module deliberately has a small, fixed parser for the two producer
namespaces in ``device-timing.patch``.  The old profile is read by the
unchanged context-profile analyzer; its records are never interpreted as
DEVICE_TIMING or PROCESS_COMPONENTS records.

The result is diagnostic preparation only.  Host microseconds and SYCL event
nanoseconds remain separate clock domains, library submissions remain
host-only/unavailable, and no whole-drafter GPU duration or speed ratio is
created.
"""
from collections import defaultdict
import json
import re
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTEXT_DIR = HERE.parent / "20261020-lilicorr-context-profile"
CONTEXT_ANALYZER = runpy.run_path(str(CONTEXT_DIR / "analyze-profile.py"))
BASE = CONTEXT_ANALYZER["BASE"]

DEVICE_PREFIX = "LLAMA_LILICORR_DEVICE_TIMING"
DEVICE_MARKER = DEVICE_PREFIX + " "
COMPONENT_PREFIX = "LLAMA_LILICORR_PROCESS_COMPONENTS"
COMPONENT_MARKER = COMPONENT_PREFIX + " "

_INT = re.compile(r"-?(?:0|[1-9][0-9]*)$")
_UINT = re.compile(r"(?:0|[1-9][0-9]*)$")
_PTR = re.compile(r"0x[0-9a-fA-F]+$")
_KEY = re.compile(r"[a-z][a-z0-9_]*$")

DEVICE_EVENT_FIELDS = frozenset({
    "v", "stage", "graph", "node", "ordinal", "backend_ctx", "device",
    "route", "kernel", "name", "weight", "k", "rows", "ncols",
    "weight_k", "weight_m", "input_cols", "coverage", "status",
    "submit_begin_us", "submit_end_us", "submit_host_us", "device_start_ns",
    "device_end_ns", "device_elapsed_ns",
})
DEVICE_COVERAGE_FIELDS = frozenset({
    "v", "stage", "graph", "backend_ctx", "device", "graph_nodes", "status",
    "coverage", "submitted", "retained", "dropped", "submit_errors",
    "other_device_work",
})
DEVICE_DRAIN_FIELDS = frozenset({
    "v", "stage", "status", "attribution", "wait_begin_us", "wait_end_us",
    "wait_host_us", "query_and_log_host_us", "records", "other_device_work",
})
DEVICE_DISCARD_FIELDS = frozenset({"v", "stage", "status", "records", "device_time"})

COMPONENT_CHUNK_FIELDS = frozenset({
    "v", "stage", "target_ctx", "draft_ctx", "seq", "batch_rows", "seq_rows",
    "offset", "rows", "batch_begin_us", "chunk_begin_us", "status",
    "getter_calls", "target_getter_host_us", "getter_attribution",
    "feature_copy_calls", "feature_copy_bytes", "feature_copy_loop_host_us",
    "process_begin_us", "process_end_us", "process_host_us", "process_rc",
    "process_attribution", "projection_seen", "projected_present",
    "projection_begin_us", "projection_end_us", "projection_getter_host_us",
    "projection_attribution", "ring_copy_calls", "ring_copy_bytes",
    "ring_copy_loop_host_us", "device_time",
})
COMPONENT_LOGGING_FIELDS = frozenset({
    "v", "stage", "draft_ctx", "seq", "offset", "batch_begin_us",
    "chunk_begin_us", "log_host_us", "coverage",
})

DEVICE_INT_FIELDS = frozenset({
    "v", "graph", "node", "ordinal", "device", "k", "rows", "ncols",
    "weight_k", "weight_m", "input_cols", "submit_begin_us", "submit_end_us",
    "submit_host_us", "graph_nodes", "submitted", "retained", "dropped",
    "submit_errors", "wait_begin_us", "wait_end_us", "wait_host_us",
    "query_and_log_host_us", "records",
})
COMPONENT_INT_FIELDS = frozenset({
    "v", "seq", "batch_rows", "seq_rows", "offset", "rows", "batch_begin_us",
    "chunk_begin_us", "getter_calls", "target_getter_host_us",
    "feature_copy_calls", "feature_copy_bytes", "feature_copy_loop_host_us",
    "process_begin_us", "process_end_us", "process_rc", "projection_seen",
    "projected_present", "projection_begin_us", "projection_end_us",
    "ring_copy_calls", "ring_copy_bytes", "log_host_us",
})

LIBRARY_ROUTES = frozenset({
    "LIBRARY_DNNL_F16", "LIBRARY_MKL_F16", "LIBRARY_DNNL_F32", "LIBRARY_MKL_F32",
})
EVENT_ROUTES = frozenset({"REORDERED_Q8_QUANTIZER", "MMVQ_REORDERED"})
EVENT_STATUSES = frozenset({
    "ok", "foreign_queue", "unsupported_profiling", "pending",
    "not_complete_after_wait", "invalid_timestamps", "profiling_query_error",
})


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _parse_int(value, key, unsigned=False):
    pattern = _UINT if unsigned else _INT
    _assert(pattern.fullmatch(value) is not None, f"malformed integer {key}={value!r}")
    return int(value)


def _parse_ptr(value, key):
    _assert(_PTR.fullmatch(value) is not None, f"malformed pointer {key}={value!r}")
    return int(value, 16)


def _parse_fields(line, marker):
    _assert(line.count(marker) == 1, f"duplicate {marker.strip()} marker")
    tail = line.split(marker, 1)[1].strip()
    _assert(tail, f"empty {marker.strip()} record")
    # Producer values contain no spaces (tensor names replace spaces and '=');
    # splitting at a key boundary catches malformed fields without interpreting
    # legacy log text as part of this namespace.
    parts = re.split(r"\s+(?=[a-z][a-z0-9_]*=)", tail)
    row = {}
    for part in parts:
        key, separator, value = part.partition("=")
        _assert(separator and key and value, f"malformed field {part!r}")
        _assert(_KEY.fullmatch(key) is not None, f"malformed field key {key!r}")
        _assert(key not in row, f"duplicate field {key}")
        _assert("=" not in value and not any(ch.isspace() for ch in value),
                f"malformed field value {key}={value!r}")
        row[key] = value
    return row


def _typed_device(row):
    stage = row.get("stage")
    schemas = {
        "event": DEVICE_EVENT_FIELDS,
        "library_submit": DEVICE_EVENT_FIELDS,
        "coverage": DEVICE_COVERAGE_FIELDS,
        "drain": DEVICE_DRAIN_FIELDS,
        "discard": DEVICE_DISCARD_FIELDS,
    }
    _assert(stage in schemas, f"unexpected device stage {stage!r}")
    _assert(set(row) == schemas[stage],
            f"unexpected device fields for {stage}: {sorted(set(row) ^ schemas[stage])}")
    for key in DEVICE_INT_FIELDS & set(row):
        row[key] = _parse_int(row[key], key, unsigned=key in {"v", "graph"})
    if stage in ("event", "library_submit"):
        row["backend_ctx"] = _parse_ptr(row["backend_ctx"], "backend_ctx")
    elif stage == "coverage":
        row["backend_ctx"] = _parse_ptr(row["backend_ctx"], "backend_ctx")
    _assert(row["v"] == 1, "device record version is not 1")

    if stage in ("event", "library_submit"):
        _validate_device_event(row)
    elif stage == "coverage":
        _validate_coverage(row)
    elif stage == "drain":
        _validate_drain(row)
    else:
        _validate_discard(row)
    return row


def _validate_device_event(row):
    _assert(row["graph"] > 0 and row["node"] >= 0 and row["ordinal"] > 0,
            "invalid device graph/node/ordinal")
    _assert(row["backend_ctx"] > 0 and row["device"] >= 0, "invalid device ownership")
    _assert(row["k"] > 0 and row["rows"] > 0 and row["ncols"] > 0,
            "invalid device launch shape")
    _assert(row["weight_k"] > 0 and row["weight_m"] > 0 and row["input_cols"] > 0,
            "invalid copied weight/input shape")
    _assert(row["input_cols"] == row["ncols"], "input_cols/ncols shape mismatch")
    _assert(row["name"] and row["weight"], "empty copied tensor name")
    _assert(row["submit_begin_us"] >= 0 and row["submit_end_us"] >= row["submit_begin_us"],
            "invalid host submit interval")
    _assert(row["submit_host_us"] == row["submit_end_us"] - row["submit_begin_us"],
            "submit_host_us is not the host interval")

    if row["stage"] == "event":
        _assert(row["route"] in EVENT_ROUTES, "unexpected selected Q8 route")
        _assert(row["coverage"] == "selected_kernel_only", "event coverage is not selected_kernel_only")
        _assert(2 <= row["ncols"] <= 8, "selected event width is outside 2..8")
        expected_kernel = {
            "REORDERED_Q8_QUANTIZER": "quantize_reorder_q8_1",
            "MMVQ_REORDERED": "mmvq_reordered_q8_0",
        }[row["route"]]
        _assert(row["kernel"] == expected_kernel, "route/kernel mismatch")
        _assert(row["status"] in EVENT_STATUSES, "unexpected event status")
        device_values = (row["device_start_ns"], row["device_end_ns"], row["device_elapsed_ns"])
        if row["status"] == "ok":
            _assert(all(_UINT.fullmatch(value) for value in device_values),
                    "successful event has malformed device timestamps")
            start, end, elapsed = map(int, device_values)
            _assert(start > 0 and end >= start and elapsed == end - start,
                    "invalid successful device timestamp interval")
        else:
            _assert(device_values == ("unavailable", "unavailable", "unavailable"),
                    "failed event fabricated device time")
    else:
        _assert(row["route"] in LIBRARY_ROUTES, "unexpected library route")
        _assert(row["kernel"] == "unmeasured", "library kernel must remain unmeasured")
        _assert(row["weight"] == "fc.weight" and row["ncols"] == 46 and row["coverage"] == "host_only_library_call",
                "library marker is not the width-46 host-only branch")
        _assert(row["status"] == "device_unavailable_library",
                "library marker must explicitly remain device-unavailable")
        _assert((row["device_start_ns"], row["device_end_ns"], row["device_elapsed_ns"])
                == ("unavailable", "unavailable", "unavailable"),
                "library marker fabricated device time")


def _validate_coverage(row):
    _assert(row["graph"] > 0 and row["backend_ctx"] > 0 and row["device"] >= 0,
            "invalid coverage ownership")
    _assert(row["graph_nodes"] > 0, "coverage has no graph nodes")
    _assert(row["coverage"] == "q8_reordered_2to8_fc46_host_only",
            "unexpected device coverage label")
    _assert(row["status"] in ("partial", "graphs_enabled_unsupported"),
            "unexpected coverage status")
    for key in ("submitted", "retained", "dropped", "submit_errors"):
        _assert(row[key] >= 0, f"negative coverage count {key}")
    _assert(row["retained"] + row["dropped"] + row["submit_errors"] == row["submitted"],
            "coverage counts do not account for every submission")
    _assert(row["retained"] <= row["submitted"] and row["dropped"] <= row["submitted"],
            "coverage count exceeds submissions")
    _assert(row["other_device_work"] == "unmeasured", "other device work was silently measured")
    if row["status"] == "graphs_enabled_unsupported":
        _assert(row["submitted"] == row["retained"] == row["dropped"] == row["submit_errors"] == 0,
                "unsupported graph coverage reported submissions")


def _validate_drain(row):
    _assert(row["status"] == "existing_wait_returned",
            "drain is not after the existing wait")
    _assert(row["attribution"] == "shared_queue_not_pure_drafter",
            "drain was mislabeled as pure drafter time")
    _assert(row["other_device_work"] == "unmeasured", "drain claimed other device work")
    _assert(row["wait_begin_us"] >= 0 and row["wait_end_us"] >= row["wait_begin_us"],
            "invalid existing wait interval")
    _assert(row["wait_host_us"] == row["wait_end_us"] - row["wait_begin_us"],
            "wait_host_us is not the host interval")
    _assert(row["query_and_log_host_us"] >= 0 and row["records"] >= 0,
            "invalid drain host/count fields")


def _validate_discard(row):
    _assert(row["status"] == "no_existing_synchronize", "unexpected discard status")
    _assert(row["device_time"] == "unavailable", "discard fabricated device time")
    _assert(row["records"] > 0, "discard record count must be positive")


def _typed_component(row):
    stage = row.get("stage")
    schemas = {"chunk": COMPONENT_CHUNK_FIELDS, "logging": COMPONENT_LOGGING_FIELDS}
    _assert(stage in schemas, f"unexpected component stage {stage!r}")
    _assert(set(row) == schemas[stage],
            f"unexpected component fields for {stage}: {sorted(set(row) ^ schemas[stage])}")
    for key in COMPONENT_INT_FIELDS & set(row):
        row[key] = _parse_int(row[key], key)
    if "target_ctx" in row:
        row["target_ctx"] = _parse_ptr(row["target_ctx"], "target_ctx")
    row["draft_ctx"] = _parse_ptr(row["draft_ctx"], "draft_ctx")
    _assert(row["v"] == 1, "component record version is not 1")
    if stage == "chunk":
        _validate_chunk(row)
    else:
        _validate_logging(row)
    return row


def _validate_host_duration(value, key, begin, end):
    _assert(value != "unavailable", f"{key} is unavailable")
    parsed = _parse_int(value, key)
    _assert(parsed >= 0 and end >= begin and parsed == end - begin,
            f"{key} is not the host interval")
    return parsed


def _validate_chunk(row):
    _assert(row["target_ctx"] > 0 and row["draft_ctx"] > 0, "invalid component context ownership")
    _assert(row["seq"] >= 0 and row["batch_rows"] > 0 and row["seq_rows"] > 0,
            "invalid component batch shape")
    _assert(row["offset"] >= 0 and row["rows"] > 0,
            "invalid component offset/rows")
    _assert(row["seq_rows"] >= row["offset"] + row["rows"] and row["batch_rows"] >= row["seq_rows"],
            "component rows do not contain the chunk")
    _assert(row["batch_begin_us"] >= 0 and row["chunk_begin_us"] >= row["batch_begin_us"],
            "invalid component host clocks")
    _assert(row["status"] in ("ok", "process_error", "not_submitted"),
            "unexpected component status")
    _assert(row["getter_attribution"] == "target_wait_and_reorder_possible",
            "getter attribution lost")
    _assert(row["process_attribution"] == "host_submit_and_internal_waits",
            "process attribution lost")
    _assert(row["projection_attribution"] == "injection_or_shared_queue_drain",
            "projection attribution lost")
    _assert(row["device_time"] == "unmeasured", "host component claimed device time")
    _assert(row["getter_calls"] > 0 and row["target_getter_host_us"] >= 0,
            "invalid target getter accounting")
    _assert(row["feature_copy_calls"] > 0 and row["feature_copy_bytes"] > 0
            and row["feature_copy_loop_host_us"] >= 0,
            "invalid feature-copy byte accounting")

    if row["status"] == "ok":
        _assert(row["process_rc"] == 0, "successful component has nonzero process rc")
        process_host = _validate_host_duration(row["process_host_us"], "process_host_us",
                                               row["process_begin_us"], row["process_end_us"])
        _assert(process_host >= 0, "negative process host duration")
    elif row["status"] == "process_error":
        _assert(row["process_rc"] != 0 and row["process_host_us"] != "unavailable",
                "process error lost its return code/interval")
        _validate_host_duration(row["process_host_us"], "process_host_us",
                                row["process_begin_us"], row["process_end_us"])
    else:
        _assert(row["process_rc"] == -1 and row["process_begin_us"] == 0
                and row["process_end_us"] == 0 and row["process_host_us"] == "unavailable",
                "not-submitted component fabricated process time")

    _assert(row["projection_seen"] in (0, 1) and row["projected_present"] in (0, 1),
            "invalid projection flags")
    if not row["projection_seen"]:
        _assert(row["projected_present"] == 0 and row["projection_begin_us"] == 0
                and row["projection_end_us"] == 0 and row["projection_getter_host_us"] == "unavailable",
                "missing projection fabricated duration")
    else:
        _assert(row["projection_begin_us"] >= 0 and row["projection_end_us"] >= row["projection_begin_us"],
                "invalid projection interval")
        _validate_host_duration(row["projection_getter_host_us"], "projection_getter_host_us",
                                row["projection_begin_us"], row["projection_end_us"])

    _assert(row["ring_copy_calls"] >= 0 and row["ring_copy_bytes"] >= 0,
            "negative ring byte accounting")
    if row["ring_copy_calls"] == 0:
        _assert(row["ring_copy_bytes"] == 0, "ring bytes without ring copies")
    else:
        _assert(row["ring_copy_bytes"] > 0, "ring copies without bytes")
    if row["projected_present"]:
        _assert(row["ring_copy_loop_host_us"] != "unavailable",
                "projected result lost ring-loop duration")
        _parse_int(row["ring_copy_loop_host_us"], "ring_copy_loop_host_us")
        _assert(int(row["ring_copy_loop_host_us"]) >= 0, "negative ring-loop duration")
    else:
        _assert(row["ring_copy_calls"] == row["ring_copy_bytes"] == 0
                and row["ring_copy_loop_host_us"] == "unavailable",
                "missing projected result fabricated ring work")


def _validate_logging(row):
    _assert(row["draft_ctx"] > 0 and row["seq"] >= 0 and row["offset"] >= 0,
            "invalid logging ownership")
    _assert(row["batch_begin_us"] >= 0 and row["chunk_begin_us"] >= row["batch_begin_us"],
            "invalid logging clocks")
    _assert(row["log_host_us"] >= 0, "negative logging host cost")
    _assert(row["coverage"] == "chunk_record_format_and_log_only",
            "logging row claimed component coverage")


def _parse_records(stderr):
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", "strict")
    _assert(isinstance(stderr, str), "stderr must be text or UTF-8 bytes")
    device, components = [], []
    for line_number, line in enumerate(stderr.splitlines(), 1):
        has_device = DEVICE_PREFIX in line
        has_components = COMPONENT_PREFIX in line
        _assert(not (has_device and has_components), "two diagnostic namespaces share one line")
        if has_device:
            _assert(line.count(DEVICE_PREFIX) == 1 and DEVICE_MARKER in line,
                    "malformed device marker")
            row = _parse_fields(line, DEVICE_MARKER)
            row = _typed_device(row)
            row["line"] = line_number
            device.append(row)
        elif has_components:
            _assert(line.count(COMPONENT_PREFIX) == 1 and COMPONENT_MARKER in line,
                    "malformed process-components marker")
            row = _parse_fields(line, COMPONENT_MARKER)
            row = _typed_component(row)
            row["line"] = line_number
            components.append(row)
    return device, components


def _row_key(row):
    return row["graph"], row["ordinal"]


def _component_key(row):
    return (row["draft_ctx"], row["seq"], row["offset"],
            row["batch_begin_us"], row["chunk_begin_us"])


def _host_value(values):
    """Return a sum only when every observed component is measured."""
    if not values or any(value is None for value in values):
        return None
    return sum(values)


def _duration_field(row, field):
    value = row[field]
    return None if value == "unavailable" else int(value)


def _component_report(chunks, logging):
    process_values = [_duration_field(row, "process_host_us") for row in chunks]
    projection_values = [_duration_field(row, "projection_getter_host_us") for row in chunks]
    ring_values = [_duration_field(row, "ring_copy_loop_host_us") for row in chunks]
    totals = {
        "target_getter": sum(row["target_getter_host_us"] for row in chunks),
        "feature_cpu_memcpy_loop": sum(row["feature_copy_loop_host_us"] for row in chunks),
        "injection_submission": _host_value(process_values),
        "projection_getter": _host_value(projection_values),
        "ring_cpu_loop": _host_value(ring_values),
        "separate_log_cost": sum(row["log_host_us"] for row in logging),
    }
    return {
        "scope": "per-request",
        "unit": "us for host durations; byte fields are bytes",
        "chunk_count": len(chunks),
        "logging_record_count": len(logging),
        "totals_us": totals,
        "available": {
            "target_getter": True,
            "feature_cpu_memcpy_loop": True,
            "injection_submission": all(value is not None for value in process_values),
            "projection_getter": all(value is not None for value in projection_values),
            "ring_cpu_loop": all(value is not None for value in ring_values),
            "separate_log_cost": True,
        },
        "bytes": {
            "feature_copy": sum(row["feature_copy_bytes"] for row in chunks),
            "ring_copy": sum(row["ring_copy_bytes"] for row in chunks),
        },
        "calls": {
            "target_getter": sum(row["getter_calls"] for row in chunks),
            "feature_copy": sum(row["feature_copy_calls"] for row in chunks),
            "ring_copy": sum(row["ring_copy_calls"] for row in chunks),
        },
        "attribution": {
            "target_getter": "target queue wait/output reorder may be included",
            "feature_cpu_memcpy_loop": "existing CPU memcpy/index loop",
            "injection_submission": "host submission and internal waits may be included",
            "projection_getter": "injection/shared-queue drain may be included",
            "ring_cpu_loop": "existing projected-ring copy/position loop",
            "separate_log_cost": "format/log host cost, excluding its own reporting call",
        },
        "unavailable_is_not_zero": True,
    }


def _timing_group(rows):
    groups = {}
    for row in rows:
        key = (row["route"], row["weight"], row["ncols"])
        item = groups.setdefault(key, {
            "route": row["route"], "weight": row["weight"], "width": row["ncols"],
            "count": 0, "device_elapsed_ns": {"unit": "ns", "sum": 0, "min": None, "max": None},
            "submission_host_us": {"unit": "us", "sum": 0}, "shape_variants": [],
        })
        device_elapsed = int(row["device_elapsed_ns"])
        item["count"] += 1
        item["device_elapsed_ns"]["sum"] += device_elapsed
        item["device_elapsed_ns"]["min"] = (device_elapsed if item["device_elapsed_ns"]["min"] is None
                                              else min(item["device_elapsed_ns"]["min"], device_elapsed))
        item["device_elapsed_ns"]["max"] = (device_elapsed if item["device_elapsed_ns"]["max"] is None
                                              else max(item["device_elapsed_ns"]["max"], device_elapsed))
        item["submission_host_us"]["sum"] += row["submit_host_us"]
        shape = {"k": row["k"], "rows": row["rows"], "input_cols": row["input_cols"],
                 "weight_k": row["weight_k"], "weight_m": row["weight_m"]}
        if shape not in item["shape_variants"]:
            item["shape_variants"].append(shape)
    return sorted(groups.values(), key=lambda item: (item["route"], item["weight"], item["width"]))


def _library_group(rows):
    groups = {}
    for row in rows:
        key = (row["route"], row["weight"], row["ncols"])
        item = groups.setdefault(key, {
            "route": row["route"], "weight": row["weight"], "width": row["ncols"],
            "count": 0, "device_elapsed_ns": "unavailable",
            "submission_host_us": {"unit": "us", "sum": 0}, "coverage": "host_only_library_call",
        })
        item["count"] += 1
        item["submission_host_us"]["sum"] += row["submit_host_us"]
    return sorted(groups.values(), key=lambda item: (item["route"], item["weight"], item["width"]))


def _device_report(device_rows, coverage_rows, drain_rows):
    events = [row for row in device_rows if row["stage"] == "event"]
    libraries = [row for row in device_rows if row["stage"] == "library_submit"]
    widths = sorted({row["ncols"] for row in events})
    expected_widths = list(range(2, 9))
    return {
        "scope": "selected diagnostic records in one request",
        "event_record_count": len(events),
        "successful_event_record_count": sum(row["status"] == "ok" for row in events),
        "library_submit_record_count": len(libraries),
        "observed_q8_widths": widths,
        "unmeasured_q8_widths": [width for width in expected_widths if width not in widths],
        "observed_q8_width_presence": {str(width): width in widths for width in expected_widths},
        "selected_q8": {
            "by_route_weight_width": _timing_group(events),
            "clock_domain": "SYCL command profiling timestamps, nanoseconds",
            "submission_host_clock_domain": "ggml_time_us host intervals, microseconds",
        },
        "library_submit_host_only": {
            "by_route_weight_width": _library_group(libraries),
            "observed_fc46": bool(libraries),
            "observed_routes": sorted({row["route"] for row in libraries}),
            "device_elapsed_ns": "unavailable",
            "interpretation": "host submission interval only; not GPU time",
        },
        "coverage": [dict(row) for row in coverage_rows],
        "drains": [dict(row) for row in drain_rows],
        "clock_domains_not_combined": True,
        "other_device_work": "unmeasured",
        "no_whole_drafter_gpu_time": True,
    }


def check_request(stderr):
    """Parse one already-framed public request and return a validated report.

    The caller must pass only one request's stderr when possible.  This function
    intentionally ignores the legacy ``LLAMA_LILICORR_PROFILE`` and
    ``LLAMA_LILICORR_SYCL_PROFILE`` namespaces; the full analyzer separately
    sends the same request through the unchanged context-profile reader.
    """
    device_rows, component_rows = _parse_records(stderr)
    _assert(device_rows, "request has no new device timing records")
    _assert(component_rows, "request has no new process component records")

    coverage = [row for row in device_rows if row["stage"] == "coverage"]
    events = [row for row in device_rows if row["stage"] == "event"]
    libraries = [row for row in device_rows if row["stage"] == "library_submit"]
    drains = [row for row in device_rows if row["stage"] == "drain"]
    discards = [row for row in device_rows if row["stage"] == "discard"]
    _assert(coverage, "request has no device coverage rows")
    _assert(drains, "request has no existing-wait drain rows")
    _assert(not discards, "pending device records were discarded without synchronization")
    _assert(not any(row["status"] == "graphs_enabled_unsupported" for row in coverage),
            "device profiling was unsupported by graph configuration")
    _assert(all(row["status"] == "partial" for row in coverage),
            "request is not partial selected-Q8 coverage")
    _assert(not any(row["dropped"] or row["submit_errors"] for row in coverage),
            "device records were dropped or submission failed")

    coverage_by_key = defaultdict(list)
    for row in coverage:
        key = (row["graph"], row["backend_ctx"], row["device"])
        _assert(not coverage_by_key[key], "duplicate coverage owner")
        coverage_by_key[key].append(row)
    seen_events = set()
    for row in events + libraries:
        key = _row_key(row)
        _assert(key not in seen_events, "duplicate device record ordinal")
        seen_events.add(key)
        owner = coverage_by_key.get((row["graph"], row["backend_ctx"], row["device"]), [])
        _assert(len(owner) == 1, "device record has no unique coverage owner")
        _assert(0 <= row["node"] < owner[0]["graph_nodes"], "device node is outside graph")
        _assert(row["ordinal"] <= owner[0]["submitted"], "device ordinal exceeds submissions")
    for owner in coverage:
        rows = [row for row in events + libraries
                if (row["graph"], row["backend_ctx"], row["device"])
                == (owner["graph"], owner["backend_ctx"], owner["device"])]
        _assert(len(rows) == owner["retained"], "retained count does not match emitted records")
    _assert(sum(row["records"] for row in drains) == len(events) + len(libraries),
            "drain record count does not match emitted device records")

    _assert(events, "request has no selected Q8 device event")
    _assert(all(row["status"] == "ok" for row in events),
            "selected Q8 event is unavailable or incomplete")
    _assert(any(row["ncols"] == 8 for row in events),
            "request has no successfully timed selected n8 Q8 event")

    chunks = [row for row in component_rows if row["stage"] == "chunk"]
    logging = [row for row in component_rows if row["stage"] == "logging"]
    _assert(chunks, "request has no process component chunk rows")
    _assert(all(row["status"] == "ok" and row["process_rc"] == 0 for row in chunks),
            "process component row is not a successful timed submission")
    chunk_keys = set()
    for row in chunks:
        key = _component_key(row)
        _assert(key not in chunk_keys, "duplicate process component chunk")
        chunk_keys.add(key)
    logging_keys = set()
    for row in logging:
        key = _component_key(row)
        _assert(key not in logging_keys, "duplicate process component logging row")
        logging_keys.add(key)
    _assert(logging_keys == chunk_keys, "process chunk/logging ownership mismatch")

    report = {
        "status": "ok",
        "records": {
            "device": len(device_rows),
            "components": len(component_rows),
            "legacy_namespaces_ignored": True,
        },
        "device": _device_report(device_rows, coverage, drains),
        "host": _component_report(chunks, logging),
        "guard": {
            "selected_q8_event": True,
            "selected_n8_q8_event": True,
            "process_components": True,
            "successful_existing_wait_drain": True,
            "library_unavailable_accepted_without_zero": bool(libraries),
        },
        "raw_line_refs": {
            "device": [row["line"] for row in device_rows],
            "components": [row["line"] for row in component_rows],
        },
    }
    return report


def _marker_count(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8", "strict")
    return {
        "device": sum(DEVICE_PREFIX in line for line in text.splitlines()),
        "components": sum(COMPONENT_PREFIX in line for line in text.splitlines()),
    }


def _read_stderr(directory):
    retained = CONTEXT_ANALYZER.get("RetainedPath", Path)
    return retained(directory / "server.stderr.log").read_text()


def _run_context_validation(out, c):
    """Run the unchanged full context analyzer without replacing its result."""
    saved = []
    shadow = dict(c)
    shadow["save"] = lambda path, value: saved.append((Path(path), value))
    result = CONTEXT_ANALYZER["analyze"](Path(out), shadow)
    _assert(result.get("status") == "analyzed", "unchanged context profile validation failed")
    _assert(not result.get("errors"), "unchanged context profile reported errors")
    _assert(len(result.get("requests", [])) == 24,
            "unchanged profile did not validate all 24 measured public outputs")
    return result


def _context_graph_record(context_result, meta):
    records = context_result.get("sycl_diagnostic", {}).get("graphs", [])
    matches = [record for record in records
               if all(record.get(key) == meta.get(key)
                      for key in ("cell", "point", "phase", "ordinal"))]
    _assert(len(matches) == 1, "missing/duplicate old context graph record for selected request")
    return matches[0]


def _graph_round_index(context_record):
    scope = context_record.get("context_scope") or {}
    index = {}
    for item in scope.get("rounds", []):
        graphs = list(item.get("graphs", []))
        for graph in graphs:
            index[graph] = {"scope": "draft_round", "round": item.get("round")}
        for key in ("head_graph", "decoder_graph"):
            if key in item:
                index[item[key]] = {"scope": "draft_round", "round": item.get("round")}
    for item in scope.get("injections", []):
        for graph in item.get("graphs", []):
            index[graph] = {"scope": "injection_owner", "ubatch_line": item.get("ubatch_line")}
    return index




def _validate_graph_links(parsed, context_record):
    old_graphs = {row["graph"]: row for row in context_record.get("graphs", [])
                  if "graph" in row and "graph_nodes" in row}
    if not old_graphs:
        return
    round_index = _graph_round_index(context_record)
    for row in parsed["device"]["coverage"]:
        _assert(row["graph"] in old_graphs, "device graph is absent from old context scope")
        _assert(row["graph_nodes"] == old_graphs[row["graph"]]["graph_nodes"],
                "device graph node count disagrees with old context graph")
        if row["graph"] in round_index:
            row["context_scope"] = round_index[row["graph"]]

def _selected_requests(out, c):
    """Use the existing request framing and schedule; never select tail rows."""
    selected = []
    excluded = {"startup": {"device": 0, "components": 0},
                "public": {"device": 0, "components": 0},
                "warmup": {"device": 0, "components": 0},
                "baseline": {"device": 0, "components": 0}}
    for cell in c["CELLS"]:
        directory = Path(out) / cell
        stderr = _read_stderr(directory)
        requests, _startup = BASE["requests"](stderr)
        jobs = [job for job in c["schedule"]() if job["cell"] == cell]
        _assert(len(jobs) == 8, f"unexpected schedule for {cell}")
        first_line = requests[0]["first_line"]
        startup_text = "\n".join(stderr.splitlines()[:first_line - 1])
        start_counts = _marker_count(startup_text)
        for key in excluded["startup"]:
            excluded["startup"][key] += start_counts[key]
        if c["role_for"](cell) == "A":
            counts = _marker_count(stderr)
            _assert(not counts["device"] and not counts["components"],
                    "baseline cell contains opt-in diagnostic records")
            excluded["baseline"]["device"] += counts["device"]
            excluded["baseline"]["components"] += counts["components"]
            continue
        public_text = "\n".join(line for request in requests[:25] for line in request["lines"])
        public_counts = _marker_count(public_text)
        excluded["public"]["device"] += public_counts["device"]
        excluded["public"]["components"] += public_counts["components"]
        for index, job in enumerate(jobs, 25):
            request = requests[index]
            text = "\n".join(request["lines"])
            if job["phase"] != "measured":
                counts = _marker_count(text)
                excluded["warmup"]["device"] += counts["device"]
                excluded["warmup"]["components"] += counts["components"]
                continue
            selected.append({
                "cell": cell, "point": job["point"], "phase": job["phase"],
                "ordinal": job["ordinal"], "request_index": index,
                "first_line": request["first_line"],
                "last_line": request.get("last_line"), "text": text,
            })
    _assert(selected, "no measured W3 requests selected")
    return selected, excluded


def _aggregate(requests):
    event_rows = [row for request in requests for row in request["device"]["_device_rows"]
                  if row["stage"] == "event"]
    library_rows = [row for request in requests for row in request["device"]["_device_rows"]
                    if row["stage"] == "library_submit"]
    coverage_rows = [row for request in requests for row in request["device"]["_device_rows"]
                     if row["stage"] == "coverage"]
    widths = sorted({row["ncols"] for row in event_rows})
    return {
        "request_count": len(requests),
        "event_record_count": len(event_rows),
        "library_submit_record_count": len(library_rows),
        "observed_q8_widths": widths,
        "unmeasured_q8_widths": [width for width in range(2, 9) if width not in widths],
        "observed_q8_width_presence": {str(width): width in widths for width in range(2, 9)},
        "observed_fc46": bool(library_rows),
        "observed_fc46_library_routes": sorted({row["route"] for row in library_rows}),
        "coverage_rows": len(coverage_rows),
        "dropped_records": sum(row["dropped"] for row in coverage_rows),
        "submit_errors": sum(row["submit_errors"] for row in coverage_rows),
        "selected_q8_device_ns_by_route_weight_width": _timing_group(event_rows),
        "selected_q8_submission_host_us_by_route_weight_width": [
            {"route": item["route"], "weight": item["weight"], "width": item["width"],
             "count": item["count"], "submission_host_us": item["submission_host_us"]}
            for item in _timing_group(event_rows)
        ],
        "library_submission_host_us_by_route_weight_width": _library_group(library_rows),
        "clock_domains": {
            "device": "SYCL command profiling timestamps, nanoseconds",
            "host": "ggml_time_us host intervals, microseconds",
            "combined": False,
        },
        "per_request_not_per_round": True,
        "no_drafter_gpu_time_or_speed_ratio": True,
    }


def analyze(out, c):
    """Validate the existing profile and write only ``analysis-r1/device-analysis.json``."""
    out = Path(out)
    report = {
        "status": "failed",
        "tier": "development-diagnostic",
        "diagnostic_only": True,
        "gpu_time_claim": False,
        "speed_comparison_permitted": False,
        "optimization_claim": False,
        "requests": [],
        "errors": [],
        "limitations": [
            "selected reordered Q8 kernel event records only; other device work is unmeasured",
            "width-46 fc.weight library markers are host-only and device time is unavailable",
            "host intervals can include queue waits, output reorder, internal waits, or shared-queue drain",
            "device nanoseconds and host microseconds are never subtracted or summed",
            "this diagnostic cannot establish whole-drafter GPU time or a speed ratio",
        ],
    }
    try:
        context_result = _run_context_validation(out, c)
        report["context_profile"] = {
            "checked_through_unchanged_analyzer": True,
            "status": context_result["status"],
            "measured_public_outputs": len(context_result["requests"]),
            "base_result_path": "analysis-r1/analysis.json",
            "old_profile_result_path": "analysis-r1/profile.json",
        }
        selected, excluded = _selected_requests(out, c)
        report["excluded_records"] = excluded
        request_reports = []
        for meta in selected:
            parsed = check_request(meta["text"])
            context_record = _context_graph_record(context_result, meta)
            _validate_graph_links(parsed, context_record)
            # Keep raw rows private to this in-memory aggregation; normalized
            # output below retains only the compact per-request report.
            parsed["device"]["_device_rows"] = [row for row in _parse_records(meta["text"])[0]]
            request_reports.append({
                key: meta[key] for key in
                ("cell", "point", "phase", "ordinal", "request_index", "first_line", "last_line")
                if meta.get(key) is not None
            } | {
                "guard": parsed["guard"],
                "device": {key: value for key, value in parsed["device"].items()
                           if key != "_device_rows"},
                "host": parsed["host"],
                "raw_line_refs": parsed["raw_line_refs"],
            })
        report["requests"] = request_reports
        report["coverage"] = _aggregate([{
            "device": {"_device_rows": _parse_records(meta["text"])[0]},
        } for meta in selected])
        report["host_components"] = {
            "scope": "per-request; these component totals are not per-round sums",
            "requests": [{
                "cell": request["cell"], "point": request["point"],
                "phase": request["phase"], "ordinal": request["ordinal"],
                "components": request["host"],
            } for request in request_reports],
            "legacy_t_draft_us_excludes_process": True,
        }
        report["status"] = "analyzed"
    except (AssertionError, KeyError, OSError, TypeError, ValueError, UnicodeError) as exc:
        report["errors"].append(repr(exc))

    analysis_dir = out / "analysis-r1"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output = analysis_dir / "device-analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _load_contract_for_cli():
    runner = runpy.run_path(str(CONTEXT_DIR / "run-profile.py"))
    return runner["bind"]()[2]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default=None,
                        help="existing profile-r1 directory (defaults to the context runner OUT)")
    args = parser.parse_args(argv)
    c = _load_contract_for_cli()
    out = Path(args.out) if args.out else Path(c["OUT"])
    result = analyze(out, c)
    print(json.dumps({"status": result["status"], "output": str(out / "analysis-r1/device-analysis.json"),
                      "errors": result["errors"]}, sort_keys=True))
    return 0 if result["status"] == "analyzed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
