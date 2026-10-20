#!/usr/bin/env python3
"""CPU-only parser checks for the committed device timing diagnostic."""
import gzip
import json
import runpy
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
A = runpy.run_path(str(HERE / "analyze-device.py"))


def _device_line(stage, **fields):
    values = {"v": 1, "stage": stage}
    values.update(fields)
    return A["DEVICE_MARKER"] + " ".join(f"{key}={value}" for key, value in values.items())


def _component_line(stage, **fields):
    values = {"v": 1, "stage": stage}
    values.update(fields)
    return A["COMPONENT_MARKER"] + " ".join(f"{key}={value}" for key, value in values.items())


def valid_request(include_library=True):
    rows = [
        "legacy LLAMA_LILICORR_SYCL_PROFILE v=1 stage=gate graph=10 requested=1 enabled=1 "
        "marker=1 graph_nodes=59 marker_name=lilicorr.context",
        _device_line(
            "coverage", graph=10, backend_ctx="0xabc", device=0, graph_nodes=59,
            status="partial", coverage="q8_reordered_2to8_fc46_host_only", submitted=3 if include_library else 2,
            retained=3 if include_library else 2, dropped=0, submit_errors=0,
            other_device_work="unmeasured",
        ),
        _device_line(
            "event", graph=10, node=7, ordinal=1, backend_ctx="0xabc", device=0,
            route="MMVQ_REORDERED", kernel="mmvq_reordered_q8_0", name="lilicorr.out",
            weight="fc.weight", k=24576, rows=32, ncols=8, weight_k=24576, weight_m=4096,
            input_cols=8, coverage="selected_kernel_only", status="ok",
            submit_begin_us=100, submit_end_us=105, submit_host_us=5,
            device_start_ns=100000, device_end_ns=100120, device_elapsed_ns=120,
        ),
        _device_line(
            "event", graph=10, node=8, ordinal=2, backend_ctx="0xabc", device=0,
            route="REORDERED_Q8_QUANTIZER", kernel="quantize_reorder_q8_1", name="q8_1",
            weight="fc.weight", k=24576, rows=32, ncols=4, weight_k=24576, weight_m=4096,
            input_cols=4, coverage="selected_kernel_only", status="ok",
            submit_begin_us=110, submit_end_us=116, submit_host_us=6,
            device_start_ns=200000, device_end_ns=200050, device_elapsed_ns=50,
        ),
    ]
    if include_library:
        rows.append(_device_line(
            "library_submit", graph=10, node=9, ordinal=3, backend_ctx="0xabc", device=0,
            route="LIBRARY_DNNL_F16", kernel="unmeasured", name="fc.out", weight="fc.weight",
            k=24576, rows=32, ncols=46, weight_k=24576, weight_m=4096, input_cols=46,
            coverage="host_only_library_call", status="device_unavailable_library",
            submit_begin_us=120, submit_end_us=132, submit_host_us=12,
            device_start_ns="unavailable", device_end_ns="unavailable",
            device_elapsed_ns="unavailable",
        ))
    rows.extend([
        _device_line(
            "drain", status="existing_wait_returned", attribution="shared_queue_not_pure_drafter",
            wait_begin_us=1000, wait_end_us=1020, wait_host_us=20, query_and_log_host_us=6,
            records=3 if include_library else 2, other_device_work="unmeasured",
        ),
        _component_line(
            "chunk", target_ctx="0x111", draft_ctx="0x222", seq=0, batch_rows=37,
            seq_rows=34, offset=2, rows=32, batch_begin_us=2000, chunk_begin_us=2020,
            status="ok", getter_calls=6, target_getter_host_us=72,
            getter_attribution="target_wait_and_reorder_possible", feature_copy_calls=192,
            feature_copy_bytes=3145728, feature_copy_loop_host_us=6, process_begin_us=2030,
            process_end_us=2054, process_host_us=24, process_rc=0,
            process_attribution="host_submit_and_internal_waits", projection_seen=1,
            projected_present=1, projection_begin_us=2060, projection_end_us=2098,
            projection_getter_host_us=38, projection_attribution="injection_or_shared_queue_drain",
            ring_copy_calls=4, ring_copy_bytes=65536, ring_copy_loop_host_us=7,
            device_time="unmeasured",
        ),
        _component_line(
            "logging", draft_ctx="0x222", seq=0, offset=2, batch_begin_us=2000,
            chunk_begin_us=2020, log_host_us=8, coverage="chunk_record_format_and_log_only",
        ),
    ])
    return "\n".join(rows)


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


class DeviceAnalysisTests(unittest.TestCase):
    def assertRejects(self, text, message=None):
        with self.assertRaises(AssertionError, msg=message):
            A["check_request"](text)

    def test_positive_selected_q8_and_partial_fc46_library(self):
        report = A["check_request"](valid_request())
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["device"]["observed_q8_widths"], [4, 8])
        self.assertEqual(report["device"]["unmeasured_q8_widths"], [2, 3, 5, 6, 7])
        self.assertTrue(report["device"]["library_submit_host_only"]["observed_fc46"])
        self.assertEqual(report["device"]["library_submit_host_only"]["device_elapsed_ns"], "unavailable")
        self.assertEqual(report["host"]["totals_us"]["injection_submission"], 24)
        self.assertEqual(report["host"]["totals_us"]["separate_log_cost"], 8)
        self.assertTrue(report["guard"]["library_unavailable_accepted_without_zero"])
        self.assertTrue(report["device"]["clock_domains_not_combined"])

    def test_legacy_namespace_is_separate_and_request_isolation_is_exact(self):
        good = valid_request()
        self.assertEqual(A["check_request"](good)["status"], "ok")
        self.assertRejects(good + "\n" + A["DEVICE_MARKER"] +
                           "v=1 stage=discard status=no_existing_synchronize records=1 device_time=unavailable")
        self.assertEqual(A["check_request"](valid_request(include_library=False))["status"], "ok")

    def test_malformed_duplicate_unexpected_and_nan_fields_rejected(self):
        self.assertRejects(valid_request().replace("v=1 stage=coverage", "v=1 v=1 stage=coverage", 1))
        self.assertRejects(valid_request().replace("other_device_work=unmeasured", "other_device_work=unmeasured extra=1", 1))
        self.assertRejects(valid_request().replace("device_elapsed_ns=120", "device_elapsed_ns=nan", 1))
        self.assertRejects(valid_request().replace("device_elapsed_ns=120", "device_elapsed_ns=-1", 1))

    def test_status_unsupported_dropped_submit_error_and_discard_rejected(self):
        bad = replace_once(valid_request(), "status=ok submit_begin_us=100", "status=unsupported_profiling submit_begin_us=100")
        bad = bad.replace("device_start_ns=100000 device_end_ns=100120 device_elapsed_ns=120",
                          "device_start_ns=unavailable device_end_ns=unavailable device_elapsed_ns=unavailable", 1)
        self.assertRejects(bad)
        self.assertRejects(valid_request().replace("dropped=0 submit_errors=0", "dropped=1 submit_errors=0", 1))
        self.assertRejects(valid_request().replace("dropped=0 submit_errors=0", "dropped=0 submit_errors=1", 1))
        self.assertRejects(valid_request() + "\n" + A["DEVICE_MARKER"] +
                           "v=1 stage=discard status=no_existing_synchronize records=1 device_time=unavailable")

    def test_graph_backend_shape_and_host_byte_ownership_rejected(self):
        self.assertRejects(valid_request().replace("event graph=10", "event graph=11", 1))
        self.assertRejects(valid_request().replace("backend_ctx=0xabc", "backend_ctx=0xdef", 1))
        self.assertRejects(valid_request().replace("ncols=8 weight_k", "ncols=9 weight_k", 1))
        self.assertRejects(valid_request().replace("feature_copy_bytes=3145728", "feature_copy_bytes=0", 1))
        self.assertRejects(valid_request().replace("ring_copy_bytes=65536", "ring_copy_bytes=-1", 1))

    def test_process_failure_and_missing_n8_guard_rejected(self):
        bad = valid_request().replace("status=ok getter_calls=6", "status=process_error getter_calls=6", 1)
        bad = bad.replace("process_rc=0", "process_rc=7", 1)
        self.assertRejects(bad)
        no_n8 = valid_request().replace("ncols=8 weight_k", "ncols=7 weight_k", 1)
        no_n8 = no_n8.replace("input_cols=8", "input_cols=7", 1)
        self.assertRejects(no_n8)

    def test_clock_domain_and_exact_interval_contracts(self):
        self.assertRejects(valid_request().replace("submit_host_us=5", "submit_host_us=500000", 1))
        self.assertRejects(valid_request().replace("device_end_ns=100120 device_elapsed_ns=120",
                                                    "device_end_ns=100120 device_elapsed_ns=20", 1))
        report = A["check_request"](valid_request())
        group = report["device"]["selected_q8"]["by_route_weight_width"][0]
        self.assertIn("device_elapsed_ns", group)
        self.assertIn("submission_host_us", group)
        self.assertNotIn("drafter_gpu_time", report)

    def test_existing_old_reader_fixture_keeps_request_framing_and_phase_selection(self):
        fixture = HERE.parent / "20261020-lilicorr-context-profile/fixtures/combined-startup-measured.json.gz"
        payload = json.loads(gzip.decompress(fixture.read_bytes()))
        self.assertEqual(len(payload["requests"]), 8)
        measured = 0
        for request in payload["requests"]:
            lines = request["lines"]
            starts = [index for index, line in enumerate(lines) if A["BASE"]["START"].search(line)]
            ends = [index for index, line in enumerate(lines) if A["BASE"]["END"].search(line)]
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(ends), 1)
            self.assertLess(starts[0], ends[0])
            legacy = [index for index, line in enumerate(lines)
                      if A["CONTEXT_ANALYZER"]["MARKER"] in line]
            self.assertTrue(all(starts[0] < index < ends[0] for index in legacy))
            measured += request["phase"] == "measured"
        self.assertEqual(measured, 6)

    def test_analyze_only_writes_its_new_json(self):
        context_result = {
            "status": "analyzed", "errors": [], "requests": [{}] * 24,
            "sycl_diagnostic": {"graphs": [{
                "cell": "VSPF-W3a", "point": "short", "phase": "measured", "ordinal": 0,
                "graphs": [],
            }]},
        }
        selected = [{
            "cell": "VSPF-W3a", "point": "short", "phase": "measured", "ordinal": 0,
            "request_index": 25, "first_line": 10, "last_line": 30, "text": valid_request(),
        }]
        globals_dict = A["analyze"].__globals__
        old_context = globals_dict["_run_context_validation"]
        old_selected = globals_dict["_selected_requests"]
        globals_dict["_run_context_validation"] = lambda out, c: context_result
        globals_dict["_selected_requests"] = lambda out, c: (selected, {
            "startup": {"device": 0, "components": 0},
            "public": {"device": 0, "components": 0},
            "warmup": {"device": 0, "components": 0},
            "baseline": {"device": 0, "components": 0},
        })
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                (out / "analysis-r1").mkdir()
                (out / "analysis-r1/analysis.json").write_text("BASE-SENTINEL\n")
                (out / "analysis-r1/profile.json").write_text("PROFILE-SENTINEL\n")
                c = {}
                result = A["analyze"](out, c)
                self.assertEqual(result["status"], "analyzed", result["errors"])
                self.assertEqual((out / "analysis-r1/analysis.json").read_text(), "BASE-SENTINEL\n")
                self.assertEqual((out / "analysis-r1/profile.json").read_text(), "PROFILE-SENTINEL\n")
                output = out / "analysis-r1/device-analysis.json"
                self.assertTrue(output.is_file())
                saved = json.loads(output.read_text())
                self.assertFalse(saved["gpu_time_claim"])
                self.assertEqual(saved["coverage"]["observed_q8_widths"], [4, 8])
        finally:
            globals_dict["_run_context_validation"] = old_context
            globals_dict["_selected_requests"] = old_selected


if __name__ == "__main__":
    unittest.main(verbosity=2)
