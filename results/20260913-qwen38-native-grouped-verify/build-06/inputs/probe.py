"""Lead-run in the pinned image on inference-host. No serving installation."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import platform
import statistics
import sys
import traceback

import torch
from vllm_xpu_kernels import flash_attn_interface as fa

PAGE = 1664
LENGTHS = (0, 1, 2, 3, 4, 5, 6, 63, 64, 65, 1663, 1664, 1665, 1668,
           1984, 1985, 2048, 2049, 2053, 8197, 32773, 65541)
TIMED = (8197, 32773, 65541)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reject(*args, **kwargs):
    raise RuntimeError("Unexpected fallback in eligible operator qualification")


class Routes:
    def __init__(self, candidate, output):
        self.candidate = candidate
        self.original = fa._spec_decode_varlen_fwd
        source = inspect.getsource(self.original)
        (output / "installed-spec-helper.py").write_text(source)
        marker = "None,  # num_splits (let the kernel pick via get_num_splits)"
        assert source.count(marker) == 1, "installed native helper drift"
        scope = dict(fa.__dict__)
        exec(compile(source.replace(marker, "32,  # fixed native control"),
                     "<native-stable32>", "exec"), scope)
        self.native32 = scope["_spec_decode_varlen_fwd"]
        self.fallback = fa._fallback_varlen_attn
        fa._fallback_varlen_attn = reject
        self.eligible_calls = 0

    def packed(self, *args, **kwargs):
        assert self.candidate.unsupported_reason(*args, **kwargs) is None
        self.eligible_calls += 1
        return self.candidate.dispatch(reject, *args, **kwargs)

    def forward(self, c, route="candidate"):
        # Intercept the REAL public API immediately before native helper expansion.
        # The baseline still executes the installed _vllm_fa2_C operator.
        helper = {"candidate": self.packed, "native": self.native32,
                  "auto": self.original}[route]
        previous = fa._spec_decode_varlen_fwd
        fa._spec_decode_varlen_fwd = helper
        try:
            y = fa.flash_attn_varlen_func(
                q=c.q, k=c.k, v=c.v, out=c.out, cu_seqlens_q=c.cu,
                seqused_k=c.used, max_seqlen_q=5, max_seqlen_k=c.maximum,
                block_table=c.table, softmax_scale=0.0625, causal=True,
                k_descale=c.ks, v_descale=c.vs)
            assert c.out is None or y.data_ptr() == c.out.data_ptr()
            return y
        finally:
            fa._spec_decode_varlen_fwd = previous

    def close(self):
        fa._spec_decode_varlen_fwd = self.original
        fa._fallback_varlen_attn = self.fallback


def capture(c, routes, route):
    for _ in range(3):
        routes.forward(c, route)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        y = routes.forward(c, route)
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    return graph, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    assert out.is_dir()
    assert torch.__version__ == "2.13.0+xpu"
    torch.set_num_threads(12)
    torch.manual_seed(42)
    candidate = load("b70_grouped_verify_seam", Path(__file__).with_name("grouped_verify.py"))
    # Reuse only Case, independent CPU FP32 reference and comparison helpers.
    # Importing this module does NOT run/import its old Triton implementation.
    checks = load("grouped_checks", Path(__file__).with_name("check-grouped-split-k.py"))
    checks.torch = torch
    torch.ops.load_library(str(args.library))
    result = {"tier": "development", "status": "running", "cases": [], "timings": [],
              "tolerance": {"rtol": 0.02, "atol": 1e-4},
              "fp32_policy": "strict diagnostics retained; separate max(1e-4,eps_fp16*max|descaledV|) allowance",
              "seed": 42, "splits": 32, "rounds": 12, "replays_per_batch": 16,
              "timing_scope": "public API graphs INCLUDING pack/unpack/device metadata",
              "torch": torch.__version__, "kernels": importlib.metadata.version("vllm-xpu-kernels"),
              "platform": platform.platform(), "device": str(torch.xpu.get_device_properties(0)),
              "command": sys.argv, "schema": str(torch.ops.b70_grouped_verify.forward.default._schema)}
    package = Path(fa.__file__).parent
    result["hashes"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        args.library, Path(fa.__file__), package / "_vllm_fa2_C.abi3.so",
        package / "libattn_kernels_xe_2.so")}

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    routes = None
    try:
        save()
        routes = Routes(candidate, out)
        # Observed serving layout: K/V share an interleaved last dimension.
        # Retain older HND qualification in operator-01/02; qualify this class anew.
        kv = torch.randn((152, PAGE, 4, 512), dtype=torch.float16, device="xpu").mul_(0.5).to(torch.float8_e4m3fn)
        k, v = kv[..., :256], kv[..., 256:]

        def case(length, *, head_pitch=True, nonunit=True):
            q = (torch.randn((5, 24, 512), dtype=torch.float16, device="xpu")[..., :256]
                 if head_pitch else
                 torch.randn((5, 48, 256), dtype=torch.float16, device="xpu")[:, :24])
            return checks.Case(
                q, k, v, torch.empty((5, 24, 256), dtype=torch.float16, device="xpu"),
                torch.tensor([0, 5], dtype=torch.int32, device="xpu"),
                torch.tensor([length], dtype=torch.int32, device="xpu"),
                torch.randperm(152, device="xpu")[:128].to(torch.int32)[None, :],
                torch.tensor(0.75 if nonunit else 1.0, device="xpu").expand(1, 4),
                torch.tensor(1.25 if nonunit else 1.0, device="xpu").expand(1, 4),
                212992)

        def reference(c):
            # Native uniform helper uses cu.numel(), not its values. Its padded
            # cu=[0,0], used=0 semantics are first-key attention, NOT zero output.
            canonical = replace(c, cu=torch.tensor([0, 5], dtype=torch.int32))
            return checks.reference(canonical)

        def comparisons(c, actual=None):
            ref = reference(c)
            if c.out is not None:
                c.out.fill_(float("nan"))
            native = routes.forward(c, "native").clone()
            if actual is None and c.out is not None:
                c.out.fill_(float("nan"))
            y = routes.forward(c).clone() if actual is None else actual
            return {"candidate_vs_native": checks.compare(y, native),
                    "native_fp32_supplemental": checks.compare_fp32(native, ref, c),
                    "candidate_fp32_supplemental": checks.compare_fp32(y, ref, c)}

        def test(name, fn):
            entry = {"name": name}
            result["cases"].append(entry)
            try:
                fn(entry)
                entry["status"] = "passed"
            except Exception:
                entry.update(status="failed", error=traceback.format_exc())
                raise
            finally:
                save()
                print(json.dumps(entry), flush=True)

        def eager(entry, c):
            entry["inputs"] = c.metadata()
            if c.out is not None:
                c.out.fill_(float("nan"))
            checks.assert_comparisons(entry, comparisons(c))

        for length in LENGTHS:
            test(f"eager/{length}", lambda e, n=length: eager(e, case(n)))
        test("token_pitch_unit_scales", lambda e: eager(e, case(1665, head_pitch=False, nonunit=False)))
        contiguous = case(65541)
        test("serving_contiguous_q", lambda e: eager(e, replace(contiguous, q=contiguous.q.contiguous())))
        test("allocated_output", lambda e: eager(e, replace(case(5), out=None)))

        def future(entry):
            c = case(1985)  # Last 64-key tile is entirely masked for rows t<4.
            last = 1984
            page = int(c.table[0, last // PAGE].item())
            old_k, old_v = k[page, last % PAGE].clone(), v[page, last % PAGE].clone()
            before = {r: routes.forward(c, r).clone() for r in ("native", "candidate")}
            ref_before = reference(c)
            try:
                k[page, last % PAGE].copy_(torch.full((4, 256), 4.0, device="xpu").to(k.dtype))
                v[page, last % PAGE].copy_(torch.full((4, 256), 8.0, device="xpu").to(v.dtype))
                ref_after = reference(c)
                checks_out = comparisons(c)
                for r in before:
                    after = routes.forward(c, r).clone()
                    checks_out[r + "_future_invariance"] = checks.compare(after[:4], before[r][:4], rtol=0, atol=0)
                entry["visible_control_change"] = (ref_before[4] - ref_after[4]).abs().max().item()
                assert entry["visible_control_change"] > 0
                checks.assert_comparisons(entry, checks_out)
            finally:
                k[page, last % PAGE].copy_(old_k)
                v[page, last % PAGE].copy_(old_v)
        test("future_key_and_empty_split", future)

        def mutation(entry):
            c = case(65541)
            graphs = {r: capture(c, routes, r) for r in ("native", "candidate")}
            records = entry["replays"] = []
            for length in (32773, 1665, 2049, 4, 0, 65541):
                c.q.mul_(-0.5).add_(0.125)
                c.used.fill_(length)
                c.table.copy_(c.table.flip(1))
                c.cu.copy_(torch.tensor([0, 0] if length == 0 else [0, 5], dtype=torch.int32, device="xpu"))
                state = {"inputs": c.metadata()}
                records.append(state)
                ys = {}
                for route, (graph, y) in graphs.items():
                    c.out.fill_(float("nan"))
                    graph.replay()
                    torch.xpu.synchronize()
                    ys[route] = y.clone()
                cmps = comparisons(c, ys["candidate"])
                cmps["graph_candidate_vs_graph_native"] = checks.compare(ys["candidate"], ys["native"])
                checks.assert_comparisons(state, cmps)
        test("graphs_mutate_q_used_pages_and_dummy_cu", mutation)

        def dummy(entry):
            c = case(0)
            c.cu.zero_()
            c.table.zero_()  # Safe, valid page zero; never run unsafe native page IDs.
            c.q.zero_()
            eager(entry, c)
        test("dummy_zero_metadata", dummy)

        def seam(entry):
            c = case(5)
            called = []
            def fallback(*args, **kwargs):
                called.append(True)
                return "unsupported"
            for bad in (replace(c, q=c.q[:4]), replace(c, table=c.table.repeat(5, 1)),
                        replace(c, ks=torch.ones((1, 4), device="xpu")),
                        replace(c, out=c.q), replace(c, maximum=212993)):
                assert candidate.unsupported_reason(*bad.helper_args()) is not None
                assert candidate.dispatch(fallback, *bad.helper_args()) == "unsupported"
            original = candidate.packed_forward
            def broken(*args, **kwargs):
                raise RuntimeError("injected eligible failure")
            candidate.packed_forward = broken
            try:
                try:
                    candidate.dispatch(fallback, *c.helper_args())
                except RuntimeError as exc:
                    assert str(exc) == "injected eligible failure"
                else:
                    raise AssertionError("eligible failure swallowed")
            finally:
                candidate.packed_forward = original
            assert len(called) == 5
            entry["unsupported_fallbacks"] = len(called)
            entry["eligible_failure_propagated"] = True
        test("pre_expansion_dispatch_contract", seam)

        result["summary"] = {}
        for length in TIMED:
            c = case(length)
            native = routes.forward(c, "native").clone()
            auto = routes.forward(c, "auto").clone()
            test(f"auto_vs_stable32/{length}", lambda e: checks.assert_comparisons(
                e, {"auto_vs_stable32": checks.compare(native, auto)}))
            graphs = {r: capture(c, routes, r) for r in ("native", "candidate")}
            graph_outputs = {}
            for route, (graph, y) in graphs.items():
                c.out.fill_(float("nan"))
                graph.replay()
                torch.xpu.synchronize()
                graph_outputs[route] = y.clone()
            test(f"timing_graph_outputs/{length}", lambda e: checks.assert_comparisons(
                e, {r: checks.compare(y, native) for r, y in graph_outputs.items()}))
            # Record a real dispatch trace once, outside the timing window.
            if length == 65541:
                for route in graphs:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.XPU], record_shapes=True) as prof:
                        routes.forward(c, route)
                        torch.xpu.synchronize()
                    prof.export_chrome_trace(str(out / f"dispatch-{route}.json"))
            for round_id in range(12):
                order = ("native", "candidate") if round_id % 2 == 0 else ("candidate", "native")
                for route in order:
                    graph, _ = graphs[route]
                    start, end = (torch.xpu.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(16):
                        graph.replay()
                    end.record()
                    torch.xpu.synchronize()
                    result["timings"].append({"length": length, "round": round_id, "order": order,
                        "route": route, "replays": 16, "ms_per_replay": start.elapsed_time(end) / 16})
                    save()
            medians = {}
            summary = {}
            for route in graphs:
                times = [s["ms_per_replay"] for s in result["timings"]
                         if s["length"] == length and s["route"] == route]
                q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
                medians[route] = statistics.median(times)
                summary[route] = {"median_ms": medians[route], "iqr_ms": q3 - q1}
            ratio = medians["candidate"] / medians["native"]
            summary.update(candidate_over_native=ratio, reduction_percent=100 * (1 - ratio),
                           operator_gate_passed=ratio <= (1.05 if length == 8197 else 0.95))
            result["summary"][str(length)] = summary
            save()
        result["operator_qualified"] = all(s["operator_gate_passed"] for s in result["summary"].values())
        result["status"] = "qualified" if result["operator_qualified"] else "correct_but_no_operator_win"
        result["serving_tested"] = False
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if routes is not None:
            result["eligible_eager_or_capture_calls"] = routes.eligible_calls
            routes.close()
        save()
    return 0 if result["operator_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
