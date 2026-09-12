"""Development qualification for the native DSpark noncausal Split-K wrapper.

This probe is intentionally GPU-only when executed in the pinned image.  It
never substitutes a fallback and writes an incremental JSON record on errors.
"""
import argparse
import functools
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import traceback

PAGE = 1664
RTOL, ATOL = .02, .002
SEED = 20260912
LENGTHS = (1, 7, 1663, 1664, 1665, 8192, 65562)


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")


def compare(actual, expected):
    import torch
    a, b = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    result = {"finite": finite, "rtol": RTOL, "atol": ATOL,
              "max_abs": float((a-b).abs().max()) if finite else None,
              "rmse": float((a-b).square().mean().sqrt()) if finite else None}
    try:
        assert finite, "nonfinite output/reference"
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
        result["passed"] = True
    except Exception as exc:
        result.update(passed=False, error=str(exc))
    return result


def reference(q, k, v, used, table, scale=128 ** -.5):
    """CPU FP32 full attention; retain the independent KV-group axis."""
    import torch
    n = max(0, min(int(used), table.shape[1] * PAGE))
    if not n:
        return torch.zeros_like(q, dtype=torch.float32, device="cpu")
    pos = torch.arange(n)
    ids = table.detach().cpu()[0, pos // PAGE].long()
    kk = k.detach().cpu().float()[ids, pos % PAGE]
    vv = v.detach().cpu().float()[ids, pos % PAGE]
    qq = q.detach().cpu().float()
    result = torch.empty_like(qq)
    groups = qq.shape[1] // kk.shape[1]
    for g in range(kk.shape[1]):
        sl = slice(g * groups, (g + 1) * groups)
        scores = torch.einsum("bhd,td->bht", qq[:, sl], kk[:, g]) * scale
        probs = torch.softmax(scores, dim=-1)
        result[:, sl] = torch.einsum("bht,td->bhd", probs, vv[:, g])
    return result


def make_case(torch, length, layout="hnd", pitched=False, fixture=False):
    torch.manual_seed(SEED + length + (31 if layout == "combined" else 0))
    pages = math.ceil(max(1, length) / PAGE) + 2
    if layout == "combined":
        # Real combined HND: [pages, KV, PAGE, 2D], then transpose to NHD.
        both = torch.randn((pages, 8, PAGE, 256), device="xpu", dtype=torch.bfloat16)
        k = both[..., :128].transpose(1, 2)
        v = both[..., 128:].transpose(1, 2)
    else:
        k = torch.randn((pages, PAGE, 8, 128), device="xpu", dtype=torch.bfloat16)
        v = torch.randn((pages, PAGE, 8, 128), device="xpu", dtype=torch.bfloat16)
    q0 = torch.randn((7, 32, 136), device="xpu", dtype=torch.bfloat16)
    q = q0[..., :128] if pitched else q0
    table = torch.randperm(pages, device="xpu", dtype=torch.int32)[:math.ceil(max(1, length) / PAGE)].unsqueeze(0)
    used = torch.tensor([length], device="xpu", dtype=torch.int32)
    cu = torch.tensor([0, 7], device="xpu", dtype=torch.int32)
    # Poison logical padding in the final physical page.  A valid final-key
    # fixture makes a causal length decrement observable in every row.
    if length and length % PAGE:
        page = int(table[0, (length - 1) // PAGE])
        start = (length - 1) % PAGE + 1
        k[page, start:, :, :].fill_(100)
        v[page, start:, :, :].fill_(100)
    if fixture:
        page, off = int(table[0, (length - 1) // PAGE]), (length - 1) % PAGE
        k[page, off].zero_(); v[page, off].fill_(17)
    return [q, k, v, torch.empty_like(q), cu, used, table]


def invoke(fn, c, out=True, **overrides):
    q, k, v, o, cu, used, table = c
    args = dict(q=q, k=k, v=v, out=o if out else None,
        cu_seqlens_q=cu, seqused_k=used, max_seqlen_q=7,
        max_seqlen_k=table.shape[1] * PAGE, block_table=table,
        softmax_scale=128 ** -.5, causal=False, window_size=(-1, -1),
        is_mix_batch=False)
    args.update(overrides)
    return fn(**args)


def guard_checks(torch, fa, native, prototype):
    calls = []
    sentinel = object()
    @functools.wraps(native)
    def spy(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return sentinel
    wrapped = prototype.make_wrapper(spy)
    c = make_case(torch, 7)
    # These values are actual public overrides, not ignored test-only flags.
    for overrides in ({"causal": True}, {"window_size": (0, 0)},
                      {"dropout_p": .1}, {"deterministic": True},
                      {"is_mix_batch": True}):
        assert invoke(wrapped, c, **overrides) is sentinel
    assert wrapped.dispatches == 0
    assert len(calls) == 5
    return {"dispatches": wrapped.dispatches, "fallthrough_calls": len(calls),
            "unsupported_overrides": [sorted(x[1]) for x in calls]}


def graph_check(torch, fa, native, candidate):
    c = make_case(torch, 65562, fixture=True)
    def run(fn, x): return invoke(fn, x)
    graphs, results = {}, {}
    for name, fn in (("native", native), ("candidate", candidate)):
        for _ in range(3): run(fn, c)
        torch.xpu.synchronize(); graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph): run(fn, c)
        rows = []
        for i, used in enumerate((65562, 8192, 1665)):
            c[0].add_(torch.randn_like(c[0]) * .01)
            c[1].add_(torch.randn_like(c[1]) * .01)
            c[2].add_(torch.randn_like(c[2]) * .01)
            c[5].fill_(used); c[6].copy_(torch.roll(c[6], 1, 1))
            graph.replay(); torch.xpu.synchronize()
            expected = reference(*c[:3], used, c[6])
            actual = c[3].clone()
            rows.append({"used": used, "table": c[6].cpu().tolist(), "comparison": compare(actual, expected)})
            assert rows[-1]["comparison"]["passed"], f"{name} graph mutation {i} failed"
        graphs[name] = {"captured": True, "rows": rows}
    results["graphs"] = graphs
    assert candidate.dispatches > 0, "candidate graph captured no eligible dispatch"
    results["candidate_dispatches"] = candidate.dispatches
    return results


def timings(torch, native, candidate):
    c = make_case(torch, 65562, fixture=True)
    ng, cg = torch.xpu.XPUGraph(), torch.xpu.XPUGraph()
    for _ in range(3): invoke(native, c); invoke(candidate, c)
    torch.xpu.synchronize()
    with torch.xpu.graph(ng): invoke(native, c)
    with torch.xpu.graph(cg): invoke(candidate, c)
    torch.xpu.synchronize(); samples = []
    for i in range(12):
        order = ("native", "candidate") if i % 2 == 0 else ("candidate", "native")
        row = {"order": order}
        for name, graph in ((order[0], ng if order[0] == "native" else cg),
                            (order[1], ng if order[1] == "native" else cg)):
            start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(16): graph.replay()
            end.record(); end.synchronize()
            total = float(start.elapsed_time(end))
            row[name + "_total_ms"] = total
            row[name + "_ms_per_call"] = total / 16
        samples.append(row)
    nm = [x["native_ms_per_call"] for x in samples]; cm = [x["candidate_ms_per_call"] for x in samples]
    def summary(xs):
        q = sorted(xs)
        return {"samples_ms_per_call": xs, "median_ms_per_call": statistics.median(xs), "iqr_ms_per_call": q[9] - q[3]}
    return {"warm": 3, "replays": 16, "intervals": 12, "compile_included": False,
            "samples": samples, "native": summary(nm), "candidate": summary(cm),
            "speedup": statistics.median(nm) / statistics.median(cm)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args(); path = ns.out / "noncausal-split-k.json"
    report = {"status": "running", "tier": "development operator qualification; not publication compliant",
        "command": sys.orig_argv, "baseline": "unmodified native public flash_attn_varlen_func",
        "candidate": "B70_DSPARK_NONCAUSAL_SPLIT_K native Q=1 row expansion",
        "geometry": {"C": 1, "Q": 7, "H": 32, "KV": 8, "D": 128, "dtype": "bfloat16", "page": PAGE},
        "tolerance": {"rtol": RTOL, "atol": ATOL, "silently_relaxed": False}, "checks": [],
        "test_process": {"boundary": "fa.flash_attn_varlen_func", "metadata": "cu_q=[0,7], used=[L], table=[1,N]",
          "reference": "independent CPU FP32 full noncausal grouped attention", "graph": "Q/K/V/used/table mutation",
          "timing": "64K graphs warm3, 12 interleaved intervals x16 replays", "evidence": str(path)}}
    save(path, report); candidate = None
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vllm_xpu_kernels import flash_attn_interface as fa
        import qwen38_noncausal_split_k as prototype
        native = fa.flash_attn_varlen_func
        os.environ["B70_DSPARK_NONCAUSAL_SPLIT_K"] = "1"
        candidate = prototype.make_wrapper(native)
        fa.flash_attn_varlen_func = candidate
        report["environment"] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "native": fa.__file__, "prototype": prototype.__file__}
        for pkg in ("vllm", "vllm-xpu-kernels"):
            try: report["environment"][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError: pass
        assert torch.xpu.is_available(), "XPU unavailable"
        report["guards"] = guard_checks(torch, fa, native, prototype)
        for length in LENGTHS:
            for layout, pitched in (("hnd", False), ("combined", True)):
                c = make_case(torch, length, layout, pitched, fixture=True)
                ref = reference(*c[:3], length, c[6])
                before = candidate.dispatches
                c[3].fill_(float("nan")); baseline = invoke(native, c).clone()
                assert candidate.dispatches == before, "baseline unexpectedly dispatched candidate"
                c[3].fill_(float("nan")); actual_result = invoke(candidate, c)
                assert actual_result is c[3], "caller output identity was not preserved"
                actual = actual_result.clone()
                assert candidate.dispatches == before + 1, "eligible public call did not increment dispatches"
                allocated = invoke(candidate, c, out=False).clone()
                row = {"length": length, "layout": layout, "pitched_q": pitched,
                    "baseline_vs_reference": compare(baseline, ref), "candidate_vs_reference": compare(actual, ref),
                    "candidate_vs_native": compare(actual, baseline), "allocated_out": compare(allocated, ref),
                    "candidate_dispatches": candidate.dispatches}
                report["checks"].append(row); save(path, report)
                assert all(row[x]["passed"] for x in ("baseline_vs_reference", "candidate_vs_reference", "candidate_vs_native", "allocated_out")), f"numeric gate failed at {length}/{layout}"
        report["zero_length"] = {}
        z = make_case(torch, 0)
        z[3].fill_(3); baseline_result = invoke(native, z)
        assert baseline_result is z[3], "baseline zero-length out identity failed"
        baseline_zero = baseline_result.clone()
        z[3].fill_(3); candidate_result = invoke(candidate, z)
        assert candidate_result is z[3], "candidate zero-length out identity failed"
        candidate_zero = candidate_result.clone()
        report["zero_length"] = {"baseline": compare(baseline_zero, baseline_zero), "candidate": compare(candidate_zero, baseline_zero), "finite": bool(torch.isfinite(candidate_zero).all())}
        assert report["zero_length"]["candidate"]["passed"] and report["zero_length"]["finite"]
        report.update(graph_check(torch, fa, native, candidate))
        report["timing"] = timings(torch, native, candidate)
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
    finally:
        report["candidate_dispatches"] = getattr(candidate, "dispatches", 0)
        save(path, report)
    print(f"{report['status']}: {path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
