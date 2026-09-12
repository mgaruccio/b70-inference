"""Native noncausal DSpark Split-K qualification (development only).

Runs through vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func,
with an opt-in qwen38_noncausal_split_k wrapper.  No fallback is accepted.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

PAGE = 1664
RTOL, ATOL = .02, .002
SEED = 20260912
LENGTHS = (1, 7, 1663, 1664, 1665, 8192, 65562)


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")


def _stats(actual, expected):
    import torch
    a, b = actual.detach().cpu().float(), expected.detach().cpu().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    d = a - b
    out = {"finite": finite, "rtol": RTOL, "atol": ATOL,
           "max_abs": float(d.abs().max()) if finite else None,
           "rmse": float(d.square().mean().sqrt()) if finite else None}
    try:
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
        out["passed"] = finite
    except AssertionError as e:
        out.update(passed=False, error=str(e))
    return out


def _reference(q, k, v, used, table, scale=.0625):
    """Independent CPU FP32 full (noncausal) paged GQA attention."""
    import torch
    n = max(0, min(int(used), table.shape[1] * PAGE))
    if n == 0:
        return torch.zeros_like(q, device="cpu", dtype=torch.float32)
    pos = torch.arange(n)
    ids = table.detach().cpu()[0, pos // PAGE].long()
    kp = k.detach().cpu().float()[ids, pos % PAGE]
    vp = v.detach().cpu().float()[ids, pos % PAGE]
    qr = q.detach().cpu().float()
    heads = torch.arange(qr.shape[1]) // (qr.shape[1] // kp.shape[1])
    kh = kp[:, heads]
    vh = vp[:, heads]
    score = torch.einsum("bhd,thd->bt", qr[...], kh) * scale
    return torch.einsum("bt,thd->bhd", torch.softmax(score, -1), vh)


def _case(torch, length, layout="hnd", pitched=False):
    torch.manual_seed(SEED + length + (13 if layout != "hnd" else 0))
    pages = math.ceil(max(1, length) / PAGE) + 2
    base = torch.randn((pages, PAGE, 8, 128), device="xpu", dtype=torch.bfloat16)
    if layout == "hnd":
        k, v = base, base.clone()
    else:  # combined HND storage with separate views and nontrivial storage offsets
        both = torch.randn((pages, PAGE, 16, 128), device="xpu", dtype=torch.bfloat16)
        k, v = both[..., :8, :], both[..., 8:, :]
    q0 = torch.randn((7, 32, 128 + (8 if pitched else 0)), device="xpu", dtype=torch.bfloat16)
    q = q0[..., :128] if pitched else q0
    out = torch.empty_like(q)
    table = torch.randperm(pages, device="xpu", dtype=torch.int32)[:math.ceil(max(1, length) / PAGE)].unsqueeze(0)
    used = torch.tensor([length], device="xpu", dtype=torch.int32)
    cu = torch.arange(8, device="xpu", dtype=torch.int32)
    return q, k, v, out, cu, used, table


def _call(fa, c, *, out=True, causal=False, dtype=None, batch=False, window=(-1, -1)):
    q, k, v, o, cu, used, table = c
    return fa.flash_attn_varlen_func(q=q, k=k, v=v, out=o if out else None,
        cu_seqlens_q=cu, seqused_k=used, max_seqlen_q=1,
        max_seqlen_k=table.shape[1] * PAGE, block_table=table,
        softmax_scale=.0625, causal=causal, window_size=window,
        is_mix_batch=batch)


def _validate_fallthrough(fa, wrapper):
    calls = []
    original = wrapper.original
    def spy(*a, **kw):
        calls.append(kw.copy())
        return original(*a, **kw)
    wrapper.original = spy
    torch = sys.modules["torch"]
    c = _case(torch, 7)
    for kw in ({"causal": True}, {"window": (0, 0)}, {"dtype": torch.float16}, {"batch": True}):
        try:
            _call(fa, c, **kw)
        except Exception:
            pass
    wrapper.original = original
    return {"fallthrough_calls": len(calls), "unsupported_options": [sorted(x) for x in calls]}


def _timing(torch, fa, c, route, repeats=16):
    events = []
    for _ in range(3):
        route(c)
    torch.xpu.synchronize()
    for i in range(12):
        s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        s.record();
        for _ in range(repeats): route(c)
        e.record(); e.synchronize()
        events.append(float(s.elapsed_time(e)))
    med = statistics.median(events)
    q = sorted(events)
    return {"samples_ms": events, "median_ms": med,
            "iqr_ms": q[9] - q[3], "replays": repeats, "warm": 3}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); path = args.out / "noncausal-split-k.json"
    report = {"status": "running", "tier": "development; not publication compliant",
      "command": sys.orig_argv, "baseline": "unmodified native flash_attn_varlen_func",
      "candidate": "opt-in native full noncausal row expansion Split-K",
      "intentional_configuration_differences": ["max_seqlen_q=1", "cu_q=arange(8)", "seven repeated full used lengths", "block table repeated seven times"],
      "geometry": {"C": 1, "Q": 7, "H": 32, "KV": 8, "D": 128, "dtype": "bfloat16", "page": PAGE},
      "tolerance": {"rtol": RTOL, "atol": ATOL, "silently_relaxed": False},
      "checks": [], "timing": None, "dispatches": 0,
      "test_process": {"boundary": "public fa.flash_attn_varlen_func", "reference": "CPU FP32 grouped full noncausal attention", "graph": "mutable Q/used/table/K/V replay", "timing": "64K paired graph, warm3, 12 interleaved samples x16 replays"}}
    _write(path, report); routes = None
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vllm_xpu_kernels import flash_attn_interface as fa
        import qwen38_noncausal_split_k as prototype
        original = fa.flash_attn_varlen_func
        os.environ["B70_DSPARK_NONCAUSAL_SPLIT_K"] = "1"
        routes = prototype.make_wrapper(original)
        fa.flash_attn_varlen_func = routes
        report["environment"] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "native": fa.__file__, "prototype": prototype.__file__}
        for n in ("vllm", "vllm-xpu-kernels"):
            try: report["environment"][n] = importlib.metadata.version(n)
            except importlib.metadata.PackageNotFoundError: pass
        assert torch.xpu.is_available(), "XPU unavailable"
        assert routes.eligible({}) is False
        report["guards"] = _validate_fallthrough(fa, routes)
        for length in LENGTHS:
            for layout, pitched in (("hnd", False), ("combined", True)):
                c = _case(torch, length, layout, pitched)
                ref = _reference(*c[:3], length, c[6])
                base_out = original(q=c[0], k=c[1], v=c[2], out=c[3], cu_seqlens_q=c[4], seqused_k=c[5], max_seqlen_q=1, max_seqlen_k=c[6].shape[1]*PAGE, block_table=c[6], softmax_scale=.0625, causal=False, window_size=(-1,-1), is_mix_batch=False).clone()
                c[3].fill_(float("nan")); cand = _call(fa, c, out=True).clone()
                allocated = _call(fa, (*c[:3], None, *c[4:]), out=True).clone()
                entry = {"length": length, "layout": layout, "pitched_q": pitched, "reference": _stats(base_out, ref), "candidate": _stats(cand, ref), "allocated": _stats(allocated, ref), "candidate_vs_native": _stats(cand, base_out), "dispatches": routes.dispatches}
                report["checks"].append(entry); _write(path, report)
                assert entry["candidate"]["passed"] and entry["allocated"]["passed"], f"candidate failed length {length}"
        report["dispatches"] = routes.dispatches
        # Explicit zero-length behavior is observed, never clamped by this harness.
        z = _case(torch, 0); z[3].fill_(3)
        try:
            native_zero = run_zero = original(q=z[0], k=z[1], v=z[2], out=z[3], cu_seqlens_q=z[4], seqused_k=z[5], max_seqlen_q=1, max_seqlen_k=z[6].shape[1]*PAGE, block_table=z[6], softmax_scale=.0625, causal=False, window_size=(-1,-1), is_mix_batch=False)
            candidate_zero = _call(fa, z)
            report["zero_length"] = {"native_finite": bool(torch.isfinite(native_zero).all()), "candidate_finite": bool(torch.isfinite(candidate_zero).all()), "same_shape": native_zero.shape == candidate_zero.shape}
        except Exception:
            report["zero_length"] = {"unsupported_or_nonfinite": traceback.format_exc()}
        # Graph replay is deliberately after every numeric gate; metadata and KV are mutated in place.
        c = _case(torch, 65562, "hnd", False)
        def run_native(x):
            q, k, v, o, cu, used, table = x
            return original(q=q, k=k, v=v, out=o, cu_seqlens_q=cu, seqused_k=used, max_seqlen_q=1, max_seqlen_k=table.shape[1]*PAGE, block_table=table, softmax_scale=.0625, causal=False, window_size=(-1,-1), is_mix_batch=False)
        def run_candidate(x): return _call(fa, x)
        graphs = {}
        for name, fn in (("native", run_native), ("candidate", run_candidate)):
            for _ in range(3): fn(c)
            torch.xpu.synchronize(); graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph): fn(c)
            c[0].add_(torch.randn_like(c[0]).mul_(.01)); c[1].add_(torch.randn_like(c[1]).mul_(.01)); c[2].add_(torch.randn_like(c[2]).mul_(.01))
            c[5].fill_(8192); c[6].copy_(torch.roll(c[6], 1, 1)); graph.replay(); torch.xpu.synchronize()
            replay = c[3].clone(); expected = _reference(*c[:3], 8192, c[6])
            graphs[name] = {"replay": _stats(replay, expected), "captured": True}
            assert graphs[name]["replay"]["passed"], f"{name} graph replay failed"
        report["graph"] = graphs
        # Paired graph timings: interleaved native/candidate, compile excluded.
        c = _case(torch, 65562, "hnd", False)
        native_graph = torch.xpu.XPUGraph(); candidate_graph = torch.xpu.XPUGraph()
        for _ in range(3): run_native(c); run_candidate(c)
        torch.xpu.synchronize()
        with torch.xpu.graph(native_graph): run_native(c)
        with torch.xpu.graph(candidate_graph): run_candidate(c)
        torch.xpu.synchronize(); samples = []
        for i in range(12):
            row = {"order": ["native", "candidate"] if i % 2 == 0 else ["candidate", "native"]}
            for name, graph in ((row["order"][0], native_graph if row["order"][0] == "native" else candidate_graph), (row["order"][1], native_graph if row["order"][1] == "native" else candidate_graph)):
                s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True); s.record()
                for _ in range(16): graph.replay()
                e.record(); e.synchronize(); row[name + "_ms"] = float(s.elapsed_time(e))
            samples.append(row)
        report["timing"] = {"samples": samples, "warm": 3, "replays": 16, "compile_included": False,
                             "native_median_ms": statistics.median([x["native_ms"] for x in samples]),
                             "candidate_median_ms": statistics.median([x["candidate_ms"] for x in samples])}
        report["timing"]["speedup"] = report["timing"]["native_median_ms"] / report["timing"]["candidate_median_ms"]
        report["status"] = "passed"
        report.update(status="failed", error=traceback.format_exc())
    finally:
        if routes is not None:
            report["dispatches"] = getattr(routes, "dispatches", 0)
        _write(path, report)
    print(f"{report['status']}: {path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
