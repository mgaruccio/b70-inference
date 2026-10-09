#!/usr/bin/env python3
"""Development-only synthetic XPU W4A16 M5-vs-padding operator probe.
This tests the exact FP16 symmetric GPTQ-G128 route used by XPUwNa16LinearKernel;
it is synthetic operator-only work, not a serving speedup or MTP result.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, shlex, statistics, sys, time, traceback
from pathlib import Path
GROUP = 128
ROUTES = ("m5", "m8", "m16")
SHAPES = (("gate_up", 5120, 34816), ("down", 17408, 5120),
          ("gdn_in", 5120, 16384), ("output", 6144, 5120), ("qkv", 5120, 14336))
SOURCE = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/mixed_precision/xpu.py"
def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
def _meta(t):
    return {"shape": [int(x) for x in t.shape], "stride": [int(x) for x in t.stride()],
            "dtype": str(t.dtype), "device": str(t.device), "numel": int(t.numel()),
            "nbytes": int(t.numel() * t.element_size())}
def _runtime(torch):
    available = bool(torch.xpu.is_available())
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "vllm": _version("vllm"), "vllm_xpu_kernels": _version("vllm-xpu-kernels"),
           "triton_xpu": _version("triton-xpu"), "xpu_available": available,
           "ONEDNN_VERBOSE": os.environ.get("ONEDNN_VERBOSE", "0")}
    if available:
        env["device_name"] = torch.xpu.get_device_name(0)
        props = torch.xpu.get_device_properties(0)
        env["device_properties"] = {k: getattr(props, k) for k in
            ("name", "total_memory", "driver_version", "major", "minor", "gpu_slices", "gpu_subslices_per_slice", "max_compute_units")
            if isinstance(getattr(props, k, None), (str, int, float, bool))}
    return env
def _register(torch):
    present = lambda: hasattr(torch.ops._xpu_C, "int4_gemm_w4a16")
    errors = []
    for module in ("vllm._xpu_ops", "vllm._C"):
        if present():
            break
        try:
            __import__(module)
        except Exception as exc:
            errors.append(f"{module}: {type(exc).__name__}: {exc}")
    if not present():
        raise RuntimeError("torch.ops._xpu_C.int4_gemm_w4a16 unavailable; registration=" + "; ".join(errors))
    return torch.ops._xpu_C.int4_gemm_w4a16
def _gemm(op, x, q, scales, zero):
    # No bias or g_idx: this is the XPUwNa16LinearKernel call shape.
    return op(x, q, None, scales, zero, GROUP, None)
def _check(torch, label, y, native):
    finite = bool(torch.isfinite(y).all().item() and torch.isfinite(native).all().item())
    return {"label": label, "finite": finite, "torch_equal": bool(torch.equal(y, native)),
            "max_abs_diff": float((y.float() - native.float()).abs().max().item()) if finite else None}
def _case(torch, op, name, k, n, seed, args, mismatches):
    torch.manual_seed(seed); torch.xpu.manual_seed_all(seed)
    device = torch.device("xpu")
    x = torch.randn((5, k), device=device, dtype=torch.float16).mul_(0.125).contiguous()
    storage = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), device=device, dtype=torch.int32)
    q = storage.t()  # logical [K/8,N], physical storage [N,K/8], stride (1,K/8)
    scales = torch.empty((k // GROUP, n), device=device, dtype=torch.float16).uniform_(0.002, 0.02).contiguous()
    zero = torch.tensor([8], device=device, dtype=torch.int8)
    pads = {route: torch.empty((int(route[1:]), k), device=device, dtype=torch.float16) for route in ROUTES[1:]}
    for pad in pads.values():
        pad.zero_(); pad[:5].copy_(x)
    if q.shape != (k // 8, n) or q.stride() != (1, k // 8) or not scales.is_contiguous():
        raise RuntimeError(f"{name}: packed layout contract failed: q={q.shape}/{q.stride()} scales={scales.stride()}")
    if not all(bool(torch.equal(pad[:5], x)) for pad in pads.values()):
        raise RuntimeError(f"{name}: padding changed the first five FP16 rows")
    def invoke(route):
        if route == "m5":
            return _gemm(op, x, q, scales, zero)
        pad = pads[route]; pad.zero_(); pad[:5].copy_(x)
        return _gemm(op, pad, q, scales, zero)
    for _ in range(args.warmup):
        for route in ROUTES:
            invoke(route)  # Warm native M5/M8/M16 before capture/timing.
    torch.xpu.synchronize()
    native, direct = invoke("m5").clone(), {}
    for route in ROUTES[1:]:
        direct[route] = invoke(route)[:5].clone()
    torch.xpu.synchronize()
    native_finite = bool(torch.isfinite(native).all().item())
    correctness = {"m5": {"label": "native", "finite": native_finite,
                            "torch_equal": True, "max_abs_diff": 0.0}}
    if not native_finite:
        mismatches.append(f"{name}/m5: native output is non-finite")
    for route, value in direct.items():
        check = _check(torch, route + "_vs_native5", value, native)
        correctness[route] = check
        if not check["finite"] or not check["torch_equal"]:
            mismatches.append(f"{name}/{route}: {check}")
    tensors = {"x_m5": x, "x_m8": pads["m8"], "x_m16": pads["m16"],
               "qweight_storage": storage, "qweight": q, "scales": scales, "zero": zero}
    case = {"name": name, "seed": seed, "logical": {
        "matmul": {route: [int(route[1:]), k, n] for route in ROUTES},
        "qweight": [k // 8, n], "scales": [k // GROUP, n], "zero_point": [1], "zero_point_value": 8},
        "physical": {"qweight_storage": list(storage.shape), "qweight_view": list(q.shape),
                     "qweight_packing": "uint4 nibbles in int32 words; K/8 contiguous", "activation_dtype": "float16"},
        "allocations": {key: _meta(value) for key, value in tensors.items()},
        "correctness": correctness, "graphs": {}, "timing": {"routes": {}}}
    graphs, outputs, stream = {}, {}, torch.xpu.current_stream()
    for route in ROUTES:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.stream(stream):
            with torch.xpu.graph(graph):
                capture_stream = str(torch.xpu.current_stream())
                output = invoke(route)
        graphs[route], outputs[route] = graph, output  # Persistent graph outputs stay live.
        case["graphs"][route] = {"capture_stream": capture_stream, "output": _meta(output)}
    for _ in range(args.warmup):
        for route in ROUTES:
            graphs[route].replay()
    torch.xpu.synchronize()
    for route in ROUTES:
        check = _check(torch, "graph_" + route + "_vs_native5", outputs[route][:5], native)
        case["correctness"]["graph_" + route] = check
        if not check["finite"] or not check["torch_equal"]:
            mismatches.append(f"{name}/graph_{route}: {check}")
    events, round_wall, started = [], [], time.perf_counter()
    for round_index in range(args.rounds):
        order = ROUTES[round_index % 3:] + ROUTES[:round_index % 3]
        round_start = time.perf_counter()
        for route in order:
            start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graphs[route].replay()
            end.record(); events.append((route, start, end))
        torch.xpu.synchronize()  # Outside the per-replay loop.
        round_wall.append(time.perf_counter() - round_start)
    timing = case["timing"]
    timing.update({"rounds": args.rounds, "replays_per_event": args.replays,
        "orders": [list(ROUTES[r % 3:] + ROUTES[:r % 3]) for r in range(args.rounds)],
        "round_host_wall_seconds": round_wall, "total_host_wall_seconds": time.perf_counter() - started,
        "ONEDNN_VERBOSE": os.environ.get("ONEDNN_VERBOSE", "0")})
    for route in ROUTES:
        batches = [float(s.elapsed_time(e)) for r, s, e in events if r == route]
        samples = [value / args.replays for value in batches]
        quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
        timing["routes"][route] = {"event_batch_ms": batches, "per_replay_ms": samples,
            "median_ms": float(statistics.median(samples)), "iqr_ms": float(quartiles[2] - quartiles[0])}
    baseline = timing["routes"]["m5"]["median_ms"]
    for route in ROUTES:
        timing["routes"][route]["median_delta_vs_m5_ms"] = timing["routes"][route]["median_ms"] - baseline
    timing["interpretation"] = "Synthetic operator-only timings; deltas are not serving speedups."
    del graphs, outputs
    torch.xpu.synchronize()
    return case
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new JSON result path")
    parser.add_argument("--seed", type=int, default=20261009); parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5); parser.add_argument("--replays", type=int, default=25)
    args = parser.parse_args(argv)
    if not (0 <= args.warmup <= 20 and 1 <= args.rounds <= 20 and 1 <= args.replays <= 100):
        parser.error("bounded values required: warmup 0..20, rounds 1..20, replays 1..100")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        parser.error("output already exists; retain it and choose a new path")
    report = {"schema_version": 1, "status": "running", "tier": "development", "command": shlex.join(sys.argv),
        "source": {"path": SOURCE}, "contract": {
            "scope": "synthetic operator-only; not a serving gain or MTP result", "baseline": "native M5 int4_gemm_w4a16",
            "candidate": "same first five FP16 rows zero-padded to M8/M16", "intentional_difference": "only M; all operands otherwise identical",
            "precision": "FP16 activations/output, symmetric GPTQ G128; no alternate dtype, bias, or g_idx",
            "test_process": "direct finite/exact checks, separate XPU graph capture, rotated interleaved event timing"},
        "research": {"pinned_kernel": SOURCE, "official_refs": [
            "https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/csrc/xpu/onednn/int4_gemm_w4a16.h",
            "https://raw.githubusercontent.com/vllm-project/vllm-xpu-kernels/v0.1.12/tests/test_int4_gemm_onednn.py"],
            "registration": "vllm._xpu_ops or vllm._C"}, "cases": []}
    source_path = Path(SOURCE)
    report["source"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.is_file() else None
    report["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["configuration"] = {"seed": args.seed, "warmup": args.warmup, "rounds": args.rounds, "replays": args.replays}
    def save():
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    save(); mismatches = []
    try:
        import torch
        report["environment"] = _runtime(torch)
        if not report["environment"]["xpu_available"]:
            raise RuntimeError("an available PyTorch XPU device is required")
        op = _register(torch); report["environment"]["operator_schema"] = str(op.default._schema)
        for index, (name, k, n) in enumerate(SHAPES):
            case = _case(torch, op, name, k, n, args.seed + index * 1009, args, mismatches)
            report["cases"].append(case)
            print(f"{name}: " + ", ".join(f"{r}={case['timing']['routes'][r]['median_ms']:.6f}ms" for r in ROUTES), flush=True)
        report["mismatches"] = mismatches; report["status"] = "passed" if not mismatches else "mismatch"
        if mismatches:
            report["error"] = "correctness mismatch; see mismatches (not silently dropped)"
    except Exception as exc:
        report["status"] = "failed"; report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc(limit=8)
    save(); print(f"{report['status']}: {args.out}", file=sys.stderr, flush=True)
    return 0 if report["status"] == "passed" else 1
if __name__ == "__main__":
    raise SystemExit(main())
