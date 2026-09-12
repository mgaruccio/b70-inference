"""Development operator qualification, NOT serving/performance promotion.

Staged invocation (no install):
  /opt/venv/bin/python -P /experiment/check-grouped-split-k.py --out /output
Stage qwen38_grouped_split_k.py beside this script or on mounted PYTHONPATH.
The CLI explicitly enables the prototype in this disposable process only.

Preconditions: immutable native image0.1.12.3, torch 2.13 XPU, idle B70 at the
lead's recorded 275W setting. Baseline is the unmodified pre-expansion native
helper; the candidate changes only grouped-query KV reuse and Split-K choice.
Journey: real flash_attn_varlen_func -> installed helper -> Triton, compared
with native and independent CPU FP32 paged attention; then mutable-input XPU
graph replay and paired 64K graph timings (warm3, measured12/20, compile excluded).
JSON retains failures, fixed tolerances, environment and every event sample.
The lead separately owns HTTP/MTP4 qualification and container cleanup. These
operator results cannot satisfy the repository's publication checklist.
"""

import argparse
from dataclasses import dataclass, replace
import functools
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
import traceback

RTOL = 0.02
ATOL = 1e-4
PAGE = 1664
LIVE_64K = 65541
SEED = 20260912


@dataclass
class Case:
    q: object
    k: object
    v: object
    out: object
    cu: object
    used: object
    table: object
    ks: object
    vs: object
    maximum: int

    def helper_args(self):
        return (self.q, self.k, self.v, self.out, self.cu, self.used, self.table,
                self.maximum, self.ks, self.vs, 0.0625, None, (-1, -1), 0.0)

    def metadata(self):
        return {"q_shape": list(self.q.shape), "q_stride": list(self.q.stride()),
                "k_shape": list(self.k.shape), "k_stride": list(self.k.stride()),
                "v_stride": list(self.v.stride()), "v_offset": self.v.storage_offset(),
                "out_stride": list(self.out.stride()) if self.out is not None else None,
                "kv_dtype": str(self.k.dtype), "used_k": self.used.cpu().tolist(),
                "cu_q": self.cu.cpu().tolist(), "max_seqlen_k": self.maximum,
                "block_table": self.table.cpu().tolist(),
                "table_stride": list(self.table.stride()),
                "k_descale": self.ks.flatten()[0].item(),
                "v_descale": self.vs.flatten()[0].item(),
                "descale_shape": list(self.ks.shape),
                "descale_stride": list(self.ks.stride())}


def make_case(length, *, layout="hnd", nonunit=True, permuted=True):
    torch.manual_seed(SEED + length)
    pages_used = math.ceil(max(length, 5) / PAGE)
    pages = pages_used + 2  # Spare physical pages make permutations nontrivial.
    if layout == "hnd":
        storage = torch.randn((pages, 4, PAGE, 512), dtype=torch.float16,
                              device="xpu").mul_(0.5).to(torch.float8_e4m3fn)
        k = storage[..., :256].transpose(1, 2)
        v = storage[..., 256:].transpose(1, 2)
    else:
        k = torch.randn((pages, PAGE, 4, 256), dtype=torch.float16,
                        device="xpu").mul_(0.5).to(torch.float8_e4m3fn)
        v = torch.randn((pages, PAGE, 4, 256), dtype=torch.float16,
                        device="xpu").mul_(0.5).to(torch.float8_e4m3fn)
    # Pitched Q/out additionally test the strides used by the two kernels.
    q = torch.randn((5, 24, 512), dtype=torch.float16, device="xpu")[..., :256]
    out = torch.empty((5, 24, 512), dtype=torch.float16, device="xpu")[..., :256]
    ids = torch.randperm(pages)[:pages_used] if permuted else torch.arange(pages_used)
    table = ids.to(device="xpu", dtype=torch.int32).unsqueeze(0)
    ks = torch.tensor(0.75 if nonunit else 1.0, device="xpu").expand(1, 4)
    vs = torch.tensor(1.25 if nonunit else 1.0, device="xpu").expand(1, 4)
    return Case(q, k, v, out,
                torch.tensor([0, 5], dtype=torch.int32, device="xpu"),
                torch.tensor([length], dtype=torch.int32, device="xpu"),
                table, ks, vs, pages_used * PAGE)


def reference(case):
    """Independent CPU FP32 math, no SDPA, native or FA reference fallback.

    Invalid page IDs are omitted. Malformed cumulative Q lengths are a dummy
    zero-output request. Used lengths are safely capacity-clipped; valid and
    short lengths follow native max(used_k - 4 + j, 1) exactly.
    """
    if case.cu.cpu().tolist() != [0, 5]:
        return torch.zeros((5, 24, 256), dtype=torch.float32)
    used = int(case.used.item())
    capacity = min(case.maximum, case.table.shape[1] * PAGE)
    n = min(max(used, 1), capacity)
    pos = torch.arange(n)
    pages = case.table.cpu()[0, pos // PAGE].long()
    valid = (pages >= 0) & (pages < case.k.shape[0])
    safe_pages = pages.clamp(0, case.k.shape[0] - 1)
    # Conversion occurs on CPU, independently of the candidate's FP8 cast.
    k = case.k.detach().cpu().float()[safe_pages, pos % PAGE]
    v = case.v.detach().cpu().float()[safe_pages, pos % PAGE]
    k *= case.ks.flatten()[0].item()
    v *= case.vs.flatten()[0].item()
    q = case.q.detach().cpu().float()
    limits = (used - 4 + torch.arange(5, dtype=torch.int64)).clamp(1, capacity)
    mask = valid[None, :] & (pos[None, :] < limits.repeat_interleave(6)[:, None])
    has_keys = mask.any(dim=1)
    result = torch.empty((5, 24, 256), dtype=torch.float32)
    for head in range(4):
        qr = q[:, head * 6:(head + 1) * 6].reshape(30, 256)
        scores = (qr @ k[:, head].T) * 0.0625
        scores.masked_fill_(~mask, -float("inf"))
        scores = torch.where(has_keys[:, None], scores, 0.0)
        probs = torch.softmax(scores, dim=-1) * has_keys[:, None]
        result[:, head * 6:(head + 1) * 6] = (probs @ v[:, head]).reshape(5, 6, 256)
    return result


def compare(actual, expected, *, rtol=RTOL, atol=ATOL):
    a = actual.detach().cpu().float()
    b = expected.detach().cpu().float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    metrics = {"finite": finite, "rtol": rtol, "atol": atol,
               "max_abs": None, "rmse": None, "cosine": None}
    if finite:
        diff = a - b
        norms = a.norm().item() * b.norm().item()
        metrics.update(max_abs=diff.abs().max().item(),
                       rmse=diff.square().mean().sqrt().item(),
                       cosine=(torch.dot(a.flatten(), b.flatten()).item() / norms
                               if norms else float(torch.equal(a, b))))
    try:
        assert finite, "nonfinite actual/reference output"
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        metrics["passed"] = True
    except AssertionError as exc:
        metrics.update(passed=False, error=str(exc))
    return metrics


def assert_comparisons(entry, comparisons):
    entry["comparisons"] = comparisons
    assert all(m["passed"] for m in comparisons.values()), "fixed-tolerance comparison failed"


class Routes:
    """Harness-only counters prove the real public call takes the intended hook."""

    def __init__(self, fa, prototype):
        self.fa, self.prototype = fa, prototype
        self.original = fa._spec_decode_varlen_fwd
        self.original_launch = prototype._launch
        self.counts = {"native": 0, "candidate": 0}

        @functools.wraps(self.original)
        def native(*args, **kwargs):
            self.counts["native"] += 1
            return self.original(*args, **kwargs)

        @functools.wraps(self.original_launch)
        def launch(*args, **kwargs):
            self.counts["candidate"] += 1
            return self.original_launch(*args, **kwargs)

        self.native = native
        prototype._launch = launch

    def enable(self, splits, stages):
        self.fa._spec_decode_varlen_fwd = self.native
        os.environ.update(B70_GROUPED_SPLIT_K="1", B70_GROUPED_SPLIT_K_SPLITS=str(splits),
                          B70_GROUPED_SPLIT_K_STAGES=str(stages))
        assert self.prototype.install()
        self.candidate = self.fa._spec_decode_varlen_fwd
        assert self.candidate._b70_grouped_split_k == (splits, stages)

    def forward(self, case, route="candidate", *, expected=None, **flags):
        self.fa._spec_decode_varlen_fwd = self.native if route == "native" else self.candidate
        before = self.counts.copy()
        kwargs = dict(q=case.q, k=case.k, v=case.v, out=case.out,
                      cu_seqlens_q=case.cu, seqused_k=case.used,
                      max_seqlen_q=case.q.shape[0], max_seqlen_k=case.maximum,
                      block_table=case.table, softmax_scale=0.0625, causal=True,
                      k_descale=case.ks, v_descale=case.vs)
        kwargs.update(flags)
        try:
            y = self.fa.flash_attn_varlen_func(**kwargs)
        finally:
            self.fa._spec_decode_varlen_fwd = self.candidate
        target = expected or route
        assert {key: self.counts[key] - before[key] for key in before} == {
            "native": int(target == "native"), "candidate": int(target == "candidate")
        }, f"unexpected public dispatch: {before} -> {self.counts}"
        if case.out is not None:
            assert y is case.out, "caller out identity was not preserved"
        assert y.shape == case.q.shape and y.dtype == case.q.dtype
        return y

    def close(self):
        self.fa._spec_decode_varlen_fwd = self.original
        self.prototype._launch = self.original_launch


def correctness(entry, case, routes):
    entry["inputs"] = case.metadata()
    assert routes.prototype.unsupported_reason(*case.helper_args()) is None
    ref = reference(case)
    case.out.fill_(float("nan"))
    native = routes.forward(case, "native").clone()
    case.out.fill_(float("nan"))
    candidate = routes.forward(case).clone()
    comparisons = {"native_vs_fp32": compare(native, ref),
                   "candidate_vs_fp32": compare(candidate, ref),
                   "candidate_vs_native": compare(candidate, native)}
    # Verify allocation as well as preservation of a caller-supplied output.
    if case.used.item() == 5:
        allocated = routes.forward(replace(case, out=None))
        comparisons["allocated_vs_fp32"] = compare(allocated, ref)
    entry["out_identity"] = True
    assert_comparisons(entry, comparisons)


def causal_invariance(entry, case, routes):
    """Change the last key, future to rows j=0..3 but visible to row j=4."""
    entry["inputs"] = case.metadata()
    before_ref = reference(case)
    before = {name: routes.forward(case, name).clone() for name in ("native", "candidate")}
    last = int(case.used.item()) - 1
    page = int(case.table[0, last // PAGE].item())
    # Deliberately large, finite changes to both the future K and V.
    case.k[page, last % PAGE].copy_(torch.full((4, 256), 4.0, device="xpu").to(case.k.dtype))
    case.v[page, last % PAGE].copy_(torch.full((4, 256), 8.0, device="xpu").to(case.v.dtype))
    after_ref = reference(case)
    comparisons = {"reference_future_rows": compare(before_ref[:4], after_ref[:4], rtol=0, atol=0)}
    for name in ("native", "candidate"):
        y = routes.forward(case, name).clone()
        comparisons[name + "_future_rows"] = compare(y[:4], before[name][:4], rtol=0, atol=0)
        comparisons[name + "_vs_changed_reference"] = compare(y, after_ref)
    entry["last_row_reference_change"] = (after_ref[4] - before_ref[4]).abs().max().item()
    assert entry["last_row_reference_change"] > 0, "causal mutation had no visible-key control effect"
    assert_comparisons(entry, comparisons)


def capture(case, routes, route):
    for _ in range(3):
        routes.forward(case, route)
    torch.xpu.synchronize()  # Compile/initial allocation excluded from capture/timing.
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        y = routes.forward(case, route)
    for _ in range(3):
        graph.replay()
    torch.xpu.synchronize()
    return graph, y  # Keep graph, input and output owners alive through replay.


def graph_mutation(entry, case, routes, route):
    entry["capture_inputs"] = case.metadata()
    graph, y = capture(case, routes, route)
    comparisons = {"initial": compare(y, reference(case))}
    states = []
    for length in (1665, 5, LIVE_64K):
        case.q.mul_(-0.5).add_(0.125)
        case.used.fill_(length)
        case.table.copy_(case.table.flip(1))
        graph.replay()
        torch.xpu.synchronize()
        states.append(case.metadata())
        comparisons[f"changed_q_length_pages_{length}"] = compare(y, reference(case))
    if route == "candidate":
        case.cu.zero_()
        graph.replay()
        torch.xpu.synchronize()
        comparisons["dummy_cumulative_lengths"] = compare(y, reference(case), rtol=0, atol=0)
        case.cu.copy_(torch.tensor([0, 5], dtype=torch.int32, device="xpu"))
        graph.replay()
        torch.xpu.synchronize()
        comparisons["restored_cumulative_lengths"] = compare(y, reference(case))
    entry["replay_inputs"] = states
    entry["out_identity"] = y is case.out
    assert_comparisons(entry, comparisons)


def dummy_safety(entry, routes):
    # Native is intentionally NOT called with unsafe/malformed page metadata.
    # Normal 0/1/4 short lengths are separately compared against native.
    c = make_case(5)
    comparisons = {}
    for label, cu, used, page in (
            ("dummy_cu", [0, 0], 5, 0),
            ("invalid_cu", [1, 6], 5, 0),
            ("negative_page", [0, 5], 5, -1),
            ("past_storage_page", [0, 5], 5, c.k.shape[0]),
            ("int32_min_used", [0, 5], -(2**31), 0),
            ("int32_max_used", [0, 5], 2**31 - 1, 0)):
        c.cu.copy_(torch.tensor(cu, dtype=torch.int32, device="xpu"))
        c.used.fill_(used)
        c.table.fill_(page)
        c.out.fill_(float("nan"))
        y = routes.forward(c)
        comparisons[label] = compare(y, reference(c))
    # Entire invalid first page, valid second page: rows with zero keys must
    # stay zero while a later causal row can have a nonzero partial result.
    c = make_case(1665)
    c.table[0, 0] = -1
    y = routes.forward(c)
    comparisons["partially_masked_pages"] = compare(y, reference(c))
    entry["native_unsafe_metadata_executed"] = False
    assert_comparisons(entry, comparisons)


def fallback_checks(entry, routes):
    c = make_case(5)
    comparisons = {}
    # Real native execution through the public API for an unsupported Q shape
    # and a flag the native helper accepts. The candidate must not launch.
    q4 = replace(c, q=c.q[:4], out=c.out[:4],
                 cu=torch.tensor([0, 4], dtype=torch.int32, device="xpu"))
    for label, case, flags in (("q4", q4, {}), ("window", c, {"window_size": (8, 0)})):
        baseline = routes.forward(case, "native", **flags).clone()
        actual = routes.forward(case, expected="native", **flags).clone()
        comparisons[label] = compare(actual, baseline, rtol=0, atol=0)
    # Direct helper delegation contract for flags/shapes unsafe to execute in
    # native. A spy verifies *all original argument identities*, not GPU math.
    observed = []
    sentinel = object()

    def native_probe(*args):
        observed.append(args)
        return sentinel

    probe = routes.prototype._make_wrapper(native_probe, 16, 1)
    args = list(c.helper_args())
    bad_cases = (("sink", 11, c.q), ("softcap", 13, 1.0),
                 ("window", 12, (4, 0)), ("q_dtype", 0, c.q.float()),
                 ("non_scalar_descale", 8, torch.ones((1, 4), device="xpu")),
                 ("batch2", 4, torch.tensor([0, 2, 5], dtype=torch.int32, device="xpu")),
                 ("out_dtype", 3, c.out.float()))
    delegated = []
    for label, index, value in bad_cases:
        trial = args.copy()
        trial[index] = value
        assert probe(*trial) is sentinel
        assert all(a is b for a, b in zip(observed[-1], trial))
        delegated.append(label)
    # Eligible compilation/runtime failures MUST escape through the public API.
    launch = routes.prototype._launch
    native_before = routes.counts["native"]

    def fail_launch(*args, **kwargs):
        raise RuntimeError("qualification-injected-supported-launch-failure")

    routes.prototype._launch = fail_launch
    try:
        try:
            routes.forward(c)
        except RuntimeError as exc:
            assert str(exc) == "qualification-injected-supported-launch-failure"
        else:
            raise AssertionError("eligible launch failure was concealed")
    finally:
        routes.prototype._launch = launch
    assert routes.counts["native"] == native_before
    entry.update(helper_delegation_spy_checks=delegated, supported_failure_propagated=True)
    assert_comparisons(entry, comparisons)


def timings(entry, case, routes, repetitions):
    entry["inputs"] = case.metadata()
    graphs = {}
    outputs = {}
    # Separate graph pools and outputs, identical input tensors and warmup.
    owners = {name: replace(case, out=torch.empty_like(case.q)) for name in ("native", "candidate")}
    for name, owner in owners.items():
        graphs[name], outputs[name] = capture(owner, routes, name)
    ref = reference(case)
    comparisons = {name: compare(y, ref) for name, y in outputs.items()}
    assert_comparisons(entry, comparisons)
    events = []
    for i in range(repetitions):
        order = ("native", "candidate") if i % 2 == 0 else ("candidate", "native")
        pair = {name: (torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True))
                for name in order}
        events.append((order, pair))
    torch.xpu.synchronize()
    for order, pair in events:
        for name in order:
            start, end = pair[name]
            start.record()
            graphs[name].replay()
            end.record()
    torch.xpu.synchronize()  # Synchronize the entire measured batch before reading events.
    samples = []
    for order, pair in events:
        sample = {"order": list(order)}
        for name, (start, end) in pair.items():
            value = start.elapsed_time(end)
            # Preserve anomalous driver values without making strict JSON fail
            # while reporting the error (they still fail the timing check).
            sample[name + "_ms"] = value if math.isfinite(value) else str(value)
        samples.append(sample)
    entry.update(raw_samples=samples, warm_calls=3, warm_replays=3,
                 repetitions=repetitions, compile_included=False)
    for name in ("native", "candidate"):
        values = [sample[name + "_ms"] for sample in samples]
        assert all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in values), "invalid event timing"
        entry[name + "_median_ms"] = statistics.median(values)
    entry["native_over_candidate"] = entry["native_median_ms"] / entry["candidate_median_ms"]
    # Also check the values actually left by the measured replay batch.
    assert_comparisons(entry, {name: compare(y, ref) for name, y in outputs.items()})


def save(report, path):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def run_check(report, path, name, fn):
    entry = {"name": name, "status": "running"}
    report["checks"].append(entry)
    save(report, path)
    try:
        fn(entry)
        entry["status"] = "passed"
    except Exception:
        entry.update(status="failed", error=traceback.format_exc())
    save(report, path)
    print(f"{name}: {entry['status']}", flush=True)
    return entry["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, nargs="+", choices=(16, 32), default=[16, 32])
    parser.add_argument("--stages", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repetitions", type=int, choices=(12, 20), default=12)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "grouped-split-k.json"
    report = {"tier": "development operator qualification; not standard-compliant",
              "status": "running", "command": shlex.join(sys.orig_argv),
              "baseline": "image0.1.12.3 unmodified _spec_decode_varlen_fwd",
              "candidate": "opt-in grouped q5*GQA6, FP16 DPAS/FP32 accum, Split-K",
              "intentional_differences": ["grouped KV reuse", "explicit splits 16/32, BN32, warps4"],
              "splits": args.splits, "stages": args.stages, "seed": SEED,
              "tolerance": {"rtol": RTOL, "atol": ATOL, "silently_relaxed": False},
              "reference": "independent CPU FP32 PyTorch paged causal attention",
              "fallback_forbidden": True, "checks": [],
              "test_process": {"boundary": "flash_attn_varlen_func through installed helper",
                               "correctness": "boundary lengths, HND strides, permuted pages, nonunit descales",
                               "graphs": "in-place Q, used lengths, page table, candidate cumulative lengths",
                               "timing": f"64K live65541 paired alternating graphs; warm3; measured{args.repetitions}",
                               "cleanup": "process exit releases graph/tensor owners; lead removes disposable container",
                               "evidence": str(path)}}
    save(report, path)
    print(json.dumps({k: report[k] for k in ("tier", "baseline", "candidate", "test_process")}), flush=True)
    routes = None
    try:
        global torch
        import torch
        import triton
        from vllm_xpu_kernels import flash_attn_interface as fa
        # -P deliberately excludes script cwd; explicitly trust only the staged
        # experiment directory, or let the mounted PYTHONPATH resolve the module.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen38_grouped_split_k as prototype

        report["environment"] = {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "triton": triton.__version__, "torch_config": torch.__config__.show(),
            "prototype_path": prototype.__file__, "native_path": fa.__file__,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("B70_", "VLLM_XPU_", "TRITON_"))
                    or k in ("ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK", "IGC_ForceOCLSIMDWidth")}}
        for package in ("vllm", "vllm-xpu-kernels", "pytorch-triton-xpu"):
            try:
                report["environment"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["environment"][package] = "distribution metadata unavailable"
        assert "IGC_ForceOCLSIMDWidth" not in os.environ, "SIMD width override is not authorized"
        assert torch.xpu.is_available(), "XPU unavailable: GPU qualification was not run"
        report["environment"].update(device=torch.xpu.get_device_name(),
                                      device_properties=str(torch.xpu.get_device_properties(0)))
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        assert not getattr(fa._spec_decode_varlen_fwd, "_b70_grouped_split_k", None), "baseline already patched"
        original_fallback = fa._fallback_varlen_attn
        original_threshold = fa._SPEC_DECODE_MAX_QLEN

        def reject_fallback(*unused_args, **unused_kwargs):
            raise RuntimeError("FORBIDDEN: FA PyTorch fallback cannot qualify a native/candidate path")

        fa._fallback_varlen_attn = reject_fallback
        fa._SPEC_DECODE_MAX_QLEN = 16
        report["native_threshold"] = {"incoming": original_threshold, "tested": 16}
        routes = Routes(fa, prototype)
        for splits in dict.fromkeys(args.splits):
            routes.enable(splits, args.stages)
            prefix = f"split{splits}-stage{args.stages}"
            qualified = True
            for length, layout, nonunit, permuted in (
                    (5, "nhd", False, False), (5, "hnd", True, True),
                    (0, "hnd", True, True), (1, "hnd", True, True), (4, "hnd", True, True),
                    (1663, "hnd", True, True), (1664, "hnd", True, True),
                    (1665, "hnd", True, True), (LIVE_64K, "hnd", True, True)):
                def test(entry):
                    c = make_case(length, layout=layout, nonunit=nonunit, permuted=permuted)
                    correctness(entry, c, routes)
                qualified &= run_check(report, path, f"{prefix}/length{length}-{layout}", test)
            for length in (1665, LIVE_64K):
                qualified &= run_check(report, path, f"{prefix}/causal-{length}",
                                       lambda e: causal_invariance(e, make_case(length), routes))
            qualified &= run_check(report, path, f"{prefix}/dummy-safety",
                                   lambda e: dummy_safety(e, routes))
            qualified &= run_check(report, path, f"{prefix}/fallback-and-fail-loud",
                                   lambda e: fallback_checks(e, routes))
            for route in ("native", "candidate"):
                qualified &= run_check(report, path, f"{prefix}/graph-mutation-{route}",
                                       lambda e: graph_mutation(e, make_case(LIVE_64K), routes, route))
            if qualified:
                run_check(report, path, f"{prefix}/paired-graph-timing",
                          lambda e: timings(e, make_case(LIVE_64K), routes, args.repetitions))
            else:
                report["checks"].append({"name": f"{prefix}/paired-graph-timing", "status": "skipped",
                                         "reason": "correctness/dispatch/graph gate failed; no speed claim"})
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["checks"]) else "failed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
    finally:
        if routes is not None:
            routes.close()
            fa._fallback_varlen_attn = original_fallback
            fa._SPEC_DECODE_MAX_QLEN = original_threshold
        save(report, path)
    print(f"{report['status']}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
