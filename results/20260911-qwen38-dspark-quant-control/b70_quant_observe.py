"""Experiment-only identity and one-shot load inspection; no numerical overrides."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

REPOSITORY = "Qwen/Qwen3.8-27B-FP8"
REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
FP8_CONFIG_SHA = "74227dd615bf1ea975aa676bdf355a0379858c12f394b5365cd9dfa5fc2c70bc"
FP8_INDEX_SHA = "f0838c766951bdfe76d6afbdb2771a8f67aaa2231dedb3d33cebd817729843a2"
FP8_QUANT_SHA = "a8794dc9a10580bc7c93fa7ee176d0bc62b9194799d8f3918f811ed4cc6facc7"
FP8_METHOD = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod"
FP8_KERNEL = "vllm.model_executor.kernels.linear.scaled_mm.xpu.XPUFp8BlockScaledMMKernel"
GPTQ_METHOD = "vllm.model_executor.layers.quantization.auto_gptq.AutoGPTQLinearMethod"
GPTQ_KERNEL = "vllm.model_executor.kernels.linear.mixed_precision.xpu.XPUwNa16LinearKernel"
# No strong tensor references: inspection must not keep offloaded buffers alive.
_OFFLOADED = {}
_REOFFLOADED = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError("quant-control: " + message)


def qualified(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def validate_fp8_target(target):
    require(os.environ.get("B70_QUANT_CONTROL_ARM") == "fp8", "FP8 arm not explicitly enabled")
    require(target.quantization == "fp8", "target quantization must remain fp8")
    require(str(target.model) == "/model", "FP8 target must be the read-only /model snapshot")
    require(digest("/model/config.json") == FP8_CONFIG_SHA, "wrong pinned FP8 config")
    require(digest("/model/model.safetensors.index.json") == FP8_INDEX_SHA, "wrong pinned FP8 index")
    qc = target.hf_config.quantization_config
    qhash = hashlib.sha256(json.dumps(qc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(qhash == FP8_QUANT_SHA, "FP8 quantization config changed, including exclusion list")
    return True


def target_identity(target, arm, manifest_path=None):
    """Recheck small-file identity and shard sizes; download manifest owns full hashes."""
    target = Path(target).resolve(strict=True)
    config = target / "config.json"
    index = target / "model.safetensors.index.json"
    qc = json.loads(config.read_text())["quantization_config"]
    result = {"path": str(target), "config_sha256": digest(config), "index_sha256": digest(index),
              "quantization_config": qc}
    if arm == "fp8":
        require(result["config_sha256"] == FP8_CONFIG_SHA, "wrong FP8 config")
        require(result["index_sha256"] == FP8_INDEX_SHA, "wrong FP8 index")
        manifest_path = Path(manifest_path).resolve(strict=True)
        manifest = json.loads(manifest_path.read_text())
        require(manifest["repository"] == REPOSITORY and manifest["revision"] == REVISION,
                "wrong download revision/repository")
        rows = {row["file"]: row for row in manifest["files"]}
        require(len(rows) == len(manifest["files"]), "duplicate download manifest entries")
        shards = set(json.loads(index.read_text())["weight_map"].values())
        for name in {"config.json", "model.safetensors.index.json", *shards}:
            require(Path(name).name == name and name in rows, "missing/unsafe manifest entry: " + name)
            path = target / name
            row = rows[name]
            require(path.is_file() and path.stat().st_size == row["bytes"], "missing/wrong-sized file: " + name)
            require(bool(re.fullmatch(r"[0-9a-f]{64}", row["sha256"])), "invalid recorded hash: " + name)
            if name in shards:
                require(row["lfs_sha256_verified"] is True, "unverified downloaded shard: " + name)
            else:
                require(digest(path) == row["sha256"], "small-file manifest mismatch: " + name)
        result.update(repository=REPOSITORY, revision=REVISION, manifest_path=str(manifest_path),
                      manifest_sha256=digest(manifest_path), indexed_shards=len(shards),
                      weight_identity="lead download manifest SHA256 verification; runner rechecks sizes, not 31GB hashes")
    else:
        require(qc.get("quant_method") == "gptq" and qc.get("bits") == 4
                and qc.get("group_size") == 128 and qc.get("sym") is True
                and qc.get("desc_act") is False and qc.get("lm_head") is False, "wrong GPTQ target")
        result["revision"] = "existing local GPTQ snapshot; config/index hashes, no invented upstream revision"
    return result


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return (str(tensor.device), storage.data_ptr(), storage.nbytes())


def record_offload(offloader, param):
    require(offloader.uva_offloading and offloader.pin_memory, "native pinned UVA is required; no fallback")
    _OFFLOADED[storage_key(param)] = param.numel() * param.element_size()


def record_reoffload(param):
    # Loader re-offloads replacement Parameters after native packing. This is
    # distinct from the initial budget counter; don't mistake new UVA storage
    # for device materialization or trust a marker on an in-place .data change.
    _REOFFLOADED[storage_key(param)] = param.numel() * param.element_size()

def tensor_info(tensor):
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "stride": list(tensor.stride()),
            "device": str(tensor.device), "contiguous": tensor.is_contiguous(),
            "bytes": tensor.numel() * tensor.element_size(),
            "uva_marker": bool(getattr(tensor, "_vllm_is_uva_offloaded", False)),
            "original_offloaded_storage": storage_key(tensor) in _OFFLOADED,
            "postprocess_reoffloaded_storage": storage_key(tensor) in _REOFFLOADED}


def memory_snapshot():
    import torch
    torch.xpu.synchronize()
    free, total = torch.xpu.mem_get_info()
    status = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in Path("/proc/self/status").read_text().splitlines() if ":" in line}
    return {"xpu_allocated_bytes": torch.xpu.memory_allocated(),
            "xpu_reserved_bytes": torch.xpu.memory_reserved(),
            "xpu_peak_allocated_bytes": torch.xpu.max_memory_allocated(),
            "xpu_free_bytes": free, "xpu_total_bytes": total,
            "process_VmRSS": status.get("VmRSS"), "process_VmHWM": status.get("VmHWM")}


def write_report(name, report):
    path = Path("/output") / name
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("B70_QUANT_CONTROL " + name + " " + str(report.get("passed", "recorded")), flush=True)


def check_linear(method, module, arm):
    """Check the *selected* native implementation and all its execution tensors."""
    import torch
    if arm == "fp8":
        require(qualified(method) == FP8_METHOD, "unexpected FP8 linear method")
        kernel = method.fp8_linear
        require(qualified(kernel) == FP8_KERNEL, "unexpected FP8 native kernel: " + qualified(kernel))
        require(method.block_quant and list(method.weight_block_size) == [128, 128]
                and not method.use_marlin and not method.act_q_static, "wrong FP8 dispatch mode")
        c = kernel.config
        require(kernel.apply_input_quant is True and not c.activation_quant_key.scale.static
                and tuple(c.activation_quant_key.scale.group_shape) == (1, 128)
                and tuple(kernel.weight_group_shape) == (128, 128), "wrong block W8A8 activation/weight mode")
        require(c.input_dtype == torch.float16 and c.out_dtype == torch.float16, "wrong FP8 compute/output dtype")
        n, k = c.weight_shape
        w, s = module.weight, module.weight_scale_inv
        gn = math.gcd(n, 128)
        require(k % 128 == 0 and gn % 16 == 0, "unsupported native FP8 block layout")
        require(w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (n, k)
                and w.is_contiguous(), "wrong native FP8 weight dtype/NK layout")
        require(s.dtype == torch.float32 and tuple(s.shape) == (n // gn, k // 128)
                and s.t().is_contiguous(), "wrong native FP8 scale dtype/transposed KN layout")
        mode = "block W8A8 dynamic GroupShape(1,128), _xpu_C.fp8_gemm, FP16 output"
    else:
        require(qualified(method) == GPTQ_METHOD, "unexpected GPTQ linear method")
        kernel = method.kernel
        require(qualified(kernel) == GPTQ_KERNEL, "unexpected GPTQ native kernel: " + qualified(kernel))
        c = kernel.config
        require(c.act_type == torch.float16 and c.group_size == 128
                and not c.zero_points and not c.has_g_idx, "wrong native GPTQ mode")
        k, n = c.partition_weight_shape
        w, s, zp, gidx = kernel._get_weight_params(module)
        require(w.dtype == torch.int32 and tuple(w.shape) == (n, k // 8)
                and w.is_contiguous(), "wrong native GPTQ packed NK/8 layout")
        require(s.dtype == torch.float16 and tuple(s.shape) == (k // 128, n)
                and s.is_contiguous(), "wrong native GPTQ scale layout/dtype")
        require(zp.dtype == torch.int8 and zp.numel() == 1 and gidx is None,
                "wrong native GPTQ symmetric zero point/act-order")
        mode = "GPTQ W4A16 symmetric G128, _xpu_C.int4_gemm_w4a16"
    require(w.device.type == "xpu" and s.device.type == "xpu", "non-XPU execution tensors")
    return {"method": qualified(method), "kernel": qualified(kernel), "mode": mode,
            "config": repr(c), "parameters": {name: tensor_info(p)
                for name, p in module.named_parameters(recurse=False)}}


def inspect_target(model, model_config):
    if str(model_config.model) != "/model":
        return  # Do not mistake the BF16 draft for the target.
    from vllm.model_executor.offloader.base import get_offloader
    arm = os.environ["B70_QUANT_CONTROL_ARM"]
    offloader = get_offloader()
    report = {"arm": arm, "passed": False, "errors": [], "linear_groups": [], "memory": memory_snapshot()}
    groups, layer_ids = {}, set()
    expected = FP8_METHOD if arm == "fp8" else GPTQ_METHOD
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is None:
            continue
        qname = qualified(method)
        # Unquantized linears and FP8 KV-cache methods are not quantized GEMMs.
        if qname.endswith("UnquantizedLinearMethod"):
            continue
        if "LinearMethod" not in qname and not any(hasattr(method, attr) for attr in ("kernel", "fp8_linear")):
            continue
        try:
            require(qname == expected, name + ": unexpected target quantized linear: " + qname)
            row = check_linear(method, module, arm)
            key = json.dumps(row, sort_keys=True)
            groups.setdefault(key, {**row, "modules": []})["modules"].append(name)
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if match:
                layer_ids.add(int(match.group(1)))
        except Exception as exc:
            report["errors"].append(name + ": " + str(exc))
    report["linear_groups"] = list(groups.values())
    if layer_ids != set(range(64)):
        report["errors"].append("quantized linear coverage must include all 64 target decoder layers")
    params = list(model.named_parameters())
    observed_uva = _OFFLOADED.keys() | _REOFFLOADED.keys()
    retained = {storage_key(p) for _, p in params if storage_key(p) in observed_uva}
    dtype_bytes = {}
    for _, p in params:
        key = str(p.dtype) + "@" + str(p.device)
        dtype_bytes[key] = dtype_bytes.get(key, 0) + p.numel() * p.element_size()
    report["parameter_bytes_by_dtype_device"] = dtype_bytes
    report["offload"] = {"class": qualified(offloader),
        "budget_bytes": getattr(offloader, "cpu_offload_max_bytes", None),
        "accounted_offloaded_parameter_bytes": getattr(offloader, "cpu_offload_bytes", None),
        "uva": getattr(offloader, "uva_offloading", None), "pinned": getattr(offloader, "pin_memory", None),
        "original_backing_storage_bytes": sum(key[2] for key in _OFFLOADED),
        "postprocess_reoffloaded_storage_bytes": sum(key[2] for key in _REOFFLOADED),
        "target_retained_observed_uva_storage_bytes": sum(key[2] for key in retained),
        "target_retained_offloaded_parameter_bytes": sum(p.numel() * p.element_size() for _, p in params
                                                          if storage_key(p) in observed_uva),
        "target_uva_marker_parameter_bytes": sum(p.numel() * p.element_size() for _, p in params
                                                if getattr(p, "_vllm_is_uva_offloaded", False)),
        "note": "Pointer-matched initial/re-offloaded UVA after packing; markers alone are not residency. Not assumed 8GiB relief."}
    if (getattr(offloader, "cpu_offload_max_bytes", None) != 8 << 30
            or not getattr(offloader, "cpu_offload_bytes", 0)
            or not getattr(offloader, "uva_offloading", False)):
        report["errors"].append("missing actual 8GiB-budget UVA offload")
    report["passed"] = not report["errors"]
    write_report("target-native.json", report)
    require(report["passed"], "; ".join(report["errors"]))


def inspect_loaded_runner(runner):
    write_report("memory-after-load.json", {"phase": "target plus draft loaded, before KV allocation",
                 "arm": os.environ["B70_QUANT_CONTROL_ARM"], "memory": memory_snapshot()})
