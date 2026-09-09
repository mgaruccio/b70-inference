"""CPU-focused contract tests for the standalone Qwen lm_head GPTQ utility."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/experiments/qwen38_calibrate_lmhead.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qwen38_calibrate_lmhead", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def quantizer():
    pytest.importorskip("torch")
    return _load_module()


def test_full_off_diagonal_gptq_changes_sequential_error_update(quantizer):
    torch = pytest.importorskip("torch")
    weight = torch.zeros((1, 128), dtype=torch.float16)
    weight[0, 0] = 1.4
    weight[0, 1] = 1.4
    weight[0, 127] = 7.0

    diagonal_hessian = torch.eye(128, dtype=torch.float32)
    correlated_hessian = diagonal_hessian.clone()
    correlated_hessian[0, 1] = 0.5
    correlated_hessian[1, 0] = 0.5

    diagonal = quantizer.gptq_quantize(weight, diagonal_hessian, damp=0.0, device="cpu")
    correlated = quantizer.gptq_quantize(weight, correlated_hessian, damp=0.0, device="cpu")
    diagonal_codes = quantizer.unpack_qweight(diagonal["qweight"], k=128)
    correlated_codes = quantizer.unpack_qweight(correlated["qweight"], k=128)

    assert diagonal["metadata"]["hessian"] == "full_off_diagonal"
    assert diagonal_codes[0, 1].item() != correlated_codes[0, 1].item()


def test_nibble_order_serialization_and_layout(quantizer, tmp_path):
    torch = pytest.importorskip("torch")
    pattern = torch.tensor(
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.int8,
    )
    signed = pattern.repeat(2, 8)
    packed = quantizer.pack_signed_nibbles(signed)
    assert torch.equal(quantizer.unpack_signed_nibbles(packed, k=128), signed)

    artifact = quantizer.rtn_quantize(signed.to(dtype=torch.float16), device="cpu")
    assert tuple(artifact["qweight"].shape) == (16, 2)
    assert tuple(artifact["qweight"].stride()) == (1, 16)
    assert tuple(artifact["scales"].shape) == (1, 2)
    assert artifact["scales"].is_contiguous()
    assert artifact["qzeros"].dtype == torch.int8
    assert tuple(artifact["qzeros"].shape) == (1,)
    assert artifact["qzeros"].item() == 8

    destination = tmp_path / "lmhead-gptq.pt"
    quantizer.save_quantized_artifact(artifact, destination)
    payload = torch.load(destination, map_location="cpu")
    assert set(payload) == {"qweight", "scales", "qzeros", "group_size", "metadata"}
    assert tuple(payload["qweight"].stride()) == (1, 16)
    assert payload["qzeros"].tolist() == [8]


def test_zero_weights_are_safe_and_nonfinite_weights_are_rejected(quantizer):
    torch = pytest.importorskip("torch")
    weight = torch.zeros((2, 128), dtype=torch.float16)
    hessian = torch.eye(128, dtype=torch.float32)
    artifact = quantizer.gptq_quantize(weight, hessian, damp=0.01, device="cpu")
    assert torch.equal(artifact["scales"], torch.ones_like(artifact["scales"]))
    assert torch.equal(quantizer.unpack_qweight(artifact["qweight"], k=128), torch.zeros_like(weight, dtype=torch.int8))
    assert torch.equal(quantizer.dequantize_weight(artifact), torch.zeros((2, 128), dtype=torch.float32))

    bad = weight.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(quantizer.CalibrationError, match="NaN|infinity"):
        quantizer.rtn_quantize(bad, device="cpu")


def test_hessian_formula_and_capture_disjointness(quantizer, tmp_path):
    torch = pytest.importorskip("torch")
    calibration = tmp_path / "calibration"
    heldout = tmp_path / "eval"
    calibration.mkdir()
    heldout.mkdir()
    hidden = torch.arange(256, dtype=torch.float16).reshape(2, 128)
    calibration_file = calibration / "part.pt"
    eval_file = heldout / "part.pt"
    torch.save({"hidden_states": hidden}, calibration_file)
    torch.save({"hidden_states": hidden + 1}, eval_file)

    hessian, count = quantizer.compute_activation_hessian([calibration_file], 128, chunk_rows=1)
    expected = 2.0 * hidden.to(torch.float32).transpose(0, 1).matmul(hidden.to(torch.float32)) / 2.0
    assert count == 2
    assert torch.allclose(hessian, expected)
    _, _, calibration_rows, eval_rows = quantizer.validate_capture_sets(calibration, heldout, 128)
    assert (calibration_rows, eval_rows) == (2, 2)

    with pytest.raises(quantizer.CalibrationError, match="disjoint"):
        quantizer.validate_capture_sets(calibration, calibration, 128)


def test_rtn_default_packing_is_bounded_independently_of_gptq(quantizer, monkeypatch):
    torch = pytest.importorskip("torch")
    weight = torch.zeros((5000, 128), dtype=torch.float16)
    seen_rows = []
    original_pack = quantizer.pack_signed_nibbles

    def recording_pack(values):
        seen_rows.append(int(values.shape[0]))
        return original_pack(values)

    monkeypatch.setattr(quantizer, "pack_signed_nibbles", recording_pack)
    quantizer.rtn_quantize(weight, row_chunk_rows=0, device="cpu")
    assert max(seen_rows) <= quantizer.DEFAULT_RTN_ROW_CHUNK
    assert max(seen_rows) == 4096
