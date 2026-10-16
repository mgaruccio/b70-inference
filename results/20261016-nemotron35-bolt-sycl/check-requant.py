#!/usr/bin/env python3
"""Host/container-only checks for this exact Bolt selective-requant experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, dequantize

ROOT = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
MODEL = Path('/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf')
SSM = re.compile(r'blk\.[0-9]+\.ssm_(in|out)\.weight')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def metadata(reader):
    result = {}
    for name, field in reader.fields.items():
        if name.startswith('GGUF.'):
            continue
        encoded = json.dumps(field.contents(), ensure_ascii=False, separators=(',', ':')).encode()
        result[name] = {'types': [int(t) for t in field.types],
                        'sha256': hashlib.sha256(encoded).hexdigest()}
    return result


def tensor_info(t):
    return {'name': t.name, 'type': t.tensor_type.name,
            'shape': t.shape.tolist(), 'bytes': int(t.n_bytes)}


def error(original, converted):
    # Whole tensor, bounded chunks aligned to both quant block sizes.
    count, energy, squared_error, max_abs = 0, 0.0, 0.0, 0.0
    width = int(original.shape[0])
    # One or more complete rows avoids assumptions about quant block byte sizes.
    rows_per_chunk = max(1, 262144 // width)
    nrows = int(original.n_elements) // width
    a = original.data.view(np.uint8).reshape(nrows, -1)
    b = converted.data.view(np.uint8).reshape(nrows, -1)
    for row in range(0, nrows, rows_per_chunk):
        x = dequantize(a[row:row + rows_per_chunk], original.tensor_type).astype(np.float64)
        y = dequantize(b[row:row + rows_per_chunk], converted.tensor_type).astype(np.float64)
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), original.name
        delta = y - x
        count += x.size
        energy += float(np.sum(x * x))
        squared_error += float(np.sum(delta * delta))
        max_abs = max(max_abs, float(np.max(np.abs(delta))))
    assert count == original.n_elements
    return {'elements': count, 'nmse': squared_error / max(energy, 1e-300),
            'rmse': (squared_error / count) ** 0.5, 'max_absolute_error': max_abs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=('q8_0', 'q6_k'), required=True)
    parser.add_argument('--converted', type=Path)
    args = parser.parse_args()
    target = GGMLQuantizationType[args.target.upper()]
    original = GGUFReader(MODEL)
    tensors = {t.name: t for t in original.tensors}
    assert len(tensors) == len(original.tensors) == 843
    selected = {n for n, t in tensors.items() if t.tensor_type == GGMLQuantizationType.NVFP4}
    assert len(selected) == 80 and all(SSM.fullmatch(n) for n in selected)
    assert all(tensors[n].shape.tolist() in ([4096, 18560], [8192, 4096]) for n in selected)
    assert sum(t.tensor_type == GGMLQuantizationType.IQ2_XXS for t in tensors.values()) == 160
    assert sum(t.tensor_type == GGMLQuantizationType.Q8_0 for t in tensors.values()) == 114
    expected = {n: target if n in selected else t.tensor_type for n, t in tensors.items()}
    if args.converted is None:
        lines = [f'^{re.escape(n)}$={expected[n].name.lower()}' for n in sorted(tensors)]
        (ROOT / f'selective-types-{args.target}.txt').write_text('\n'.join(lines) + '\n')
        save(ROOT / 'requant-original-structure.json',
             {'tensors': [tensor_info(t) for t in original.tensors], 'metadata': metadata(original)})
        print(f'Prepared {len(lines)} exact anchored overrides; only {len(selected)} NVFP4 weights change.')
        return
    converted = GGUFReader(args.converted)
    actual = {t.name: t for t in converted.tensors}
    assert len(actual) == len(converted.tensors) and actual.keys() == tensors.keys()
    checks, errors = [], []
    for name, src in tensors.items():
        dst = actual[name]
        assert dst.shape.tolist() == src.shape.tolist(), name
        assert dst.tensor_type == expected[name], (name, dst.tensor_type, expected[name])
        item = tensor_info(dst)
        if name in selected:
            item['error_vs_dequantized_original'] = error(src, dst)
            errors.append(item['error_vs_dequantized_original']['nmse'])
        else:
            before = hashlib.sha256(memoryview(src.data).cast('B')).hexdigest()
            after = hashlib.sha256(memoryview(dst.data).cast('B')).hexdigest()
            assert src.n_bytes == dst.n_bytes and before == after, name
            item['unchanged_payload_sha256'] = before
        checks.append(item)
        print(name, 'checked', flush=True)
    before, after = metadata(original), metadata(converted)
    allowed = {'general.file_type', 'general.quantization_version',
               'split.no', 'split.count', 'split.tensors.count'}
    changes = [n for n in before.keys() | after.keys() if before.get(n) != after.get(n)]
    assert set(changes) <= allowed, changes
    result = {'structure_and_untouched_payloads_pass': True, 'target': args.target,
              'converted_path': str(args.converted), 'converted_bytes': args.converted.stat().st_size,
              'changed_tensors': len(selected), 'unchanged_tensors': len(tensors) - len(selected),
              'metadata_changes': changes, 'max_tensor_nmse': max(errors),
              'median_tensor_nmse': float(np.median(errors)), 'tensors': checks,
              'note': 'Weight error is against the existing dequantized NVFP4, not original full-precision weights; model API quality/performance validation remains separate.'}
    save(ROOT / f'requant-{args.target}-verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
