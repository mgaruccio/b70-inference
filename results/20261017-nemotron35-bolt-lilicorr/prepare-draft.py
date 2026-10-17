#!/usr/bin/env python3
"""Remote-only pinned official download and GGUF header compatibility inspection."""
import collections
import hashlib
import json
from pathlib import Path
import socket
import struct
import urllib.request

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr')
REPO = 'nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF'
REV = '182d2987939e12f458298b5d5f6ef98912f4ad6c'
NAME = 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf'
SHA = '080e1ea41c9562a7804dbf30c6fbd4d17e216c6b8ff666f88e6edb1816010194'
SIZE = 998242560
TARGET = Path('/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def header(path):
    with path.open('rb') as f:
        def u(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]
        def s():
            return f.read(u('Q')).decode()
        def value(t):
            if t == 8:
                return s()
            if t == 9:
                kind, count = u('I'), u('Q')
                return [value(kind) for _ in range(count)]
            return u({0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[t])
        assert f.read(4) == b'GGUF'
        version, nt, nk = u('I'), u('Q'), u('Q')
        assert version == 3
        meta = {}
        for _ in range(nk):
            key = s()
            meta[key] = value(u('I'))
        tensors = []
        for _ in range(nt):
            name = s()
            dims = [u('Q') for _ in range(u('I'))]
            tensors.append(dict(name=name, dimensions=dims, type=u('I'), offset=u('Q')))
        return meta, tensors


def main():
    assert socket.gethostname() == 'inference-host'
    directory = ROOT / 'draft'
    directory.mkdir(exist_ok=False)
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{NAME}'
    save(ROOT / 'download-request.json', dict(url=url, revision=REV, bytes=SIZE, sha256=SHA))
    for name in ('README.md', 'LICENSE'):
        with urllib.request.urlopen(f'https://huggingface.co/{REPO}/raw/{REV}/{name}', timeout=60) as response:
            (ROOT / ('draft-' + name)).write_bytes(response.read())
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true', timeout=60) as response:
        (ROOT / 'draft-repository.json').write_bytes(response.read())
    path = directory / NAME
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, path.open('xb') as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
            digest.update(block)
    receipt = dict(path=str(path), revision=REV, bytes=path.stat().st_size, sha256=digest.hexdigest())
    receipt['pass'] = receipt['bytes'] == SIZE and receipt['sha256'] == SHA
    save(ROOT / 'draft-verified.json', receipt)
    assert receipt['pass'], receipt
    target_meta, target_tensors = header(TARGET)
    draft_meta, draft_tensors = header(path)
    tokenizer = sorted(k for k in target_meta if k.startswith('tokenizer.ggml.'))
    comparison = {k: target_meta[k] == draft_meta.get(k) for k in tokenizer}
    for label, meta, tensors in [('target', target_meta, target_tensors), ('draft', draft_meta, draft_tensors)]:
        save(ROOT / (label + '-header.json'), dict(
            metadata={k:v for k,v in meta.items() if not k.startswith('tokenizer.')},
            tensors=tensors, tensor_types=dict(collections.Counter(t['type'] for t in tensors)),
            tokenizer_fields=sorted(k for k in meta if k.startswith('tokenizer.'))))
    save(ROOT / 'tokenizer-comparison.json', comparison)
    assert all(comparison.values()), comparison
    assert draft_meta['general.architecture'] == 'dflash'
    assert draft_meta.get('lilicorr.candidate_topk', 0) > 0
    print(json.dumps(dict(draft=receipt, metadata={k:v for k,v in draft_meta.items() if not k.startswith('tokenizer.')}, tokenizer=comparison), indent=2), flush=True)
    print('BOLT_LILICORR_PREPARATION_PASSED', flush=True)


if __name__ == '__main__':
    main()
