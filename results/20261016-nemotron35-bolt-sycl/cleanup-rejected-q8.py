#!/usr/bin/env python3
"""Host-only removal of our rejected Q8 scratch copy; preserve originals/evidence."""
import json
import os
from pathlib import Path
import stat
import subprocess

root = Path('/home/mike/b70-evals/20261016-nemotron35-bolt-sycl')
directory = Path('/home/mike/inference/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF-ssm-q8_0')
file = directory / 'NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf'
manifest = json.loads((root / 'model-verified-ssm-q8_0.json').read_text())
assert manifest['pass_'] and manifest['path'] == str(file)
assert manifest['sha256'] == 'd037cbbba6b0c0b27b67c585062ae3bd52fe602d5eacc19b94e8b6efb476163b'
assert directory.resolve() == directory and not directory.is_symlink()
assert set(directory.iterdir()) == {file}
assert stat.S_ISREG(file.lstat().st_mode) and file.stat().st_size == manifest['bytes'] == 24209850368
assert not os.path.ismount(directory) and not os.path.ismount(file)
ids = subprocess.check_output(['docker', 'ps', '-aq'], text=True).split()
containers = json.loads(subprocess.check_output(['docker', 'inspect', *ids], text=True)) if ids else []
for container in containers:
    for mount in container['Mounts']:
        source = Path(mount['Source']).resolve()
        assert not (directory == source or directory.is_relative_to(source) or source.is_relative_to(directory)), (container['Name'], str(source))
file.unlink()
directory.rmdir()
print('Removed only rejected SSM-Q8 scratch GGUF and its now-empty directory; original models and all logs/manifests preserved.')
subprocess.run(['df', '-B1', str(directory.parent)], check=True)
