#!/usr/bin/env python3
"""Case-local bindings of the passed CPU-only 105-object host build driver."""
import hashlib
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/mike/b70-evals/20261020-lilicorr-iq2-packed')
TEMPLATE = HERE.parent / '20261020-lilicorr-host-boundary/build-candidate.sh'
TEMPLATE_SHA256 = '83ac40c38fdcfee1ca021f0cc25cea02cb687e908d12a7f4ccee8bd78e4aa24d'
OLD_POST = """pins = s['load_pins']()
expected_delta = sorted(name for name in pins['diagnostic'] if pins['parent'].get(name) != pins['diagnostic'][name])
assert source['changed_relative_to_parent'] == expected_delta"""


def render():
    data = TEMPLATE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == TEMPLATE_SHA256, 'frozen build template drift'
    text = data.decode()
    changes = (
        ('20261020-lilicorr-host-boundary', '20261020-lilicorr-iq2-packed', 1),
        ('llama.cpp-bolt-b6334e37-nvfp4-lilicorr-device-timing-r1', 'llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-batch4-r1', 1),
        ('llama.cpp-bolt-b6334e37-nvfp4-lilicorr-host-boundary-r1', 'llama.cpp-bolt-b6334e37-nvfp4-lilicorr-iq2-packed-r1', 1),
        ('20261020-lilicorr-device-timing', '20261020-lilicorr-iq2-batch4', 1),
        ('bolt-lilicorr-host-boundary-r1', 'bolt-lilicorr-iq2-packed-r1', 1),
        ('disable-device-profiling.patch', 'iq2-packed.patch', 2),
        ("s['HELPER']", "s['SEED_SOURCE_FILE']", 2),
        ('DEVICE_CASE', 'SEED_CASE', 2),
        ('completed device', 'completed clean batch4', 2),
        (OLD_POST, "assert source['changed_relative_to_parent'] == sorted(s['SOURCE_SCOPE'])", 1),
    )
    for old, new, count in changes:
        assert text.count(old) == count, ('template substitution drift', old, text.count(old), count)
        text = text.replace(old, new)
    return text


def main():
    text = render()
    if sys.argv[1:] == ['--print']:
        print(text, end='')
        return
    assert sys.argv[1:] == [] and socket.gethostname() == 'inference-host' and HERE == ROOT
    generated = HERE / 'build-candidate.generated.sh'
    assert not generated.exists(), 'preserve the existing build driver'
    generated.write_text(text)
    subprocess.run(['bash', str(generated)], check=True)


if __name__ == '__main__':
    main()
