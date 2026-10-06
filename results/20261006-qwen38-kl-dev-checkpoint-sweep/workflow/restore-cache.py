#!/usr/bin/env python3
"""One-run streamed readback/extraction of the verified full CE archive."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import urllib.request

os.umask(0o077)
HOME = Path.home()
EXPECTED = '86632b201f09facabf33824d75a062f364ab67eeb67b2e19a39413bfb6539244'

class Parts(io.RawIOBase):
    def __init__(self, index):
        self.index = index
        self.number = 0
        self.response = None
        self.whole = hashlib.sha256()
        self.total = 0
        self.part_hash = None
        self.part_size = 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Unbounded restore reads forbidden')
        if size == 0:
            return b''
        while self.number < len(self.index['parts']):
            part = self.index['parts'][self.number]
            if self.response is None:
                self.response = urllib.request.urlopen(part['url'], timeout=180)
                self.part_hash = hashlib.sha256()
                self.part_size = 0
            block = self.response.read(size)
            if block:
                self.part_hash.update(block)
                self.whole.update(block)
                self.part_size += len(block)
                self.total += len(block)
                if self.part_size > part['bytes']:
                    raise ValueError('Oversized archive part')
                return block
            self.response.close()
            self.response = None
            if self.part_size != part['bytes'] or self.part_hash.hexdigest() != part['sha256']:
                raise ValueError('Archive part readback mismatch')
            print('RESTORED_SOURCE_PART_VERIFIED=' + part['path'], flush=True)
            self.number += 1
        return b''

    def verify(self):
        while self.read(8 * 1024**2):
            pass
        if self.total != self.index['bytes'] or self.whole.hexdigest() != EXPECTED:
            raise ValueError('Whole source readback mismatch')

    def close(self):
        if self.response is not None:
            self.response.close()
        super().close()


def main():
    index = json.loads((HOME / 'qwen-mtp-upload/restore-private.json').read_text())
    assert index['sha256'] == EXPECTED and index['bytes'] == 14339297280
    assert len(index['parts']) == 7
    for i, part in enumerate(index['parts']):
        assert part['path'] == 'final.tgz.part' + str(i).zfill(2)
        assert re.fullmatch('[0-9a-f]{64}', part['sha256'])
    dest = HOME / 'qwen-mtp-cache-restore'
    dest.mkdir(mode=0o700, exist_ok=False)
    # Extraction occurs only into a fresh private staging directory. Publication
    # into the new run happens after ALL parts and the whole hash verify.
    with Parts(index) as stream:
        with tarfile.open(fileobj=stream, mode='r|') as archive:
            for member in archive:
                # Restore only public source captures; do not unpack old results,
                # private manifests, checkpoints, or arbitrary archive links.
                name = member.name.removeprefix('./')
                if name.startswith(('capture-train/', 'capture-dev/')):
                    if member.issym() or member.islnk():
                        raise ValueError('Capture archive link forbidden')
                    archive.extract(member, path=dest, filter='data')
        stream.verify()
    counts = {}
    for split, folder, n in [('train', 'capture-train/train', 374), ('dev', 'capture-dev/heldout', 64)]:
        source = dest / folder
        counts[split] = len(list(source.glob('*.pt')))
        if counts[split] != n:
            raise ValueError('Missing full-corpus captures')
        target = HOME / 'qwen-mtp-run' / folder
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('Refuse to overwrite capture cache')
        shutil.move(str(source), str(target))
    report = {'source_archive_sha256': EXPECTED, 'bytes_verified': index['bytes'],
              'parts_verified': 7, 'restored_sequences': counts, 'recaptured': False}
    (HOME / 'qwen-mtp-run/cache-restore.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FULL_CACHED_CORPUS374_64_SOURCE_READBACK_AND_RESTORE_VERIFIED', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not leak signed GET URLs in exception text/tracebacks.
        print('CACHE_RESTORE_FAILED=' + type(error).__name__, flush=True)
        raise SystemExit(1)
