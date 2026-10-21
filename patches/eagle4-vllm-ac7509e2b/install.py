#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply the exact-base EAGLE4 generic engine hook overlay to an offline package.

No vLLM import, downloads, version rewriting, plugin installation, or GPU use.
Stop the target process first. --check validates every file and the full replay
without writing. --apply stages all outputs after validation, then replaces files.
Run --check again after --apply to verify the idempotent installed state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

HERE = Path(__file__).resolve().parent
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class OverlayError(RuntimeError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        old = lines[i].rstrip("\n")
        if not old.startswith("--- a/vllm/") or i + 1 >= len(lines):
            raise OverlayError(f"Malformed patch header: {old}")
        path = old.removeprefix("--- a/vllm/")
        if lines[i + 1].rstrip("\n") != "+++ b/vllm/" + path:
            raise OverlayError(f"Mismatched patch path: {path}")
        if path in sections:
            raise OverlayError(f"Duplicate patch path: {path}")
        i += 2
        start = i
        while i < len(lines) and not lines[i].startswith("--- a/vllm/"):
            i += 1
        sections[path] = lines[start:i]
    return sections


def replay(before: bytes, hunks: list[str]) -> bytes:
    source = before.decode("utf-8").splitlines(keepends=True)
    result: list[str] = []
    cursor, i = 0, 0
    while i < len(hunks):
        match = HUNK.match(hunks[i])
        if not match:
            raise OverlayError(f"Invalid hunk: {hunks[i]!r}")
        start, old_count, new_start, new_count = (
            int(match[1]), int(match[2] or 1), int(match[3]), int(match[4] or 1)
        )
        offset = start - 1 if old_count else start
        if offset < cursor or offset > len(source):
            raise OverlayError("Overlapping or out-of-range hunk")
        result.extend(source[cursor:offset])
        cursor = offset
        if len(result) != (new_start - 1 if new_count else new_start):
            raise OverlayError("Invalid new-file hunk offset")
        i += 1
        old_seen, new_seen = 0, 0
        while i < len(hunks) and not hunks[i].startswith("@@ "):
            line = hunks[i]
            tag, content = line[:1], line[1:]
            if tag in (" ", "-"):
                if cursor >= len(source) or source[cursor] != content:
                    raise OverlayError("Hunk context differs from exact source")
                cursor += 1
                old_seen += 1
            if tag in (" ", "+"):
                result.append(content)
                new_seen += 1
            if tag not in (" ", "+", "-"):
                raise OverlayError(f"Invalid patch operation {tag!r}")
            i += 1
        if (old_seen, new_seen) != (old_count, new_count):
            raise OverlayError("Hunk count mismatch")
    result.extend(source[cursor:])
    return "".join(result).encode("utf-8")


def plan(package_root: Path, overlay: Path = HERE) -> tuple[dict, list[tuple[Path, bytes]], int]:
    root = package_root.resolve(strict=True)
    if not root.is_dir():
        raise OverlayError("--package-root must be the vllm package directory")
    manifest = json.loads((overlay / "manifest.json").read_text())
    patch = (overlay / "engine.patch").read_bytes()
    if sha(patch) != manifest["patch_sha256"]:
        raise OverlayError("Overlay patch digest mismatch")
    sections = patch_sections(patch.decode("utf-8"))
    entries = manifest["files"]
    names = [e["path"] for e in entries]
    if len(names) != len(set(names)) or not set(sections).issubset(names):
        raise OverlayError("Duplicate/unmanifested patch paths")
    writes, installed = [], 0
    for entry in entries:
        name = PurePosixPath(entry["path"])
        if name.is_absolute() or ".." in name.parts or not name.parts or name.suffix != ".py":
            raise OverlayError(f"Unsafe manifest path: {name}")
        path = root / name
        for component in (path, *path.parents):
            if component == root:
                break
            if component.is_symlink():
                raise OverlayError(f"Symlink in target: {component}")
        if path.exists() and not path.is_file():
            raise OverlayError(f"Not a regular file: {path}")
        before = path.read_bytes() if path.exists() else None
        current = sha(before) if before is not None else None
        if current == entry["after_sha256"]:
            installed += entry["before_sha256"] != entry["after_sha256"]
            continue
        if current != entry["before_sha256"]:
            raise OverlayError(f"Unsupported source {path}: {current}; expected exact frozen or installed digest")
        if entry["path"] not in sections:
            raise OverlayError(f"Missing patch for {path}")
        after = replay(before or b"", sections[entry["path"]])
        if sha(after) != entry["after_sha256"]:
            raise OverlayError(f"Output digest mismatch: {path}")
        compile(after, str(path), "exec")
        writes.append((path, after))
    return manifest, writes, installed


def apply(package_root: Path, overlay: Path = HERE, *, check: bool = False) -> dict:
    manifest, writes, installed = plan(package_root, overlay)
    # Validation and in-memory replay of ALL files finished before the first write.
    staged = []
    try:
        if not check:
            for path, data in writes:
                if not path.parent.is_dir():
                    raise OverlayError(f"Missing parent directory: {path.parent}")
                fd, tmp = tempfile.mkstemp(prefix=".eagle4-", suffix=".tmp", dir=path.parent)
                staged.append((Path(tmp), path))
                with os.fdopen(fd, "wb") as out:
                    out.write(data)
                os.chmod(tmp, path.stat().st_mode & 0o777 if path.exists() else 0o644)
            for tmp, path in staged:
                os.replace(tmp, path)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
    return dict(base_version=manifest["base_version"], files_validated=len(manifest["files"]),
                already_installed=installed, pending=len(writes) if check else 0,
                written=0 if check else len(writes), check=check)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(apply(args.package_root, check=args.check), sort_keys=True))
    except (OverlayError, OSError, ValueError) as exc:
        parser.exit(1, f"EAGLE4 overlay refused: {exc}\n")


if __name__ == "__main__":
    main()
