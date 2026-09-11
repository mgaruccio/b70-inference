#!/usr/bin/env python3
"""Stage the public pinned FP8 control; weights stay on inference-host, never Git."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "target-fp8"
REPO = "Qwen/Qwen3.8-27B-FP8"
REV = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
SMALL = {"config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json",
         "tokenizer_config.json", "merges.txt", "vocab.json", "chat_template.jinja", "README.md",
         "preprocessor_config.json", "video_preprocessor_config.json"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    assert metadata["sha"] == REV and not metadata.get("private") and not metadata.get("gated")
    files = [f for f in metadata["siblings"] if f["rfilename"] in SMALL or
             re.fullmatch(r"(?:layers-\d+|mtp|outside)\.safetensors", f["rfilename"])]
    assert len([f for f in files if f["rfilename"].endswith(".safetensors")]) == 66
    assert all(f.get("size", 0) > 0 for f in files)
    assert all(f.get("lfs", {}).get("sha256") for f in files if f["rfilename"].endswith(".safetensors"))
    assert shutil.disk_usage(ROOT).free > sum(f["size"] for f in files) + (20 << 30)
    MODEL.mkdir(exist_ok=True)
    (ROOT / "download-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def download(info):
        name = info["rfilename"]
        path = MODEL / name
        expected = info.get("lfs", {}).get("sha256")
        if not path.exists():
            part = path.with_name(path.name + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                            "--retry-delay", "3", "--max-time", "1800", "--continue-at", "-", "--output", str(part),
                            f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"], check=True, timeout=7500)
            assert part.stat().st_size == info["size"], name
            actual = digest(part)
            assert expected is None or actual == expected, name
            part.rename(path)
        else:
            actual = digest(path)
            assert path.stat().st_size == info["size"] and (expected is None or actual == expected), name
        row = {"file": name, "bytes": path.stat().st_size, "sha256": actual, "lfs_sha256_verified": expected is not None}
        print(json.dumps(row), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(download, files))
    (ROOT / "download-result.json").write_text(json.dumps({"repository": REPO, "revision": REV,
        "total_bytes": sum(r["bytes"] for r in rows), "files": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
