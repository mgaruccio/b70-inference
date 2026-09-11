"""Pin and replay-check the existing GDN prefill patch in the disposable image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy

root = Path(importlib.util.find_spec("vllm").origin).parent
if not str(root).startswith("/opt/venv/"):
    raise RuntimeError(f"not the installed serving package: {root}")
pins = {
    "v1/attention/backends/gdn_attn.py": "f78ae94fa99bfc13a705c3fd319347d55163420bc5dd6aceaf60f95171c54e4c",
    "v1/attention/backends/utils.py": "8fa9cbdec4b0327d34a243b7beb3783983d0fcb70a90a05a2f5217b4fed325eb",
}
for relative, expected in pins.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"pinned source changed: {relative}")
patch_path = Path("/experiment/patch_xpu_prefill.py")
if hashlib.sha256(patch_path.read_bytes()).hexdigest() != "2e206c150de41c66db7d2568911933188c080ac3ebba35172c1027b31fae43ea":
    raise RuntimeError("prefill patch changed")
patch = runpy.run_path(str(patch_path))
p = root / "v1/attention/backends/gdn_attn.py"
before = p.read_text()
after = patch["patch_text"](before)
assert patch["patch_text"](after) == after
assert after.replace(patch["NEW"], patch["OLD"]).replace(
    patch["IMPORT"] + "from vllm.platforms import current_platform\n", patch["IMPORT"]
) == before
compile(after, str(p), "exec")
p.write_text(after)
record = {"root": str(root), "before": pins, "patched_gdn_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
          "exact_reverse_and_replay": True}
Path("/output/effective-prefill-source.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
