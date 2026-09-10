"""Disposable cold-cache benchmark view of the existing MTP4 reference launcher."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import qwen38_dflash2_probe as dflash

IMAGE = "vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f"
PATCH_ROOT = Path.home() / "inference/src/intel-arc-pro-b70-inference-cookbook/patches"
PATCHES = ("patch_mtp_nightly.py", "patch_mtp_boundary.py", "patch_gdn_mixed_split_v5.py",
           "patch_draft_lmhead_int4.py", "patch_draft_mtp_int4.py")


def launcher_text(original, out):
    """Closed, reviewable changes to the known launcher; never rewrite the original."""
    if hashlib.sha256(original).hexdigest() != dflash.probe.LAUNCHER_SHA:
        raise RuntimeError("persistent MTP4 launcher changed")
    text = original.decode()
    changes = [
        ('-p "0.0.0.0:${PORT:-8000}:8000"', '-p "127.0.0.1:8000:8000"'),
        ('--enable-prefix-caching', '--no-enable-prefix-caching'),
        (r'enable_thinking\":true', r'enable_thinking\":false'),
        ('COOKBOOK="$ROOT/src/intel-arc-pro-b70-inference-cookbook"', 'COOKBOOK=' + shlex.quote(str(out / 'reference-source'))),
        ('exec docker run --rm --name qwen38 --ipc=host',
         'exec docker run --rm --name qwen38 --ipc=host -v ' + shlex.quote(str(out) + ':/profile')
         + ' -v ' + shlex.quote(str(out / 'patch_uniform_decode_prefill.py') + ':/prefill_guard.py:ro')),
        ('exec vllm serve /model ',
         'python /prefill_guard.py; exec vllm serve /model --performance-mode balanced '
         '--compilation-config "{\\"cudagraph_capture_sizes\\":[1,2,4,8]}" --cudagraph-metrics '),
    ]
    for old, new in changes:
        if text.count(old) != 1:
            raise RuntimeError(f"MTP launcher anchor changed: {old}")
        text = text.replace(old, new)
    return text


class MTPReferenceCell(dflash.DFlashCell):
    def start(self):
        if self.power.read_text().strip() != "275000000":
            raise RuntimeError("power cap is not 275 W")
        if dflash.probe.command("docker", "ps", "--format", "{{.Names}}").strip():
            raise RuntimeError("reference requires idle host")
        glimmer = dflash.probe.command("docker", "inspect", "-f", "{{.State.Running}}", "glimmer-tb21-prefix-c8").strip()
        if glimmer != "false":
            raise RuntimeError("Glimmer must remain stopped")
        guard = self.args.guard.read_bytes()
        if hashlib.sha256(guard).hexdigest() != dflash.probe.GUARD_SHA:
            raise RuntimeError("prefill guard changed")
        (self.out / "patch_uniform_decode_prefill.py").write_bytes(guard)
        (self.out / "persistent-launcher.sh").write_bytes(self.original)
        source = self.out / "reference-source" / "patches"
        source.mkdir(parents=True)
        patch_hashes = {}
        for name in PATCHES:
            shutil.copy2(PATCH_ROOT / name, source / name)
            patch_hashes[name] = hashlib.sha256((source / name).read_bytes()).hexdigest()
        for module in (Path(__file__), Path(dflash.__file__), Path(dflash.probe.__file__)):
            shutil.copy2(module, self.out / module.name)
        launcher = self.out / "launcher.sh"
        launcher.write_text(launcher_text(self.original, self.out))
        dflash.probe.command("bash", "-n", str(launcher))
        self.summary.update(mode="mtp4-reference", image=IMAGE, context=212992,
                            speculative_tokens=4, draft_quantization="S+M1 INT4", draft_dtype=None,
                            reference_patch_sha256=patch_hashes, prefill_guard_sha256=dflash.probe.GUARD_SHA,
                            glimmer_running_before=glimmer, model_runner="pinned reference default")
        self.log = (self.out / "server.log").open("w")
        self.proc = subprocess.Popen(["bash", str(launcher)], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"MTP reference exited during startup: {self.proc.returncode}")
            try:
                dflash.probe.get("/health")
                break
            except OSError:
                time.sleep(3)
        else:
            raise TimeoutError("MTP reference startup exceeded 900 seconds")
        models = json.loads(dflash.probe.get("/v1/models"))
        dflash.probe.save(self.out / "models.json", models)
        if not any(m["id"] == "qwen38" and m["max_model_len"] == 212992 for m in models["data"]):
            raise RuntimeError("wrong reference model/context")
        self.summary.update(models=models, status="running")
        print("CELL_READY=" + str(self.out), flush=True)
