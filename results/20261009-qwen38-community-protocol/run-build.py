#!/usr/bin/env python3
"""Benchmark our shared-KV build with the baseline's serving settings and inputs."""
import hashlib
import json
from pathlib import Path
import runpy
import shlex

ROOT = Path('/home/mike/b70-evals/qwen38-b70-gptq-int4-mtp4')
CAMPAIGN = ROOT / '20261009-qwen38-community-protocol'
PRIOR = ROOT / '20260913-qwen38-native-grouped-verify'


def main():
    previous = ROOT / '20261009-qwen38-native-shared-kv/run-cell.py'
    assert hashlib.sha256(previous.read_bytes()).hexdigest() == '58fc22ce410d84c6559c07109973568078e6ac80004a2c8c0769a25dc277fd09'
    sources = runpy.run_path(str(previous))['verify_sources']()
    adapter = runpy.run_path(str(PRIOR / 'run-serving.py'))
    patch, library = adapter['CANONICAL_PATCH'], adapter['DEFAULT_LIBRARY']
    assert adapter['sha256_file'](patch) == adapter['EXPECTED_CANONICAL_PATCH_SHA256']
    assert adapter['sha256_file'](library) == adapter['EXPECTED_LIBRARY_SHA256']
    production = Path('/home/mike/inference/launchers/start-qwen38.sh')
    original = production.read_text()
    assert hashlib.sha256(production.read_bytes()).hexdigest() == '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4'
    mounts = adapter['_candidate_mounts'](PRIOR / 'serving-overlay.py', patch, PRIOR / 'grouped_verify.py', library)
    environment = ['PYTHONPATH=/experiment', 'B70_STEP_TIMING=1', 'B70_GROUPED_SERVING=1',
                   'B70_GROUPED_SERVING_LIBRARY=/candidate/libb70_grouped_verify.so']
    additions = ''.join('  -v ' + shlex.quote(f"{m['host']}:{m['container']}:ro") + ' \\\n' for m in mounts)
    additions += ''.join('  -e ' + shlex.quote(value) + ' \\\n' for value in environment)
    marker = '  --entrypoint bash'
    hook = '; /opt/venv/bin/python -P /experiment/qwen38_step_timing_patch.py; exec vllm serve '
    assert original.count(marker) == original.count('; exec vllm serve ') == 1
    modified = original.replace(marker, additions + marker).replace('; exec vllm serve ', hook)
    assert modified.replace(additions, '', 1).replace(hook, '; exec vllm serve ', 1) == original
    launcher = CAMPAIGN / 'shared-kv-launch.sh'
    with launcher.open('x') as destination:
        destination.write(modified)
    launcher.chmod(0o700)
    run = runpy.run_path(str(CAMPAIGN / 'run.py'))
    ns = run['main'].__globals__
    out = CAMPAIGN / 'shared-kv-serving-01'
    ns.update(OUT=out, LAUNCHER=launcher,
              PREFILL_PROMPTS=CAMPAIGN / 'current-serving-02/prefill-prompts.json')

    def check_build(directory, final):
        text = (directory / 'server.log').read_text(errors='replace')
        eligible = [line for line in text.splitlines()
                    if '[B70_GROUPED_SERVING]' in line and '"event":"eligible-dispatch"' in line]
        assert eligible, 'Refusing to benchmark: no shared-KV dispatch evidence'
        assert any(adapter['EXPECTED_LIBRARY_SHA256'] in line for line in eligible), 'Wrong shared-KV library'
        assert '"event":"unsupported-q5"' not in text, 'Refusing native fallback instead of our build'
        (directory / 'build-contract.json').write_text(json.dumps({
            'candidate_library_sha256': adapter['EXPECTED_LIBRARY_SHA256'],
            'source_hashes': sources, 'mounts': mounts, 'environment': environment,
            'base_launcher_sha256': hashlib.sha256(production.read_bytes()).hexdigest(),
            'serving_settings': 'unchanged except shared-KV worker hook and library',
            'startup_eligible_dispatch': eligible, 'final_check': final}, indent=2) + '\n')
        print('Verified shared-KV build dispatch ' + ('after benchmark' if final else 'before measurements'), flush=True)
        if final:
            adapter['_candidate_execution_evidence'](directory)
            assert hashlib.sha256(production.read_bytes()).hexdigest() == '63b61b16bfcdb44bb5df9e0a7b1ee0b2666101951d9229b8b263c2c42fb38de4'
            runpy.run_path(str(previous))['verify_sources']()
    ns['CHECK_BUILD'] = check_build
    try:
        run['main']()
    finally:
        if out.is_dir():
            (out / 'run-build.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
