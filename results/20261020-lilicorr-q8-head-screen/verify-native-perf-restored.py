#!/usr/bin/env python3
"""Read-only recheck using the existing exact-Id false/null Docker comparator."""
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-native-perf.py'))
OLD = next(HERE.parent / name / 'verify-native-perf-restored.py'
           for name in ('20261020-lilicorr-iq2-packed', '20261020-lilicorr-iq2-packed-r2')
           if (HERE.parent / name / 'verify-native-perf-restored.py').is_file())
V = runpy.run_path(str(OLD))


def main():
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == R['ROOT']
    runtime, c, _capture, cls = R['bind']()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == R['R']['S']['PARENT']
    before = c['load'](R['OUT'] / 'VSPF-W3a/container-before.json')
    assert before['Name'] == '/' + c['name']('VSPF-W3a')
    c['same_container'] = V['native_oom_comparator'](c['same_container'], before['Id'])
    destination = R['OUT'] / 'independent-original-restoration-oom-spelling.json'
    assert not destination.exists(), 'preserve all earlier verification records'
    record = cls(h, R['OUT']).verify_restored(destination)
    assert record['passed'] and not record['errors']
    record['comparison_spelling'] = dict(
        scope='only exact stopped native container Id', container_id=before['Id'],
        field='HostConfig.OomKillDisable', values=[False, None],
        original_container_unchanged=True, captured_records_unchanged=True,
        every_other_field_and_check='unchanged', comparator_sha256=c['sha'](OLD),
        verifier_sha256=c['sha'](Path(__file__)))
    c['save'](destination, record)
    print('Q8_HEAD_SCREEN_STRICT_ORIGINAL_AND_EXACT_ID_OOM_SPELLING_RECHECK_PASSED')


if __name__ == '__main__':
    main()
