#!/usr/bin/env python3
"""Repeat strict live restoration; normalize only stopped native false/null OOM metadata."""
import copy
import runpy
import socket
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = runpy.run_path(str(HERE / 'run-native-perf.py'))


def native_oom_comparator(original, ident):
    def compare(expected, actual):
        if expected.get('Id') == actual.get('Id') == ident:
            expected, actual = copy.deepcopy(expected), copy.deepcopy(actual)
            for record in (expected, actual):
                value = record['HostConfig']['OomKillDisable']
                assert value is None or value is False, value
                record['HostConfig']['OomKillDisable'] = None
        return original(expected, actual)
    return compare


def main():
    assert __debug__ and socket.gethostname() == 'inference-host' and HERE == R['R']['ROOT']
    runtime, c, _capture, cls = R['bind']()
    p, _k, h = runtime['load_dependencies']()
    assert p['CANDIDATE'] == R['R']['S']['PARENT']
    before = c['load'](R['OUT'] / 'VSPF-W3a/container-before.json')
    assert before['Name'] == '/' + c['name']('VSPF-W3a')
    c['same_container'] = native_oom_comparator(c['same_container'], before['Id'])
    destination = R['OUT'] / 'independent-original-restoration-oom-normalized.json'
    assert not destination.exists(), 'preserve earlier verification records'
    record = cls(h, R['OUT']).verify_restored(destination)
    assert record['passed'] and not record['errors']
    record['normalization'] = dict(scope='only exact stopped native container Id',
        container_id=before['Id'], field='HostConfig.OomKillDisable', values=[False, None],
        original_container_unchanged=True, every_other_field_and_check='unchanged',
        verifier_sha256=c['sha'](Path(__file__)))
    c['save'](destination, record)
    print('NATIVE_PERF_NEW_PROCESS_STRICT_ORIGINAL_AND_NORMALIZED_STOPPED_CONTAINER_PASSED')


if __name__ == '__main__':
    main()
