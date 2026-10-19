#!/usr/bin/env python3
"""Lead-only ABBA timing runner for the qualified W3 singleton adapter.

All public requests, native/startup checks, raw evidence, and cleanup are
inherited from the existing width-sweep lifecycle.  The adapter changes only
cell naming, current qualified references, and the four-cell W3 timing plan.
"""
import argparse
import json
import runpy
import socket
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
C = runpy.run_path(str(HERE / 'contracts.py'))
P = None
K = None

# Importing the old controller is read-only.  It can be unavailable in a
# developer checkout that has not staged the remote sibling; pure CPU
# fixtures still load this module and exercise the contracts.
try:
    _BOOT_K = runpy.run_path(str(C['WIDTH_SWEEP_PACKAGE'] / 'controller.py'))
except (FileNotFoundError, OSError, AssertionError, ImportError):
    _BOOT_K = None


if _BOOT_K is not None:
    _RuntimeBase = _BOOT_K['Runtime']
else:
    _RuntimeBase = object

if _BOOT_K is not None:
    _BOOT_K['C'] = C


def load_dependencies():
    """Load the qualified gate and patch only this adapter's K namespace."""
    global P, K
    if P is None:
        P = runpy.run_path(str(C['QUALIFIED_PACKAGE'] / 'public-gate.py'))
    if K is None:
        K = _BOOT_K if _BOOT_K is not None else runpy.run_path(
            str(C['WIDTH_SWEEP_PACKAGE'] / 'controller.py'))
    C['configure'](P)
    # K functions resolve C through their own globals.  This adapter-local
    # substitution gives inherited cleanup the new C.PREFIX without touching
    # the qualified P namespace or its original ancestry.
    K['C'] = C
    return P, K, P['C']['helpers']()


class Runtime(_RuntimeBase):
    """One exclusive timing lifecycle with fresh names and current references."""

    def __init__(self, h, out):
        if _BOOT_K is None:
            raise RuntimeError('remote width-sweep lifecycle is not staged')
        super().__init__(h, out, 'timing')
        self.mode = 'timing'
        self.h = h
        self.out = Path(out)
        self.off = h['off']
        self.reference_dir = None
        self.reference_label = 'qualified VSB-A'
        self.prevalidation_reports = {}
        self.adapter_hashes = {}

    def _cell_name(self, cell):
        return C['name'](cell)

    def _base_role(self, cell):
        return C['role_for'](cell)

    def _base_cell(self, cell):
        return C['base_cell'](cell)

    def preflight(self):
        """Run every immutable P proof before touching the original service."""
        assert P is not None and K is not None
        self.capacity('initial')
        # These calls intentionally remain the qualified public-gate routines;
        # no P.ROOT/ROLES/ancestry monkeypatch is permitted.
        candidate = P['candidate_proof']()
        n9 = P['check_n9_equivalence'](self.h)
        frozen = P['freeze'](self.h)
        tested = P['check_tested'](self.h)
        qualification = {}
        for base, role in (('VSB-A', 'A'), ('VSB-W3', 'W3'), ('VSB-W7', 'W7')):
            report = C['load'](C['qualified_dir'](role) / 'qualification.json')
            qualification[base] = dict(passed=report['passed'], jobs=len(report['jobs']),
                                      fidelity_failures=report.get('fidelity_failures', []))
        assert qualification['VSB-A']['passed'] and qualification['VSB-A']['jobs'] == 25
        assert qualification['VSB-W3']['passed'] and qualification['VSB-W3']['jobs'] == 25
        assert not qualification['VSB-W7']['passed'], 'W7 is excluded from timing'
        C['save'](self.out / 'qualified-eligibility.json', qualification)
        C['eligible_order']({3})
        self.adapter_hashes = C['adapter_code_hashes']()
        self.hashes = C['source_binary_hashes'](self.h)
        self.before_hw = self.h['runner']['hardware']()
        reference_hw = C['load'](self.off['REFERENCE'] / 'run-r1/hardware-before.json')
        assert self.before_hw == reference_hw, 'hardware/capability snapshot differs from prerequisite'
        live = self.inspect(self.original_name)
        frozen_original = C['load'](self.off['ROOT'] / 'run-r1/original-container.json')
        assert C['same_container'](frozen_original, live)
        assert live['State']['Running']
        self.original = live
        self.original_was_running = True
        assert self.h['api']['request'](self.off['OLD']['URL'], '/health')['status'] == 'ok'
        existing = self.command('docker', 'ps', '-a', '--format', '{{.Names}}').splitlines()
        C['save'](self.out / 'planned-containers.json', C['check_name_collisions'](existing))
        C['save'](self.out / 'original-container.json', live)
        C['save'](self.out / 'hash-checks-before.json', self.hashes)
        C['save'](self.out / 'adapter-code-hashes-before.json', self.adapter_hashes)
        C['save'](self.out / 'source-contracts.json', C['source_evidence'](self.h))
        C['save'](self.out / 'qualified-proof.json', dict(candidate=candidate, n9=n9,
                                                         freeze=frozen, tested=tested))
        env = dict(
            kernel=self.command('uname', '-a'),
            os_release=Path('/etc/os-release').read_text(),
            cpu=self.command('lscpu'),
            memory=Path('/proc/meminfo').read_text(),
            driver=self.command('modinfo', 'xe'),
            source_ids={str(source): (source / 'UPSTREAM_COMMIT').read_text()
                        for source in (P['N']['ACCEPTED'], P['PARENT'], P['CANDIDATE'])},
        )
        C['save'](self.out / 'environment.json', env)
        C['save'](self.out / 'original-live.json', live)
        self.observe('run-before', self.original_name)

    def begin(self, cell):
        assert cell in C['CELLS'] and self.active is None
        self.capacity(cell)
        if not self.prefixes:
            self._stop_original_for_case()
        else:
            assert not self.inspect(self.original_name)['State']['Running']
            self.ownership(None)
        directory = self.out / cell
        directory.mkdir()
        argv = C['make_create'](cell, self.h)
        base_argv = P['make_create'](self._base_cell(cell), self.h)
        C['save'](directory / 'create-argv.json', argv)
        C['save'](directory / 'argv-diff.json', C['argv_diff'](base_argv, argv))
        # Account for successful docker create even if start/ready/transport
        # fails; inherited restore must attempt every created name.
        self.created.append(cell)
        self.command(*argv, timeout=60)
        live = self.inspect(self._cell_name(cell))
        C['check_container'](live, cell, self.h)
        self.command('docker', 'start', self._cell_name(cell))
        C['save'](directory / 'models.json', self.ready(self.h['api'], self._cell_name(cell)))
        self.save_logs(cell, 'native-and-startup', self._cell_name(cell))
        raw = subprocess.run(['docker', 'logs', self._cell_name(cell)], capture_output=True,
                             timeout=90, check=True)
        startup = (raw.stdout + raw.stderr).decode('utf-8')
        contract = P['check_startup'](startup, self._base_cell(cell), self.h)
        self.preserve_native(cell, 'ready', required=True)
        contract['timing_cell'] = cell
        contract['pid_argv'] = self.pid(cell, 'before')['argv']
        C['save'](directory / 'startup.json', contract)
        self.observe(cell + '-ready', self._cell_name(cell))
        C['save'](directory / 'startup-buffers.json', [
            line for line in startup.splitlines()
            if any(word in line.lower() for word in ('buffer size', 'kv cache size', 'recurrent'))])
        C['save'](directory / 'container-before.json', self.inspect(self._cell_name(cell)))
        self.active = cell

    def pid(self, cell, phase):
        name = self._cell_name(cell)
        probe = P['C']['PID_PROBE']
        record = json.loads(self.command('docker', 'exec', name, 'python3', '-B', '-c', probe))
        C['save'](self.out / cell / ('pid-' + phase + '.json'), record)
        # Validate the actual unmodified record against the role's qualified
        # PID contract; never rewrite its argv or environment for a pass.
        P['C']['check_pid'](record, self._base_role(cell), self.h)
        maps = self.command('docker', 'exec', name, 'cat', '/proc/1/maps')
        (self.out / cell / ('libraries-' + phase + '.txt')).write_text(
            '\n'.join(sorted({line.split()[-1] for line in maps.splitlines() if '.so' in line})) + '\n')
        return record

    def preserve_native(self, cell, phase, required=False):
        errors = []
        for tag, filename in P['NATIVE_LOGS'].items():
            destination = self.out / cell / (phase + '-' + filename)
            try:
                assert not destination.exists()
                self.command('docker', 'cp', self._cell_name(cell) + ':/tmp/' + filename,
                             str(destination))
            except Exception as exc:  # Preserve every missing artifact as evidence.
                errors.append(dict(tag=tag, error=repr(exc)))
        C['save'](self.out / cell / (phase + '-native-copy.json'), dict(errors=errors))
        if required:
            assert not errors, errors
            for tag, filename in P['NATIVE_LOGS'].items():
                P['check_native_log'](tag, (self.out / cell / (phase + '-' + filename)).read_bytes().decode())

    def _reference(self, job):
        assert self.reference_dir is not None
        stem = self.reference_dir / job['stem']
        raw = self.h['analysis']['read_http'](stem, job['payload'], '/v1/chat/completions')
        if C['qualification_kinds'](job) == 'diagnostic':
            prompt = self.off['reference_prompt']('A0', job)
            return raw, self.off['diagnostic_view'](json.loads(raw), prompt)
        return raw, self.h['analysis']['normalize'](raw, job['payload'])

    def _reference_result(self, directory, job):
        return P['read_result'](self.h, directory, job)

    def _reference_directories(self, cell):
        frozen = self.off['ROOT'] / 'run-r1/A0'
        if cell == 'VSPF-A1':
            fresh = C['qualified_dir']('A')
            label = 'qualified VSB-A'
        else:
            fresh = self.out / 'VSPF-A1'
            label = 'fresh timing VSPF-A1'
        return frozen, fresh, label

    def prevalidate_timing_cell(self, cell):
        """Run all 25 P fidelity/function checks before any timed request."""
        base = self._base_cell(cell)
        frozen_dir, fresh_dir, label = self._reference_directories(cell)
        # K.run_job uses this matched reference for its ordinary/diagnostic
        # output checks.  For the first A load it is the qualified VSB-A; all
        # later arms use the already-captured fresh A1 raw responses.
        self.reference_dir = fresh_dir
        self.reference_label = label
        report = P['new_report'](base)
        report.update(timing_cell=cell, timing_reference=label,
                      frozen_reference='original A0', actual_cell=cell)
        try:
            for job in C['qualification_jobs'](self.h):
                try:
                    result = self.run_job(cell, job, expected_for_compare=False)
                except C['FidelityFailure']:
                    # The transport/raw artifacts remain authoritative.  Read
                    # them back through P rather than manufacturing a result.
                    result = P['read_result'](self.h, self.out / cell, job)
                frozen = self._reference_result(frozen_dir, job)
                fresh = self._reference_result(fresh_dir, job)
                P['add_result'](report, job, result, frozen, fresh)
                C['save'](self.out / cell / 'prevalidation.json', report)
            P['finish_report'](report)
            assert len(report['jobs']) == 25
            assert report['passed'], report
        except BaseException as exc:
            report['errors'].append(dict(error=repr(exc)))
            C['save'](self.out / cell / 'prevalidation.json', report)
            raise C['InfraFailure']('timing prevalidation failed for ' + cell) from exc
        C['save'](self.out / cell / 'prevalidation.json', report)
        C['save'](self.out / cell / 'timing-gate.json', dict(
            passed=True, timing_not_started=True, timing_cell=cell,
            reference=label, jobs=25, rejected_draft_observed=(
                report['behavior']['rejected_draft_observed'] > 0)))
        self.prevalidation_reports[cell] = report
        return report

    def _preserve_restore_log(self, cell, container_name):
        try:
            super()._preserve_restore_log(cell, container_name)
        finally:
            # docker cp works on stopped containers, including partial startup.
            self.preserve_native(cell, 'restored')

    def stopped_cells(self):
        result = {}
        for cell in C['CELLS']:
            directory = self.out / cell
            if not directory.exists():
                continue
            live = self.inspect(self._cell_name(cell))
            C['check_container'](live, cell, self.h)
            assert not live['State']['Running']
            if (directory / 'container-before.json').exists():
                assert C['same_container'](C['load'](directory / 'container-before.json'), live)
            result[cell] = live
        return result

    def verify_restored(self, destination):
        """Use inherited live restoration, then independently repeat P proofs."""
        record = super().verify_restored(destination)
        errors = []
        checks = (
            ('p_candidate_proof', lambda: P['candidate_proof']()),
            ('p_n9_equivalence', lambda: P['check_n9_equivalence'](self.h)),
            ('p_tested_code', lambda: P['check_tested'](self.h)),
            ('adapter_code', lambda: C['adapter_code_hashes']() ==
             C['load'](self.out / 'adapter-code-hashes-before.json')),
            ('stopped_cells', self.stopped_cells),
        )
        for key, fn in checks:
            try:
                record[key] = fn()
            except BaseException as exc:
                errors.append(dict(check=key, error=repr(exc)))
        record['errors'] += errors
        record['passed'] = not record['errors']
        C['save'](destination, record)
        if errors:
            raise C['InfraFailure']('restoration/qualified proof failed: ' + repr(errors))
        return record


def capture(runtime):
    """Delegate lifecycle/signal/finally semantics to the existing K.capture."""
    if K is None:
        raise RuntimeError('remote width-sweep lifecycle is not staged')
    return K['capture'](runtime, C['CELLS'], lambda rt, cell: rt.run_timing_cell(cell))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lead-run-timing', action='store_true', required=True)
    parser.parse_args()
    assert __debug__ and socket.gethostname() == 'inference-host'
    assert HERE == C['ROOT'], (HERE, C['ROOT'])
    p, k, h = load_dependencies()
    out = C['ROOT'] / 'run-r1'
    out.mkdir(exist_ok=False)
    (out / 'declaration.md').write_text((HERE / 'README.md').read_text())
    summary = capture(Runtime(h, out))
    assert summary['status'] == 'captured' and summary['restored'], summary
    print('W3_ABBA_TIMING_CAPTURED_RESTORED_ANALYSIS_PENDING', flush=True)


if __name__ == '__main__':
    main()
