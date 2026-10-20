#!/usr/bin/env python3
"""CPU-only guard tests for the frozen GGML-SYCL module build boundary."""
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
M = runpy.run_path(str(HERE / 'build-module.py'))
R = M['expected']()
S = runpy.run_path(str(HERE / 'source-proof.py'))


class ModuleBuildTests(unittest.TestCase):
    def closure(self, **changes):
        data = dict(link=R['link'], targets=R['targets'], queries=R['queries'],
                    commands='\n'.join(R['commands']), reference=R)
        data.update(changes)
        return M['validate_closure'](**data)

    def test_real_link_selects_all_105_with_the_frozen_split(self):
        argv, objects, commands = self.closure()
        self.assertEqual(R['objects'], objects)
        self.assertEqual(105, len(commands))
        self.assertEqual(59, sum('/template-instances/' not in x for x in objects))
        self.assertEqual(46, sum('/template-instances/' in x for x in objects))
        self.assertEqual(M['COMPILER'], argv[0])
        self.assertNotIn('&&', argv)
        self.assertEqual('-Xs', argv[4])
        self.assertEqual('-ze-intel-greater-than-4GB-buffer-required', argv[5])
        M['check_plan'](R['plan'], objects)

    def test_link_rejects_missing_foreign_duplicate_unsafe_and_flag_drift(self):
        first = R['objects'][0]
        bad = [
            R['link'].replace(first + ' ', '', 1),
            R['link'].replace(first, R['objects'][1]),
            R['link'].replace(first, 'ggml/src/ggml-cpu/foreign.cpp.o'),
            R['link'].replace(first, '@objects.rsp'),
            R['link'].replace(first, '../outside.cpp.o'),
            R['link'].replace('-O3', '-O2'),
            R['link'].replace('compiler/2026.0/bin/icpx', 'compiler/other/bin/icpx'),
            R['link'].replace('bin/libggml-base.so.0.27.0', '/foreign/libggml-base.so'),
            R['link'] + ' && echo unexpected',
            R['link'].replace(first, first + ' ; echo unexpected ;'),
        ]
        for line in bad:
            with self.subTest(line=line[:120]), self.assertRaises(AssertionError):
                M['link_argv'](line)
        with self.assertRaises(AssertionError):
            self.closure(link=R['link'].replace('mmvq.cpp.o', 'foreign.cpp.o'))

    def test_targets_queries_and_commands_are_a_closed_105_object_set(self):
        first = R['objects'][0]
        for targets in (
            R['targets'].replace(first + ': ' + M['RULE'] + '\n', ''),
            R['targets'] + M['PREFIX'] + 'new.cpp.o: ' + M['RULE'] + '\n',
            R['targets'].replace(first + ': ' + M['RULE'], first + ': WRONG_RULE'),
        ):
            with self.subTest(targets=targets[-100:]), self.assertRaises(AssertionError):
                self.closure(targets=targets)
        for queries in (
            R['queries'].replace('    ' + first + '\n', '', 1),
            R['queries'].replace('/src/ggml/src/ggml-sycl/ggml-sycl.cpp', '/foreign/input.cpp'),
            R['queries'].replace('CXX_COMPILER__ggml-sycl_unscanned_Release', 'FOREIGN_RULE', 1),
        ):
            with self.subTest(queries=queries[:120]), self.assertRaises(AssertionError):
                self.closure(queries=queries)
        for commands in (
            '\n'.join(R['commands'][1:]),
            '\n'.join(R['commands']) + '\necho foreign command',
            '\n'.join(R['commands']).replace('-O3', '-O2', 1),
            '\n'.join([R['commands'][1], R['commands'][0], *R['commands'][2:]]),
        ):
            with self.subTest(commands=commands[:120]), self.assertRaises(AssertionError):
                self.closure(commands=commands)

    def test_plan_rejects_widening_reordering_and_incomplete_actions(self):
        objects = R['objects']
        bad = (
            'ninja: no work to do.\n',
            '\n'.join(R['plan'].splitlines()[1:]),
            R['plan'] + '[106/106] Linking CXX shared library bin/libggml-sycl.so.0.27.0\n',
            '[0/1] Re-running CMake...\n' + R['plan'],
            R['plan'].replace(objects[0], 'ggml/src/foreign.cpp.o'),
            R['plan'].replace(objects[0], 'tools/server/ui.cpp.o'),
            R['plan'] + 'cp foreign.o /src/build-sycl/\n',
        )
        for plan in bad:
            with self.subTest(plan=plan[:100]), self.assertRaises(AssertionError):
                M['check_plan'](plan, objects)
        M['check_plan'](R['plan'] + 'compiler warning: example\n', objects, completed=True)
        with self.assertRaises(AssertionError):
            M['check_plan'](R['plan'] + '[106/106] unexpected action\n', objects, completed=True)

    def simulated_main(self, bad_plan=False, corrupt_protected=False, bad_query=False):
        """Exercise the driver boundary without Ninja, compiler, or a container."""
        calls = []
        frozen_tools = R['tools'].split('libcurl4-openssl-dev:', 1)[0]
        compiler, remainder = frozen_tools.split('1.11.1\n', 1)

        def run(label, argv):
            calls.append((label, argv))
            return {
                'compiler-version': compiler,
                'ninja-version': '1.11.1\n',
                'cmake-version': remainder,
                'generated-commands': 'irrelevant earlier dependency commands\n' + R['link'] + '\n',
                'targets': R['targets'],
                'queries': R['queries'] if not bad_query else R['queries'].replace(
                    '    ' + R['objects'][0] + '\n', '', 1),
                'module-commands': '\n'.join(R['commands']),
                'module-plan': ('[0/1] Re-running CMake...\n' if bad_plan else R['plan']),
                'module-build': R['plan'],
                'module-link': '',
            }[label]

        def files(root, names):
            result = dict(names) if names == R['graph_sha256'] else {n: 'hash:' + n for n in names}
            if corrupt_protected and root == M['BUILD'] and M['PRESERVED']:
                result[M['PRESERVED'][0]] = 'changed'
            return result

        with tempfile.TemporaryDirectory(prefix='.cpu-iq2-', dir=HERE) as directory:
            with mock.patch.dict(M['main'].__globals__, OUT=Path(directory), run=run,
                                 file_map=files, symlink_map=lambda root: {}), \
                 mock.patch.object(Path, 'cwd', return_value=M['BUILD']):
                if bad_plan or corrupt_protected or bad_query:
                    with self.assertRaises(AssertionError):
                        M['main']()
                    self.assertNotIn('module-build', [label for label, _ in calls])
                    self.assertNotIn('module-link', [label for label, _ in calls])
                else:
                    M['main']()
                    self.assertEqual(['ninja', '-j6', *R['objects']], dict(calls)['module-build'])
                    self.assertEqual(M['link_argv'](R['link'])[0], dict(calls)['module-link'])
        return calls

    def test_driver_stops_before_build_on_bad_plan_library_or_query(self):
        self.simulated_main(bad_plan=True)
        self.simulated_main(corrupt_protected=True)
        self.simulated_main(bad_query=True)

    def test_driver_uses_only_direct_argv_and_preserves_metadata(self):
        calls = self.simulated_main()
        labels = [label for label, _ in calls]
        self.assertLess(labels.index('module-plan'), labels.index('module-build'))
        source = (HERE / 'build-module.py').read_text()
        self.assertNotIn('shell=True', source)
        self.assertNotIn('os.utime', source)
        self.assertNotIn('.ninja_log', source)
        self.assertNotIn('.ninja_deps', source)
        self.assertNotIn('cmake --build', source)
        self.assertNotIn('cmake -S', source)

    def test_case_input_and_source_patch_are_pinned(self):
        self.assertEqual(105, len(R['objects']))
        self.assertEqual(105, len(R['commands']))
        self.assertEqual(59, sum('/template-instances/' not in x for x in R['objects']))
        self.assertEqual(46, sum('/template-instances/' in x for x in R['objects']))
        self.assertEqual(['libggml-sycl.so.0'], S['CHANGED_BINARIES'])
        self.assertEqual(['ggml/src/ggml-sycl/ggml-sycl.cpp',
                          'ggml/src/ggml-sycl/mmvq.cpp',
                          'ggml/src/ggml-sycl/mmvq.hpp'], S['SOURCE_SCOPE'])
        self.assertEqual(74, S['patch_stats']()['additions'])
        self.assertEqual(0, S['patch_stats']()['deletions'])

    def test_shell_wrapper_is_lead_only_and_cpu_safe(self):
        path = HERE / 'build-candidate.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        script = path.read_text()
        self.assertIn('iq2-batch4-r1', script)
        self.assertIn('aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0', script)
        self.assertIn('cp -a --reflink=auto "$PARENT" "$CANDIDATE"', script)
        self.assertIn('python3 -B /package/build-module.py', script)
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged|cmake --build|cmake -S')
        self.assertEqual(2, script.count('set -eo pipefail\nsource /opt/intel/oneapi/setvars.sh --force\nset -u'))
        self.assertNotIn('docker push', script)
        self.assertNotIn('ssh ', script)


if __name__ == '__main__':
    unittest.main(verbosity=2)
