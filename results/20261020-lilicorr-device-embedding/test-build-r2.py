#!/usr/bin/env python3
"""Local CPU-only tests against recorded R4 linker/commands and generated graph."""
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
M = runpy.run_path(str(HERE / 'build-module-r2.py'))
R = M['expected']()


class ModuleBuildTests(unittest.TestCase):
    def closure(self, **changes):
        data = dict(link=R['link'], targets=R['targets'], queries=R['queries'],
                    commands='\n'.join(R['commands']), reference=R)
        data.update(changes)
        return M['validate_closure'](**data)

    def test_real_r4_link_selects_all_44_including_11_unity(self):
        argv, objects, commands = self.closure()
        self.assertEqual(R['objects'], objects)
        self.assertEqual(44, len(commands))
        self.assertEqual(11, sum('/Unity/' in x for x in objects))
        self.assertEqual(33, sum(x.endswith('.cpp.o') for x in objects))
        self.assertEqual(M['COMPILER'], argv[0])
        self.assertNotIn('&&', argv)
        M['check_plan'](R['plan'], objects)

    def test_missing_foreign_duplicate_unsafe_link_and_target_drift(self):
        first = R['objects'][0]
        bad = [
            R['link'].replace(first + ' ', '', 1),
            R['link'].replace(first, R['objects'][1]),
            R['link'].replace(first, 'ggml/src/ggml-sycl/kernel.cpp.o'),
            R['link'].replace(first, '@objects.rsp'),
            R['link'].replace(first, '../outside.cpp.o'),
            R['link'].replace('-o bin/libllama.so.0.7.0', '-o bin/libggml-sycl.so.0.27.0'),
            R['link'].replace('-O3', '-O2'),
            R['link'].replace('compiler/2026.0/bin/icpx', 'compiler/other/bin/icpx'),
            R['link'].replace('bin/libggml-base.so.0.27.0', '/foreign/libggml-base.so'),
            R['link'] + ' && echo unexpected',
            R['link'].replace(first, first + ' ; echo unexpected ;'),
        ]
        for line in bad:
            with self.subTest(line=line[:100]), self.assertRaises(AssertionError):
                M['link_argv'](line)
        # Same object count but altered object/order still fails the recorded exact linker.
        with self.assertRaises(AssertionError):
            self.closure(link=R['link'].replace('llama-context.cpp.o', 'foreign.cpp.o'))

    def test_targets_queries_and_compile_flags_must_agree(self):
        first = R['objects'][0]
        for targets in (R['targets'].replace(first + ': ' + M['RULE'] + '\n', ''),
                        R['targets'] + 'src/CMakeFiles/llama.dir/new.cpp.o: ' + M['RULE'] + '\n'):
            with self.assertRaises(AssertionError): self.closure(targets=targets)
        with self.assertRaises(AssertionError):
            self.closure(queries=R['queries'].replace('    ' + first + '\n', '', 1))
        with self.assertRaises(AssertionError):
            self.closure(queries=R['queries'].replace('/src/src/llama-context.cpp', '/foreign/input.cpp'))
        for commands in ('\n'.join(R['commands'][11:]),
                         '\n'.join(R['commands']) + '\necho foreign command',
                         '\n'.join(R['commands']).replace('-O3', '-O2', 1)):
            with self.assertRaises(AssertionError): self.closure(commands=commands)

    def test_plan_rejects_every_widening_or_incomplete_object_build(self):
        plan = R['plan']; objects = R['objects']
        for bad in ('ninja: no work to do.\n', '\n'.join(plan.splitlines()[11:]),
                    plan + '[45/45] Linking CXX shared library bin/libllama.so.0.7.0\n',
                    '[0/1] Re-running CMake...\n' + plan,
                    plan.replace(objects[0], 'ggml/src/kernel.cpp.o'),
                    plan.replace(objects[0], 'tools/server/ui.cpp.o'),
                    plan.replace(objects[0], 'common/common.cpp.o'),
                    plan + 'cp foreign.o /src/build-sycl/\n'):
            with self.assertRaises(AssertionError): M['check_plan'](bad, objects)
        M['check_plan'](plan + 'compiler warning: example\n', objects, completed=True)
        with self.assertRaises(AssertionError):
            M['check_plan'](plan + '[45/45] unexpected action\n', objects, completed=True)

    def simulated_main(self, bad_plan=False, corrupt_protected=False):
        """Exercise the actual driver boundary without any compiler/Ninja process."""
        calls = []
        def run(label, argv):
            calls.append((label, argv))
            return {
                'compiler-version': R['tools'].split('1.11.1\n')[0],
                'ninja-version': '1.11.1\n',
                'cmake-version': R['tools'].split('1.11.1\n')[1].split('libcurl4-openssl-dev:')[0],
                'generated-commands': 'irrelevant earlier dependency commands\n' + R['link'] + '\n',
                'targets': R['targets'], 'queries': R['queries'],
                'module-commands': '\n'.join(R['commands']),
                'module-plan': '[0/1] Re-running CMake...\n' if bad_plan else R['plan'],
                'module-build': R['plan'], 'module-link': '',
            }[label]
        def files(root, names):
            result = dict(names) if names == R['graph_sha256'] else {n: 'hash:' + n for n in names}
            if corrupt_protected and root == M['BUILD'] and M['PROTECTED'][0] in result:
                result[M['PROTECTED'][0]] = 'changed'
            return result
        with tempfile.TemporaryDirectory(prefix='.cpu-r2-', dir=HERE) as directory:
            with mock.patch.dict(M['main'].__globals__, OUT=Path(directory), run=run, file_map=files), \
                    mock.patch.object(Path, 'cwd', return_value=M['BUILD']):
                if bad_plan or corrupt_protected:
                    with self.assertRaises(AssertionError): M['main']()
                    self.assertNotIn('module-build', [label for label, _ in calls])
                    self.assertNotIn('module-link', [label for label, _ in calls])
                else:
                    M['main']()
                    self.assertEqual(['ninja', '-j6', *R['objects']], dict(calls)['module-build'])
                    self.assertEqual(M['link_argv'](R['link'])[0], dict(calls)['module-link'])
        return calls

    def test_driver_stops_before_compilation_on_bad_plan_or_library(self):
        self.simulated_main(bad_plan=True)
        self.simulated_main(corrupt_protected=True)

    def test_driver_executes_only_complete_module_then_validated_link_argv(self):
        calls = self.simulated_main()
        self.assertLess([x[0] for x in calls].index('module-plan'), [x[0] for x in calls].index('module-build'))
        source = (HERE / 'build-module-r2.py').read_text()
        self.assertNotIn('shell=True', source)
        self.assertNotIn('os.utime', source)
        self.assertNotIn('.ninja_log', source)
        self.assertNotIn('.ninja_deps', source)

    def test_r2_paths_frozen_sdk_and_nounset_safe_setvars(self):
        path = HERE / 'build-candidate-r2.sh'
        subprocess.run(['bash', '-n', str(path)], check=True)
        script = path.read_text()
        self.assertNotRegex(script, r'--device|/dev/dri|--privileged|cmake --build|cmake -S')
        self.assertIn('cp -a --reflink=auto "$PARENT" "$CANDIDATE"', script)
        self.assertIn('ATTEMPT=$ROOT/build-attempt-r2', script)
        self.assertIn('lilicorr-device-embedding-r2', script)
        self.assertNotIn('build-attempt-r1', script)
        self.assertEqual(2, script.count('set -eo pipefail\nsource /opt/intel/oneapi/setvars.sh --force\nset -u'))
        self.assertIn('aeb924ed73a4707576dcc6e9d9afb0f24435390a3f6596cdd88cee967dede0a0', script)
        runner = runpy.run_path(str(HERE / 'run-performance.py'))
        self.assertTrue(str(runner['S']['CANDIDATE']).endswith('device-embedding-r3'))
        self.assertEqual(['libllama.so.0.7.0'], runner['S']['CHANGED_BINARIES'])
        self.assertEqual(6, len(runner['S']['ARTIFACTS']))
        self.assertTrue(all(n.startswith('build-attempt-r3/') for n in runner['BUILD_FILES']))
        for name in ('build-candidate-r2.sh', 'build-module-r2.py', 'build-r2-inputs.json.gz'):
            self.assertIn(name, runner['OWN_FILES'])
        r3 = HERE / 'build-candidate-r3.sh'
        subprocess.run(['bash', '-n', str(r3)], check=True)
        expected = script.replace('build-attempt-r2', 'build-attempt-r3').replace('device-embedding-r2', 'device-embedding-r3')
        expected = expected.replace('device-embedding.patch', 'device-embedding-f32.patch')
        expected = expected.replace('Fresh R2 source; preserve failed R1 and all original source/build bytes.',
                                    'Fresh F32 R3 source; preserve R1/R2 and all original source/build bytes.')
        self.assertEqual(expected, r3.read_text())  # No new build recipe or relaxed closure.
        for name in ('build-candidate-r3.sh', 'source-hashes-f32.json', 'device-embedding-f32.patch'):
            self.assertIn(name, runner['OWN_FILES'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
