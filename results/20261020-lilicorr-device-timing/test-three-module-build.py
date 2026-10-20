#!/usr/bin/env python3
"""CPU-only checks for the three-module frozen-SDK build boundary."""
import io
import runpy
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
M = runpy.run_path(str(HERE / 'build-three-modules.py'))
R = M['expected']()


def plan(objects):
    total = len(objects)
    return '\n'.join(
        f'[{number}/{total}] Building CXX object {name}'
        for number, name in enumerate(objects, 1)
    ) + '\n'


class ThreeModuleBuildTests(unittest.TestCase):
    def test_reference_has_the_three_real_closures(self):
        self.assertEqual(
            M['MODULE_ORDER'],
            (
                'bin/libggml-sycl.so.0.27.0',
                'bin/libllama.so.0.7.0',
                'bin/libllama-common.so.0.7.0',
            ),
        )
        self.assertEqual(
            [len(R['modules'][target]['objects']) for target in M['MODULE_ORDER']],
            [105, 44, 55],
        )
        self.assertEqual(204, sum(len(R['modules'][target]['objects']) for target in M['MODULE_ORDER']))
        ggml = R['modules'][M['MODULE_ORDER'][0]]['objects']
        llama = R['modules'][M['MODULE_ORDER'][1]]['objects']
        self.assertEqual(59, sum('/template-instances/' not in name for name in ggml))
        self.assertEqual(46, sum('/template-instances/' in name for name in ggml))
        self.assertEqual(11, sum('/Unity/' in name for name in llama))
        self.assertTrue(any('/Unity/unity_10_cxx.cxx.o' in name for name in llama))
        self.assertFalse(set(M['MODULE_ORDER']) & set(M['PRESERVED']))
        self.assertEqual(
            set(R['graph_sha256']),
            {
                'build.ninja',
                'CMakeFiles/rules.ninja',
                'CMakeCache.txt',
                'CMakeFiles/3.28.3/CMakeCXXCompiler.cmake',
            },
        )

    def test_each_real_link_argv_is_exact_and_rejects_drift(self):
        for target in M['MODULE_ORDER']:
            with self.subTest(target=target):
                frozen = R['modules'][target]['link']
                argv, objects = M['link_argv'](frozen, target, R)
                self.assertEqual(objects, R['modules'][target]['objects'])
                self.assertEqual(M['COMPILER'], argv[0])
                first, second = objects[:2]
                bad = (
                    frozen.replace('-O3', '-O2', 1),
                    frozen.replace(first, 'foreign.cpp.o', 1),
                    frozen.replace(f'{first} {second}', f'{second} {first}', 1),
                    frozen.replace(first + ' ', '', 1),
                    frozen + ' && echo unexpected',
                    frozen.replace(M['COMPILER'], '/other/icpx', 1),
                    frozen.replace(target, 'bin/foreign.so', 1),
                )
                for line in bad:
                    with self.subTest(line=line[:100]), self.assertRaises(AssertionError):
                        M['link_argv'](line, target, R)

    def test_each_module_edges_commands_and_rules_match_capture(self):
        for target in M['MODULE_ORDER']:
            with self.subTest(target=target):
                module = R['modules'][target]
                M['validate_closure'](
                    module['link'],
                    R['targets'],
                    module['queries'],
                    '\n'.join(module['commands']),
                    R,
                    target,
                )
                first = module['objects'][0]
                bad_queries = module['queries'].replace('    ' + first + '\n', '', 1)
                with self.assertRaises(AssertionError):
                    M['validate_closure'](
                        module['link'], R['targets'], bad_queries,
                        '\n'.join(module['commands']), R, target,
                    )
                with self.assertRaises(AssertionError):
                    M['validate_closure'](
                        module['link'], R['targets'], module['queries'],
                        '\n'.join(module['commands']).replace('-O3', '-O2', 1), R, target,
                    )
                target_line = first + ': ' + M['MODULES'][target]['rule'] + '\n'
                with self.assertRaises(AssertionError):
                    M['validate_closure'](
                        module['link'], R['targets'].replace(target_line, '', 1),
                        module['queries'], '\n'.join(module['commands']), R, target,
                    )
                extra = M['MODULES'][target]['prefix'] + 'foreign.cpp.o: ' + M['MODULES'][target]['rule']
                with self.assertRaises(AssertionError):
                    M['validate_closure'](
                        module['link'], R['targets'] + extra + '\n', module['queries'],
                        '\n'.join(module['commands']), R, target,
                    )

    def test_plan_is_module_only_and_requires_every_object(self):
        for target in M['MODULE_ORDER']:
            objects = R['modules'][target]['objects']
            with self.subTest(target=target):
                M['check_plan'](plan(objects), objects)
                M['check_plan'](plan(objects) + 'compiler warning: retained', objects, completed=True)
                bad = (
                    '\n'.join(plan(objects).splitlines()[1:]) + '\n',
                    plan(objects) + '[999/1] Linking CXX shared library bin/foreign.so\n',
                    plan(objects).replace(objects[0], 'foreign.cpp.o', 1),
                    plan(objects).replace(objects[0] + '\n', '', 1),
                    'ninja: no work to do.\n',
                    'Re-running CMake...\n' + plan(objects),
                )
                for text in bad:
                    with self.subTest(text=text[:100]), self.assertRaises(AssertionError):
                        M['check_plan'](text, objects)

    def test_driver_stops_before_build_on_bad_metadata_or_plan(self):
        def run_driver(mode=None):
            calls = []
            tools = R['tools'].split('libcurl4-openssl-dev:', 1)[0]
            compiler, remainder = tools.split('1.11.1\n', 1)
            link_lines = {target: R['modules'][target]['link'] for target in M['MODULE_ORDER']}

            def fake_run(label, argv):
                calls.append((label, list(argv)))
                if argv == [M['COMPILER'], '--version']:
                    return compiler
                if argv == ['ninja', '--version']:
                    return '1.11.1\n'
                if argv == ['cmake', '--version']:
                    return remainder
                if argv == ['ninja', '-t', 'targets', 'all']:
                    return R['targets']
                if argv[:3] == ['ninja', '-t', 'commands'] and len(argv) == 4:
                    target = argv[3]
                    return 'earlier compile command\n' + link_lines[target] + '\n'
                if argv[:3] == ['ninja', '-t', 'query']:
                    target = argv[3]
                    text = R['modules'][target]['queries']
                    if mode == 'bad-query' and target == M['MODULE_ORDER'][0]:
                        text = text.replace(
                            '    ' + R['modules'][target]['objects'][0] + '\n', '', 1
                        )
                    return text
                if argv[:3] == ['ninja', '-t', 'commands']:
                    objects = argv[3:]
                    target = next(
                        target for target in M['MODULE_ORDER']
                        if R['modules'][target]['objects'] == objects
                    )
                    return '\n'.join(R['modules'][target]['commands']) + '\n'
                if argv[:5] == ['ninja', '-n', '-d', 'explain', '-j6']:
                    objects = argv[5:]
                    text = plan(objects)
                    if mode == 'bad-plan' and objects == R['modules'][M['MODULE_ORDER'][0]]['objects']:
                        text += '[999/1] Linking CXX shared library bin/foreign.so\n'
                    return text
                if argv[:2] == ['ninja', '-j6']:
                    return plan(argv[2:])
                if argv and argv[0] == M['COMPILER']:
                    return ''
                raise AssertionError(('unexpected driver call', label, argv))

            def files(root, names):
                if names == R['graph_sha256']:
                    return dict(names)
                values = {name: 'hash:' + name for name in names}
                if mode == 'bad-preserved' and root == M['BUILD']:
                    values[M['PRESERVED'][0]] = 'changed'
                return values

            symlinks = {'bin/libllama.so.0': 'libllama.so.0.7.0'}
            with tempfile.TemporaryDirectory(prefix='.cpu-three-module-', dir=HERE) as directory:
                with mock.patch.dict(
                    M['main'].__globals__,
                    OUT=Path(directory), run=fake_run, file_map=files,
                    symlink_map=lambda root: symlinks,
                ), mock.patch.object(Path, 'cwd', return_value=M['BUILD']):
                    if mode:
                        with self.assertRaises(AssertionError):
                            with redirect_stdout(io.StringIO()):
                                M['main']()
                    else:
                        with redirect_stdout(io.StringIO()):
                            M['main']()
            return calls

        calls = run_driver()
        builds = [argv for label, argv in calls if label.startswith('module-build-')]
        links = [argv for label, argv in calls if label.startswith('module-link-')]
        self.assertEqual(
            builds,
            [['ninja', '-j6', *R['modules'][target]['objects']] for target in M['MODULE_ORDER']],
        )
        self.assertEqual(
            links,
            [M['link_argv'](R['modules'][target]['link'], target, R)[0]
             for target in M['MODULE_ORDER']],
        )
        self.assertEqual(
            [label for label, _ in calls if label.startswith('module-link-')],
            ['module-link-' + target.replace('/', '-').replace('.', '-') for target in M['MODULE_ORDER']],
        )
        for mode in ('bad-plan', 'bad-query', 'bad-preserved'):
            failed = run_driver(mode)
            self.assertFalse(any(label.startswith('module-build-') for label, _ in failed))
            self.assertFalse(any(label.startswith('module-link-') for label, _ in failed))

    def test_driver_source_is_cpu_safe_and_does_not_regenerate_the_graph(self):
        source = (HERE / 'build-three-modules.py').read_text()
        for forbidden in (
            'shell=True',
            'os.utime',
            '.ninja_log',
            '.ninja_deps',
            'cmake -S',
            'cmake --build',
            '/dev/dri',
            'docker',
            'ssh ',
            'DPCT_PROFILING_ENABLED',
            'DNNL_EXPERIMENTAL_PROFILING',
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("subprocess.run(argv, cwd=BUILD", source)
        self.assertIn('ninja', source)
        self.assertIn('assert total_objects == 204', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
