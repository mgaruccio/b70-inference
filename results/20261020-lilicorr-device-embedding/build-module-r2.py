#!/usr/bin/env python3
"""Case-local SDK driver: existing 44 llama object rules + existing generated link."""
import gzip
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path('/src/build-sycl')
PARENT = Path('/qualified/build-sycl')
OUT = Path('/build-evidence')
COMPILER = '/opt/intel/oneapi/compiler/2026.0/bin/icpx'
TARGET = 'bin/libllama.so.0.7.0'
PREFIX = 'src/CMakeFiles/llama.dir/'
RULE = 'CXX_COMPILER__llama_unscanned_Release'
OBJECT = re.compile(r'src/CMakeFiles/llama\.dir/(?:Unity/unity_\d+_cxx\.cxx|[A-Za-z0-9_-]+\.cpp)\.o')
LIBRARIES = ['bin/libggml.so.0.27.0', 'bin/libggml-cpu.so.0.27.0',
             'bin/libggml-sycl.so.0.27.0', 'bin/libggml-base.so.0.27.0']
PROTECTED = ['bin/' + n for n in ('llama-server', 'libllama-server-impl.so',
             'libggml-sycl.so.0', 'test-backend-ops', 'libllama-common.so')]
INPUTS_SHA256 = 'f7339edafa6a3d83e434a1f49dac379ee2044fa7c331660d9d951f4090d90ca4'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected():
    path = HERE / 'build-r2-inputs.json.gz'
    assert sha(path) == INPUTS_SHA256
    return json.loads(gzip.decompress(path.read_bytes()))


def link_argv(line):
    tokens = shlex.split(line)
    assert tokens[:2] == [':', '&&'] and tokens[-2:] == ['&&', ':'], 'unexpected link wrapper'
    argv = tokens[2:-2]
    prefix = [COMPILER, '-fPIC', '-O3', '-DNDEBUG', '-shared',
              '-Wl,-soname,libllama.so.0', '-o', TARGET]
    suffix = ['-Wl,-rpath,/src/build-sycl/bin:/opt/onednn/lib:', *LIBRARIES,
              '-Wl,-rpath-link,/opt/onednn/lib']
    assert argv[:len(prefix)] == prefix and argv[-len(suffix):] == suffix, 'link/compiler/library drift'
    objects = argv[len(prefix):-len(suffix)]
    assert len(objects) == len(set(objects)) == 44, 'incomplete/duplicate module closure'
    assert all(OBJECT.fullmatch(name) for name in objects), 'foreign object or unsafe link argument'
    assert sum('/Unity/' in name for name in objects) == 11, 'Unity objects are required'
    return argv, objects


def query_map(text):
    result = {}; current = None; section = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(' '):
            assert line.endswith(':') and line[:-1] not in result
            current = result[line[:-1]] = dict(rule=None, inputs=[], outputs=[])
            section = None
        elif line.startswith('  input: '):
            assert current is not None and current['rule'] is None
            current['rule'] = line.split(': ', 1)[1]; section = 'inputs'
        elif line == '  outputs:':
            section = 'outputs'
        else:
            assert current is not None and section and line.startswith('    '), line
            current[section].append(line.strip())
    for value in result.values():
        value['inputs'].sort(); value['outputs'].sort()
    return result


def validate_closure(link, targets, queries, commands, reference):
    argv, objects = link_argv(link)
    assert (argv, objects) == link_argv(reference['link']), 'generated linker differs from R4'
    assert objects == reference['objects']
    module_targets = {}
    for line in targets.splitlines():
        name, rule = line.rsplit(': ', 1)
        if name.startswith(PREFIX) and name.endswith('.o'):
            assert name not in module_targets and rule == RULE
            module_targets[name] = rule
    assert set(module_targets) == set(objects), 'link/target object set mismatch'
    assert query_map(queries) == query_map(reference['queries']), 'link/object query drift'
    actual = [shlex.split(line) for line in commands.splitlines() if line.strip()]
    wanted = [shlex.split(line) for line in reference['commands']]
    assert len(actual) == 44 and sorted(actual) == sorted(wanted), 'nonmodule command or compile flag drift'
    assert {a[a.index('-o') + 1] for a in actual} == set(objects)
    return argv, objects, actual


def check_plan(text, objects, completed=False):
    actions = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\[(\d+)/44\] Building CXX object (.+)', line)
        if match:
            actions.append((int(match[1]), match[2]))
        else:
            # Real compiler warnings may appear only AFTER the read-only plan passed.
            assert completed and not line.startswith('[') and 'Re-running CMake' not in line, line
    assert [n for n, _ in actions] == list(range(1, 45)), 'must rebuild all 44 objects'
    assert len({name for _, name in actions}) == 44
    assert {name for _, name in actions} == set(objects), 'nonmodule or missing build action'


def file_map(root, names):
    result = {}
    for name in names:
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()), ('escaped build input', path)
        result[name] = sha(path)
    return result


def run(label, argv):
    print('+ ' + shlex.join(argv), flush=True)
    (OUT / (label + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
    with (OUT / (label + '.stdout.log')).open('w') as stdout, (OUT / (label + '.stderr.log')).open('w') as stderr:
        result = subprocess.run(argv, cwd=BUILD, stdout=stdout, stderr=stderr, text=True)
    result.check_returncode()
    return (OUT / (label + '.stdout.log')).read_text()


def main():
    assert __debug__ and Path.cwd() == BUILD
    reference = expected()
    graph = reference['graph_sha256']
    assert file_map(PARENT, graph) == file_map(BUILD, graph) == graph
    tools = (run('compiler-version', [COMPILER, '--version']) +
             run('ninja-version', ['ninja', '--version']) + run('cmake-version', ['cmake', '--version']))
    # R2 diagnosis includes package versions after this exact version-output prefix.
    assert tools == reference['tools'].split('libcurl4-openssl-dev:', 1)[0], 'frozen SDK/tool version drift'
    generated = run('generated-commands', ['ninja', '-t', 'commands', TARGET])
    link = generated.splitlines()[-1]
    _, objects = link_argv(link)  # Derive from the real final linker, never a *.cpp.o glob.
    targets = run('targets', ['ninja', '-t', 'targets', 'all'])
    queries = run('queries', ['ninja', '-t', 'query', TARGET, *objects])
    commands = run('module-commands', ['ninja', '-t', 'commands', *objects])
    argv, objects, compile_argv = validate_closure(link, targets, queries, commands, reference)
    (OUT / 'module-objects.json').write_text(json.dumps(objects, indent=2) + '\n')
    # Unity wrappers are generated build inputs, outside the normal source-map scope.
    unity = [str(Path(a[a.index('-c') + 1]).relative_to(BUILD)) for a in compile_argv
             if a[a.index('-c') + 1].startswith(str(BUILD) + '/')]
    assert len(unity) == 11
    wrappers = file_map(PARENT, unity)
    assert file_map(BUILD, unity) == wrappers
    inputs = file_map(PARENT, [*LIBRARIES, *PROTECTED])
    assert file_map(BUILD, inputs) == inputs
    assert (BUILD / TARGET).resolve().is_relative_to(BUILD)
    plan = run('module-plan', ['ninja', '-n', '-d', 'explain', '-j6', *objects])
    check_plan(plan, objects)  # No generation, linking, kernels, common, tools, or UI.
    compiled = run('module-build', ['ninja', '-j6', *objects])
    check_plan(compiled, objects, completed=True)
    assert file_map(BUILD, graph) == graph and file_map(BUILD, unity) == wrappers
    assert file_map(BUILD, inputs) == file_map(PARENT, inputs) == inputs
    run('module-link', argv)  # No shell: strip ONLY the validated ': && ... && :' wrapper.
    assert file_map(BUILD, graph) == file_map(PARENT, graph) == graph
    assert file_map(BUILD, unity) == wrappers
    assert file_map(BUILD, inputs) == file_map(PARENT, inputs) == inputs
    print('LILICORR_DEVICE_EMBEDDING_MODULE_R2_PASSED', flush=True)


if __name__ == '__main__':
    main()
