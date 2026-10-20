#!/usr/bin/env python3
"""Case-local frozen-SDK driver for the complete GGML-SYCL module.

The driver consumes the already-generated Ninja graph.  It never regenerates
CMake metadata, edits timestamps, or asks Ninja for a wider target.  The lead
runs it inside the frozen SDK after the source candidate and the graph have
been independently captured.
"""
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = Path('/src/build-sycl')
PARENT = Path('/qualified/build-sycl')
OUT = Path('/build-evidence')
COMPILER = '/opt/intel/oneapi/compiler/2026.0/bin/icpx'
TARGET = 'bin/libggml-sycl.so.0.27.0'
PREFIX = 'ggml/src/ggml-sycl/CMakeFiles/ggml-sycl.dir/'
RULE = 'CXX_COMPILER__ggml-sycl_unscanned_Release'
LINK_RULE = 'CXX_SHARED_LIBRARY_LINKER__ggml-sycl_Release'
OBJECT = re.compile(r'ggml/src/ggml-sycl/CMakeFiles/ggml-sycl\.dir/(?:[A-Za-z0-9_.-]+\.cpp|template-instances/[A-Za-z0-9_.-]+\.cpp)\.o')
TOP_LEVEL_COUNT = 59
TEMPLATE_COUNT = 46

# The six public artifacts are checked by source-proof.py.  These are the
# unrelated artifacts that this module build must not rewrite.
PROTECTED = [
    'bin/llama-server',
    'bin/libllama-server-impl.so',
    'bin/test-backend-ops',
    'bin/libllama-common.so',
    'bin/libllama.so.0.7.0',
]
PRESERVED = [
    'bin/libggml.so.0.27.0', 'bin/libggml.so.0', 'bin/libggml.so',
    'bin/libggml-cpu.so.0.27.0', 'bin/libggml-cpu.so.0', 'bin/libggml-cpu.so',
    'bin/libggml-base.so.0.27.0', 'bin/libggml-base.so.0', 'bin/libggml-base.so',
    'bin/libllama-common.so.0.7.0',
    *PROTECTED,
]
# Filled after the case-local compressed input is created.
INPUTS_SHA256 = 'cd31dea2a1ec7b8f3ef37267c517609e4e26e7165e2ec8fd39813612235b7694'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def expected():
    path = HERE / 'build-inputs.json.gz'
    assert sha(path) == INPUTS_SHA256, 'case-local input capture changed'
    return json.loads(gzip.decompress(path.read_bytes()))


def link_argv(line):
    tokens = shlex.split(line)
    assert tokens[:2] == [':', '&&'] and tokens[-2:] == ['&&', ':'], 'unexpected linker wrapper'
    argv = tokens[2:-2]
    prefix = [
        COMPILER, '-fPIC', '-O3', '-DNDEBUG', '-Xs',
        '-ze-intel-greater-than-4GB-buffer-required', '-fsycl', '-shared',
        '-Wl,-soname,libggml-sycl.so.0', '-o', TARGET,
    ]
    assert argv[:len(prefix)] == prefix, 'compiler or link flags drift'
    try:
        first_non_object = next(i for i, name in enumerate(argv[len(prefix):], len(prefix))
                                if not OBJECT.fullmatch(name))
    except StopIteration:
        raise AssertionError('link has no frozen library tail')
    objects = argv[len(prefix):first_non_object]
    suffix = argv[first_non_object:]
    expected_suffix = [
        '-Wl,-rpath,/src/build-sycl/bin:/opt/onednn/lib:',
        'bin/libggml-base.so.0.27.0',
        '/usr/lib/x86_64-linux-gnu/libze_loader.so',
        '/opt/onednn/lib/libdnnl.so.3.14',
        '/opt/intel/oneapi/compiler/2026.0/lib/libsycl.so',
        '-ldl', '-fsycl', '-fsycl-allow-device-image-dependencies',
        '-Wl,-rpath=/opt/intel/oneapi/mkl/2026.0/lib',
        '/opt/intel/oneapi/mkl/2026.0/lib/libmkl_sycl_blas.so',
        '/opt/intel/oneapi/mkl/2026.0/lib/libmkl_intel_ilp64.so',
        '/opt/intel/oneapi/mkl/2026.0/lib/libmkl_tbb_thread.so',
        '/opt/intel/oneapi/mkl/2026.0/lib/libmkl_core.so',
        '/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8/libtbb.so.12',
        '-Wl,-rpath,/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8',
        '-L/opt/intel/oneapi/tbb/2023.0/lib/intel64/gcc4.8',
        '-ltbb', '-lm', '-ldl', '-lpthread',
    ]
    assert suffix == expected_suffix, 'library or runtime link flags drift'
    assert len(objects) == len(set(objects)) == 105, 'incomplete or duplicate module closure'
    assert all(OBJECT.fullmatch(name) for name in objects), 'foreign object or unsafe link argument'
    assert sum('/template-instances/' not in name for name in objects) == TOP_LEVEL_COUNT
    assert sum('/template-instances/' in name for name in objects) == TEMPLATE_COUNT
    return argv, objects


def query_map(text):
    result = {}
    current = None
    section = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(' '):
            assert line.endswith(':') and line[:-1] not in result, 'malformed duplicate query target'
            current = result[line[:-1]] = dict(rule=None, inputs=[], outputs=[])
            section = None
        elif line.startswith('  input: '):
            assert current is not None and current['rule'] is None
            current['rule'] = line.split(': ', 1)[1]
            section = 'inputs'
        elif line == '  outputs:':
            assert current is not None
            section = 'outputs'
        else:
            assert current is not None and section and line.startswith('    '), line
            current[section].append(line.strip())
    return result


def validate_queries(text, reference, objects):
    actual = query_map(text)
    contract = reference['query_contract']
    required = {TARGET, *objects}
    assert required <= set(actual), 'query omitted a module target'
    module_names = {name for name in actual if name.startswith(PREFIX)}
    assert module_names == set(objects), 'query contains a foreign or missing module object'
    target = actual[TARGET]
    assert target['rule'] == LINK_RULE
    direct = [name for name in target['inputs'] if not name.startswith('|')]
    assert direct == objects, 'query/link direct object closure drift'
    assert target['inputs'] == contract['target_inputs'], 'link dependency graph drift'
    assert target['outputs'] == contract['target_outputs'], 'link output graph drift'
    for name in objects:
        record = actual[name]
        frozen = contract['objects'][name]
        assert record['rule'] == RULE
        assert record['inputs'] == frozen['inputs'], ('compile input graph drift', name)
        assert record['outputs'] == [TARGET], ('compile output graph drift', name)


def validate_closure(link, targets, queries, commands, reference):
    argv, objects = link_argv(link)
    ref_argv, ref_objects = link_argv(reference['link'])
    assert (argv, objects) == (ref_argv, ref_objects), 'generated linker differs from frozen capture'
    assert objects == reference['objects']

    module_targets = {}
    for line in targets.splitlines():
        if ': ' not in line:
            continue
        name, rule = line.rsplit(': ', 1)
        if name.startswith(PREFIX) and name.endswith('.o'):
            assert name not in module_targets, 'duplicate module target'
            module_targets[name] = rule
    assert set(module_targets) == set(objects), 'link/target object set mismatch'
    assert all(rule == RULE for rule in module_targets.values()), 'module rule drift'
    assert f'{TARGET}: {LINK_RULE}' in targets.splitlines(), 'module linker target missing'

    validate_queries(queries, reference, objects)
    actual = [shlex.split(line) for line in commands.splitlines() if line.strip()]
    wanted = [shlex.split(line) for line in reference['commands']]
    assert actual == wanted, 'nonmodule command, order, or compile flag drift'
    assert len(actual) == 105
    assert {a[a.index('-o') + 1] for a in actual} == set(objects)
    assert all(a[a.index('-c') + 1].startswith('/src/ggml/src/ggml-sycl/') for a in actual)
    return argv, objects, actual


def check_plan(text, objects, completed=False):
    actions = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\[(\d+)/105\] Building CXX object (.+)', line)
        if match:
            actions.append((int(match[1]), match[2]))
        else:
            # Compiler warnings can appear after the complete plan passed.
            assert completed and not line.startswith('[') and 'Re-running CMake' not in line, line
    assert [number for number, _ in actions] == list(range(1, 106)), 'must rebuild all 105 objects'
    assert len({name for _, name in actions}) == 105
    assert [name for _, name in actions] == objects, 'nonmodule, reordered, or missing build action'


def file_map(root, names):
    root = Path(root).resolve()
    result = {}
    for name in names:
        path = root / name
        resolved = path.resolve(strict=True)
        assert resolved.is_relative_to(root), ('escaped build input', name, resolved)
        assert path.is_file(), ('missing build input', name)
        result[name] = sha(path)
    return result


def symlink_map(root):
    root = Path(root).resolve()
    result = {}
    bindir = root / 'bin'
    if not bindir.exists():
        return result
    for path in bindir.rglob('*'):
        if not path.is_symlink():
            continue
        resolved = path.resolve(strict=False)
        assert resolved.is_relative_to(root), ('escaped build symlink', path)
        result[str(path.relative_to(root))] = os.readlink(path)
    return result


def run(label, argv):
    print('+ ' + shlex.join(argv), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (label + '.argv.json')).write_text(json.dumps(argv, indent=2) + '\n')
    with (OUT / (label + '.stdout.log')).open('w') as stdout, (OUT / (label + '.stderr.log')).open('w') as stderr:
        result = subprocess.run(argv, cwd=BUILD, stdout=stdout, stderr=stderr, text=True)
    result.check_returncode()
    return (OUT / (label + '.stdout.log')).read_text()


def main():
    assert __debug__, 'assertions are required'
    assert Path.cwd() == BUILD
    reference = expected()
    graph = reference['graph_sha256']
    assert file_map(PARENT, graph) == file_map(BUILD, graph) == graph
    before_links = symlink_map(BUILD)
    preserved = file_map(PARENT, PRESERVED)
    assert file_map(BUILD, PRESERVED) == preserved

    compiler = run('compiler-version', [COMPILER, '--version'])
    ninja = run('ninja-version', ['ninja', '--version'])
    cmake = run('cmake-version', ['cmake', '--version'])
    tools = compiler + ninja + cmake
    assert tools == reference['tools'].split('libcurl4-openssl-dev:', 1)[0], 'frozen SDK/tool version drift'

    generated = run('generated-commands', ['ninja', '-t', 'commands', TARGET])
    link = generated.splitlines()[-1]
    _, objects = link_argv(link)
    targets = run('targets', ['ninja', '-t', 'targets', 'all'])
    queries = run('queries', ['ninja', '-t', 'query', TARGET, *objects])
    commands = run('module-commands', ['ninja', '-t', 'commands', *objects])
    argv, objects, compile_argv = validate_closure(link, targets, queries, commands, reference)
    (OUT / 'module-objects.json').write_text(json.dumps(objects, indent=2) + '\n')

    plan = run('module-plan', ['ninja', '-n', '-d', 'explain', '-j6', *objects])
    check_plan(plan, objects)
    assert file_map(BUILD, graph) == graph and file_map(BUILD, PRESERVED) == preserved
    assert symlink_map(BUILD) == before_links

    compiled = run('module-build', ['ninja', '-j6', *objects])
    check_plan(compiled, objects, completed=True)
    assert file_map(BUILD, graph) == graph
    assert file_map(BUILD, PRESERVED) == preserved
    assert symlink_map(BUILD) == before_links

    # Execute only the already-captured target linker argv, without a shell.
    run('module-link', argv)
    assert file_map(BUILD, graph) == graph
    assert file_map(BUILD, PRESERVED) == preserved
    assert symlink_map(BUILD) == before_links
    assert (BUILD / TARGET).resolve().is_relative_to(BUILD.resolve())
    print('LILICORR_IQ2_BATCH4_GGML_SYCL_MODULE_PASSED', flush=True)


if __name__ == '__main__':
    main()
