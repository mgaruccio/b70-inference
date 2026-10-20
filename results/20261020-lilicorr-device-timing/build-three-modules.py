#!/usr/bin/env python3
"""Case-local frozen-SDK driver for the three affected shared modules.

The lead supplies a complete composed source tree and a copy of the already
captured Ninja graph at runtime.  This driver deliberately consumes only the
three module object closures from that graph: it never regenerates CMake,
changes timestamps, asks Ninja for a target, or builds a server/test target.
The captured link argv and every compile command are treated as a closure
contract; a future source change that changes the closure stops the run rather
than becoming an unsupported speed claim.
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

MODULE_ORDER = (
    'bin/libggml-sycl.so.0.27.0',
    'bin/libllama.so.0.7.0',
    'bin/libllama-common.so.0.7.0',
)
MODULES = {
    'bin/libggml-sycl.so.0.27.0': {
        'prefix': 'ggml/src/ggml-sycl/CMakeFiles/ggml-sycl.dir/',
        'rule': 'CXX_COMPILER__ggml-sycl_unscanned_Release',
        'link_rule': 'CXX_SHARED_LIBRARY_LINKER__ggml-sycl_Release',
        'count': 105,
        'top_level': 59,
        'template_instances': 46,
        'object_re': re.compile(
            r'ggml/src/ggml-sycl/CMakeFiles/ggml-sycl\.dir/'
            r'(?:[A-Za-z0-9_.-]+\.cpp|template-instances/[A-Za-z0-9_.-]+\.cpp)\.o'
        ),
    },
    'bin/libllama.so.0.7.0': {
        'prefix': 'src/CMakeFiles/llama.dir/',
        'rule': 'CXX_COMPILER__llama_unscanned_Release',
        'link_rule': 'CXX_SHARED_LIBRARY_LINKER__llama_Release',
        'count': 44,
        'unity': 11,
        'object_re': re.compile(
            r'src/CMakeFiles/llama\.dir/'
            r'(?:Unity/unity_[0-9]+_cxx\.cxx|[A-Za-z0-9_.-]+\.cpp)\.o'
        ),
    },
    'bin/libllama-common.so.0.7.0': {
        'prefix': 'common/CMakeFiles/llama-common.dir/',
        'rule': 'CXX_COMPILER__llama-common_unscanned_Release',
        'link_rule': 'CXX_SHARED_LIBRARY_LINKER__llama-common_Release',
        'count': 55,
        'object_re': re.compile(
            r'common/CMakeFiles/llama-common\.dir/'
            r'(?:[A-Za-z0-9_.-]+|parsers/[A-Za-z0-9_.-]+|jinja/[A-Za-z0-9_.-]+)\.cpp\.o'
        ),
    },
}

# These are intentionally not the three outputs above.  The versioned link
# inputs and private archives are also protected because the direct links use
# them but this case does not rebuild them.
PROTECTED = [
    'bin/llama-server',
    'bin/libllama-server-impl.so',
    'bin/test-backend-ops',
]
PRESERVED = [
    'bin/libggml.so.0.27.0',
    'bin/libggml-cpu.so.0.27.0',
    'bin/libggml-base.so.0.27.0',
    'common/libllama-common-base.a',
    'vendor/cpp-httplib/libcpp-httplib.a',
    *PROTECTED,
]

# The raw JSON is the read-only capture supplied for this case.  The
# compressed file is case-local only; checking both hashes prevents a silently
# edited fixture from turning into a new build contract.
REFERENCE_JSON_SHA256 = '77973dc1fb94f3bb33656366dab7f6d8ea11ff50782ffb5b2cf46e968b645b2d'
INPUTS_SHA256 = '9f7d5ce2719a74cc272c46eecc2ec028d73586a2e7b2d451ada7d2a1712cfd92'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def expected():
    path = HERE / 'build-three-module-inputs.json.gz'
    assert sha(path) == INPUTS_SHA256, 'case-local input capture changed'
    raw = gzip.decompress(path.read_bytes())
    assert sha_bytes(raw) == REFERENCE_JSON_SHA256, 'read-only metadata capture changed'
    reference = json.loads(raw)
    assert tuple(reference['modules']) == MODULE_ORDER, 'module order drift'
    assert set(reference['graph_sha256']) == {
        'build.ninja',
        'CMakeFiles/rules.ninja',
        'CMakeCache.txt',
        'CMakeFiles/3.28.3/CMakeCXXCompiler.cmake',
    }
    assert sum(len(reference['modules'][target]['objects']) for target in MODULE_ORDER) == 204
    for target in MODULE_ORDER:
        module = reference['modules'][target]
        spec = MODULES[target]
        assert len(module['objects']) == spec['count']
        assert len(module['commands']) == spec['count']
        assert len(query_map(module['queries'])) == spec['count'] + 1
    return reference


def _module(target_or_module):
    if isinstance(target_or_module, str):
        target = target_or_module
    elif isinstance(target_or_module, dict):
        target = target_or_module.get('target')
    else:
        target = None
    assert target in MODULES, ('unknown module', target)
    return target, MODULES[target]


def _split_shell_line(line):
    try:
        return shlex.split(line)
    except ValueError as error:
        raise AssertionError(f'malformed shell argv: {error}') from error


def link_argv(line, target_or_module=None, reference=None):
    """Parse one generated Ninja linker line and return (argv, object list).

    The frozen capture is compared as tokenized argv, not as a hand-written
    library recipe.  This keeps each target's basename, object prefix, flags,
    runtime libraries, and link order tied to its actual generated command.
    """
    tokens = _split_shell_line(line)
    assert len(tokens) >= 5
    assert tokens[:2] == [':', '&&'] and tokens[-2:] == ['&&', ':'], \
        'unexpected linker wrapper'
    argv = tokens[2:-2]
    try:
        output_index = argv.index('-o')
        target = argv[output_index + 1]
    except (ValueError, IndexError) as error:
        raise AssertionError('link has no unique output target') from error
    if target_or_module is not None:
        requested, _ = _module(target_or_module)
        assert target == requested, 'link target drift'
    target, spec = _module(target)
    if reference is None:
        reference = expected()
    frozen = _split_shell_line(reference['modules'][target]['link'])
    assert argv == frozen[2:-2], 'compiler, flags, libraries, or link order drift'
    assert argv[0] == COMPILER
    assert argv.count('-o') == 1 and argv[output_index + 1] == target

    objects = [name for name in argv if spec['object_re'].fullmatch(name)]
    positions = [index for index, name in enumerate(argv)
                 if spec['object_re'].fullmatch(name)]
    assert positions, 'link has no module objects'
    assert positions == list(range(positions[0], positions[-1] + 1)), \
        'module objects are not one contiguous link closure'
    assert objects == reference['modules'][target]['objects'], \
        'link object closure drift'
    assert len(objects) == len(set(objects)) == spec['count'], \
        'incomplete or duplicate module closure'
    if 'top_level' in spec:
        assert sum('/template-instances/' not in name for name in objects) == spec['top_level']
        assert sum('/template-instances/' in name for name in objects) == spec['template_instances']
    if 'unity' in spec:
        assert sum('/Unity/' in name for name in objects) == spec['unity']
    return argv, objects


def query_map(text):
    """Parse the stable structure emitted by ``ninja -t query``."""
    result = {}
    current = None
    section = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(' '):
            assert line.endswith(':') and line[:-1] not in result, \
                'malformed duplicate query target'
            current = result[line[:-1]] = {'rule': None, 'inputs': [], 'outputs': []}
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
    assert result and all(value['rule'] is not None for value in result.values()), \
        'query omitted an input rule'
    return result


def _command_argv(text):
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        result.append(_split_shell_line(line))
    return result


def validate_targets(text, reference):
    """Validate every captured module object rule and linker target."""
    module_targets = {}
    link_targets = {}
    for line in text.splitlines():
        if ': ' not in line:
            continue
        name, rule = line.rsplit(': ', 1)
        for target in MODULE_ORDER:
            spec = MODULES[target]
            if name == target:
                assert name not in link_targets, 'duplicate module linker target'
                link_targets[name] = rule
            elif name.startswith(spec['prefix']) and name.endswith('.o'):
                assert name not in module_targets, 'duplicate module object target'
                module_targets[name] = rule
    expected_objects = {
        name
        for target in MODULE_ORDER
        for name in reference['modules'][target]['objects']
    }
    assert set(module_targets) == expected_objects, 'link/target object set mismatch'
    assert set(link_targets) == set(MODULE_ORDER), 'module linker target set mismatch'
    for target in MODULE_ORDER:
        module = reference['modules'][target]
        query = query_map(module['queries'])
        assert link_targets[target] == query[target]['rule'] == MODULES[target]['link_rule']
        expected = set(module['objects'])
        actual = {name for name in module_targets if name in expected}
        assert actual == expected
        assert all(module_targets[name] == MODULES[target]['rule'] for name in expected)


def validate_queries(text, reference, target):
    target, _ = _module(target)
    actual = query_map(text)
    frozen = query_map(reference['modules'][target]['queries'])
    assert actual == frozen, 'query edge or rule closure drift'
    objects = reference['modules'][target]['objects']
    assert set(actual) == {target, *objects}
    assert [name for name in actual[target]['inputs'] if not name.startswith('|')] == objects


def validate_commands(text, reference, target):
    target, spec = _module(target)
    actual = _command_argv(text)
    frozen = [_split_shell_line(line) for line in reference['modules'][target]['commands']]
    assert actual == frozen, 'compile command, order, or flag drift'
    objects = reference['modules'][target]['objects']
    assert len(actual) == spec['count']
    outputs = []
    for argv in actual:
        try:
            output = argv[argv.index('-o') + 1]
            source = argv[argv.index('-c') + 1]
        except (ValueError, IndexError) as error:
            raise AssertionError('compile command lacks -c/-o') from error
        assert source.startswith('/src/'), 'compile source is outside supplied /src tree'
        outputs.append(output)
    assert outputs == objects


def validate_closure(link, targets, queries, commands, reference, target=None):
    if target is None:
        tokens = _split_shell_line(link)
        target = tokens[tokens.index('-o') + 1]
    target, _ = _module(target)
    argv, objects = link_argv(link, target, reference)
    assert objects == reference['modules'][target]['objects']
    validate_targets(targets, reference)
    validate_queries(queries, reference, target)
    validate_commands(commands, reference, target)
    return argv, objects, _command_argv(commands)


def check_plan(text, objects, completed=False):
    """Require exactly one compile action for every object, and no target plan."""
    actions = []
    total = len(objects)
    pattern = re.compile(rf'\[(\d+)/{total}\] Building CXX object (.+)')
    for line in text.splitlines():
        if not line.strip():
            continue
        match = pattern.fullmatch(line)
        if match:
            actions.append((int(match[1]), match[2]))
        else:
            # Compiler diagnostics can appear after execution.  A configure,
            # link, copy, or unrelated Ninja action is never accepted.
            assert completed and not line.startswith('[') and 'Re-running CMake' not in line, line
    assert [number for number, _ in actions] == list(range(1, total + 1)), \
        'must rebuild every object exactly once'
    assert len({name for _, name in actions}) == total
    assert {name for _, name in actions} == set(objects), 'nonmodule or missing build action'


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
    bindir = root / 'bin'
    if not bindir.exists():
        return {}
    result = {}
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
    with (OUT / (label + '.stdout.log')).open('w') as stdout, \
            (OUT / (label + '.stderr.log')).open('w') as stderr:
        result = subprocess.run(argv, cwd=BUILD, stdout=stdout, stderr=stderr, text=True)
    result.check_returncode()
    return (OUT / (label + '.stdout.log')).read_text()


def _label(target):
    return target.replace('/', '-').replace('.', '-')


def _assert_unchanged(graph, preserved, links):
    assert file_map(BUILD, graph) == graph
    assert file_map(BUILD, PRESERVED) == preserved
    assert symlink_map(BUILD) == links


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
    assert tools == reference['tools'].split('libcurl4-openssl-dev:', 1)[0], \
        'frozen SDK/tool version drift'

    targets = run('targets', ['ninja', '-t', 'targets', 'all'])
    validate_targets(targets, reference)
    _assert_unchanged(graph, preserved, before_links)

    total_objects = 0
    for target in MODULE_ORDER:
        label = _label(target)
        module = reference['modules'][target]
        generated = run('generated-commands-' + label, ['ninja', '-t', 'commands', target])
        lines = [line for line in generated.splitlines() if line.strip()]
        assert lines, 'Ninja emitted no linker command'
        link = lines[-1]
        argv, objects = link_argv(link, target, reference)
        assert objects == module['objects']
        total_objects += len(objects)

        queries = run('queries-' + label, ['ninja', '-t', 'query', target, *objects])
        commands = run('module-commands-' + label, ['ninja', '-t', 'commands', *objects])
        validate_closure(link, targets, queries, commands, reference, target)
        (OUT / ('module-objects-' + label + '.json')).write_text(
            json.dumps(objects, indent=2) + '\n'
        )
        _assert_unchanged(graph, preserved, before_links)

        plan = run('module-plan-' + label, ['ninja', '-n', '-d', 'explain', '-j6', *objects])
        check_plan(plan, objects)
        _assert_unchanged(graph, preserved, before_links)

        compiled = run('module-build-' + label, ['ninja', '-j6', *objects])
        check_plan(compiled, objects, completed=True)
        _assert_unchanged(graph, preserved, before_links)

        # Execute only the exact generated linker argv, without a shell.  The
        # dependency order is the explicit MODULE_ORDER above.
        run('module-link-' + label, argv)
        assert (BUILD / target).resolve().is_relative_to(BUILD.resolve())
        _assert_unchanged(graph, preserved, before_links)

    assert total_objects == 204
    print('LILICORR_THREE_MODULES_BUILD_PASSED objects=204', flush=True)


if __name__ == '__main__':
    main()
