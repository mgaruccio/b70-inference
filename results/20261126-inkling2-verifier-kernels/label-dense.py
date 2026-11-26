"""Annotate only a disposable eager-profile plugin copy; do not benchmark these sources."""
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
if root.parts[-3:] != ('dense', 'plugin', 'vllm_inkling2_xpu'):
    raise ValueError(f'not the disposable dense probe: {root}')
sites = {
    'attention': {'Inkling2Attention': ('_project', 'forward')},
    'sconv': {'Inkling2ShortConv': ('dynamic_taps',)},
    'moe': {'Inkling2MoE': ('_select_experts',)},
    'loop': {'*': ('blend',)},
    'model': {'*': ('compute_logits',)},
    'dspark': {'Inkling2DSparkModel': ('forward', 'markov_step', 'markov_logits', 'markov_argmax')},
}
for module, targets in sites.items():
    path = root / f'{module}.py'
    text = path.read_text()
    if 'record_function("ink2/' in text:
        raise RuntimeError(f'already labeled: {module}')
    edits = []
    for cls in ast.parse(text).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for f in cls.body:
            if isinstance(f, ast.FunctionDef) and f.name in targets.get(cls.name, targets.get('*', ())):
                label = f'ink2/{module}/{cls.name}.{f.name}'
                edits.append((f.lineno-1, ' '*f.col_offset + f'@torch.profiler.record_function("{label}")\n'))
    if len(edits) != sum(map(len, targets.values())):
        raise ValueError((path, edits))
    lines = text.splitlines(keepends=True)
    for i, line in sorted(edits, reverse=True):
        lines.insert(i, line)
    text = ''.join(lines)
    compile(text, str(path), 'exec')
    path.write_text(text)
