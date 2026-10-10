#!/usr/bin/env python3
"""Render the two social cards from committed measurements (no invented chart data).
Requires Python 3 and librsvg's rsvg-convert. Run from any working directory.
"""
import json
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/20261009-qwen38-community-protocol'
OUT = ROOT / 'docs/images'
BG, PANEL, GRID = '#0b1220', '#132237', '#27364b'
WHITE, MUTED, NATIVE, GREEN = '#f5f8ff', '#a3b3c8', '#71839b', '#34d6a2'
FONT = 'DejaVu Sans, Arial, sans-serif'


def text(x, y, value, size=28, color=WHITE, weight=400, anchor='start'):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, w, h, color, radius=0):
    return f'<rect x="{x}" y="{y}" width="{w:.3f}" height="{h}" rx="{radius}" fill="{color}"/>'


def line(x1, y1, x2, y2, color=GRID):
    return f'<path d="M{x1} {y1} L{x2} {y2}" stroke="{color}" stroke-width="1.5"/>'


def shell(title, description):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            rect(0, 0, 1600, 900, BG),
            '<rect x="24" y="24" width="1552" height="852" rx="22" fill="none" stroke="#203047"/>',
            rect(80, 57, 6, 26, GREEN, 3),
            text(104, 78, 'B70 INFERENCE / RESEARCH UPDATE', 22, MUTED, 600),
            text(1514, 78, 'OCT 2026', 20, MUTED, anchor='end')]


def save(name, parts):
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f'{name}.svg'
    svg.write_text('\n'.join(parts + ['</svg>']) + '\n')
    subprocess.run(['rsvg-convert', '--output', str(OUT / f'{name}.png'), str(svg)], check=True)
    print(svg.relative_to(ROOT), 'and PNG rendered')


def main():
    short = json.loads((RESULTS / 'comparison-230w.json').read_text())
    long = json.loads((RESULTS / 'comparison-long-230w.json').read_text())
    for run in (short, long):
        assert run['baseline_runner_exit']['success'] and run['candidate_runner_exit']['success']
        assert run['candidate_execution_evidence']['full_graph_run_seen']
    cells = {c['cell']: c for run in (short, long) for c in run['cells']}
    full = cells['p212864-g128']
    base, candidate = full['baseline']['median'], full['shared_kv']['median']
    gain = (candidate / base - 1) * 100
    assert abs(gain - full['delta_percent']) < 1e-10
    assert 212864 + 128 == 212992

    card = shell('About 10% faster full-context decode on one B70',
                 f'Qwen3.8-27B at 230 W. Native MTP4 {base:.2f} versus shared-KV {candidate:.2f} decode tokens per second; {gain:.2f}% gain. 212864 input plus 128 output tokens. Five matched samples per arm. Experimental; quality not yet cleared. Not a measured whole-agent speedup.')
    card += [text(80, 164, 'Faster decode. Same single GPU.', 64, weight=700),
             text(84, 217, 'Qwen3.8-27B  ·  Intel Arc Pro B70  ·  230 W', 29, MUTED),
             text(76, 418, f'+{gain:.1f}%', 148, GREEN, 700),
             text(84, 479, 'decode throughput', 39, weight=600),
             text(84, 526, 'at the full supported context', 29, MUTED),
             text(84, 582, 'Same target. Same prompts. Same power cap.', 24, MUTED),
             line(820, 273, 820, 609),
             text(904, 306, 'Native MTP4', 29, MUTED, 600),
             text(904, 367, f'{base:.2f}', 50, weight=700),
             text(1090, 365, 'tok/s', 26, MUTED),
             rect(904, 390, base / 60 * 540, 22, NATIVE, 5),
             text(904, 473, 'Shared-KV MTP4', 29, GREEN, 600),
             text(904, 534, f'{candidate:.2f}', 50, weight=700),
             text(1090, 532, 'tok/s', 26, MUTED),
             rect(904, 557, candidate / 60 * 540, 22, GREEN, 5),
             text(904, 611, '0', 20, MUTED), text(1444, 611, '60 tok/s', 20, MUTED, anchor='end'),
             rect(80, 654, 1440, 80, PANEL, 12),
             text(108, 706, '212,864 input + 128 generated = 212,992 tokens', 31, weight=600),
             text(84, 779, 'Matched C1 medians · n=5 · FP8 KV · INT4 target + draft · Decode only', 22, MUTED),
             text(84, 815, 'Experimental: quality not yet cleared. Not a measured whole-agent speedup.', 22, WHITE),
             text(84, 857, 'github.com/mgaruccio/b70-inference', 22, MUTED),
             text(1514, 857, 'DATA: 7927610d', 20, MUTED, anchor='end')]
    save('qwen38-shared-kv-full-context', card)

    chart = shell('Shared-KV decode gains persist through full context',
                  'Matched 230 W comparison: about 7.92% at 130944 input tokens, 9.35% at 163840, 10.25% at 196608, and 10.08% at 212864. All generate 128 tokens. Five measured samples per point; 128K comes from the preceding run. Experimental; quality not yet cleared.')
    chart += [text(80, 161, 'The gain holds at full context.', 66, weight=700),
              text(84, 216, 'Shared-KV vs native MTP4 · Qwen3.8-27B · One B70 at 230 W', 28, MUTED)]
    x, width = 480, 810
    for tick in (0, 4, 8, 12):
        pos = x + tick / 12 * width
        chart += [line(pos, 266, pos, 697), text(pos, 734, f'{tick}%', 22, MUTED, anchor='middle')]
    points = [('~128K*', 'p130944-g128'), ('160K', 'p163840-g128'),
              ('192K', 'p196608-g128'), ('Full window', 'p212864-g128')]
    for index, (label, key) in enumerate(points):
        c = cells[key]
        b, s = c['baseline']['median'], c['shared_kv']['median']
        delta = (s / b - 1) * 100
        assert abs(delta - c['delta_percent']) < 1e-10
        y = 280 + index * 110
        bar_width = delta / 12 * width
        chart += [text(84, y + 31, label, 35, weight=600),
                  text(86, y + 66, f'{b:.2f} → {s:.2f} tok/s', 24, MUTED),
                  rect(x, y + 2, bar_width, 43, GREEN, 5),
                  text(1514, y + 36, f'+{delta:.2f}%', 37, GREEN, 700, anchor='end')]
    chart += [text(84, 774, 'n=5 medians · 128 output tokens · *128K from the preceding matched run', 21, MUTED),
              text(84, 807, 'Full window: 212,864 input + 128 output. Decode only; not whole-agent speed.', 21, MUTED),
              text(84, 850, 'github.com/mgaruccio/b70-inference', 22, MUTED),
              text(1514, 850, 'EXPERIMENTAL · QUALITY NOT YET CLEARED', 20, WHITE, anchor='end')]
    save('qwen38-shared-kv-context-gains', chart)


if __name__ == '__main__':
    main()
