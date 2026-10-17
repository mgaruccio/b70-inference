#!/usr/bin/env python3
"""Read-only diagnostic comparison of the rejected partial-routing control."""
import runpy
from pathlib import Path

ROOT = Path('/home/mike/b70-evals/20261017-nemotron35-bolt-lilicorr-attention-dmmv2')


def main():
    control = runpy.run_path(str(ROOT / 'run-trace.py'))
    analysis = runpy.run_path(str(ROOT / 'analyze-trace.py'))
    continuation = runpy.run_path(str(ROOT / 'continue-c0.py'))
    old = runpy.run_path(str(control['WIDTH1'] / 'run-controls.py'))
    native = runpy.run_path(str(control['WIDTH1'] / 'analyze-controls.py'))
    runner = runpy.run_path(str(old['OLD'] / 'run-bolt-cell-ssm-prefill-r1.py'))
    outputs = [ROOT / 'run-r1', ROOT / 'continuation-c0/run-r1']
    continuation['check_restoration'](control, old, runner, outputs)
    arms, route_errors = {}, {}
    for label, out in zip(('S0', 'C0'), outputs):
        cell = out / label
        assert control['load'](out / 'summary.json')['status'] == 'failed'
        control['check_startup']((cell / 'server.log').read_text(), label, old, native)
        quality = control['load'](cell / 'quality/summary.json')
        assert quality['pass'] and len(quality['checks']) == 20 and all(x['pass'] for x in quality['checks'])
        ordinary = control['load'](cell / 'ordinary.result.json')
        diagnostic = control['load'](cell / 'diagnostic.response.json')
        for stem in ('warmup', 'ordinary'):
            old['check_stream_result'](control['load'](cell / (stem + '.result.json')))
            assert control['load'](cell / (stem + '.request.json')) == control['load'](control['WIDTH1'] / 'run-r1/frozen-payload.json')
        tokens = old['check_diagnostic_result'](diagnostic)
        reference = control['WIDTH1'] / 'run-r1' / ('A0' if label == 'S0' else 'C0')
        assert control['load'](cell / 'diagnostic.request.json') == control['load'](reference / 'diagnostic.request.json')
        analysis['within_arm_text_consistency'](ordinary, diagnostic)
        log = (cell / 'diagnostic-window.log').read_text()
        assert log == control['diagnostic_window']((cell / 'pre-diagnostic.log').read_text(), (cell / 'through-diagnostic.log').read_text())
        result = analysis['align'](analysis['records'](log), tokens, cell / 'capture')
        result['capture_path'] = str(cell / 'capture')
        routes = analysis['select_route_batches'](log, label)
        try:
            analysis['assert_actual_route_gates'](routes, label)
        except AssertionError as exc:
            route_errors[label] = repr(exc)
        parent = analysis['report_parent_comparison'](result, label, ordinary, diagnostic)
        if label == 'S0':
            analysis['assert_parent_observer_invariant'](parent)
        accepted = sum(x['accepted'] for x in result['verifications'])
        proposed = sum(len(x['drafts']) for x in result['verifications'])
        assert (accepted, proposed) == (diagnostic['timings'].get('draft_n_accepted', 0), diagnostic['timings'].get('draft_n', 0))
        result.update(within_arm_text_passed=True, route_analysis=routes, parent_comparison=parent,
                      accepted=accepted, proposed=proposed, prior_reference=analysis['prior_reference_flags'](
                          ordinary, diagnostic, control['load'](reference / 'repeat-00.result.json'),
                          control['load'](reference / 'diagnostic.response.json')))
        arms[label] = result
    assert set(route_errors) == {'C0'} and 'blk.7.attn_output.weight' in route_errors['C0']
    result = analysis['compare'](arms['S0'], arms['C0'])
    result.update(attribution_gates_passed=False, planned_route_control_passed=False,
                  timing_permitted=False, promotion_permitted=False, route_gate_errors=route_errors,
                  restoration_verified=True, original_failures_preserved=True,
                  sources={label: str(out / label) for label, out in zip(('S0', 'C0'), outputs)},
                  route_analysis={label: arm['route_analysis'] for label, arm in arms.items()},
                  parent_comparison={label: arm['parent_comparison'] for label, arm in arms.items()},
                  acceptance={label: {'accepted': arm['accepted'], 'proposed': arm['proposed']} for label, arm in arms.items()})
    control['save'](ROOT / 'failed-route-comparison.json', result)
    print({k: result[k] for k in ('first_output_difference_ordinal', 'compared_through_ordinal',
                                 'attribution_gates_passed', 'restoration_verified', 'acceptance')})


if __name__ == '__main__':
    main()
