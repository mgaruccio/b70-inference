"""CPU regressions; these fixtures are not evidence of live Glimmer training."""
import copy

import pytest
import torch

from test_glimmer_mtp_training import (
    ContractDataset, TinyTarget, load, pilot, runtime, train_args,
)


@pytest.mark.parametrize('scale,expected', [(1., 0.), (2., 1.), (.5, .25)])
def test_rms_loss_is_scale_sensitive_with_detached_teacher(scale, expected):
    teacher = torch.ones(2, 12, requires_grad=True)
    predicted = torch.full((2, 12), scale, requires_grad=True)
    angular_mse, cosine = pilot.state_errors(predicted, teacher)
    assert float(angular_mse.detach()) == pytest.approx(0., abs=1e-6)
    assert float(cosine.detach()) == pytest.approx(0., abs=1e-6)
    error = pilot.state_rms_error(predicted, teacher)
    assert float(error.detach()) == pytest.approx(expected, abs=1e-6)
    error.backward()
    assert teacher.grad is None
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.parametrize('student,teacher', [(0., 1.), (1e-12, 1.), (1., 0.), (0., 0.)])
def test_rms_loss_near_zero_is_finite(student, teacher):
    p = torch.full((2, 12), student, requires_grad=True)
    t = torch.full((2, 12), teacher, requires_grad=True)
    loss = pilot.state_rms_error(p, t)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    assert t.grad is None


@pytest.mark.parametrize('depth', [1, 2, 4])
def test_amplitude_loss_adds_only_declared_weight_and_preserves_freeze(depth):
    target = TinyTarget()
    data = ContractDataset()
    states = data.states[:2].clone().requires_grad_()
    head = pilot.make_head(12, 64, True, max_depth=4)
    options = dict(state_weight=.2, kl_weight=1., ce_weight=.25, depth=depth)
    old, metrics = pilot.training_loss(head, states, data.tokens[:2], target, **options)
    new, _ = pilot.training_loss(head, states, data.tokens[:2], target, **options, state_norm_weight=.2)
    expected = .2 * sum(w * m['rms_ratio_error'] for w, m in zip(pilot.WEIGHTS[:depth], metrics)) / sum(pilot.WEIGHTS[:depth])
    assert float((new - old).detach()) == pytest.approx(expected, abs=1e-6)
    legacy = sum(w * (.25 * m['ce'] + m['teacher_kl'] + .2 * (m['normalized_mse'] + m['cosine_distance']))
                 for w, m in zip(pilot.WEIGHTS[:depth], metrics)) / sum(pilot.WEIGHTS[:depth])
    assert float(old.detach()) == pytest.approx(legacy, abs=1e-6)
    new.backward()
    assert states.grad is None
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    target.assert_frozen()


def checkpoint(target, variant='shared-state'):
    head = pilot.make_head(12, 64, variant != 'fixed-ce', max_depth=1)
    return {'model': pilot.MODEL, 'revision': pilot.REVISION,
            'state_transition': pilot.STATE_TRANSITION, 'variant': variant,
            'max_depth': 1, 'hidden_size': 12, 'rank': 64, 'head': head.state_dict()}


def test_shared_head_can_be_probed_deeper_without_rewriting_checkpoint():
    target = TinyTarget()
    saved = checkpoint(target)
    original = copy.deepcopy(saved)
    head = pilot.head_from_checkpoint(saved, target, evaluation_depth=4)
    state = torch.ones(12)
    for depth in range(1, 5):
        state = head.step(state, torch.ones(12), depth, target.norm)
    assert saved['max_depth'] == original['max_depth'] == 1
    assert len(head.blocks) == 1
    for name, value in saved['head'].items():
        assert torch.equal(value, original['head'][name])


def test_fixed_head_cannot_invent_untrained_blocks():
    with pytest.raises(ValueError, match='untrained fixed-depth'):
        pilot.head_from_checkpoint(checkpoint(TinyTarget(), 'fixed-ce'), TinyTarget(), evaluation_depth=4)


def test_real_cpu_decoder_probe_reports_recursive_depth_and_untimed_drift():
    target = TinyTarget()
    head = pilot.head_from_checkpoint(checkpoint(target), target, evaluation_depth=4)
    rows = [{'id': f'{category}-{i}', 'category': category, 'token_ids': [1, 2]}
            for category in sorted(pilot.CATEGORIES) for i in range(2)]
    report = pilot.validation_probe(target, head, rows, 6, True, depth=4)
    assert report['depth'] == 4 and report['diagnostics_timed'] is False
    assert len(report['conditional_acceptance']) == 4
    assert 0 <= report['mean_accepted_drafts_per_pass'] <= 4
    diagnostics = [p for p in report['pairs'] if 'chosen_path_drift_raw' in p]
    assert len(diagnostics) == 6
    assert all('predicted_rms' in r and 'rms_ratio_error' in r for p in diagnostics for r in p['chosen_path_drift_raw'])
    target.assert_frozen()


def test_norm_variant_train_save_reload_and_resume(runtime, tmp_path):
    index, _ = runtime
    output = tmp_path / 'amplitude'
    pilot.train_command(train_args(index, output, '--variants', 'shared-state', 'shared-state-norm',
        '--updates', 2, '--schedule-updates', 4, '--batch-size', 2, '--rank', 64,
        '--train-depth', 1, '--state-weight', .2, '--validation-every', 1,
        '--validation-roots', 4, '--validation-batch-size', 2, '--checkpoint-every', 1))
    for arm, norm in [('shared-state', 0.), ('shared-state-norm', .2)]:
        saved = load(output / f'{arm}-last.pt')
        assert saved['state_weight'] == .2 and saved['state_norm_weight'] == norm
        assert saved['update'] == 2
    pilot.train_command(train_args(index, output, '--resume', output / 'shared-state-norm-last.pt', '--updates', 4))
    assert load(output / 'shared-state-norm-last.pt')['update'] == 4
    assert pilot.TRAIN_DEFAULTS['variants'] == ['fixed-ce', 'shared-ce', 'shared-state']


def test_validation_cli_labels_trained_vs_probed_depth(runtime, tmp_path):
    index, _ = runtime
    training = tmp_path / 'trained'
    pilot.train_command(train_args(index, training, '--variants', 'shared-state', '--updates', 1,
        '--batch-size', 2, '--rank', 64, '--train-depth', 1, '--checkpoint-every', 1))
    out = tmp_path / 'validation.json'
    args = pilot.parser().parse_args(['validate-head', '--capture', str(index),
        '--head', str(training / 'checkpoint-last.pt'), '--output', str(out),
        '--validation-roots', '4', '--batch-size', '2', '--max-new-tokens', '4',
        '--probe-depth', '4', '--record-divergence'])
    pilot.validate_head_command(args)
    import json
    report = json.loads(out.read_text())
    assert report['trained_depth'] == 1 and report['evaluation_depths'] == [1, 2, 4]
    assert report['checkpoint_selection_depths'] == [1]
    assert report['probe']['depth'] == 4
