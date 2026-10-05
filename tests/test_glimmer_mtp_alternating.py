"""Alternating-head CPU regressions, not real Glimmer training evidence."""
import pytest
import torch

from test_glimmer_mtp_training import (
    TinyTarget, assert_nested_equal, checkpoint, load, pilot, runtime, train_args,
)


@pytest.mark.parametrize("depth", [2, 4, 8])
def test_alternating_routing_feedback_and_constant_parameter_budget(depth):
    head = pilot.make_head(12, 64, True, depth, alternating=True)
    routes, inputs = [], []
    for i, block in enumerate(head.blocks):
        block.register_forward_pre_hook(lambda module, args, i=i: (routes.append(i), inputs.append(args[0])) and None)
    state, embedding = torch.randn(2, 12), torch.randn(2, 12)
    for d in range(1, depth + 1):
        previous = state
        state = head.step(state, embedding, d, None)
        assert inputs[-1] is previous
    assert routes == [i % 2 for i in range(depth)]
    assert sum(v.numel() for v in head.parameters()) == 2 * (3 * 12 * 64 + 12)
    assert head.blocks[0].up.weight.data_ptr() != head.blocks[1].up.weight.data_ptr()
    with pytest.raises(ValueError, match="out of range"):
        head.step(state, embedding, depth + 1, None)


def test_alternating_loss_backpropagates_both_blocks_without_teacher_gradients():
    target = TinyTarget()
    head = pilot.make_head(12, 64, True, 4, alternating=True)
    states = torch.randn(2, 5, 12, requires_grad=True)
    tokens = torch.randint(0, target.lm_head.weight.shape[0], (2, 6))
    loss, metrics = pilot.training_loss(head, states, tokens, target, .2, 1., .25, 1., 4,
                                        state_norm_weight=.2)
    loss.backward()
    assert torch.isfinite(loss) and len(metrics) == 4 and states.grad is None
    for block in head.blocks:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        assert float(block.up.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for module in (target.embedding, target.norm, target.lm_head) for p in module.parameters())


def test_shared_source_initializes_independent_alternating_blocks_and_roundtrips():
    source = pilot.make_head(12, 64, True, 2)
    saved = checkpoint(source, variant="shared-state-norm", depth=2, rank=64)
    head = pilot.make_head(12, 64, True, 4, alternating=True)
    pilot.initialize_head(head, saved, 64, 12)
    for block in head.blocks:
        assert_nested_equal(block.state_dict(), source.blocks[0].state_dict())
    reference = pilot.head_from_checkpoint(saved, TinyTarget(), evaluation_depth=4)
    a = b = torch.randn(2, 12)
    embedding = torch.randn(2, 12)
    for depth in range(1, 5):
        a = reference.step(a, embedding, depth, None)
        b = head.step(b, embedding, depth, None)
        assert torch.equal(a, b)
    head.blocks[1].up.weight.data.add_(1.)
    alternate = checkpoint(head, variant="alternating-state-norm", depth=4, rank=64)
    restored = pilot.head_from_checkpoint(alternate, TinyTarget(), evaluation_depth=8)
    assert_nested_equal(restored.state_dict(), head.state_dict())
    destination = pilot.make_head(12, 64, True, 4, alternating=True)
    pilot.initialize_head(destination, alternate, 64, 12)
    assert_nested_equal(destination.state_dict(), head.state_dict())
    with pytest.raises(ValueError, match="alternating checkpoint"):
        pilot.initialize_head(source, alternate, 64, 12)
    with pytest.raises(ValueError, match="recurrent blocks"):
        pilot.make_head(12, 64, False, 4, alternating=True)


def test_alternating_public_training_warm_start_checkpoint_and_exact_resume(runtime, tmp_path):
    index, _ = runtime
    source = pilot.make_head(12, 64, True, 2)
    initial = checkpoint(source, variant="shared-state-norm", depth=2, rank=64)
    path = tmp_path / "source.pt"
    torch.save(initial, path)
    output = tmp_path / "alternating"
    pilot.train_command(train_args(index, output, "--init-head", path,
        "--variants", "alternating-state-norm", "--rank", 64, "--train-depth", 4,
        "--batch-size", 64, "--updates", 2, "--schedule-updates", 4,
        "--state-weight", .2, "--checkpoint-every", 1))
    saved = load(output / "checkpoint-last.pt")
    assert saved["variant"] == "alternating-state-norm" and saved["max_depth"] == 4
    assert saved["head_parameters"] == 2 * (3 * 12 * 64 + 12)
    assert saved["state_weight"] == saved["state_norm_weight"] == .2
    for i in (0, 1):
        assert any(not torch.equal(saved["head"][f"blocks.{i}." + k], v) for k, v in source.blocks[0].state_dict().items())
    restored = pilot.head_from_checkpoint(saved, TinyTarget(), evaluation_depth=8)
    assert len(restored.blocks) == 2
    pilot.train_command(train_args(index, output, "--resume", output / "checkpoint-last.pt", "--updates", 4))
    final = load(output / "checkpoint-last.pt")
    assert final["update"] == 4 and final["root_exposures"] == 256 and final["loss_position_exposures"] == 1024
    assert all(float(v["step"]) == 4 for v in final["optimizer"]["state"].values())
    assert pilot.TRAIN_DEFAULTS["variants"] == ["fixed-ce", "shared-ce", "shared-state"]
