"""CheckpointGuard scope/location round-trip + bracket rollback (P3a)."""

import copy

import pytest
import torch

from conftest import make_tiny_supermodel
from mimarsinan.tuning.learning_rate_explorer import clone_state_for_trainer
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard


class _Stub:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device


def _mutate(model, delta=1.0):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(delta)


def test_invalid_scope_or_location_raises():
    stub = _Stub(make_tiny_supermodel())
    with pytest.raises(ValueError):
        CheckpointGuard(stub, scope="bogus")
    with pytest.raises(ValueError):
        CheckpointGuard(stub, location="bogus")


def test_full_device_matches_legacy_and_round_trips():
    model = make_tiny_supermodel()
    stub = _Stub(model)
    guard = CheckpointGuard(stub, scope="full", location="device")

    legacy = clone_state_for_trainer(stub)
    handle = guard.snapshot()
    assert set(handle.state) == set(legacy)
    for k in legacy:
        assert torch.allclose(handle.state[k], legacy[k])

    before = {k: v.clone() for k, v in model.state_dict().items()}
    _mutate(model)
    guard.restore(handle)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)


def test_tunable_scope_skips_frozen_params():
    model = make_tiny_supermodel()
    params = list(model.named_parameters())
    frozen_name = params[0][0]
    params[0][1].requires_grad_(False)

    stub = _Stub(model)
    guard = CheckpointGuard(stub, scope="tunable", location="device")
    handle = guard.snapshot()

    assert frozen_name not in handle.state          # frozen param not captured
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all(n in handle.state for n in trainable)

    snap_trainable = {n: model.state_dict()[n].clone() for n in trainable}
    _mutate(model)  # mutates everything, incl. the frozen param
    guard.restore(handle)

    sd = model.state_dict()
    for n in trainable:
        assert torch.allclose(sd[n], snap_trainable[n], atol=1e-6)


def test_cpu_pinned_round_trips():
    model = make_tiny_supermodel()
    stub = _Stub(model, device="cpu")
    guard = CheckpointGuard(stub, scope="full", location="cpu_pinned")
    handle = guard.snapshot()
    assert all(v.device.type == "cpu" for v in handle.state.values())

    before = {k: v.clone() for k, v in model.state_dict().items()}
    _mutate(model)
    guard.restore(handle)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)


def test_live_rollback_routes_through_guard(tmp_path):
    """A rolled-back cycle restores via the (always-on) CheckpointGuard."""
    from conftest import MockPipeline, make_tiny_supermodel, default_config
    from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
        SmoothAdaptationTuner,
    )

    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()

    class _T(SmoothAdaptationTuner):
        def _update_and_evaluate(self, rate):
            with torch.no_grad():  # simulate a destructive transform
                for p in self.model.parameters():
                    p.add_(1.0)
            return 0.85

        def _find_lr(self):
            return 0.001

    tuner = _T(pipeline, model, target_accuracy=0.9, lr=0.001)
    assert tuner._checkpoint_guard is not None
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9
    seq = iter([0.9, 0.1])  # pre=0.9, post=0.1 → rollback
    tuner.trainer.validate_n_batches = lambda n: next(seq, 0.1)
    tuner.trainer.train_steps_until_target = lambda *a, **k: None

    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = tuner._adaptation(0.5)

    assert result == tuner._committed_rate == 0.0
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)


def test_bracket_restores_on_rollback():
    model = make_tiny_supermodel()
    stub = _Stub(model)
    guard = CheckpointGuard(stub)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with guard.bracket() as handle:
        _mutate(model)
        guard.restore(handle)  # caller restores on rollback signal
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)
