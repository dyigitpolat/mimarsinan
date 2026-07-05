"""[MBH] D-hat probe isolation for the closure-apply tuner families (NAPQ/Pruning).

``full_transform_acc_on_clone`` prepares the deployed transformation through
``axis.probe_replica(clone, ...)``. The closure-apply axes (NAPQ / Pruning /
ActivationShift) drive a live-bound apply closure, so a replica dispatching to
it would transform the LIVE model and evaluate the UNTRANSFORMED clone (theory
§5g-v incidental (ii)). Contract pinned here: a probe replica applies the
transform to the CLONE via a model-targeted replica apply — the live model,
trainer wiring, and persistent tuner state stay bit-identical — and an axis
without a replica apply fails LOUD instead of corrupting the live ramp.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.axes import NAPQAxis, PruningAxis
from mimarsinan.tuning.axes.activation_shift_axis import ActivationShiftAxis
from mimarsinan.tuning.axes.perceptron_transform_axis import PerceptronTransformAxis
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.mbh_ledger import full_transform_acc_on_clone
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)
from mimarsinan.tuning.tuners.pruning.pruning_tuner import PruningTuner


def _state_dict_clone(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_state_dicts_equal(pre, model, what):
    post = model.state_dict()
    assert pre.keys() == post.keys()
    for key in pre:
        assert torch.equal(pre[key], post[key]), (
            f"{what}: state_dict[{key}] diverged — the probe touched live state"
        )


def _napq_tuner(tmp_path):
    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    torch.manual_seed(0)
    model = make_tiny_supermodel()
    return NormalizationAwarePerceptronQuantizationTuner(
        pipeline, model, quantization_bits=4, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=AdaptationManager(),
    )


def _pruning_tuner(tmp_path):
    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    torch.manual_seed(0)
    model = make_tiny_supermodel()
    tuner = PruningTuner(
        pipeline=pipeline, model=model, target_accuracy=0.0, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(), pruning_fraction=0.5,
    )
    tuner._init_original_weights()
    perceptrons = model.get_perceptrons()
    tuner._persistent_pruned_rows = [set() for _ in perceptrons]
    tuner._persistent_pruned_cols = [set() for _ in perceptrons]
    for p in perceptrons:
        w = p.layer.weight.data
        tuner.base_row_imp.append(w.abs().sum(dim=1))
        tuner.base_col_imp.append(w.abs().sum(dim=0))
    return tuner


class TestNAPQProbeIsolation:
    def test_full_transform_probe_leaves_live_state_untouched(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            pre_model = _state_dict_clone(tuner.model)
            pre_aux = _state_dict_clone(tuner.trainer.aux_model)
            pre_transform = tuner.trainer.perceptron_transformation

            acc = full_transform_acc_on_clone(tuner)

            assert 0.0 <= acc <= 1.0
            _assert_state_dicts_equal(pre_model, tuner.model, "live model")
            _assert_state_dicts_equal(pre_aux, tuner.trainer.aux_model, "aux model")
            assert tuner.trainer.perceptron_transformation is pre_transform, (
                "the probe must not re-point the live trainer's transformation"
            )
        finally:
            tuner.close()

    def test_probe_forward_transforms_the_clone(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            pre_model = _state_dict_clone(tuner.model)
            clone = copy.deepcopy(tuner.model)

            forward = tuner._mbh_full_transform_forward(clone)

            assert forward is clone
            _assert_state_dicts_equal(pre_model, tuner.model, "live model")
            live = list(tuner.model.get_perceptrons())
            probed = list(clone.get_perceptrons())
            assert any(
                not torch.equal(lp.layer.weight.detach(), cp.layer.weight.detach())
                for lp, cp in zip(live, probed)
            ), "the D-hat clone must actually carry the rate-1.0 quantization"

            reference = copy.deepcopy(tuner.model)
            transform = tuner._mixed_transform(1.0)
            with torch.no_grad():
                for perceptron in reference.get_perceptrons():
                    transform(perceptron)
            for rp, cp in zip(reference.get_perceptrons(), probed):
                assert torch.equal(rp.layer.weight.detach(), cp.layer.weight.detach()), (
                    "clone transform must equal the tuner's own rate-1.0 transform"
                )
        finally:
            tuner.close()


class TestPruningProbeIsolation:
    def test_full_transform_probe_leaves_live_state_and_masks_untouched(self, tmp_path):
        tuner = _pruning_tuner(tmp_path)
        try:
            pre_model = _state_dict_clone(tuner.model)
            pre_rows = [set(s) for s in tuner._persistent_pruned_rows]
            pre_cols = [set(s) for s in tuner._persistent_pruned_cols]

            acc = full_transform_acc_on_clone(tuner)

            assert 0.0 <= acc <= 1.0
            _assert_state_dicts_equal(pre_model, tuner.model, "live model")
            assert tuner._persistent_pruned_rows == pre_rows, (
                "the probe must not grow the live persistent pruned-row sets"
            )
            assert tuner._persistent_pruned_cols == pre_cols, (
                "the probe must not grow the live persistent pruned-col sets"
            )
        finally:
            tuner.close()

    def test_probe_clone_is_pruned_at_rate_one(self, tmp_path):
        tuner = _pruning_tuner(tmp_path)
        try:
            pre_rows = [set(s) for s in tuner._persistent_pruned_rows]
            clone = copy.deepcopy(tuner.model)

            forward = tuner._mbh_full_transform_forward(clone)

            assert forward is clone
            row_masks, col_masks = tuner._get_masks(1.0, commit=False)
            assert tuner._persistent_pruned_rows == pre_rows, (
                "commit=False mask computation must not grow live persistent sets"
            )
            any_pruned = False
            for i, cp in enumerate(clone.get_perceptrons()):
                prune_mask = (~row_masks[i]).unsqueeze(1) | (~col_masks[i]).unsqueeze(0)
                if not bool(prune_mask.any()):
                    continue
                any_pruned = True
                masked = cp.layer.weight.detach()[prune_mask]
                assert torch.equal(masked, torch.zeros_like(masked)), (
                    f"clone perceptron {i}: pruned entries must be exactly zero"
                )
            assert any_pruned, "fixture must prune at least one perceptron"
        finally:
            tuner.close()


class _Recorder:
    def __init__(self):
        self.calls = []

    def live(self, rate):
        self.calls.append(("live", float(rate)))

    def live_zero_arg(self):
        self.calls.append(("live", None))

    def replica(self, model, rate):
        self.calls.append(("replica", model, float(rate)))


class TestClosureAxisReplicaDispatch:
    @pytest.mark.parametrize("AxisCls", [PerceptronTransformAxis, NAPQAxis])
    def test_replica_dispatches_to_replica_apply_on_attach_target(self, AxisCls):
        rec = _Recorder()
        axis = AxisCls(rec.live, replica_apply_fn=rec.replica)
        axis.attach("LIVE", None, {})

        replica = axis.probe_replica("CLONE", None, {})
        replica.set_rate(1.0)
        assert rec.calls == [("replica", "CLONE", 1.0)]

        axis.set_rate(0.5)
        assert rec.calls[-1] == ("live", 0.5), (
            "spawning a replica must not re-point the live axis"
        )

    def test_pruning_axis_replica_dispatch(self):
        rec = _Recorder()
        axis = PruningAxis(
            rec.live, recovery_hooks_fn=lambda rate: [], replica_apply_fn=rec.replica,
        )
        axis.attach("LIVE", None, {})
        replica = axis.probe_replica("CLONE", None, {})
        replica.set_rate(1.0)
        assert rec.calls == [("replica", "CLONE", 1.0)]

    @pytest.mark.parametrize(
        "make_axis",
        [
            lambda rec: PerceptronTransformAxis(rec.live),
            lambda rec: NAPQAxis(rec.live),
            lambda rec: PruningAxis(rec.live),
            lambda rec: ActivationShiftAxis(rec.live_zero_arg),
        ],
    )
    def test_replica_without_replica_apply_fails_loud(self, make_axis):
        rec = _Recorder()
        axis = make_axis(rec)
        axis.attach("LIVE", None, {})
        replica = axis.probe_replica("CLONE", None, {})
        with pytest.raises(RuntimeError, match="replica"):
            replica.set_rate(1.0)
        assert rec.calls == [], (
            "a replica must never fire the live-bound apply closure"
        )

    def test_live_shift_axis_still_fires_its_zero_arg_closure(self):
        rec = _Recorder()
        axis = ActivationShiftAxis(rec.live_zero_arg)
        axis.attach("LIVE", None, {})
        axis.set_rate(1.0)
        assert rec.calls == [("live", None)]
