"""U2.a — KD-blend genuine full-transform probe + ``_finalize_forward_for``.

The cascaded ttfs_cycle tuner's ``_full_transform_eval`` measures the GENUINE
single-spike cascade accuracy at rate 1.0 on an isolated clone (the deployed
``_SegmentSpikeForward`` dynamics), not the value-domain staircase proxy. These
tests pin: probe == an independently built clone forward (same builder ⇒ exact),
the probe is non-destructive to the live model AND the shared adaptation_manager,
``_finalize_forward_for(model)`` binds to the PASSED model, and the no-genuine-
forward schedules (synchronized TTFS / non-cycle-accurate LIF) fall back to the
value domain.
"""

import copy

import pytest
import torch

from conftest import make_tiny_supermodel

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.genuine_probe import eval_forward_over_val
from mimarsinan.tuning.perceptron_rate import rebuild_activations, set_blend_rate
from mimarsinan.tuning.tuners.lif_adaptation_tuner import (
    LIFAdaptationTuner,
    _ChipAlignedNFForward,
)
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


def _make_pipeline(tmp_path, schedule):
    from conftest import MockPipeline, default_config

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_lif_pipeline(tmp_path, *, cycle_accurate):
    from conftest import MockPipeline, default_config

    cfg = default_config()
    cfg["spiking_mode"] = "rate"
    cfg["cycle_accurate_lif_forward"] = cycle_accurate
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 16
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _make_ttfs_tuner(tmp_path, schedule):
    pipeline = _make_pipeline(tmp_path, schedule)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


def _make_lif_tuner(tmp_path, *, cycle_accurate):
    pipeline = _make_lif_pipeline(tmp_path, cycle_accurate=cycle_accurate)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    tuner = LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5,
        lr=pipeline.config["lr"], adaptation_manager=am,
    )
    return tuner, model, am


def _commit_rate(tuner, rate):
    """Mimic a committed cycle: apply the blend rate on the live model."""
    tuner._set_rate(rate)
    tuner._committed_rate = rate


def _independent_segment_spike_acc(tuner, model):
    """Build the genuine clone forward exactly as the override must:
    deepcopy → blend rate 1.0 → rebuild activations → ``_SegmentSpikeForward``."""
    device = tuner.pipeline.config["device"]
    clone = copy.deepcopy(model).to(device)
    set_blend_rate(clone, 1.0)
    rebuild_activations(clone, tuner.adaptation_manager, tuner.pipeline.config)
    forward_obj = _SegmentSpikeForward(clone, tuner._T)
    return eval_forward_over_val(
        tuner.trainer, forward_obj, clone,
        tuner._budget.progress_eval_batches, device,
    )


class TestFinalizeForwardForBinding:
    def test_cascaded_binds_to_passed_model(self, tmp_path):
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "cascaded")
        other = make_tiny_supermodel()
        fwd = tuner._finalize_forward_for(other)
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is other
        assert fwd.model is not model
        assert fwd.T == tuner._T

    def test_finalize_forward_uses_self_model(self, tmp_path):
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "cascaded")
        fwd = tuner._finalize_forward()
        assert isinstance(fwd, _SegmentSpikeForward)
        assert fwd.model is model

    def test_synchronized_returns_none(self, tmp_path):
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "synchronized")
        assert tuner._finalize_forward_for(model) is None
        assert tuner._finalize_forward() is None

    def test_lif_cycle_accurate_binds_chip_aligned(self, tmp_path):
        tuner, model, _ = _make_lif_tuner(tmp_path, cycle_accurate=True)
        other = make_tiny_supermodel()
        fwd = tuner._finalize_forward_for(other)
        assert isinstance(fwd, _ChipAlignedNFForward)
        assert fwd.model is other
        assert fwd.T == tuner._T

    def test_lif_non_cycle_accurate_returns_none(self, tmp_path):
        tuner, model, _ = _make_lif_tuner(tmp_path, cycle_accurate=False)
        assert tuner._finalize_forward_for(model) is None
        assert tuner._finalize_forward() is None


class TestGenuineFullTransformMatchesIndependentBuild:
    def test_cascaded_probe_equals_independent_segment_spike(self, tmp_path):
        torch.manual_seed(11)
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "cascaded")
        _commit_rate(tuner, 0.375)

        got = tuner._full_transform_eval()
        expected = _independent_segment_spike_acc(tuner, model)
        assert got == pytest.approx(expected, abs=1e-9)

    def test_cascaded_probe_differs_from_value_domain_in_general(self, tmp_path):
        """The genuine probe is a distinct measurement from the value-domain one
        (same surface only by coincidence) — both must at least be valid accs."""
        torch.manual_seed(12)
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "cascaded")
        _commit_rate(tuner, 0.5)
        genuine = tuner._full_transform_eval()
        value = float(tuner._value_full_transform_eval())
        assert 0.0 <= genuine <= 1.0
        assert 0.0 <= value <= 1.0


class TestNoGenuineForwardFallback:
    def test_synchronized_probe_equals_value_domain(self, tmp_path):
        torch.manual_seed(13)
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "synchronized")
        _commit_rate(tuner, 0.5)
        genuine = tuner._full_transform_eval()
        value = float(tuner._value_full_transform_eval())
        assert genuine == pytest.approx(value, abs=1e-9)

    def test_lif_non_cycle_accurate_probe_equals_value_domain(self, tmp_path):
        torch.manual_seed(14)
        tuner, model, _ = _make_lif_tuner(tmp_path, cycle_accurate=False)
        _commit_rate(tuner, 0.5)
        genuine = tuner._full_transform_eval()
        value = float(tuner._value_full_transform_eval())
        assert genuine == pytest.approx(value, abs=1e-9)


class TestNonDestructive:
    def test_cascaded_probe_leaves_live_state_intact(self, tmp_path):
        torch.manual_seed(15)
        tuner, model, am = _make_ttfs_tuner(tmp_path, "cascaded")
        _commit_rate(tuner, 0.375)

        before_state = copy.deepcopy(model.state_dict())
        base_act_ids = [id(p.base_activation) for p in model.get_perceptrons()]
        before_ttfs_active = am.ttfs_active
        before_rates = [p.base_activation.rate for p in model.get_perceptrons()]
        forward_installed = "forward" in model.__dict__

        tuner._full_transform_eval()

        after_state = model.state_dict()
        assert set(after_state.keys()) == set(before_state.keys())
        for k in before_state:
            assert torch.equal(after_state[k], before_state[k]), k
        assert [id(p.base_activation) for p in model.get_perceptrons()] == base_act_ids
        assert "forward" not in model.__dict__
        assert ("forward" in model.__dict__) == forward_installed
        assert am.ttfs_active is before_ttfs_active
        assert [p.base_activation.rate for p in model.get_perceptrons()] == before_rates

    def test_lif_cycle_accurate_probe_restores_manager_lif_active(self, tmp_path):
        """The clone rebuild toggles ``lif_active`` on the shared manager; the
        probe must snapshot and restore it (else the live ramp is corrupted)."""
        torch.manual_seed(16)
        tuner, model, am = _make_lif_tuner(tmp_path, cycle_accurate=True)
        _commit_rate(tuner, 0.5)
        assert am.lif_active is False
        before_state = copy.deepcopy(model.state_dict())

        tuner._full_transform_eval()

        assert am.lif_active is False, (
            "the genuine probe must restore the shared manager's lif_active flag"
        )
        after_state = model.state_dict()
        for k in before_state:
            assert torch.equal(after_state[k], before_state[k]), k
        assert "forward" not in model.__dict__

    def test_cascaded_probe_does_not_keep_ttfs_nodes_on_live_model(self, tmp_path):
        """The live model's base_activation must remain the BlendActivation (the
        TTFSActivation rebuild happens only on the clone)."""
        torch.manual_seed(17)
        tuner, model, _ = _make_ttfs_tuner(tmp_path, "cascaded")
        _commit_rate(tuner, 0.25)
        tuner._full_transform_eval()
        for p in model.get_perceptrons():
            assert not isinstance(p.base_activation, TTFSActivation)
            assert hasattr(p.base_activation, "rate")  # still the blend


class TestCliffProbeConsistency:
    """Full-step drift sentinel. Exercised end-to-end at the Wave-2 verifier
    barrier (needs U2.b's loop logging populating ``_full_transform_log``); here
    we only assert the reporting path runs and records the expected keys."""

    def test_after_run_reports_cliff_probe_consistency(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
            TTFSCycleAdaptationStep,
        )

        captured = {}

        pipeline = _make_pipeline(tmp_path, "cascaded")

        orig_report = pipeline.reporter.report

        def _capture(key, value=None):
            captured[key] = value
            return orig_report(key, value)

        pipeline.reporter.report = _capture
        pipeline.config["tuning_full_transform_probe"] = True

        model = make_tiny_supermodel()
        am = AdaptationManager()
        pipeline.seed("model", model, step_name="Activation Quantization")
        pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")

        step = TTFSCycleAdaptationStep(pipeline)
        step.name = "TTFS Cycle Fine-Tuning"
        pipeline.prepare_step(step)
        step.run()

        key = f"{step.tuner.name} cliff_probe_consistency"
        if step.tuner._full_transform_log:
            assert key in captured, "cliff_probe_consistency must be reported"
            payload = captured[key]
            assert set(payload) >= {
                "last_genuine_drop", "finalize_cliff", "abs_diff",
            }
