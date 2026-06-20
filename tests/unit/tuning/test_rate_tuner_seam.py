"""E1: the uniform rate-tuner seam (``ramp`` / ``recover_to`` / ``probe``).

Locks Fix A's real deliverable: every rate tuner exposes the same three driver-
facing verbs, and the verbs DELEGATE to the tuner's legacy private methods (no
behavior change). Two assertions per family:

1. CONFORMANCE — the tuner satisfies the ``RateTunerSeam`` protocol and the three
   verbs (plus ``seam_descriptor``) are callable.
2. DELEGATION — each verb routes to exactly the legacy path:
   * ``ramp(rate)``       → ``_update_and_evaluate`` (smooth) / ``_axis.set_rate`` (one-shot)
   * ``recover_to(target)`` → ``_recover_to_target`` → ``RecoveryEngine.train_to_target``
     (smooth) / ``trainer.train_steps_until_target`` (one-shot)
   * ``probe()``          → ``trainer.validate_n_batches`` (smooth) / ``trainer.validate`` (one-shot)

Covers the families the task names: the KD-blend tuners (LIF + TTFS-cycle), the
analytical clamp/shift/activation-quant chain + weight-quant, and the manager-rate
tuners. The seam being byte-identical to the cycle is locked by the SSOT extraction
(``_recover``'s recovery call == ``recover_to``'s recovery call).
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from conftest import (
    MockPipeline,
    default_config,
    make_tiny_supermodel,
    make_activation_scale_stats,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.orchestration.rate_tuner_seam import (
    OneShotRateTunerSeamMixin,
    RateTunerSeam,
    RateTunerSeamMixin,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
    TunerBase,
)


# ── per-family construction (mirrors the routing tests) ───────────────────────

def _activation_adaptation_tuner(tmp_path):
    from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
        ActivationAdaptationTuner,
    )

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, manager)


def _activation_quantization_tuner(tmp_path):
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.9, 0.001, manager)


def _noise_tuner(tmp_path):
    from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return NoiseTuner(pipeline, model, 0.9, 0.001, manager)


def _clamp_tuner(tmp_path):
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.9, 0.001, manager, scales, stats)


def _weight_quantization_tuner(tmp_path):
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner,
    )

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return NormalizationAwarePerceptronQuantizationTuner(
        pipeline, model, 8, 0.9, 0.001, manager
    )


def _pruning_tuner(tmp_path):
    from mimarsinan.tuning.tuners.pruning.pruning_tuner import PruningTuner

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = AdaptationManager()
    return PruningTuner(pipeline, model, 0.9, 0.001, manager, 0.25)


def _lif_tuner(tmp_path):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["simulation_steps"] = 4
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = AdaptationManager()
    return LIFAdaptationTuner(pipeline, model, 0.5, 0.001, manager)


def _ttfs_cycle_tuner(tmp_path):
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
        TTFSCycleAdaptationTuner,
    )

    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "cascaded"
    cfg["activation_quantization"] = True
    cfg["simulation_steps"] = 16
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = AdaptationManager()
    return TTFSCycleAdaptationTuner(pipeline, model, 0.5, 0.001, manager)


def _activation_shift_tuner(tmp_path):
    from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationShiftTuner(pipeline, model, 0.9, 0.001, manager)


SMOOTH_FAMILIES = {
    "activation_adaptation": _activation_adaptation_tuner,
    "activation_quantization": _activation_quantization_tuner,
    "noise": _noise_tuner,
    "clamp": _clamp_tuner,
    "weight_quantization": _weight_quantization_tuner,
    "pruning": _pruning_tuner,
    "lif_kd_blend": _lif_tuner,
    "ttfs_cycle_kd_blend": _ttfs_cycle_tuner,
}

ALL_FAMILIES = {**SMOOTH_FAMILIES, "activation_shift": _activation_shift_tuner}


@pytest.fixture(params=list(ALL_FAMILIES), ids=list(ALL_FAMILIES))
def tuner(request, tmp_path):
    t = ALL_FAMILIES[request.param](tmp_path)
    yield request.param, t
    if hasattr(t, "close"):
        t.close()


# ── 1. conformance: every tuner exposes the seam ──────────────────────────────

class TestSeamConformance:
    def test_smooth_tuner_base_is_a_seam(self):
        assert issubclass(SmoothAdaptationTuner, RateTunerSeamMixin)

    def test_shift_tuner_is_a_one_shot_seam(self):
        from mimarsinan.tuning.tuners.activation_shift_tuner import (
            ActivationShiftTuner,
        )

        assert issubclass(ActivationShiftTuner, OneShotRateTunerSeamMixin)
        # the one-shot tuner is deliberately NOT a smooth tuner
        assert not issubclass(ActivationShiftTuner, SmoothAdaptationTuner)

    def test_every_family_satisfies_the_protocol(self, tuner):
        _name, t = tuner
        assert isinstance(t, RateTunerSeam)

    def test_every_family_exposes_the_three_verbs(self, tuner):
        _name, t = tuner
        for verb in ("ramp", "recover_to", "probe", "seam_descriptor"):
            assert callable(getattr(t, verb)), verb

    def test_seam_descriptor_is_a_string(self, tuner):
        name, t = tuner
        desc = t.seam_descriptor()
        assert isinstance(desc, str) and desc
        shape = "one_shot" if name == "activation_shift" else "smooth"
        assert shape in desc


# ── 2. delegation: the verbs route to the legacy path ─────────────────────────

class TestSmoothSeamDelegation:
    """The smooth seam verbs delegate to the legacy private methods exactly."""

    @pytest.fixture(params=list(SMOOTH_FAMILIES), ids=list(SMOOTH_FAMILIES))
    def smooth(self, request, tmp_path):
        t = SMOOTH_FAMILIES[request.param](tmp_path)
        yield t
        t.close()

    def test_ramp_delegates_to_update_and_evaluate(self, smooth):
        seen = {}
        sentinel = object()

        def _spy(rate):
            seen["rate"] = rate
            return sentinel

        smooth._update_and_evaluate = _spy
        out = smooth.ramp(0.375)
        assert seen["rate"] == 0.375
        assert out is sentinel

    def test_probe_delegates_to_validate_n_batches(self, smooth):
        seen = {}

        def _spy(n):
            seen["n"] = n
            return 0.777

        smooth.trainer.validate_n_batches = _spy
        out = smooth.probe()
        assert out == pytest.approx(0.777)
        assert seen["n"] == smooth._budget.eval_n_batches

    def test_recover_to_delegates_to_recover_to_target(self, smooth):
        seen = {}
        sentinel = object()

        def _spy(target, rate):
            seen["target"] = target
            seen["rate"] = rate
            return ("lr", sentinel)

        smooth._recover_to_target = _spy
        smooth._committed_rate = 0.5
        out = smooth.recover_to(0.91)
        assert out is sentinel
        assert seen["target"] == pytest.approx(0.91)
        # defaults the recovery-hook rate to the live committed position
        assert seen["rate"] == pytest.approx(0.5)

    def test_recover_to_honors_explicit_rate(self, smooth):
        seen = {}

        def _spy(target, rate):
            seen["rate"] = rate
            return ("lr", None)

        smooth._recover_to_target = _spy
        smooth.recover_to(0.91, rate=0.25)
        assert seen["rate"] == pytest.approx(0.25)


class TestOneShotSeamDelegation:
    """The one-shot seam verbs delegate to ``ActivationShiftTuner``'s own calls."""

    @pytest.fixture
    def shift(self, tmp_path):
        t = _activation_shift_tuner(tmp_path)
        yield t
        t.close()

    def test_ramp_delegates_to_axis_set_rate(self, shift):
        seen = {}
        shift._axis.set_rate = lambda r: seen.setdefault("rate", r)
        assert shift.ramp(1.0) is None
        assert seen["rate"] == pytest.approx(1.0)

    def test_recover_to_delegates_to_train_steps_until_target(self, shift):
        seen = {}
        sentinel = object()
        shift._find_lr = lambda: 0.005

        def _spy(lr, max_steps, target, start, **kwargs):
            seen.update(lr=lr, target=target, kwargs=kwargs)
            return sentinel

        shift.trainer.train_steps_until_target = _spy
        out = shift.recover_to(0.88)
        assert out is sentinel
        assert seen["lr"] == pytest.approx(0.005)
        assert seen["target"] == pytest.approx(0.88)
        assert "validation_n_batches" in seen["kwargs"]

    def test_probe_delegates_to_validate(self, shift):
        shift.trainer.validate = lambda: 0.654
        assert shift.probe() == pytest.approx(0.654)


# ── 3. byte-identity: the seam recovery == the cycle recovery (SSOT) ───────────

class TestSeamMatchesCyclePath:
    """``_recover`` and the ``recover_to`` seam share one recovery primitive, so
    the seam is byte-identical to the legacy per-cycle corrector."""

    def test_recover_uses_the_same_primitive_as_the_seam(self, tmp_path):
        from mimarsinan.tuning.orchestration.adaptation_driver import CycleContext

        t = _clamp_tuner(tmp_path)
        try:
            calls = []
            t._recover_to_target = lambda target, rate: (
                calls.append((target, rate)), ("lr", None)
            )[1]
            t._get_target = lambda: 0.9

            # the per-cycle corrector routes through the SSOT primitive ...
            ctx = CycleContext(rate=0.5, t_cycle_start=0.0, pre_state=None,
                               pre_cycle_acc=0.0)
            t._recover(ctx)
            # ... and so does the driver-facing seam verb
            t._committed_rate = 0.5
            t.recover_to(0.9)

            assert calls == [(0.9, 0.5), (0.9, 0.5)]
        finally:
            t.close()
