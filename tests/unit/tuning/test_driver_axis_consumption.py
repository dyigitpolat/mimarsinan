"""EF1 — every rate-tuner family READS the pipeline-wide OptimizationDriver axis.

Fix A's consuming half: the driver is unbound into ``DeploymentPlan.optimization_driver``
(default ``controller``), and EVERY rate-tuner family resolves its controller-vs-fast
decision FROM that axis (via ``DeploymentPlan.of(pipeline)``) through the lifted
``FastLadderMixin`` + the E1 seam — instead of each family reading its own hard-coded
per-family fast switch (LIF) or defaulting to the controller with no axis read at all
(the analytical clamp/activation-quant/activation-adaptation chain, the manager-rate
family, and the one-shot shift tuner), and instead of TTFS resolving its driver purely
off its own ``ttfs_*`` flags inside ``TtfsAdaptationPlan``.

The lock is two-pronged, per family:

1. THE FAMILY READS THE AXIS — it records the resolved ``OptimizationDriver``
   (``self._optimization_driver``) and selects fast iff the axis is ``fast``; an explicit
   ``optimization_driver`` key drives the decision (the generic override).
2. DEFAULT ``controller`` ⇒ BYTE-IDENTICAL — with the axis at its default the family is
   on the controller path (``_fixed_ladder_policy`` False), reproducing today's behavior.
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.pipelining.core.deployment_plan import (
    OPTIMIZATION_DRIVER_CONTROLLER,
    OPTIMIZATION_DRIVER_FAST,
    DeploymentPlan,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.orchestration.optimization_driver import OptimizationDriver


# ── per-family construction (the axis defaults to controller unless overridden) ──

def _pipeline(tmp_path, **cfg_over):
    cfg = default_config()
    cfg.update(cfg_over)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    return pipeline


def _clamp_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    pipeline = _pipeline(tmp_path, **cfg_over)
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(pipeline.config, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, 0.001, manager, scales, stats)


def _activation_quantization_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    pipeline = _pipeline(tmp_path, **cfg_over)
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(pipeline.config, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.5, 0.001, manager)


def _activation_adaptation_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
        ActivationAdaptationTuner,
    )

    pipeline = _pipeline(tmp_path, **cfg_over)
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(pipeline.config, model)
    return ActivationAdaptationTuner(pipeline, model, 0.5, 0.001, manager)


def _activation_shift_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner

    pipeline = _pipeline(tmp_path, **cfg_over)
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(pipeline.config, model)
    return ActivationShiftTuner(pipeline, model, 0.5, 0.001, manager)


def _lif_tuner(tmp_path, **cfg_over):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    base = dict(spiking_mode="lif", firing_mode="Default", simulation_steps=4)
    base.update(cfg_over)
    pipeline = _pipeline(tmp_path, **base)
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(pipeline, model, 0.5, 0.001, AdaptationManager())


_FAMILY_BUILDERS = {
    "clamp": _clamp_tuner,
    "activation_quantization": _activation_quantization_tuner,
    "activation_adaptation": _activation_adaptation_tuner,
    "lif": _lif_tuner,
}


def _close(tuner):
    if hasattr(tuner, "close"):
        try:
            tuner.close()
        except Exception:
            pass


# ── 1. each family records the resolved axis ──────────────────────────────────

class TestEveryFamilyReadsTheAxis:
    @pytest.mark.parametrize("family", sorted(_FAMILY_BUILDERS))
    def test_default_resolves_controller(self, tmp_path, family):
        t = _FAMILY_BUILDERS[family](tmp_path)
        try:
            driver = t._optimization_driver
            assert isinstance(driver, OptimizationDriver)
            assert driver.controller is True
            assert driver.fast_ladder is False
            assert getattr(t, "_fixed_ladder_policy", False) is False
        finally:
            _close(t)

    @pytest.mark.parametrize("family", sorted(_FAMILY_BUILDERS))
    def test_explicit_fast_axis_selects_fast_ladder(self, tmp_path, family):
        t = _FAMILY_BUILDERS[family](
            tmp_path, optimization_driver=OPTIMIZATION_DRIVER_FAST,
        )
        try:
            assert t._optimization_driver.fast_ladder is True
            assert t._fixed_ladder_policy is True
            assert t._fixed_ladder_rates[-1] == pytest.approx(1.0)
        finally:
            _close(t)

    @pytest.mark.parametrize("family", sorted(_FAMILY_BUILDERS))
    def test_explicit_controller_axis_stays_controller(self, tmp_path, family):
        t = _FAMILY_BUILDERS[family](
            tmp_path, optimization_driver=OPTIMIZATION_DRIVER_CONTROLLER,
        )
        try:
            assert t._optimization_driver.controller is True
            assert getattr(t, "_fixed_ladder_policy", False) is False
        finally:
            _close(t)

    def test_shift_tuner_reads_axis_default_controller(self, tmp_path):
        # The one-shot shift tuner has no fast ladder, but still CONSUMES the axis.
        t = _activation_shift_tuner(tmp_path)
        try:
            assert t._optimization_driver.controller is True
        finally:
            _close(t)

    def test_shift_tuner_records_explicit_fast_axis(self, tmp_path):
        t = _activation_shift_tuner(
            tmp_path, optimization_driver=OPTIMIZATION_DRIVER_FAST,
        )
        try:
            assert t._optimization_driver.fast_ladder is True
        finally:
            _close(t)


# ── 2. the axis is sourced from DeploymentPlan.of(pipeline) ────────────────────

class TestAxisSourcedFromDeploymentPlan:
    @pytest.mark.parametrize("family", sorted(_FAMILY_BUILDERS))
    def test_family_driver_matches_plan_axis(self, tmp_path, family):
        # The family's controller/fast decision agrees with the plan resolved from the
        # SAME config — the family reads the pipeline-wide axis, not a private flag.
        for axis in (OPTIMIZATION_DRIVER_CONTROLLER, OPTIMIZATION_DRIVER_FAST):
            t = _FAMILY_BUILDERS[family](tmp_path, optimization_driver=axis)
            try:
                plan = DeploymentPlan.of(t.pipeline)
                assert t._optimization_driver.fast_ladder == plan.is_fast_driver
            finally:
                _close(t)

    def test_lif_legacy_switch_still_drives_axis(self, tmp_path):
        # Byte-identical back-compat: a config carrying only the legacy lif_blend_fast
        # switch (no explicit optimization_driver) still resolves to the fast ladder,
        # because the legacy switch feeds DeploymentPlan.optimization_driver.
        t = _lif_tuner(tmp_path, lif_blend_fast=True, lif_blend_fast_steps_per_rate=2)
        try:
            assert DeploymentPlan.of(t.pipeline).is_fast_driver is True
            assert t._optimization_driver.fast_ladder is True
            assert t._fixed_ladder_policy is True
        finally:
            _close(t)


# ── 3. default-off ⇒ byte-identical run (the controller path is unchanged) ─────

class TestDefaultOffRunsController:
    def test_clamp_controller_run_unchanged(self, tmp_path):
        torch.manual_seed(0)
        t = _clamp_tuner(tmp_path)
        try:
            t.run()
            assert getattr(t, "_fast_blend_path", False) is False
            assert t._committed_rate == pytest.approx(1.0)
            assert len(t._cycle_log) > 0
        finally:
            _close(t)

    def test_activation_quantization_controller_run_unchanged(self, tmp_path):
        torch.manual_seed(0)
        t = _activation_quantization_tuner(tmp_path)
        try:
            t.run()
            assert getattr(t, "_fast_blend_path", False) is False
            assert t._committed_rate == pytest.approx(1.0)
        finally:
            _close(t)

    def test_clamp_fast_axis_run_uses_fixed_ladder(self, tmp_path):
        torch.manual_seed(0)
        t = _clamp_tuner(
            tmp_path,
            optimization_driver=OPTIMIZATION_DRIVER_FAST,
            clamp_fast_steps_per_rate=2,
            clamp_fast_rates=[0.5, 1.0],
        )
        try:
            t.run()
            assert t._fast_blend_path is True
            assert t._committed_rate == pytest.approx(1.0)
            assert [e["outcome"] for e in t._cycle_log] == \
                ["commit"] * len(t._fixed_ladder_rates)
        finally:
            _close(t)
