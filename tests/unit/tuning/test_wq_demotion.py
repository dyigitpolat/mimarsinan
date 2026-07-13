"""WQ demotion (MBH X3, from 5g-v): projection + bounded high-water recovery.

The 5g decomposition proved the WQ step's gain was its accidental recovery
engine (recovery-only >= full step; quantization-only strictly negative) whose
greedy/bisect scheduler never engaged and whose target re-anchored on the
damaged local baseline. NAPQ therefore demotes to: the SAME gated fixed ladder
as the rest of the pipeline (projection-only rungs — 0 training steps at the
recipe rates), then the generic P1'' endpoint stage (train_steps_until_target
through the transform trainer, target = the D-hat high-water SSOT, steps
bounded by the wq_endpoint_recovery_steps recipe constant, keep-best kept).
The demotion knobs ride EVERY mode's recipe.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration import dhat_highwater, mbh_ledger
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

_ALL_MODES = [
    ("lif", None),
    ("ttfs", None),
    ("ttfs_quantized", None),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
]


def _napq_tuner(tmp_path, *, driver="fast", endpoint_steps=0, rates=None):
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner,
    )

    cfg = default_config()
    cfg["weight_quantization"] = True
    if driver is not None:
        cfg["optimization_driver"] = driver
    if rates is not None:
        cfg["wq_fast_rates"] = rates
    cfg["wq_endpoint_recovery_steps"] = endpoint_steps
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return NormalizationAwarePerceptronQuantizationTuner(
        pipeline, model, cfg["weight_bits"], 0.5, cfg["lr"], manager,
    )


def _inject_accepting_gate(monkeypatch):
    monkeypatch.setattr(
        mbh_ledger, "rung_measurements",
        lambda tuner: {
            "blended_fp32": 0.5, "full_acc": 0.5,
            "rho": 1.0, "grad_norm_t": 0.0,
        },
    )
    monkeypatch.setattr(
        mbh_ledger, "full_transform_measurement", lambda tuner: 0.5,
    )


class TestRecipeCarriesTheDemotion:
    @pytest.mark.parametrize("mode,schedule", _ALL_MODES)
    def test_every_mode_folds_the_wq_knobs(self, mode, schedule):
        knobs = ConversionPolicy.derive(mode, schedule).knobs
        assert knobs["wq_fast_rates"] == [0.5, 1.0]
        assert knobs["wq_fast_steps_per_rate"] == 0
        # [5u] the bit-parity-lossless ttfs row AND [5u generalized + C4] the
        # well-conditioned rows (lif/sync/cascaded/ttfs_quantized) fund a
        # lifted NAPQ endpoint from the wall headroom; the C1 convergence stop
        # makes the funding a ceiling. The generic 600 survives only for modes
        # outside both families.
        assert knobs["wq_endpoint_recovery_steps"] == 16000


class TestGeneralizedEndpointFloor:
    """[5u generalized] the WQ endpoint carries the well-conditioned floor scoped
    to itself (wq_endpoint_target_floor); the bit-parity every-endpoint floor
    (endpoint_target_floor) still flows through unchanged; the effective floor is
    the max of the two."""

    def _capture_floor(self, monkeypatch):
        import mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner as napq_mod

        seen = {}
        monkeypatch.setattr(
            napq_mod, "run_endpoint_recovery",
            lambda tuner, *, base_steps, target_floor=None: seen.update(
                base_steps=base_steps, target_floor=target_floor,
            ),
        )
        return seen

    def test_wq_scoped_floor_is_passed(self, tmp_path, monkeypatch):
        seen = self._capture_floor(monkeypatch)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner.pipeline.config["wq_endpoint_target_floor"] = 0.98
            tuner._post_stabilization_hook()
            assert seen["target_floor"] == pytest.approx(0.98)
        finally:
            tuner.close()

    def test_bit_parity_floor_still_passed(self, tmp_path, monkeypatch):
        seen = self._capture_floor(monkeypatch)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner.pipeline.config["endpoint_target_floor"] = 0.98
            tuner._post_stabilization_hook()
            assert seen["target_floor"] == pytest.approx(0.98)
        finally:
            tuner.close()

    def test_effective_floor_is_the_max(self, tmp_path, monkeypatch):
        seen = self._capture_floor(monkeypatch)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner.pipeline.config["endpoint_target_floor"] = 0.9
            tuner.pipeline.config["wq_endpoint_target_floor"] = 0.98
            tuner._post_stabilization_hook()
            assert seen["target_floor"] == pytest.approx(0.98)
        finally:
            tuner.close()

    def test_absent_floors_pass_zero(self, tmp_path, monkeypatch):
        seen = self._capture_floor(monkeypatch)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner._post_stabilization_hook()
            assert seen["target_floor"] == pytest.approx(0.0)
        finally:
            tuner.close()


class TestGeneralizedFloorRecipeKnobs:
    """[5u generalized] the well-conditioned near-lossless recipes carry the
    WQ-scoped floor; bit-parity ttfs keeps its every-endpoint floor. [C4]
    ttfs_quantized joined the well-conditioned family."""

    @pytest.mark.parametrize(
        "mode,schedule",
        [("lif", None), ("ttfs_quantized", None),
         ("ttfs_cycle_based", "cascaded"), ("ttfs_cycle_based", "synchronized")],
    )
    def test_well_conditioned_modes_carry_the_wq_scoped_floor(self, mode, schedule):
        knobs = ConversionPolicy.derive(mode, schedule).knobs
        assert knobs["wq_endpoint_target_floor"] == pytest.approx(0.98)
        # WQ-scoped: the every-endpoint floor stays OFF so only the final
        # composition lifts (one bounded wall lift).
        assert "endpoint_target_floor" not in knobs

    def test_ttfs_quantized_carries_the_generalized_floor(self):
        """[C4 pin flip] the former ``stays_off_the_generalized_floor`` pin is
        deliberately inverted: ttfs_quantized's proxy→deployed transfer is
        measured sub-SE (t0_11 +0.0007, t0_14 −0.0014, t01_06 −0.0010 vs SE
        0.0092), so a floor-funded proxy climb survives to the deployed read
        and the mode earns the well-conditioned WQ floor. C1's convergence
        stop bounds the funded burn."""
        knobs = ConversionPolicy.derive("ttfs_quantized", None).knobs
        assert knobs["wq_endpoint_target_floor"] == pytest.approx(0.98)
        assert knobs["wq_endpoint_recovery_steps"] == 16000
        assert "endpoint_target_floor" not in knobs

    def test_bit_parity_ttfs_uses_the_every_endpoint_floor(self):
        knobs = ConversionPolicy.derive("ttfs", None).knobs
        assert knobs["endpoint_target_floor"] == pytest.approx(0.98)
        assert "wq_endpoint_target_floor" not in knobs


class TestLadderWiring:
    def test_fast_driver_puts_napq_on_the_gated_fixed_ladder(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            assert tuner._fixed_ladder_policy is True
            assert tuner._fixed_ladder_rates == [0.5, 1.0]
            assert tuner._fast_steps_per_rate == 0, (
                "WQ rungs are pure projection: no accidental training engine"
            )
        finally:
            tuner.close()

    def test_custom_recipe_rates_are_normalized_to_full(self, tmp_path):
        tuner = _napq_tuner(tmp_path, rates=[0.25, 0.5])
        try:
            assert tuner._fixed_ladder_rates == [0.25, 0.5, 1.0]
        finally:
            tuner.close()

    def test_controller_driver_keeps_the_legacy_path(self, tmp_path):
        tuner = _napq_tuner(tmp_path, driver="controller")
        try:
            assert getattr(tuner, "_fixed_ladder_policy", False) is False
        finally:
            tuner.close()


class TestProjectionRun:
    def test_run_projects_without_training_and_quantizes(self, tmp_path, monkeypatch):
        _inject_accepting_gate(monkeypatch)
        torch.manual_seed(0)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner.run()
            assert tuner._committed_rate == pytest.approx(1.0)
            assert tuner._fast_optimizer_steps == 0
            assert [e["outcome"] for e in tuner._cycle_log] == \
                ["commit"] * len(tuner._fixed_ladder_rates)
        finally:
            tuner.close()

    def test_endpoint_stage_runs_with_the_wq_budget(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner as napq_mod

        calls = []
        monkeypatch.setattr(
            napq_mod, "run_endpoint_recovery",
            lambda tuner, *, base_steps, target_floor=None: calls.append(base_steps),
        )
        _inject_accepting_gate(monkeypatch)
        torch.manual_seed(0)
        tuner = _napq_tuner(tmp_path, endpoint_steps=600)
        try:
            tuner.run()
            assert calls == [600]
        finally:
            tuner.close()

    def test_controller_path_has_no_endpoint_stage(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner as napq_mod

        calls = []
        monkeypatch.setattr(
            napq_mod, "run_endpoint_recovery",
            lambda tuner, *, base_steps, target_floor=None: calls.append(base_steps),
        )
        tuner = _napq_tuner(tmp_path, driver="controller", endpoint_steps=600)
        try:
            tuner._post_stabilization_hook()
            assert calls == []
        finally:
            tuner.close()


class TestEndpointRecoveryThroughTheTransformTrainer:
    def test_recovery_trains_aux_and_keeps_main_projected(
        self, tmp_path, monkeypatch,
    ):
        # The endpoint stage's training steps must go through the transform
        # trainer (train aux -> reproject main), so the shipped model stays a
        # projection of the trained float model.
        from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
            run_endpoint_recovery,
        )

        _inject_accepting_gate(monkeypatch)
        torch.manual_seed(0)
        tuner = _napq_tuner(tmp_path)
        try:
            tuner.run()  # commit the rate-1.0 projection first
            dhat_highwater.observe(tuner.pipeline, 0.99)
            pre_aux = {
                k: v.clone() for k, v in tuner.trainer.aux_model.state_dict().items()
            }
            # Fix C anchors keep-best at the entry probe: make the recovery
            # probes improve so the TRAINED state commits and the aux-model
            # gradient routing stays observable. The +0.2 slope clears the
            # keep-best SE bar (~0.177 on this mock) in ONE read, so the
            # commit survives the [P4] widened armed eval cadence (fewer
            # checks per leg).
            rising = itertools.count()
            monkeypatch.setattr(
                tuner.trainer, "validate_n_batches",
                lambda n: min(0.5 + 0.2 * next(rising), 0.98),
            )
            report = run_endpoint_recovery(tuner, base_steps=3)
            assert report.engaged is True
            assert report.steps_used > 0
            aux_changed = any(
                not torch.equal(pre_aux[k], v)
                for k, v in tuner.trainer.aux_model.state_dict().items()
            )
            assert aux_changed, "recovery must train the float aux model"
            assert report.exit >= report.entry - tuner._rollback_tolerance
        finally:
            tuner.close()
