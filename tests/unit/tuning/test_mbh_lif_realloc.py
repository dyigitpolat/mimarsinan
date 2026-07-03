"""[MBH X2/E2] LIF wall reallocation (prediction T5, efficiency form).

Gated by ``MIMARSINAN_MBH_LIF_REALLOC`` through the env SSOT, default OFF. X1
measured the Clamp and ActivationQuantization fast ladders behaviorally inert in
LIF mode (their decorators are subsumed by the spiking node — blended == D-hat,
rho == 1 on all rungs). With the flag on AND the mode lif (spiking_semantics
predicate), those two ladders train 0 steps per rung (installation, probes, and
finalize contracts all still run) and the reclaimed budget moves into the LIF
deployed-forward fast-stabilize (600 -> 2000 steps). Non-lif modes and every
other rate ladder (noise, shift, activation adaptation) are untouched.
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
from mimarsinan.common import env
from mimarsinan.tuning.orchestration.adaptation_manager import (
    AdaptationManager,
    mbh_lif_realloc_ladder_steps,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)


def _set_flag(monkeypatch, enabled):
    if enabled:
        monkeypatch.setenv(env.MBH_LIF_REALLOC_VAR, "1")
    else:
        monkeypatch.delenv(env.MBH_LIF_REALLOC_VAR, raising=False)


# -- fixtures ----------------------------------------------------------------------

def _base_cfg(spiking_mode, steps):
    cfg = default_config()
    cfg["spiking_mode"] = spiking_mode
    cfg["optimization_driver"] = "fast"
    if spiking_mode != "lif":
        cfg["activation_quantization"] = True
    cfg["clamp_fast_rates"] = [0.5, 1.0]
    cfg["clamp_fast_steps_per_rate"] = steps
    cfg["manager_rate_fast_rates"] = [0.5, 1.0]
    cfg["manager_rate_fast_steps_per_rate"] = steps
    return cfg


def _clamp_tuner(tmp_path, *, spiking_mode="lif", steps=3):
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = _base_cfg(spiking_mode, steps)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, cfg["lr"], manager, scales, stats)


def _aq_tuner(tmp_path, *, spiking_mode="lif", steps=3):
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    cfg = _base_cfg(spiking_mode, steps)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)


def _noise_tuner(tmp_path, *, steps=3):
    from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner

    cfg = _base_cfg("lif", steps)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return NoiseTuner(pipeline, model, 0.5, cfg["lr"], manager)


def _lif_tuner(tmp_path, *, stabilize_steps=None):
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    cfg = _base_cfg("lif", 2)
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = 4
    cfg["lif_blend_fast"] = True
    cfg["lif_blend_fast_steps_per_rate"] = 2
    cfg["lif_blend_fast_rates"] = [0.5, 1.0]
    if stabilize_steps is not None:
        cfg["lif_blend_fast_stabilize_steps"] = stabilize_steps
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    return LIFAdaptationTuner(
        pipeline, model=model, target_accuracy=0.5, lr=cfg["lr"],
        adaptation_manager=AdaptationManager(),
    )


# -- the env SSOT accessor -----------------------------------------------------

class TestEnvFlag:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv(env.MBH_LIF_REALLOC_VAR, raising=False)
        assert env.mbh_lif_realloc_enabled() is False

    def test_exactly_one_enables(self, monkeypatch):
        monkeypatch.setenv(env.MBH_LIF_REALLOC_VAR, "1")
        assert env.mbh_lif_realloc_enabled() is True

    def test_other_values_stay_off(self, monkeypatch):
        for value in ("0", "true", "yes", ""):
            monkeypatch.setenv(env.MBH_LIF_REALLOC_VAR, value)
            assert env.mbh_lif_realloc_enabled() is False


# -- the spiking-semantics predicate ---------------------------------------------

class TestIsLifPredicate:
    def test_lif_is_lif(self):
        from mimarsinan.chip_simulation.spiking_semantics import is_lif

        assert is_lif("lif") is True

    def test_other_modes_are_not(self):
        from mimarsinan.chip_simulation.spiking_semantics import is_lif

        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert is_lif(mode) is False


# -- the steps helper (the subsumption-keyed SSOT) --------------------------------

class TestLadderStepsHelper:
    def test_flag_off_is_identity(self, monkeypatch):
        _set_flag(monkeypatch, False)
        cfg = {"spiking_mode": "lif"}
        for attr in ("clamp_rate", "quantization_rate", "noise_rate"):
            assert mbh_lif_realloc_ladder_steps(cfg, attr, 120) == 120

    def test_flag_on_zeroes_subsumed_rates_in_lif(self, monkeypatch):
        _set_flag(monkeypatch, True)
        cfg = {"spiking_mode": "lif"}
        assert mbh_lif_realloc_ladder_steps(cfg, "clamp_rate", 120) == 0
        assert mbh_lif_realloc_ladder_steps(cfg, "quantization_rate", 120) == 0

    def test_flag_on_leaves_unsubsumed_rates(self, monkeypatch):
        _set_flag(monkeypatch, True)
        cfg = {"spiking_mode": "lif"}
        for attr in ("noise_rate", "shift_rate", "activation_adaptation_rate"):
            assert mbh_lif_realloc_ladder_steps(cfg, attr, 120) == 120

    def test_flag_on_leaves_non_lif_modes(self, monkeypatch):
        _set_flag(monkeypatch, True)
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            cfg = {"spiking_mode": mode}
            assert mbh_lif_realloc_ladder_steps(cfg, "clamp_rate", 120) == 120
            assert mbh_lif_realloc_ladder_steps(cfg, "quantization_rate", 120) == 120


# -- tuner wiring -------------------------------------------------------------------

class TestTunerWiring:
    def test_flag_on_lif_zeroes_clamp_and_aq_only(self, tmp_path, monkeypatch):
        _set_flag(monkeypatch, True)
        clamp = _clamp_tuner(tmp_path / "clamp")
        aq = _aq_tuner(tmp_path / "aq")
        noise = _noise_tuner(tmp_path / "noise")
        try:
            assert clamp._fast_steps_per_rate == 0
            assert aq._fast_steps_per_rate == 0
            assert noise._fast_steps_per_rate == 3
        finally:
            clamp.close(); aq.close(); noise.close()

    def test_flag_off_keeps_configured_steps(self, tmp_path, monkeypatch):
        _set_flag(monkeypatch, False)
        clamp = _clamp_tuner(tmp_path / "clamp")
        aq = _aq_tuner(tmp_path / "aq")
        try:
            assert clamp._fast_steps_per_rate == 3
            assert aq._fast_steps_per_rate == 3
        finally:
            clamp.close(); aq.close()

    def test_flag_on_non_lif_untouched(self, tmp_path, monkeypatch):
        _set_flag(monkeypatch, True)
        clamp = _clamp_tuner(tmp_path / "clamp", spiking_mode="ttfs_quantized")
        aq = _aq_tuner(tmp_path / "aq", spiking_mode="ttfs_quantized")
        try:
            assert clamp._fast_steps_per_rate == 3
            assert aq._fast_steps_per_rate == 3
        finally:
            clamp.close(); aq.close()

    def test_flag_on_boosts_lif_stabilize_budget(self, tmp_path, monkeypatch):
        _set_flag(monkeypatch, True)
        tuner = _lif_tuner(tmp_path, stabilize_steps=600)
        try:
            assert tuner._fast_stabilize_steps == 2000
        finally:
            tuner.close()

    def test_flag_off_keeps_lif_stabilize_budget(self, tmp_path, monkeypatch):
        _set_flag(monkeypatch, False)
        tuner = _lif_tuner(tmp_path, stabilize_steps=600)
        try:
            assert tuner._fast_stabilize_steps == 600
        finally:
            tuner.close()


# -- run contracts -------------------------------------------------------------------

class TestRunContracts:
    def _run_seeded(self, builder, tmp_path, monkeypatch, enabled, **kw):
        _set_flag(monkeypatch, enabled)
        torch.manual_seed(0)
        tuner = builder(tmp_path, **kw)
        try:
            tuner.run()
        finally:
            tuner.close()
        return tuner

    def test_lif_inert_ladders_walk_all_rungs_without_training(
        self, tmp_path, monkeypatch,
    ):
        for name, builder in (("clamp", _clamp_tuner), ("aq", _aq_tuner)):
            tuner = self._run_seeded(builder, tmp_path / name, monkeypatch, True)
            assert tuner._fast_steps_per_rate == 0
            assert tuner._fast_optimizer_steps == 0, name
            assert tuner._committed_rate == pytest.approx(1.0), name
            assert [e["outcome"] for e in tuner._cycle_log] == \
                ["commit"] * len(tuner._fixed_ladder_rates), name

    def test_flag_on_non_lif_run_is_bit_identical(self, tmp_path, monkeypatch):
        t_off = self._run_seeded(
            _aq_tuner, tmp_path / "off", monkeypatch, False,
            spiking_mode="ttfs_quantized",
        )
        t_on = self._run_seeded(
            _aq_tuner, tmp_path / "on", monkeypatch, True,
            spiking_mode="ttfs_quantized",
        )
        sd_off, sd_on = t_off.model.state_dict(), t_on.model.state_dict()
        assert sd_off.keys() == sd_on.keys()
        for key in sd_off:
            assert torch.equal(sd_off[key], sd_on[key]), key
        assert [e["post_acc"] for e in t_off._cycle_log] == \
            [e["post_acc"] for e in t_on._cycle_log]

    def test_flag_off_matches_unset_in_lif(self, tmp_path, monkeypatch):
        t_unset = self._run_seeded(_clamp_tuner, tmp_path / "unset", monkeypatch, False)
        monkeypatch.setenv(env.MBH_LIF_REALLOC_VAR, "0")
        torch.manual_seed(0)
        t_zero = _clamp_tuner(tmp_path / "zero")
        try:
            t_zero.run()
        finally:
            t_zero.close()
        sd_a, sd_b = t_unset.model.state_dict(), t_zero.model.state_dict()
        for key in sd_a:
            assert torch.equal(sd_a[key], sd_b[key]), key
