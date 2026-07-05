"""LIF subsumed-ladder drop — recipe default (MBH X3, from X2/E2 + X1 inertness).

X1 measured the Clamp and ActivationQuantization fast ladders behaviorally inert
in LIF mode (their decorators are subsumed by the spiking node — blended == D-hat,
rho == 1 on all rungs); X2/E2 proved zeroing their training changes deployed
accuracy by nothing. The LIF recipe therefore DROPS that training by default:
in lif mode (spiking_semantics predicate) those two ladders train 0 steps per
rung, while installation, rung walk, probes, and finalize contracts all still
run. Non-lif modes and every other rate ladder (noise, shift, activation
adaptation) are untouched. The reclaimed budget funds the LIF endpoint-recovery
stage (P1'').
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
from mimarsinan.tuning.orchestration import mbh_ledger
from mimarsinan.tuning.orchestration.adaptation_manager import (
    lif_subsumed_ladder_steps,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)


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
    def test_lif_zeroes_subsumed_rates(self):
        cfg = {"spiking_mode": "lif"}
        assert lif_subsumed_ladder_steps(cfg, "clamp_rate", 120) == 0
        assert lif_subsumed_ladder_steps(cfg, "quantization_rate", 120) == 0

    def test_unsubsumed_rates_keep_their_steps(self):
        cfg = {"spiking_mode": "lif"}
        for attr in ("noise_rate", "shift_rate", "activation_adaptation_rate"):
            assert lif_subsumed_ladder_steps(cfg, attr, 120) == 120

    def test_non_lif_modes_keep_their_steps(self):
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            cfg = {"spiking_mode": mode}
            assert lif_subsumed_ladder_steps(cfg, "clamp_rate", 120) == 120
            assert lif_subsumed_ladder_steps(cfg, "quantization_rate", 120) == 120


# -- tuner wiring -------------------------------------------------------------------

class TestTunerWiring:
    def test_lif_zeroes_clamp_and_aq_only(self, tmp_path):
        clamp = _clamp_tuner(tmp_path / "clamp")
        aq = _aq_tuner(tmp_path / "aq")
        noise = _noise_tuner(tmp_path / "noise")
        try:
            assert clamp._fast_steps_per_rate == 0
            assert aq._fast_steps_per_rate == 0
            assert noise._fast_steps_per_rate == 3
        finally:
            clamp.close(); aq.close(); noise.close()

    def test_non_lif_keeps_configured_steps(self, tmp_path):
        clamp = _clamp_tuner(tmp_path / "clamp", spiking_mode="ttfs_quantized")
        aq = _aq_tuner(tmp_path / "aq", spiking_mode="ttfs_quantized")
        try:
            assert clamp._fast_steps_per_rate == 3
            assert aq._fast_steps_per_rate == 3
        finally:
            clamp.close(); aq.close()


# -- run contracts -------------------------------------------------------------------

class TestRunContracts:
    def test_lif_inert_ladders_walk_all_rungs_without_training(
        self, tmp_path, monkeypatch,
    ):
        # Never-regressing D-hat keeps the (default) gate on the accept path so
        # the whole rung walk is exercised.
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
        for name, builder in (("clamp", _clamp_tuner), ("aq", _aq_tuner)):
            torch.manual_seed(0)
            tuner = builder(tmp_path / name)
            try:
                tuner.run()
            finally:
                tuner.close()
            assert tuner._fast_steps_per_rate == 0
            assert tuner._fast_optimizer_steps == 0, name
            assert tuner._committed_rate == pytest.approx(1.0), name
            assert [e["outcome"] for e in tuner._cycle_log] == \
                ["commit"] * len(tuner._fixed_ladder_rates), name
