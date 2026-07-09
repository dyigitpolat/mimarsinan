"""SpikingDeploymentContract: the cross-side deployment-semantics SSOT."""

import re
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.deployment_contract import SpikingDeploymentContract


def _cfg(**overrides):
    cfg = {
        "spiking_mode": "ttfs_cycle_based",
        "firing_mode": "TTFS",
        "thresholding_mode": "<=",
        "spike_generation_mode": "TTFS",
        "simulation_steps": 4,
        "ttfs_cycle_schedule": "synchronized",
        "encoding_layer_placement": "offload",
    }
    cfg.update(overrides)
    return cfg


class TestFactory:
    def test_from_pipeline_config_populates_all_axes(self):
        contract = SpikingDeploymentContract.from_pipeline_config(_cfg())
        assert contract.spiking_mode == "ttfs_cycle_based"
        assert contract.firing_mode == "TTFS"
        assert contract.thresholding_mode == "<="
        assert contract.spike_generation_mode == "TTFS"
        assert contract.spike_encoding_seed is None
        assert contract.simulation_steps == 4
        assert contract.ttfs_cycle_schedule == "synchronized"
        assert contract.encoding_layer_placement == "offload"
        assert contract.bias_mode in ("on_chip", "param_encoded")

    def test_behavior_composition_round_trips(self):
        cfg = _cfg(spike_encoding_seed=7)
        contract = SpikingDeploymentContract.from_pipeline_config(cfg)
        behavior = contract.behavior
        assert isinstance(behavior, NeuralBehaviorConfig)
        assert behavior == NeuralBehaviorConfig.from_deployment_config(cfg)
        # Per-backend helpers stay reachable through the composition.
        assert behavior.nevresim_compare_policy() == "InclusiveCompare"
        assert behavior.nevresim_reset_policy() == "ZeroReset"

    def test_schedule_is_normalized(self):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(ttfs_cycle_schedule=None)
        )
        assert contract.ttfs_cycle_schedule == "cascaded"
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(ttfs_cycle_schedule="bogus")
        )
        assert contract.ttfs_cycle_schedule == "cascaded"

    def test_simulation_steps_required(self):
        cfg = _cfg()
        del cfg["simulation_steps"]
        with pytest.raises(KeyError):
            SpikingDeploymentContract.from_pipeline_config(cfg)

    def test_simulation_step_timeout_defaults_to_unset(self):
        contract = SpikingDeploymentContract.from_pipeline_config(_cfg())
        assert contract.simulation_step_timeout_s is None

    def test_simulation_step_timeout_carries_the_config_value(self):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(simulation_step_timeout_s=300)
        )
        assert contract.simulation_step_timeout_s == 300.0


class TestDerivedGettersTruthTable:
    """The D4 kill-lock: one expected answer per (mode × schedule)."""

    CASES = [
        # (spiking_mode, schedule, is_sync, quantize_grid, forward_kind)
        ("ttfs_cycle_based", "cascaded", False, False, "segment_spike"),
        ("ttfs_cycle_based", "synchronized", True, True, "analytical_staircase"),
        ("ttfs", "cascaded", False, False, "analytical_staircase"),
        ("ttfs", "synchronized", False, False, "analytical_staircase"),
        ("ttfs_quantized", "cascaded", False, False, "analytical_staircase"),
        ("ttfs_quantized", "synchronized", False, False, "analytical_staircase"),
        ("lif", "cascaded", False, False, "lif_cycle"),
        ("lif", "synchronized", False, False, "lif_cycle"),
    ]

    @pytest.mark.parametrize(
        "mode,schedule,is_sync,quantize,kind", CASES,
        ids=[f"{m}-{s}" for m, s, *_ in CASES],
    )
    def test_derived_getters(self, mode, schedule, is_sync, quantize, kind):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(spiking_mode=mode, ttfs_cycle_schedule=schedule)
        )
        assert contract.is_synchronized() is is_sync
        assert contract.is_cascaded() is (
            mode == "ttfs_cycle_based" and schedule == "cascaded"
        )
        assert contract.quantize_stage_input_to_grid() is quantize
        assert contract.training_forward_kind() == kind
        # The floor+half-step-bias convention: ttfs_quantized OR synchronized.
        assert contract.uses_ttfs_floor_ceil_convention() is (
            mode == "ttfs_quantized" or is_sync
        )


class TestReservedPerCoreSeam:
    def test_derived_getters_accept_core_and_return_global_answer(self):
        contract = SpikingDeploymentContract.from_pipeline_config(_cfg())
        sentinel = object()
        assert contract.is_synchronized(core=sentinel) == contract.is_synchronized()
        assert contract.is_cascaded(core=sentinel) == contract.is_cascaded()
        assert (
            contract.quantize_stage_input_to_grid(core=sentinel)
            == contract.quantize_stage_input_to_grid()
        )
        assert (
            contract.training_forward_kind(core=sentinel)
            == contract.training_forward_kind()
        )


class TestCalibrationPipelineAccessor:
    """E3: the contract resolves the conversion-health pipeline per (firing × sync)."""

    def test_cascaded_cycle_opts_in(self):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded")
        )
        cal = contract.calibration_pipeline({"ttfs_gain_correction": True})
        assert cal.gain_cold is True
        assert cal.gain_active is True

    def test_synchronized_cycle_is_inert(self):
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized")
        )
        cal = contract.calibration_pipeline(
            {"ttfs_gain_correction": True, "ttfs_theta_cotrain": True}
        )
        assert cal == CalibrationPipeline.inert()

    def test_lif_is_inert(self):
        from mimarsinan.tuning.orchestration.calibration_pipeline import (
            CalibrationPipeline,
        )

        contract = SpikingDeploymentContract.from_pipeline_config(_cfg(spiking_mode="lif"))
        cal = contract.calibration_pipeline({"ttfs_gain_correction": True})
        assert cal == CalibrationPipeline.inert()

    def test_distmatch_driver_threads_through(self):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded")
        )
        assert contract.calibration_pipeline({}, distmatch_driven=True).distmatch is True
        assert contract.calibration_pipeline({}, distmatch_driven=False).distmatch is False

    def test_core_kwarg_returns_global_answer(self):
        contract = SpikingDeploymentContract.from_pipeline_config(
            _cfg(spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded")
        )
        sentinel = object()
        assert (
            contract.calibration_pipeline({"ttfs_gain_correction": True}, core=sentinel)
            == contract.calibration_pipeline({"ttfs_gain_correction": True})
        )


class TestSingleReaderInvariant:
    """``from_pipeline_config`` is the only place reading these config keys."""

    # Files legitimately reading the raw config key:
    #  - the factory itself
    #  - config_schema (defines/derives the key)
    #  - deployment_specs (step selection on predicates, per the design doc)
    ALLOWLIST = (
        "chip_simulation/deployment_contract.py",
        "config_schema/",
        "pipelining/core/pipelines/deployment_specs.py",
    )
    PATTERN = re.compile(
        r"""(?:config|cfg)\s*(?:\.get\(\s*['"]ttfs_cycle_schedule['"]"""
        r"""|\[\s*['"]ttfs_cycle_schedule['"]\s*\])"""
    )

    def test_no_stray_schedule_config_reads(self):
        src_root = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"
        offenders = []
        for path in src_root.rglob("*.py"):
            rel = path.relative_to(src_root).as_posix()
            if any(rel.startswith(allowed) for allowed in self.ALLOWLIST):
                continue
            if self.PATTERN.search(path.read_text()):
                offenders.append(rel)
        assert offenders == [], (
            "ttfs_cycle_schedule must be read from pipeline config only by "
            f"SpikingDeploymentContract.from_pipeline_config; offenders: {offenders}"
        )
