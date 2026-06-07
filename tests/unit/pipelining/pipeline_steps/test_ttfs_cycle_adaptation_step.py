"""Unit tests for TTFSCycleAdaptationStep.

Final-polish fine-tuning for ``spiking_mode == 'ttfs_cycle_based'``. Both
schedules ramp the per-perceptron TTFS blend gradually in the value domain (the
golden LIF non-destructive ramp); the genuine cross-layer dynamics are installed
only at finalize. Cascaded finalizes on the single-spike cascade forward
(``_SegmentSpikeForward``); synchronized keeps the class-level analytical
staircase composition with a TTFS grid-snap STE on segment-entry inputs — the
semantics the deployed chip actually executes.
"""

import numpy as np
import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
    TTFSCycleAdaptationStep,
)
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner

SCHEDULES = ("cascaded", "synchronized")


def _seed_ttfs_cycle_step(mock_pipeline, *, schedule="cascaded", target_metric=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "ttfs_cycle_based"
    mock_pipeline.config["ttfs_cycle_schedule"] = schedule
    mock_pipeline.config["activation_quantization"] = True
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline.config.setdefault("simulation_steps", 16)
    mock_pipeline._target_metric = target_metric

    mock_pipeline.seed("model", model, step_name="Activation Quantization")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    return model, am


def _run_step(mock_pipeline):
    step = TTFSCycleAdaptationStep(mock_pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


def _encoding_input_quantizers(perceptron):
    modules = [perceptron.input_activation]
    if isinstance(perceptron.input_activation, torch.nn.Sequential):
        modules = list(perceptron.input_activation)
    return [m for m in modules if isinstance(m, TTFSInputGridQuantizer)]


class TestTunerCreation:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_tuner_created(self, mock_pipeline, schedule):
        _seed_ttfs_cycle_step(mock_pipeline, schedule=schedule)
        step = _run_step(mock_pipeline)
        assert step.tuner is not None
        assert isinstance(step.tuner, TTFSCycleAdaptationTuner)


class TestValidate:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_validate_returns_float_in_range(self, mock_pipeline, schedule):
        _seed_ttfs_cycle_step(mock_pipeline, schedule=schedule)
        step = _run_step(mock_pipeline)
        result = step.validate()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestFinalState:
    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_ttfs_active_set_and_activation_is_ttfs(self, mock_pipeline, schedule):
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule=schedule)
        _run_step(mock_pipeline)
        assert am.ttfs_active is True
        for p in model.get_perceptrons():
            # base_activation is the blend, fully ramped to the TTFS target.
            assert p.base_activation.rate == pytest.approx(1.0)
            assert p.base_activation.activation_type == "TTFS"
            assert isinstance(p.base_activation.target_activation, TTFSActivation)

    def test_genuine_spike_forward_persists_after_step_cascaded(self, mock_pipeline):
        """Cascaded mirrors LIF: the genuine cycle-based cascade forward stays
        installed as ``model.forward`` after the step, so the committed metric,
        recovery, and every downstream calibration run the exact deployed
        single-spike dynamics (fine-tune↔deploy parity)."""
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            _SegmentSpikeForward,
        )

        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        _run_step(mock_pipeline)
        assert isinstance(model.__dict__.get("forward"), _SegmentSpikeForward), (
            "TTFS-cycle fine-tuning must leave the genuine spike forward installed "
            "(like LIF's _ChipAlignedNFForward), not revert to the analytical path."
        )

    def test_synchronized_uses_analytical_forward_no_instance_override(
        self, mock_pipeline
    ):
        """Synchronized deployment composes per-group analytical staircases — the
        cascade spike walk is the WRONG dynamics for it (the 2026-06-06 incident).
        The class-level forward through the ramped TTFSActivation blend IS the
        analytical staircase; no instance override may shadow it."""
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule="synchronized")
        _run_step(mock_pipeline)
        assert "forward" not in model.__dict__, (
            "Synchronized TTFS fine-tuning must not install the cascade spike "
            "forward; the deployed schedule runs the analytical staircase."
        )
        for p in model.get_perceptrons():
            assert p.base_activation.target_activation._cycle_accurate_mode is False

    def test_synchronized_wraps_segment_entry_after_host_encoder(self, mock_pipeline):
        """The synchronized wire contract grid-quantizes every hybrid stage
        input q(x). The seam is the FIRST ON-CHIP perceptron of each segment:
        with a host encoding layer (subsume-style, tiny model marks p1), the
        snap belongs on p2's input — the value entering the first NeuralCore —
        not on the host op's input."""
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule="synchronized")
        _run_step(mock_pipeline)
        T = int(mock_pipeline.config["simulation_steps"])
        p1, p2 = model.get_perceptrons()
        assert getattr(p1, "is_encoding_layer", False) is True
        assert _encoding_input_quantizers(p1) == [], (
            "host encoding layer is not on-chip; its input is not the q(x) seam"
        )
        (quantizer,) = _encoding_input_quantizers(p2)
        assert quantizer.T == T

    def test_synchronized_wraps_raw_input_entry_under_offload_style(
        self, mock_pipeline
    ):
        """With no host encoder (offload: is_encoding_layer never set), the
        raw-input-fed perceptron is the segment entry and gets the snap —
        the 2026-06-07 regression run showed the wrap was inert under offload."""
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule="synchronized")
        for p in model.get_perceptrons():
            p.is_encoding_layer = False
        _run_step(mock_pipeline)
        p1, p2 = model.get_perceptrons()
        assert len(_encoding_input_quantizers(p1)) == 1
        assert _encoding_input_quantizers(p2) == []

    def test_cascaded_does_not_wrap_encoding_input(self, mock_pipeline):
        """Cascaded NF feeds genuine spike trains; the segment walk owns the
        value->spike encode, so no analytical grid snap may be installed."""
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        _run_step(mock_pipeline)
        for p in model.get_perceptrons():
            assert _encoding_input_quantizers(p) == []

    @pytest.mark.parametrize("schedule", SCHEDULES)
    def test_decorators_subsumed_no_clamp_quant_wrapping(self, mock_pipeline, schedule):
        # With ttfs_active, update_activation must not wrap the blend in clamp/quant
        # decorators (the TTFS kernel does that internally).
        model, am = _seed_ttfs_cycle_step(mock_pipeline, schedule=schedule)
        am.clamp_rate = 1.0
        am.quantization_rate = 1.0
        _run_step(mock_pipeline)
        for p in model.get_perceptrons():
            decorators = getattr(p.activation, "decorators", [])
            assert len(decorators) == 0


class TestCascadedGradualRamp:
    """Cascaded ramps gradually through the GENUINE deployed cascade via a
    whole-model output blend (genuine-gradual, default on): non-destructive at
    r=0 (continuous teacher), bit-exact deployed dynamics at r=1, so the gradual
    phase trains through the deployed dynamics (incl. offload's input encode)."""

    def test_cascaded_makes_natural_blend_progress(self, mock_pipeline):
        """Regression for the incident's secondary anomaly: cascaded used to pin
        rate at 1.0 (one-shot jump; ``natural adaptation reached only 0.0000``).
        The gradual ramp must make genuine blend progress on its own, like
        synchronized and LIF."""
        torch.manual_seed(7)
        _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        step = _run_step(mock_pipeline)
        assert step.tuner._natural_rate > 0.0, (
            "Cascaded adaptation made no natural blend progress — the rate-pin "
            "regression."
        )

    def test_cascaded_default_ramp_is_value_domain(self, mock_pipeline):
        """Default (genuine-gradual off): the cascaded ramp is the value-domain
        staircase proxy (no instance forward during the ramp); finalize installs
        the genuine single-spike cascade. The proxy ramp is the higher-accuracy
        default (genuine-gradual under-optimizes the single-spike cascade)."""
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            _SegmentSpikeForward,
        )

        _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        step = _run_step(mock_pipeline)
        assert step.tuner._ramp_forward() is None
        assert isinstance(step.tuner._finalize_forward(), _SegmentSpikeForward)

    def test_cascaded_committed_model_runs_genuine_cascade(self, mock_pipeline):
        """The committed model's installed forward IS the genuine single-spike
        cascade: bit-exact to a freshly built cascade forward on the same
        weights (so the committed metric and downstream steps run the deployed
        dynamics, not the staircase ramp proxy)."""
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            _SegmentSpikeForward,
        )

        model, _ = _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        _run_step(mock_pipeline)
        installed = model.__dict__.get("forward")
        assert isinstance(installed, _SegmentSpikeForward)

        T = int(mock_pipeline.config["simulation_steps"])
        x = torch.randn(3, *mock_pipeline.config["input_shape"])
        fresh = _SegmentSpikeForward(model, T)
        with torch.no_grad():
            torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)

    def test_cascaded_finalize_marks_lr_refind(self, mock_pipeline):
        """Cascaded finalize swaps in the genuine cascade forward, so
        stabilization must re-find the LR on the deployed dynamics (the
        staircase-proxy ramp's cached LR is stale)."""
        _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        step = _run_step(mock_pipeline)
        assert step.tuner._stabilization_refinds_lr is True

    def test_phase_seconds_recorded(self, mock_pipeline):
        """Phase timing is recorded for the gradual ramp + stabilization."""
        _seed_ttfs_cycle_step(mock_pipeline, schedule="cascaded")
        step = _run_step(mock_pipeline)
        assert set(step.tuner._phase_seconds) >= {"gradual", "stabilization"}


class TestSynchronizedAdaptation:
    def test_natural_adaptation_progresses(self, mock_pipeline):
        """Regression for the incident's secondary anomaly: under the cascade
        objective the KD blend never adapted naturally (rate stuck at exactly
        0.0000, forced to 1.0). The analytical-staircase objective must make
        genuine blend progress on its own (full 1.0 is budget/noise dependent
        on the 10-sample TinyDataset, so only zero progress is a failure)."""
        torch.manual_seed(7)
        _seed_ttfs_cycle_step(mock_pipeline, schedule="synchronized")
        step = _run_step(mock_pipeline)
        assert step.tuner._natural_rate > 0.0, (
            "Synchronized adaptation made no natural blend progress — the "
            "incident's cascade-objective signature."
        )


class TestSynchronizedNFMatchesContractSemantics:
    """Per-element locks: the committed NF semantics equal the numpy wire SSOT
    (the kernels the contract runner / chip simulators execute). This is the
    rung-1↔rung-2 semantic unification R1 establishes; the mapping-level
    per-neuron gate lands in R5."""

    def _committed_model(self, mock_pipeline):
        model, _ = _seed_ttfs_cycle_step(mock_pipeline, schedule="synchronized")
        _run_step(mock_pipeline)
        return model, int(mock_pipeline.config["simulation_steps"])

    def test_activation_matches_ttfs_quantized_kernel(self, mock_pipeline):
        from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation_np

        model, S = self._committed_model(mock_pipeline)
        for p in model.get_perceptrons():
            activation = p.base_activation.target_activation
            scale = float(activation.activation_scale)
            grid = torch.arange(S + 1, dtype=torch.float64) / S * scale
            dense = torch.linspace(-0.5, 1.5, 101, dtype=torch.float64) * scale
            V = torch.cat([grid, dense])
            with torch.no_grad():
                got = activation(V).numpy()
            expected = ttfs_quantized_activation_np(V.numpy(), scale, S) * scale
            np.testing.assert_allclose(
                got, expected, rtol=0, atol=1e-9,
                err_msg=(
                    "Committed synchronized NF activation diverges from the "
                    "deployment staircase kernel."
                ),
            )

    def test_encoding_snap_matches_wire_quantize(self, mock_pipeline):
        from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
            ttfs_input_grid_quantize,
        )

        model, S = self._committed_model(mock_pipeline)
        entries = [
            p for p in model.get_perceptrons() if _encoding_input_quantizers(p)
        ]
        assert entries, "the committed model must carry segment-entry q(x) snaps"
        for p in entries:
            (quantizer,) = _encoding_input_quantizers(p)
            scale = float(quantizer.activation_scale)
            ties = (torch.arange(S, dtype=torch.float64) + 0.5) / S
            dense = torch.linspace(0.0, 1.0, 101, dtype=torch.float64)
            x = torch.cat([ties, dense]) * scale
            with torch.no_grad():
                got = quantizer(x).numpy()
            expected = ttfs_input_grid_quantize((x / scale).numpy(), S) * scale
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
