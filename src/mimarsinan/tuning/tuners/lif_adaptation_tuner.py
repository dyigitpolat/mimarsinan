"""LIF adaptation tuner: blend ramp to LIFActivation with KD recovery."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import (
    ChipInputQuantizer,
    LIFActivation,
    run_cycle_accurate,
)
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    BlendActivation,
    KDBlendAdaptationTuner,
    _InstalledForward,
)


class _CycleAccurateForward(_InstalledForward):
    """Picklable ``model.forward`` override that drives ``run_cycle_accurate``
    on top of the model's class-level forward, used during the LIF blend ramp."""

    def _run(self, x):
        return run_cycle_accurate(
            self.model, x, self.T, forward_fn=self._unpatched_forward,
        )


class _ChipAlignedNFForward(_InstalledForward):
    """Picklable ``model.forward`` override installed post-blend (rate==1.0).
    Routes NF through ``chip_aligned_segment_forward`` so downstream calibrators
    (WQ, NormFusion, SCM probes) see the same forward the chip simulators run."""

    def _run(self, x):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

        return chip_aligned_segment_forward(self.model, x, self.T)


class LIFBlendActivation(BlendActivation):
    """Linear blend of old_activation and LIFActivation controlled by rate."""

    def __init__(
        self,
        old_activation: nn.Module,
        lif_activation: LIFActivation,
        rate: float = 0.0,
    ):
        super().__init__(old_activation, lif_activation, rate, target_type="LIF")

    @property
    def lif_activation(self) -> LIFActivation:
        return self.target_activation


class LIFAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp base_activation to LIFActivation with KD recovery."""

    _target_activation_type = "LIF"

    def _configure(self) -> None:
        self.name = "LIF Adaptation"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        self._cycle_accurate = bool(self.pipeline.config.get("cycle_accurate_lif_forward", False))
        # Legacy per-frame ramp leaks off the continuous teacher at rate 0 (the
        # blend is applied inside ``run_cycle_accurate``). Default OFF: the ramp
        # runs in the value domain (golden, non-destructive); the genuine
        # chip-aligned forward is installed at finalize either way. Opt back in
        # via ``legacy_lif_blend_ramp`` for the old behavior.
        self._legacy_blend_ramp = bool(
            self.pipeline.config.get("legacy_lif_blend_ramp", False)
        )
        from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode

        self._bias_mode = resolve_bias_mode(self.pipeline.config)

    def _make_target_activation(self, perceptron) -> LIFActivation:
        return LIFActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            thresholding_mode=self._thresholding_mode,
            firing_mode=str(self.pipeline.config.get("firing_mode", "Default")),
            bias_mode=self._bias_mode,
        )

    def _make_blend(self, old, target, rate):
        return LIFBlendActivation(old, target, rate)

    def _after_make_target(self, perceptron, lif) -> None:
        if self._cycle_accurate:
            lif.use_cycle_accurate_trains = True

    def _wrap_encoding_input(self, perceptron) -> None:
        if getattr(perceptron, "is_encoding_layer", False):
            quantizer = ChipInputQuantizer(
                T=self._T,
                activation_scale=perceptron.input_activation_scale,
            )
            self._append_encoding_input_module(perceptron, quantizer)

    def _ramp_forward(self):
        """Legacy cycle-accurate ramp forward (per-frame blend). Installed only
        when ``cycle_accurate_lif_forward`` AND ``legacy_lif_blend_ramp`` are set;
        otherwise the ramp runs in the value domain (no forward — reproduces the
        continuous teacher at rate 0, the golden non-destructive ramp). The
        genuine chip-aligned forward is installed at finalize either way."""
        if self._cycle_accurate and self._legacy_blend_ramp:
            return _CycleAccurateForward(self.model, self._T)
        return None

    def _finalize_forward(self):
        if self._cycle_accurate:
            return _ChipAlignedNFForward(self.model, self._T)
        return None

    def _finalize(self) -> None:
        self.adaptation_manager.lif_active = True
        self._update_target_activations()
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(self.model, True)
        fwd = self._finalize_forward()
        if fwd is not None:
            self._install_forward(fwd)
