"""LIF adaptation tuner: blend ramp to LIFActivation with KD recovery."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import (
    ChipInputQuantizer,
    LIFActivation,
)
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    BlendActivation,
    KDBlendAdaptationTuner,
    _InstalledForward,
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

    def _finalize_forward(self):
        if self._cycle_accurate:
            return _ChipAlignedNFForward(self.model, self._T)
        return None

    def _before_finalize_rebuild(self) -> None:
        # lif_active before the rebuild so the committed activations subsume the
        # clamp/quant/shift decorators (the base _finalize installs any forward).
        self.adaptation_manager.lif_active = True

    def _after_finalize_rebuild(self) -> None:
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(self.model, True)
