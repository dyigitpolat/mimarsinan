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
)


class _CycleAccurateForward:
    """Picklable ``model.forward`` override that drives ``run_cycle_accurate``
    on top of the model's class-level forward, used during the LIF blend ramp."""

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)

    def _call_unpatched_forward(self, x):
        return type(self.model).forward(self.model, x)

    def __call__(self, x):
        return run_cycle_accurate(
            self.model, x, self.T,
            forward_fn=self._call_unpatched_forward,
        )


class _ChipAlignedNFForward:
    """Picklable ``model.forward`` override installed post-blend (rate==1.0).
    Routes NF through ``chip_aligned_nf_forward`` so downstream calibrators
    (WQ, NormFusion, SCM probes) see the same forward the chip simulators run."""

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)

    def __call__(self, x):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_nf_forward

        return chip_aligned_nf_forward(self.model, x, self.T)


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
        self._patched_forward = False

    def _make_target_activation(self, perceptron) -> LIFActivation:
        return LIFActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            thresholding_mode=self._thresholding_mode,
            firing_mode=str(self.pipeline.config.get("firing_mode", "Default")),
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
            if isinstance(perceptron.input_activation, nn.Identity):
                perceptron.input_activation = quantizer
            else:
                perceptron.input_activation = nn.Sequential(
                    perceptron.input_activation, quantizer,
                )

    def _after_install_blend(self) -> None:
        if self._cycle_accurate:
            self._install_cycle_accurate_forward()

    def _install_cycle_accurate_forward(self) -> None:
        """Patch model.forward to run_cycle_accurate for the duration of the blend ramp."""
        assert "forward" not in self.model.__dict__, (
            "LIFAdaptationTuner: model.forward is already patched; double-install "
            "would shadow the prior wrapper. Call _after_run on the previous tuner "
            "first."
        )
        self._patched_forward = True
        self.model.forward = _CycleAccurateForward(model=self.model, T=self._T)

    def _after_run(self):
        try:
            self._continue_to_full_rate()
            self._set_rate(1.0)
        finally:
            if getattr(self, "_patched_forward", False):
                try:
                    del self.model.forward
                except AttributeError:
                    pass
                self._patched_forward = False

        self.adaptation_manager.lif_active = True
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(self.model, True)
            assert "forward" not in self.model.__dict__, (
                "LIFAdaptationTuner._after_run: model.forward is already patched; "
                "did the blend-ramp wrapper leak?"
            )
            self.model.forward = _ChipAlignedNFForward(self.model, self._T)

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric
