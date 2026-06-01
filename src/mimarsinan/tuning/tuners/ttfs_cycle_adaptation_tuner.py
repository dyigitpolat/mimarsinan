"""TTFS-cycle fine-tuning tuner: blend ramp to the exact single-spike TTFS kernel.

Final polish for ``spiking_mode == 'ttfs_cycle_based'``: ramp each perceptron's
activation from the clamp/quantise-decorated ReLU produced by the prior steps to
:class:`TTFSCycleActivation` (the exact ``ttfs_quantized_activation``, no half-LSB
shift, ``S`` levels), recovering accuracy with KD against the pre-swap model. The
exact kernel subsumes the clamp/quant/shift decorators, so they are disabled
(``adaptation_manager.ttfs_active``) for the duration. No forward patching is
needed — the single-spike value is a closed-form one-pass activation.
"""

from __future__ import annotations

from mimarsinan.models.nn.activations import TTFSCycleActivation
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import KDBlendAdaptationTuner


class TTFSCycleAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp base activation to TTFSCycleActivation with KD recovery."""

    _target_activation_type = "TTFS"

    def _configure(self) -> None:
        self.name = "TTFS Cycle Fine-Tuning"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        # The TTFS kernel clamps + quantises internally; disable the decorators so
        # update_activation rebuilds each activation as the bare blend.
        self.adaptation_manager.ttfs_active = True

    def _blend_old_activation(self, perceptron):
        # Old side = the perceptron's current (clamp/quant-decorated) activation, so
        # the rate-0 blend matches the teacher exactly.
        return perceptron.activation

    def _make_target_activation(self, perceptron) -> TTFSCycleActivation:
        return TTFSCycleActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            thresholding_mode=self._thresholding_mode,
        )

    def _finalize(self) -> None:
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
