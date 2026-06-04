"""TTFS-cycle fine-tuning: train through the genuine segment-aware spike forward.

The cascaded ``ttfs_cycle_based`` deployment runs each neural segment as a
single-spike, ramp-integrate, fire-once simulation with value-domain compute ops
between segments and a host-side encoding layer (value -> TTFS spike) at each
segment entry. So -- like LIF's cycle-accurate path -- fine-tuning runs the model
on actual TTFS spike trains via :class:`TTFSSegmentForward` (gradient through the
per-cycle dynamics), not a pointwise analytical surrogate.

TTFS single-spike decode is non-linear in a partial old/spike blend (unlike LIF's
mean decode), so this tuner trains **pure spike** (rate pinned at 1.0) and relies
on KD against the frozen pre-step teacher for recovery.
"""

from __future__ import annotations

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import KDBlendAdaptationTuner


class _SegmentSpikeForward:
    """Picklable ``model.forward`` override driving the segment-aware spike sim."""

    def __init__(self, mapper_repr, T: int):
        self._driver = TTFSSegmentForward(mapper_repr, T)

    def __call__(self, x):
        return self._driver(x)


class TTFSCycleAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp to the TTFS spike node, training through the segment-aware spike forward."""

    _target_activation_type = "TTFS"

    def _configure(self) -> None:
        self.name = "TTFS Cycle Fine-Tuning"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        self._firing_mode = str(self.pipeline.config.get("firing_mode", "TTFS"))
        from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode

        self._bias_mode = resolve_bias_mode(self.pipeline.config)
        self._patched_forward = False
        self.adaptation_manager.ttfs_active = True

    def _make_target_activation(self, perceptron) -> TTFSActivation:
        return TTFSActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            input_scale=perceptron.input_activation_scale,
            bias=perceptron.layer.bias,
            thresholding_mode=self._thresholding_mode,
            firing_mode=self._firing_mode,
            encoding=getattr(perceptron, "is_encoding_layer", False),
            bias_mode=self._bias_mode,
        )

    # -- pure spike: pin the blend at full rate ------------------------------
    def _install_blend(self) -> None:
        super()._install_blend()
        self._set_rate(1.0)

    def _set_rate(self, rate: float) -> None:  # noqa: ARG002 - always full rate
        super()._set_rate(1.0)

    def _get_rates(self):
        return [1.0 for _ in self.model.get_perceptrons()]

    # -- install / remove the spike-train forward ----------------------------
    def _install_spike_forward(self) -> None:
        assert "forward" not in self.model.__dict__, (
            "TTFSCycleAdaptationTuner: model.forward already patched."
        )
        self._patched_forward = True
        self.model.forward = _SegmentSpikeForward(self.model.get_mapper_repr(), self._T)

    def _remove_spike_forward(self) -> None:
        if getattr(self, "_patched_forward", False):
            try:
                del self.model.forward
            except AttributeError:
                pass
            self._patched_forward = False

    def _after_install_blend(self) -> None:
        self._install_spike_forward()

    def _after_run(self):
        """Mirror ``LIFAdaptationTuner._after_run``: subsume the decorators, then
        **re-install the genuine cascade forward** so the committed metric, the
        recovery loop, and every downstream step (WQ/NormFusion/SCM) run the exact
        deployed single-spike dynamics — not the analytical staircase. Stripping it
        here (the old behaviour) let fine-tuning commit an analytical metric that
        the chip never reproduces."""
        self._continue_to_full_rate()
        self._set_rate(1.0)
        # Rebuild the (decorator-free, ttfs_active) activations, then reinstall the
        # spike forward on top of the finalized mapper graph.
        self._remove_spike_forward()
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        self._install_spike_forward()

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric
