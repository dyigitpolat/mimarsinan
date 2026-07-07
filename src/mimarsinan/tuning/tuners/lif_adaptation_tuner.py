"""LIF adaptation tuner: blend ramp to LIFActivation with KD recovery."""

from __future__ import annotations

from typing import cast

import torch.nn as nn

from mimarsinan.models.nn.activations import (
    ChipInputQuantizer,
    LIFActivation,
)
from mimarsinan.tuning.orchestration.blend_ramp import (
    BlendActivation,
    run_teacher_distmatch,
)
from mimarsinan.spiking.gain_correction import per_perceptron_cascade_depth
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import (
    run_endpoint_recovery,
)
from mimarsinan.tuning.orchestration.install_resolution import (
    capture_install_stats,
    emit_temporal_gauge,
    lif_temporal_gauge,
)
from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)
from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan
from mimarsinan.tuning.orchestration.mbh_tanneal import (
    TAnnealRealizableRamp,
    apply_simulation_steps,
)


class _ChipAlignedNFForward(LazyExecutorForward):
    """Picklable ``model.forward`` override installed post-blend (rate==1.0).

    Routes NF through ``chip_aligned_segment_forward`` so downstream calibrators see
    the same forward the chip simulators run.
    """

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
        # The constructor only accepts a LIFActivation target.
        return cast(LIFActivation, self.target_activation)


class LIFAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp base_activation to LIFActivation with KD recovery."""

    _target_activation_type = "LIF"

    def _configure(self) -> None:
        self.name = "LIF Adaptation"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode

        self._bias_mode = resolve_bias_mode(self.pipeline.config)
        plan = LifAdaptationPlan.resolve(self.pipeline.config)
        self._adaptation_plan = plan
        self._cycle_accurate = plan.cycle_accurate
        self._consume_optimization_driver(
            rates=plan.blend_fast_rates,
            steps_per_rate=plan.blend_fast_steps_per_rate,
            eta_min_factor=plan.blend_fast_lr_eta_min,
        )
        self._tanneal = plan.tanneal_schedule(self._fixed_ladder_rates)
        self._endpoint_recovery_steps = plan.endpoint_recovery_steps
        self._lif_distmatch_stats = None
        self._theta_cotrain = plan.theta_cotrain
        self._emit_a6_temporal_gauge()

    def _emit_a6_temporal_gauge(self) -> None:
        """[MBH-A6(ii)] the lockstep window pre-flight at the LIF install anchor
        (warn-only); the capture is cursor-isolated so the trajectory is untouched."""
        stats = capture_install_stats(self)
        depths = per_perceptron_cascade_depth(self.model.get_mapper_repr())
        gauge = lif_temporal_gauge(stats, depths, window=self._T)
        emit_temporal_gauge(type(self).__name__, gauge)

    def _make_ramp_strategy(self):
        if self._tanneal is not None:
            return TAnnealRealizableRamp(self._tanneal)
        return super()._make_ramp_strategy()

    def _force_full_transform_on_clone(self, clone) -> None:
        super()._force_full_transform_on_clone(clone)
        if self._tanneal is not None:
            # D-hat is the FULL target behavior at target_T, not the rung's T.
            apply_simulation_steps(clone, self._tanneal.target_T)

    def _post_stabilization_hook(self):
        if self._adaptation_plan.distmatch:
            self._calibrate_to_teacher_distribution()
        if getattr(self, "_fixed_ladder_policy", False):
            # P1'': the chip-aligned NF forward is installed at finalize, so the
            # endpoint stage trains the deployed composition itself.
            run_endpoint_recovery(self, base_steps=self._endpoint_recovery_steps)

    def _calibrate_to_teacher_distribution(self) -> None:
        """DFQ-match the deployed LIF cascade's per-neuron mean to the frozen teacher ANN's.

        No-op unless the cycle-accurate cascade is deployed (it supplies the decoded values).
        """
        if not self._cycle_accurate:
            return
        from mimarsinan.spiking.lif_distribution_matching import (
            match_lif_activation_distributions,
        )

        plan = self._adaptation_plan
        self._lif_distmatch_stats = run_teacher_distmatch(
            self,
            match_lif_activation_distributions,
            n_batches=plan.distmatch_cal_batches,
            bias_iters=plan.distmatch_bias_iters,
            eta=plan.distmatch_bias_eta,
        )

    def _make_target_activation(self, perceptron) -> LIFActivation:
        return LIFActivation(
            T=self._T,
            activation_scale=perceptron.activation_scale,
            thresholding_mode=self._thresholding_mode,
            firing_mode=str(self.pipeline.config.get("firing_mode", "Default")),
            bias_mode=self._bias_mode,
        )

    def _make_blend(self, old, target, rate):
        # ``_make_target_activation`` above builds the LIFActivation target.
        return LIFBlendActivation(old, cast(LIFActivation, target), rate)

    def _after_make_target(self, perceptron, target) -> None:
        if self._cycle_accurate:
            cast(LIFActivation, target).use_cycle_accurate_trains = True

    def _wrap_encoding_input(self, perceptron) -> None:
        if getattr(perceptron, "is_encoding_layer", False):
            quantizer = ChipInputQuantizer(
                T=self._T,
                activation_scale=perceptron.input_activation_scale,
            )
            self._append_encoding_input_module(perceptron, quantizer)

    def _finalize_forward_for(self, model):
        if self._cycle_accurate:
            return _ChipAlignedNFForward(model, self._T)
        return None

    def _before_finalize_rebuild(self, model=None) -> None:
        self.adaptation_manager.lif_active = True

    def _after_finalize_rebuild(self, model=None) -> None:
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(
                self.model if model is None else model, True,
            )
