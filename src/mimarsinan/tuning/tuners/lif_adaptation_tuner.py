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
from mimarsinan.tuning.orchestration.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
    _InstalledForward,
)
from mimarsinan.tuning.orchestration.mbh_tanneal import (
    TAnnealRealizableRamp,
    TAnnealSchedule,
    apply_simulation_steps,
)


class _ChipAlignedNFForward(_InstalledForward):
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


def derive_lif_tanneal_schedule(config, *, ladder_rates) -> TAnnealSchedule | None:
    """The T-anneal schedule when the ``lif_tanneal`` recipe knob is on and the
    mode is lif, else None (the value-blend recipe stays bit-identical)."""
    if not bool(config.get("lif_tanneal", False)):
        return None
    # Lazy: chip_simulation has a fragile import cycle with tuning at init time.
    from mimarsinan.chip_simulation.spiking_semantics import is_lif

    if not is_lif(config.get("spiking_mode", "lif")):
        return None
    return TAnnealSchedule(
        target_T=int(config["simulation_steps"]),
        ladder_rates=tuple(float(r) for r in ladder_rates),
    )


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
        self._consume_optimization_driver(
            rates=self.pipeline.config.get(
                "lif_blend_fast_rates", [0.25, 0.5, 0.75, 1.0]
            ),
            steps_per_rate=int(
                self.pipeline.config.get("lif_blend_fast_steps_per_rate", 120)
            ),
            eta_min_factor=float(
                self.pipeline.config.get("lif_blend_fast_lr_eta_min", 0.1)
            ),
        )
        # The LIF recipe's realizable T-anneal over the SAME normalized ladder
        # (equal budget); None keeps the value-blend recipe bit-identical.
        self._tanneal = derive_lif_tanneal_schedule(
            self.pipeline.config, ladder_rates=self._fixed_ladder_rates,
        )
        self._endpoint_recovery_steps = int(
            self.pipeline.config.get("endpoint_recovery_steps", 0)
        )
        self._lif_distmatch = bool(self.pipeline.config.get("lif_distmatch", False))
        self._lif_distmatch_bias_iters = int(
            self.pipeline.config.get("lif_distmatch_bias_iters", 10)
        )
        self._lif_distmatch_eta = float(
            self.pipeline.config.get("lif_distmatch_bias_eta", 0.5)
        )
        self._lif_distmatch_cal_batches = int(
            self.pipeline.config.get("lif_distmatch_cal_batches", 8)
        )
        self._lif_distmatch_stats = None
        self._lif_theta_cotrain = bool(
            self.pipeline.config.get("lif_theta_cotrain", False)
        )
        self._theta_cotrain = self._lif_theta_cotrain

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
        if self._lif_distmatch:
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

        self._lif_distmatch_stats = run_teacher_distmatch(
            self,
            match_lif_activation_distributions,
            n_batches=self._lif_distmatch_cal_batches,
            bias_iters=self._lif_distmatch_bias_iters,
            eta=self._lif_distmatch_eta,
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
