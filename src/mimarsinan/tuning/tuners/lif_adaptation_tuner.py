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
        # Fast fixed-ladder LIF ramp (opt-in, default OFF): run the value-domain
        # blend ramp through the ONE orchestrator's fixed_ladder policy (one shared
        # optimizer + spanning cosine, KD recovery per rung, no controller) instead
        # of the greedy/bisect controller. LIF's rate code is bit-exact deployed, so
        # the value-domain ramp suffices (cliff ≈ 0). Uses the inherited _fast_loss
        # (the installed KD loss) and no genuine probe.
        #
        # EF1: LIF no longer bypasses the driver with an inline `lif_blend_fast`
        # read — the (`controller` | `fast`) decision is READ from the pipeline-wide
        # axis (`DeploymentPlan.optimization_driver` via `DeploymentPlan.of(pipeline)`),
        # the SAME axis every family consumes. The legacy `lif_blend_fast` switch still
        # feeds that axis, so a config carrying only the switch is byte-identical; an
        # explicit `optimization_driver` is the generic override. The fast switch floors
        # the spanning cosine (the value-domain endpoint needs real LIF-dynamics
        # training, unlike TTFS where genuine-CE carries it). Default `controller` ⇒
        # byte-identical.
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        driver = DeploymentPlan.of(self.pipeline).optimization_driver_for_family(
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
        self._optimization_driver = driver
        self._setup_fast_ladder(
            enabled=driver.fast_ladder,
            rates=driver.fast_ladder_rates,
            steps_per_rate=driver.fast_ladder_steps_per_rate,
            eta_min_factor=driver.fast_ladder_eta_min_factor,
        )
        # Post-finalize bounded stabilization on the deployed cycle-accurate forward
        # (the value-domain ramp under-trains the deployed LIF dynamics vs the
        # controller's stabilization; this recovers it while staying fast).
        self._fast_stabilize_steps = int(
            self.pipeline.config.get("lif_blend_fast_stabilize_steps", 0)
        )
        # DFQ per-neuron bias correction on the deployed cascade (opt-in, default
        # OFF): match each perceptron's deployed cycle-accurate channel-mean to the
        # frozen teacher ANN's, shrinking the systematic ANN->SNN first-moment
        # conversion gap. Pure bias correction — no decode-scale retune (that is
        # TTFS-only). Runs before the bounded stabilization so recovery refines the
        # calibrated init. Needs the cycle-accurate cascade to read decoded values.
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
        # Per-channel trainable firing-gain theta (opt-in, default OFF): a single
        # scalar threshold cannot serve both wide and narrow channels of a perceptron,
        # so rebind each non-encoding perceptron's activation_scale to a per-output-
        # channel param the blend ramp co-trains WITH the weights (the LIF analogue of
        # ttfs_theta_cotrain; promoted in _after_install_blend via the shared base
        # helper, before the fast-ladder optimiser captures model.parameters()).
        self._lif_theta_cotrain = bool(
            self.pipeline.config.get("lif_theta_cotrain", False)
        )

    def _after_install_blend(self) -> None:
        """Base pre-ramp setup + ramp-forward install, then (opt-in) promote the
        per-channel theta so the deployed LIF nodes reference the trainable param
        BEFORE the fast-ladder optimiser is built over ``model.parameters()``."""
        super()._after_install_blend()
        if self._lif_theta_cotrain:
            self._promote_per_channel_theta()

    def _post_stabilization_hook(self):
        if self._lif_distmatch:
            self._calibrate_to_teacher_distribution()
        if getattr(self, "_fixed_ladder_policy", False):
            self._fast_stabilize(getattr(self, "_fast_stabilize_steps", 0))

    def _calibrate_to_teacher_distribution(self) -> None:
        """DFQ-match the deployed LIF cascade's per-neuron mean to the frozen
        teacher ANN's. No-op unless the cycle-accurate cascade is deployed (the
        decoded values are read from its segment forward). Stats are stashed on
        ``self._lif_distmatch_stats`` and reported."""
        if not self._cycle_accurate:
            return
        from mimarsinan.spiking.lif_distribution_matching import (
            match_lif_activation_distributions,
        )

        cal_x = self._calibration_inputs(self._lif_distmatch_cal_batches)
        self._lif_distmatch_stats = match_lif_activation_distributions(
            self.model,
            self._teacher,
            cal_x,
            self._T,
            bias_iters=self._lif_distmatch_bias_iters,
            eta=self._lif_distmatch_eta,
        )
        self.pipeline.reporter.report(
            f"{self.name} distmatch", self._lif_distmatch_stats,
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

    def _finalize_forward_for(self, model):
        if self._cycle_accurate:
            return _ChipAlignedNFForward(model, self._T)
        return None

    def _before_finalize_rebuild(self, model=None) -> None:
        # lif_active before the rebuild so the committed activations subsume the
        # clamp/quant/shift decorators (the base _finalize installs any forward).
        self.adaptation_manager.lif_active = True

    def _after_finalize_rebuild(self, model=None) -> None:
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(
                self.model if model is None else model, True,
            )
