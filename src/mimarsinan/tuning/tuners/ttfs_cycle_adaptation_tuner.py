"""TTFS-cycle fine-tuning: gradual value-domain ramp, genuine cascade at finalize.

By default both schedules ramp the per-perceptron ``TTFSActivation`` blend in
the value domain (the plain class forward through ``BlendActivation``), exactly
the golden LIF non-destructive ramp: rate 0 reproduces the continuous teacher,
rate 1 is the pointwise on-chip staircase composition. The blend ramps
naturally under the rate scheduler's uniform ladder (no rate pin), with KD
recovery against the frozen pre-step teacher.

By default the genuine cross-layer dynamics are installed only at **finalize**:

- **cascaded** ``ttfs_cycle_based`` is a single-spike, ramp-integrate, fire-once
  cascade — the per-perceptron staircase blend cannot stay bit-identical to the
  intra-segment sub-window spike timing, so (like LIF's chip-aligned forward)
  finalize installs :class:`TTFSSegmentForward` via ``_finalize_forward`` and
  keeps it, so the committed metric, recovery, and every downstream step run
  the exact deployed single-spike dynamics.
- **synchronized** composes per-group analytical staircases — the ramped class
  forward already *is* that deployed composition, so no instance forward is
  installed; the wire contract's stage-input grid snap q(x) is trained through
  an STE on each segment entry.

The opt-in ``ttfs_genuine_annealed_ramp`` (default off, cascaded only) instead
installs the genuine single-spike cascade for the WHOLE ramp and anneals the
spike surrogate sharpness smooth→sharp (``TTFSGenuineAxis``). Since the surrogate
``alpha`` is backward-only, rate=1 is bit-identical to the deployed cascade, so
the finalize cliff is ~0 by construction.

The opt-in ``ttfs_genuine_blend_ramp`` (default off, cascaded only) calibrates
the deployed cascade to the teacher ANN's activation distribution
(``match_activation_distributions``: scale-aware [0,1] boundaries + DFQ per-neuron
bias correction), then installs a ``BlendedGenuineForward`` as the WHOLE-ramp
forward (``out = (1-rate)*teacher + rate*genuine``). The committed rate drives
the blend live via ``GenuineBlendAxis`` (rate 0 = the frozen continuous teacher,
rate 1 = the genuine cascade exactly), with KD recovery against the same teacher;
finalize deploys the PURE genuine cascade (the teacher is dropped, so the cliff is
0 by construction). It is mutually exclusive with the annealed ramp — blend wins.

The EXPERIMENTAL ``ttfs_genuine_blend_fast`` (default off, requires the blend ramp)
SKIPS the heavy SmoothAdaptation controller: ``run`` overrides the scheduler flow to
walk a FIXED rate schedule (``ttfs_blend_fast_rates``) a FIXED number of steps each
(``ttfs_blend_fast_steps_per_rate``) with ONE Adam optimizer + warmup/cosine LR, loss
``CE((1-R)*teacher + R*genuine) + 0.3*CE(genuine)`` (the validated prototype), then
runs the same finalize (pure genuine cascade). No adaptation cycles, no RecoveryEngine,
no rollback clone/restore, no stabilization pass, no per-cycle LR find.

INVARIANT CORE vs OPTIONAL CONTROLLER (the line the fast hack must not cross).
The genuine gradual-tuning *contract* (pinned in
``tests/unit/tuning/test_genuine_gradual_invariants.py``: inert@low-rate, smooth
degradation, r=1 == deployment bit-exact, tuning lifts the r=1 endpoint, rounds
converge) is a property of the MECHANISM, not the controller — so the mechanism is
installed in ``KDBlendAdaptationTuner.__init__`` (``_install_blend`` →
``_make_kd_loss`` → ``_after_install_blend``) and is therefore ACTIVE UNDER BOTH
paths, fast and slow:

  INVARIANT CORE (always on, even under fast) — DO NOT gate behind the fast flag:
    * scale-aware [0,1] boundaries + DFQ distribution matching
      (``_calibrate_to_teacher_distribution``)
    * the teacher<->genuine ``BlendedGenuineForward`` (rate 0 == teacher, rate 1 ==
      deployed cascade bit-exact)
    * the genuine-CE objective (``_BlendGenuineKDLoss`` / the fast loop's
      ``+ 0.3*CE(genuine)``) — this is what lifts the r=1 endpoint
    * the pure-genuine finalize (``_finalize``)

  OPTIONAL CONTROLLER (the SmoothAdaptation machinery; disabled under the fast hack):
    adaptive rate scheduler + bisect, recover-to-target, rollback clone/restore,
    stabilization pass, per-cycle LR finder, catastrophic gate, target adjuster.
    These add robustness/quality (the slow path), not invariant-correctness. The
    fast hack trades them for ~30-60s; the proper controller keeps them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.nn.activations.autograd import TTFSInputGridQuantizer
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.tuning.axes.blend_axis import GenuineBlendAxis, TTFSGenuineAxis
from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
    _KDClassificationLoss,
)
from mimarsinan.tuning.orchestration.ramp_strategy import RampStrategy


class _SegmentSpikeForward(LazyExecutorForward):
    """Picklable ``model.forward`` override driving the segment-aware spike sim.

    ``boundary_surrogate_temp`` (None = severed) enables the offload-boundary STE
    so the genuine backward trains every segment; the forward is unchanged."""

    def __init__(self, model, T: int, *, boundary_surrogate_temp: float | None = None):
        super().__init__(model, T)
        self.boundary_surrogate_temp = boundary_surrogate_temp

    def _build_executor(self):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )

        return TTFSSegmentForward(
            self.model.get_mapper_repr(), self.T,
            boundary_surrogate_temp=self.boundary_surrogate_temp,
        )

    def _run(self, x):
        return self._ensure_executor(self._build_executor)(x)


class _Rung2TeacherFlow(nn.Module):
    """Frozen KD teacher evaluating the identity-mapped contract semantics.

    Wraps an identity-mapped ``SpikingHybridCoreFlow`` built from the frozen
    pre-step snapshot; outputs are normalized back from the flow's count-scaled
    logits (÷T) and restored to the value domain via per-output node scales.
    """

    def __init__(self, flow, simulation_length: int, output_scales):
        super().__init__()
        self.flow = flow
        self.simulation_length = int(simulation_length)
        self.register_buffer(
            "_output_scales", torch.as_tensor(output_scales, dtype=torch.float32),
        )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.flow(x) / float(self.simulation_length)
        return logits * self._output_scales.to(logits.device, logits.dtype)


class _BlendGenuineKDLoss(_KDClassificationLoss):
    """KD (vs frozen teacher) on the blend output + a CE on the PURE genuine logits.

    Reproduces the validated genuine-blend recipe: the base KD+CE distills the
    teacher onto the ramping ``BlendedGenuineForward`` output (``model(x)``), while
    an extra ``genuine_ce_alpha · CE(genuine_logits, y)`` term sharpens the pure
    cascade so training at intermediate rates lifts the rate-1 endpoint. The blend
    forward is resolved through ``blend_forward_provider`` (a tuner-owned reference,
    NOT ``model.__dict__``), so the term is decoupled from the install mechanism;
    when the provider returns ``None`` (e.g. finalize/stabilization on the pure
    cascade) ``model(x)`` IS the genuine cascade, so the base CE already covers it
    and the extra term is skipped.
    """

    def __init__(self, teacher, *, genuine_ce_alpha: float, blend_forward_provider,
                 temperature: float = 3.0, alpha: float = 0.3):
        super().__init__(teacher, temperature=temperature, alpha=alpha)
        self.genuine_ce_alpha = float(genuine_ce_alpha)
        self._blend_forward = blend_forward_provider

    def __call__(self, model, x, y):
        loss = super().__call__(model, x, y)
        if self.genuine_ce_alpha > 0.0:
            blend = self._blend_forward()
            if blend is not None:
                loss = loss + self.genuine_ce_alpha * F.cross_entropy(
                    blend.genuine_logits(x), y,
                )
        return loss


class _StaircaseSteKDLoss(_KDClassificationLoss):
    """Straight-through estimator: forward value == the genuine single-spike cascade
    (the exact deploy path), backward == a hedge of the CLEAN complete-sum staircase
    gradient + the genuine surrogate. Fixes the deep high-S surrogate-gradient plateau
    (research: lossless cascaded TTFS in <2min). ``mix`` (0.5 default) weights the
    staircase half of the backward; the forward is genuine for any mix.

        back = mix*staircase + (1-mix)*genuine
        ste  = back + (genuine - back).detach()   # value=genuine, grad=back
    """

    def __init__(self, teacher, *, mix: float, forward_provider,
                 temperature: float = 3.0, alpha: float = 0.3):
        super().__init__(teacher, temperature=temperature, alpha=alpha)
        self.mix = float(mix)
        self._forward = forward_provider

    @staticmethod
    def _ste_logits(fwd, x, mix: float = 0.5):
        """ste = back + (genuine-back).detach(); staircase via the model's unpatched
        (analytical, cycle_accurate=False) forward, with the mode restored after."""
        genuine = fwd(x)
        nodes = [m for m in fwd.model.modules() if isinstance(m, TTFSActivation)]
        prev = [n._cycle_accurate_mode for n in nodes]
        for n in nodes:
            n.set_cycle_accurate(False)
        try:
            staircase = fwd._unpatched_forward(x)
        finally:
            for n, p in zip(nodes, prev):
                n.set_cycle_accurate(p)
        back = mix * staircase + (1.0 - mix) * genuine
        return back + (genuine - back).detach()

    def __call__(self, model, x, y):
        with torch.no_grad():
            tp = next(self.teacher.parameters(), None)
            if tp is not None and tp.device != x.device:
                self.teacher.to(x.device)
            teacher_logits = self.teacher(x)
        student = self._ste_logits(self._forward(), x, self.mix)
        ce = F.cross_entropy(student, y)
        T = self.temperature
        kd = F.kl_div(
            F.log_softmax(student / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kd


class _GenuineRamp(RampStrategy):
    """Shared base of the genuine cascade ramps: ``base_activation`` IS the bare
    ``TTFSActivation`` target (no value blend) and the deployed on-chip rebuild
    runs before the ramp forward is installed."""

    def is_bare_target(self, tuner) -> bool:
        return True

    def make_blend(self, tuner, old, target, rate):
        target.rate = float(rate)
        return target

    def after_install_blend_pre(self, tuner) -> None:
        tuner._finalize_rebuild()


class GenuineAnnealedRamp(_GenuineRamp):
    """Train through the genuine single-spike cascade for the WHOLE ramp, annealing
    the spike surrogate alpha alongside the rate (``TTFSGenuineAxis``)."""

    def make_axis(self, tuner):
        return TTFSGenuineAxis()

    def ramp_forward(self, tuner, model):
        return tuner._finalize_forward_for(model)


class GenuineBlendRamp(_GenuineRamp):
    """Ramp a teacher<->genuine OUTPUT blend (``BlendedGenuineForward``, driven by
    ``GenuineBlendAxis``) calibrated to the teacher distribution; finalize deploys
    the pure cascade."""

    def make_axis(self, tuner):
        return GenuineBlendAxis()

    def ramp_forward(self, tuner, model):
        tuner._blend_forward = BlendedGenuineForward(
            model, tuner._teacher, tuner._T, rate=0.0,
            boundary_surrogate_temp=tuner._boundary_surrogate_temp,
        )
        return tuner._blend_forward

    def make_kd_loss(self, tuner):
        return _BlendGenuineKDLoss(
            tuner._teacher,
            genuine_ce_alpha=tuner._genuine_ce_alpha,
            blend_forward_provider=lambda: tuner._blend_forward,
        )

    def after_install_blend_pre(self, tuner) -> None:
        super().after_install_blend_pre(tuner)
        tuner._calibrate_to_teacher_distribution()

    def on_remove_forward(self, tuner) -> None:
        tuner._blend_forward = None


class StaircaseSteRamp(GenuineAnnealedRamp):
    """Genuine annealed-ramp install with a straight-through estimator loss: forward
    = the genuine cascade, backward = a staircase/genuine hedge."""

    def make_kd_loss(self, tuner):
        return _StaircaseSteKDLoss(
            tuner._teacher, mix=tuner._ste_mix,
            forward_provider=lambda: tuner.model.__dict__.get("forward"),
        )


class TTFSCycleAdaptationTuner(KDBlendAdaptationTuner):
    """Ramp to the TTFS spike node, training through the schedule's NF dynamics."""

    _target_activation_type = "TTFS"

    def _configure(self) -> None:
        self.name = "TTFS Cycle Fine-Tuning"
        self._T = int(self.pipeline.config["simulation_steps"])
        self._thresholding_mode = str(self.pipeline.config.get("thresholding_mode", "<="))
        self._firing_mode = str(self.pipeline.config.get("firing_mode", "TTFS"))
        from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode

        self._bias_mode = resolve_bias_mode(self.pipeline.config)
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        contract = SpikingDeploymentContract.from_pipeline_config(self.pipeline.config)
        self._synchronized = contract.training_forward_kind() != "segment_spike"
        self._entry_perceptron_ids = None
        self.adaptation_manager.ttfs_active = True
        # The adaptation flag-thicket (ramp strategy × optimization driver × fast-ladder
        # rung) is resolved ONCE, declaratively: TtfsAdaptationPlan owns the precedence /
        # compatibility rules (blend ⊐ annealed, STE ⇒ annealed, ste_fast ⊂ ste,
        # proxy_fast excludes the genuine bare-target ramps + synchronized, ste_refine ⊂
        # proxy_fast, the fast-ladder rung). _configure just binds the validated plan to
        # the tuner's fields, so the dispatch is read in one tested place, not re-derived.
        from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import (
            TtfsAdaptationPlan,
        )

        # EF1: read the controller-vs-fast decision from the pipeline-wide axis
        # (DeploymentPlan.optimization_driver via DeploymentPlan.of(pipeline)) and
        # thread it into the plan as the GATE over TTFS's three-way fast fork; the
        # per-`ttfs_*`-flag selectors still pick WHICH fast variant. Default
        # `controller` ⇒ no fast variant ⇒ byte-identical.
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = TtfsAdaptationPlan.resolve(
            self.pipeline.config,
            synchronized=self._synchronized,
            optimization_driver=DeploymentPlan.of(self.pipeline).optimization_driver,
        )
        self._adaptation_plan = plan
        self._genuine_annealed_ramp = plan.genuine_annealed_ramp
        self._genuine_blend_ramp = plan.genuine_blend_ramp
        self._staircase_ste = plan.staircase_ste
        self._ste_mix = plan.ste_mix
        self._staircase_ste_fast = plan.staircase_ste_fast
        self._ste_steps = plan.ste_steps
        self._ste_w_lr = plan.ste_w_lr
        self._ste_theta_lr = plan.ste_theta_lr
        self._ste_init_frac = plan.ste_init_frac
        self._genuine_bare_target_ramp = plan.genuine_bare_target_ramp
        self._genuine_blend_fast = plan.genuine_blend_fast
        self._proxy_fast = plan.proxy_fast
        self._staircase_ste_refine = plan.staircase_ste_refine
        self._blend_fast_rates = plan.blend_fast_rates
        self._blend_fast_steps_per_rate = plan.blend_fast_steps_per_rate
        self._fast_stabilize_steps = plan.fast_stabilize_steps
        # The optimization-driver concern (controller vs fast-ladder rung) is owned by
        # the plan's OptimizationDriver; the controller is the default (fast_ladder
        # False → _setup_fast_ladder disables the fixed_ladder policy).
        driver = plan.driver
        self._optimization_driver = driver
        self._setup_fast_ladder(
            enabled=driver.fast_ladder,
            rates=driver.fast_ladder_rates,
            steps_per_rate=driver.fast_ladder_steps_per_rate,
            eta_min_factor=driver.fast_ladder_eta_min_factor,
        )
        self._fast_full_transform_log = []
        # The installed teacher<->genuine blend forward (owned reference so the
        # genuine-CE loss + the fast attempt resolve it without introspecting
        # model.__dict__); set in _ramp_forward, cleared in _remove_forward.
        self._blend_forward = None
        # Single read of the genuine-CE weight (canonical default in defaults.py);
        # both the KD loss and the fast loop consume this one value.
        self._genuine_ce_alpha = float(
            self.pipeline.config.get("ttfs_genuine_blend_ce_alpha", 0.3)
        )
        self._distmatch_stats = None
        # The conversion-health calibration steps (gain-correction / theta-cotrain /
        # distmatch / boundary-STE) + their compatibility (theta ⊥ gain-ramp, gain-ramp
        # ⊐ gain-cold, all cascaded-only) are resolved ONCE by the CalibrationPipeline
        # the plan carries; _configure binds the resolved decisions instead of
        # re-deriving the per-flag guards. The parity-critical APPLICATION order stays
        # here (gain trim before node build, theta promotion after scalar-theta calib).
        cal = plan.calibration
        self._calibration = cal
        # Offload-boundary straight-through estimator (default OFF, cascaded only):
        # when set, the genuine single-spike cascade backward flows through the
        # round-based re-encode at offload/host-ComputeOp segment boundaries (a soft
        # spike-time STE), so EVERY neural segment trains on the deployed dynamics
        # instead of only the last (the §7 gradient-severance root cause). Forward
        # is byte-identical, so NF↔SCM parity and the deployed metric are unchanged.
        self._boundary_surrogate_temp = cal.boundary_surrogate_temp
        self._gain_correction_stats = None
        self._maybe_apply_gain_correction()
        # Per-channel TRAINABLE theta co-training (default OFF, cascaded only): promote
        # each non-encoding perceptron's activation_scale to a per-output-channel
        # requires_grad Parameter (in _after_install_blend, after any scalar-theta
        # distmatch/gain calibration) so the genuine fine-tune co-trains the firing-gain
        # with the weights. Mutually exclusive with the per-depth gain ramp (both manage
        # theta; the ramp's scalar set would clobber the per-channel param) — gain wins.
        self._theta_cotrain = cal.theta_cotrain
        self._theta_cotrain_stats = None
        # The promoted per-channel theta Parameters (set in _maybe_promote_theta_cotrain);
        # the STE-fast loop trains them in their own LR group.
        self._theta_cotrain_params = None

    def _maybe_apply_gain_correction(self) -> None:
        """Per-cascade-depth activation_scale trim that inverts the deployed ramp
        decode's depth attenuation (the death cascade). Two modes (cascaded only; a
        pure calibration change -> decode bit-exact -> NF<->SCM parity holds):

        * COLD (``ttfs_gain_correction``): apply the full trim once before node build.
        * RATE-GATED RAMP (``ttfs_gain_correction_ramp``, wins if both set): capture
          the base scales + target factors and ramp ``theta_d -> base*g_d**rate`` on
          every ``_set_rate`` so the correction co-adapts WITH the KD blend (the model
          learns the calibration and the spiking dynamics together) instead of being a
          cold init a downstream fine-tune absorbs at the readout.

        Which mode is active (and its cascaded-only / ramp-wins-over-cold compatibility)
        is resolved by the CalibrationPipeline; this method only applies the body."""
        cal = self._calibration
        rule, c = cal.gain_rule, cal.gain_c
        self._gain_ramp = cal.gain_ramp

        if cal.gain_ramp:
            # Base scales + factors are captured in _after_install_blend (AFTER any
            # genuine-blend scale-aware-boundary calibration that also sets the
            # scales), so the gain ramp multiplies the SETTLED scales by g_d**rate.
            self._gain_ramp_rule = rule
            self._gain_ramp_c = c
            self._gain_ramp_base = None
            self._gain_ramp_factors = None
            return

        if cal.gain_cold:
            from mimarsinan.spiking.gain_correction import apply_cascaded_gain_correction

            self._gain_correction_stats = apply_cascaded_gain_correction(
                self.model, self._T, rule=rule, c=c,
            )
            self.pipeline.reporter.report(
                f"{self.name} gain_correction", self._gain_correction_stats,
            )

    def _apply_gain_at_rate(self, rate: float) -> None:
        from mimarsinan.spiking.gain_correction import apply_gain_at_rate

        apply_gain_at_rate(
            self.model, self._gain_ramp_base, self._gain_ramp_factors, rate,
        )

    def _set_rate(self, rate: float) -> None:
        super()._set_rate(rate)
        if getattr(self, "_gain_ramp", False) and self._gain_ramp_base is not None:
            self._apply_gain_at_rate(rate)

    def _invalidate_lr_cache(self):
        """Genuine controller perf: the LR finder is the dominant cost on the
        cascade (~55s/call — 8 probes × 30 steps through the S-step blend forward
        that computes teacher AND genuine), and the base loop re-finds it on every
        target relaxation (measured: 3-4 re-finds eating ~4-5 min before the rate
        left 0.125). The LR is stable across the blend ramp — the fast hack proves
        ONE LR suffices — so the genuine ramp finds it ONCE and never re-finds.
        Scoped to the opt-in genuine ramp → golden-safe for every other tuner."""
        if getattr(self, "_genuine_blend_ramp", False):
            return
        super()._invalidate_lr_cache()

    def _find_lr(self):
        """Genuine controller perf: a coarse 3×10 LR sweep instead of the default
        8×30. Each probe step runs the expensive blend forward (teacher+genuine),
        so the full 8×30 sweep costs ~55s — most of a <2-min budget — for a fine
        LR that barely matters (the fast hack converges on the bare recipe LR).
        Three probes around the anchor are enough to reject a destructive LR.
        Scoped to the opt-in genuine ramp → golden-safe."""
        if not getattr(self, "_genuine_blend_ramp", False):
            return super()._find_lr()
        import dataclasses
        steps = min(10, self._budget.lr_steps_per_probe)
        saved = self._budget
        self._budget = dataclasses.replace(
            saved, lr_num_probes=3, lr_steps_per_probe=steps,
            max_lr_exploration_steps=3 * steps,
        )
        try:
            return super()._find_lr()
        finally:
            self._budget = saved

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

    # -- genuine ramps (default OFF) -------------------------------------------
    def _make_ramp_strategy(self) -> RampStrategy:
        """Pick the ramp strategy from the flags settled in ``_configure``. The
        precedence (blend > annealed; STE forces the annealed install) is preserved
        by the flag rules there — here it is a flat, mutually-exclusive dispatch."""
        if self._genuine_blend_ramp:
            return GenuineBlendRamp()
        if self._staircase_ste:
            return StaircaseSteRamp()
        if self._genuine_annealed_ramp:
            return GenuineAnnealedRamp()
        return super()._make_ramp_strategy()

    def _after_install_blend(self) -> None:
        """Strategy pre-step (genuine rebuild / blend distmatch), then the
        rate-gated gain base capture, the ramp forward install, and finally the
        per-channel theta promotion (after every scalar-theta calibration). The
        ordering is parity-critical; the calibration wrapper stays here (P1)."""
        self._ramp.after_install_blend_pre(self)
        self._capture_gain_ramp_base()
        self._install_ramp_forward()
        self._maybe_promote_theta_cotrain()

    def _maybe_promote_theta_cotrain(self) -> None:
        """Promote per-channel TRAINABLE theta AFTER any scalar-theta calibration
        (distmatch / gain base capture, which call ``float(activation_scale)`` and
        ``propagate_boundary_input_scales`` — both scalar-only). The bare-cascade or
        blend nodes already reference ``perceptron.activation_scale`` by identity, so
        rebinding it to a per-channel param (on the perceptron AND its nodes) routes
        the optimiser's gradient into the deployed forward."""
        if not self._theta_cotrain:
            return
        from mimarsinan.spiking.theta_cotrain import (
            promote_activation_scale_per_channel,
        )

        params = promote_activation_scale_per_channel(self.model)
        self._theta_cotrain_params = list(params)
        self._theta_cotrain_stats = {"n_theta": len(params)}
        self.pipeline.reporter.report(
            f"{self.name} theta_cotrain", self._theta_cotrain_stats,
        )

    def _capture_gain_ramp_base(self) -> None:
        """Capture the (post-calibration) per-perceptron base scales + target gain
        factors for the rate-gated ramp, and set rate 0 (= base). Runs after any
        distmatch scale-aware-boundary calibration so the gain ramp composes on top."""
        if not getattr(self, "_gain_ramp", False):
            return
        from mimarsinan.spiking.gain_correction import cascaded_gain_factors

        self._gain_ramp_base = [
            float(p.activation_scale) for p in self.model.get_perceptrons()
        ]
        self._gain_ramp_factors = cascaded_gain_factors(
            self.model, self._T, rule=self._gain_ramp_rule, c=self._gain_ramp_c,
        )
        self._apply_gain_at_rate(0.0)
        self._gain_correction_stats = {
            "mode": "ramp", "rule": self._gain_ramp_rule, "S": int(self._T),
            "n_perceptrons": len(self._gain_ramp_base),
            "min_factor": round(min(self._gain_ramp_factors.values()), 4),
        }
        self.pipeline.reporter.report(
            f"{self.name} gain_correction", self._gain_correction_stats,
        )

    def _calibrate_to_teacher_distribution(self) -> None:
        """Distribution-match the deployed cascade to the frozen teacher ANN:
        scale-aware [0,1] boundaries from the teacher's per-perceptron activation
        quantile + DFQ per-neuron bias correction. Runs on the deployed-TTFS model
        (after ``_finalize_rebuild``) over a few concatenated validation batches,
        turning the full transform into a smoothly recoverable teacher->genuine
        ramp. Stats are stashed on ``self._distmatch_stats`` and reported."""
        from mimarsinan.spiking.distribution_matching import (
            match_activation_distributions,
        )

        cal = self._calibration
        cal_x = self._calibration_inputs()
        self._distmatch_stats = match_activation_distributions(
            self.model,
            self._teacher,
            cal_x,
            self._T,
            quantile=cal.distmatch_quantile,
            bias_iters=cal.distmatch_bias_iters,
            eta=cal.distmatch_bias_eta,
        )
        self.pipeline.reporter.report(
            f"{self.name} distmatch", self._distmatch_stats,
        )

    def _wrap_encoding_input(self, perceptron) -> None:
        # Synchronized wire contract: every hybrid stage input is grid-quantized
        # q(x); train through it via an STE on each segment-entry perceptron
        # (the first on-chip core of a segment — NOT ``is_encoding_layer``,
        # which is inert under offload and the wrong seam under subsume).
        # Cascaded NF feeds genuine spike trains (the segment walk encodes).
        if self._synchronized and id(perceptron) in self._segment_entry_ids():
            quantizer = TTFSInputGridQuantizer(
                T=self._T,
                activation_scale=perceptron.input_activation_scale,
            )
            self._append_encoding_input_module(perceptron, quantizer)

    def _segment_entry_ids(self) -> set[int]:
        if self._entry_perceptron_ids is None:
            from mimarsinan.torch_mapping.encoding_layers import (
                segment_entry_perceptrons,
            )

            self._entry_perceptron_ids = {
                id(p)
                for p in segment_entry_perceptrons(self.model.get_mapper_repr())
            }
        return self._entry_perceptron_ids

    # -- KD teacher: optional rung-2 (identity-mapped contract) target ---------
    def _make_kd_loss(self):
        """Synchronized + ``ttfs_finetune_kd_against_rung2``: distill against the
        frozen teacher evaluated under rung-2 semantics (identity-mapped contract
        flow) instead of its torch forward (design doc §6.1, default off). The
        genuine/STE losses are owned by their ramp strategies (the base
        delegation); synchronized always rides the value-domain proxy ramp."""
        if self._synchronized and bool(
            self.pipeline.config.get("ttfs_finetune_kd_against_rung2", False)
        ):
            return _KDClassificationLoss(self._build_rung2_teacher())
        return super()._make_kd_loss()

    def _build_rung2_teacher(self) -> _Rung2TeacherFlow:
        from mimarsinan.models.spiking.hybrid.identity_flow import (
            build_identity_spiking_flow,
        )

        cfg = self.pipeline.config
        ir_graph = self._map_teacher_to_ir()
        flow = build_identity_spiking_flow(
            cfg["input_shape"],
            ir_graph,
            self._T,
            getattr(self._teacher, "preprocessor", None),
            str(cfg.get("firing_mode", "TTFS")),
            str(cfg.get("spike_generation_mode", "TTFS")),
            self._thresholding_mode,
            spiking_mode=str(cfg.get("spiking_mode", "ttfs_cycle_based")),
            ttfs_cycle_schedule="synchronized",
        )
        mapping = flow.hybrid_mapping
        output_scales = [
            float(mapping.node_activation_scales.get(int(src.node_id), 1.0))
            for src in mapping.output_sources.flatten()
        ]
        return _Rung2TeacherFlow(flow, self._T, output_scales)

    def _map_teacher_to_ir(self):
        from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.mapping.platform.platform_constraints import (
            resolve_platform_mapping_params,
        )
        from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
        from mimarsinan.pipelining.core.platform_constraints_resolver import (
            build_platform_constraints_resolved,
        )
        from mimarsinan.transformations.quantization_bounds import quantization_bounds

        cfg = self.pipeline.config
        params = resolve_platform_mapping_params(
            build_platform_constraints_resolved(cfg)["cores"],
        )
        bits = int(cfg["weight_bits"])
        _, q_max = quantization_bounds(bits)

        mapper_repr = self._teacher.get_mapper_repr()
        if hasattr(mapper_repr, "assign_perceptron_indices"):
            mapper_repr.assign_perceptron_indices()
        compute_per_source_scales(mapper_repr)
        ir_graph = IRMapping(
            q_max=q_max,
            firing_mode=str(cfg.get("firing_mode", "TTFS")),
            max_axons=params.effective_max_axons,
            max_neurons=params.effective_max_neurons,
            hardware_bias=params.hardware_bias,
        ).map(mapper_repr)
        quantize_ir_graph(ir_graph, bits, weight_quantization=False)
        return ir_graph

    # -- Fast fixed-ladder genuine-blend ramp: TTFS-specific hooks -------------
    # The shared fixed-ladder machinery (run-reset / _driver_attempt routing /
    # _ensure_fast_optimizer / _fast_rate_attempt / _record_fast_cycle /
    # _stabilization_budget→0) lives in KDBlendAdaptationTuner. TTFS supplies the
    # validated genuine objective (plain CE on the blend + ce_alpha·CE on the PURE
    # genuine logits — lifts the r=1 endpoint, invariants 4/5) and the deployed-r=1
    # convergence probe via these two hooks.
    def _fast_rate_attempt(self, target):
        # STE-fast owns its single rung: the proven clean loop (split-LR +
        # progressive depth + cosine) instead of the base single-optimizer ladder.
        if getattr(self, "_staircase_ste_fast", False):
            return self._run_staircase_ste_fast(float(target))
        return super()._fast_rate_attempt(target)

    def _weight_params_through(self, depth):
        """Set ``requires_grad`` on the first ``depth`` perceptrons' layer weight/bias
        (shallow->deep) and return the now-trainable ones; deeper layers stay frozen.
        Progressive unfreeze grows ``depth`` to N so the final step trains all weights.
        Frozen params just receive no gradient (Adam skips them) — no optimizer rebuild
        is needed for the freeze itself; the rebuild on a depth CHANGE adds the newly
        unfrozen params to the optimizer."""
        out = []
        for i, p in enumerate(self.model.get_perceptrons()):
            req = i < depth
            p.layer.weight.requires_grad_(req)
            if p.layer.bias is not None:
                p.layer.bias.requires_grad_(req)
            if req:
                out.append(p.layer.weight)
                if p.layer.bias is not None:
                    out.append(p.layer.bias)
        return out

    def _run_staircase_ste_fast(self, target):
        """Clean fixed-step STE loop (the toy recipe that crossed the ~0.95 plateau):
        split-LR optimizer (weights @ ``ttfs_ste_w_lr``, per-channel theta @
        ``ttfs_ste_theta_lr`` when ``ttfs_theta_cotrain`` is on), progressive
        shallow->deep weight unfreeze from ``ttfs_ste_init_frac`` of the depth, cosine
        LR over ``ttfs_ste_steps``, and the installed STE+KD ``_fast_loss``. Forward is
        the genuine cascade throughout (deploy-exact); commits ``target`` (== 1.0).
        Replaces the rate-search controller, which caps the STE ~0.83 on MNIST."""
        import time

        t0 = time.time()
        device = self.pipeline.config["device"]
        self.model = self.model.to(device)
        steps = self._ste_steps
        thetas = list(getattr(self, "_theta_cotrain_params", None) or [])
        n = len(self.model.get_perceptrons())
        start = max(1, round(n * self._ste_init_frac))
        schedule = [
            min(n, start + round((n - start) * t / max(1, steps - 1)))
            for t in range(steps)
        ]
        grad_clip = float(
            (self.pipeline.config.get("tuning_recipe") or {}).get("grad_clip_norm", 1.0)
        )
        depth = -1
        opt = sched = None
        for t in range(steps):
            if schedule[t] != depth:
                depth = schedule[t]
                weights = self._weight_params_through(depth)
                groups = []
                if thetas:
                    groups.append({"params": thetas, "lr": self._ste_theta_lr})
                groups.append({"params": weights, "lr": self._ste_w_lr})
                opt = torch.optim.Adam(groups)
                for g in opt.param_groups:
                    g["initial_lr"] = g["lr"]
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=steps, last_epoch=t - 1,
                )
                # Expose the live optimizer so the shared trace (_record_fast_cycle)
                # can read the rung LR; rebuilt on each depth change.
                self._fast_optimizer = opt
            # Anneal the spike-surrogate sharpness (the annealed-ramp rate) 0->1 across
            # the budget instead of jumping to a sharp surrogate at step 0: a sharp
            # surrogate on the cold/dead cascade gives ~0 genuine gradient (the
            # controller climbs because it ramps; a single rung at 1.0 stalls at chance).
            self._set_rate(min(1.0, (t + 1) / steps))
            x, y = self.trainer.next_training_batch()
            x, y = x.to(device), y.to(device)
            self.model.train()
            loss = self._fast_loss(x, y)
            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    [p for grp in opt.param_groups for p in grp["params"]], grad_clip,
                )
            opt.step()
            sched.step()
            self._fast_optimizer_steps += 1
        self._set_rate(float(target))  # commit/eval at the deploy rate (sharp surrogate)
        self._committed_rate = float(target)
        post_acc = float(self.trainer.validate_n_batches(self._budget.eval_n_batches))
        self._record_fast_cycle(float(target), post_acc, t0)
        self._last_post_acc = post_acc
        self._fast_probe(float(target))
        self._phase_seconds["fast_blend"] = (
            self._phase_seconds.get("fast_blend", 0.0) + (time.time() - t0)
        )
        return float(target)

    def _fast_loss(self, x, y):
        # Proxy fast path: use the installed KD loss (distil the teacher onto the
        # value-domain blend) — there is no BlendedGenuineForward to read.
        if not self._genuine_blend_ramp:
            return super()._fast_loss(x, y)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        genuine = self._installed_genuine_branch()
        if self._genuine_ce_alpha > 0.0 and genuine is not None:
            loss = loss + self._genuine_ce_alpha * F.cross_entropy(genuine(x), y)
        return loss

    def _fast_probe(self, rate) -> None:
        # The genuine-blend probe reads the installed BlendedGenuineForward; the
        # proxy path has none (its deployed cascade is installed only at finalize).
        if self._genuine_blend_ramp:
            self._probe_fast_full_transform(rate)

    def _post_stabilization_hook(self):
        # Proxy fast: the value-domain ramp leaves the deployed genuine cascade
        # untrained (the proxy↔genuine cliff). A short bounded stabilization on the
        # deployed _SegmentSpikeForward closes it (LIF pattern). The genuine blend
        # trains through the cascade for the whole ramp (cliff≈0) → no stab needed.
        if not getattr(self, "_proxy_fast", False):
            return
        steps = getattr(self, "_fast_stabilize_steps", 0)
        if getattr(self, "_staircase_ste_refine", False):
            # Two-stage revive->refine: the deployed forward is now the revived genuine
            # cascade, so refine it with the STE (staircase-backward hedge) instead of
            # plain KD — the STE finally operates on an ALIVE cascade.
            ste = _StaircaseSteKDLoss(
                self._teacher, mix=self._ste_mix,
                forward_provider=lambda: self.model.__dict__.get("forward"),
            )
            self._fast_stabilize(steps, loss_fn=lambda x, y: ste(self.model, x, y))
        else:
            self._fast_stabilize(steps)

    def _installed_genuine_branch(self):
        """The PURE genuine-cascade branch of the owned ``BlendedGenuineForward``
        (``None`` when no blend forward is installed — then ``model(x)`` IS genuine)."""
        blend = self._blend_forward
        return blend.genuine_logits if blend is not None else None

    def _probe_fast_full_transform(self, rate) -> None:
        """Observability for invariant 5 (default off, ``tuning_full_transform_probe``):
        after a fast-ramp round, record the r=1 full-transform (deployed) accuracy so
        the convergence is visible even without the controller's per-cycle probe. The
        installed blend forward at rate 1.0 IS the deployed cascade, so no deepcopy or
        finalize-rebuild is needed — set 1.0, evaluate, restore the round's rate."""
        if not getattr(self, "_full_transform_probe", False):
            return
        blend = self._blend_forward
        prev = float(blend.rate) if blend is not None else rate
        device = self.pipeline.config["device"]
        self._set_rate(1.0)
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in self.trainer.iter_validation_batches(self._budget.eval_n_batches):
                pred = self.model(x.to(device)).argmax(dim=1)
                correct += int((pred == y.to(device)).sum())
                total += int(y.numel())
        self._set_rate(prev)
        acc = correct / total if total else 0.0
        self._fast_full_transform_log.append(
            {"rate": float(rate), "full_transform_acc": acc}
        )
        self.pipeline.reporter.report(
            f"{self.name} fast_full_transform",
            {"rate": round(float(rate), 4), "full_transform_acc": round(acc, 4)},
        )

    # -- finalize forward ------------------------------------------------------
    def _finalize_forward_for(self, model):
        """Cascaded builds the genuine single-spike cascade forward bound to
        ``model`` (kept at finalize, so the committed metric, recovery, and every
        downstream step — WQ/NormFusion/SCM — run the exact deployed single-spike
        dynamics; built on a clone for the genuine probe). Synchronized returns
        ``None`` — the class-level analytical staircase forward IS its deployment.
        Both ramp the value-domain blend (``_ramp_forward`` stays ``None``)."""
        if self._synchronized:
            return None
        return _SegmentSpikeForward(
            model, self._T, boundary_surrogate_temp=self._boundary_surrogate_temp,
        )
