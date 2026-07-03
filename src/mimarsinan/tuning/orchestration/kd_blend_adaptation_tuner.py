"""Shared KD blend-ramp adaptation for the LIF/TTFS conversion tuners."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from mimarsinan.common.best_effort import best_effort
from mimarsinan.tuning.forward_install import CascadeForwardInstall, LazyExecutorForward
from mimarsinan.tuning.orchestration.blend_ramp import (
    BlendActivation,
    KDClassificationLoss,
    kd_loss_from_config,
)
from mimarsinan.tuning.orchestration.genuine_probe import (
    eval_forward_over_val,
    genuine_acc_on_clone,
    iter_val_batches,
)
from mimarsinan.tuning.orchestration.ramp_strategy import (
    RampStrategy,
    ValueDomainProxyRamp,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.perceptron_rate import rebuild_activations, set_blend_rate
from mimarsinan.tuning.teacher import snapshot_frozen_teacher

__all__ = ["BlendActivation", "KDBlendAdaptationTuner"]

_InstalledForward = LazyExecutorForward
_KDClassificationLoss = KDClassificationLoss


class KDBlendAdaptationTuner(CascadeForwardInstall, SmoothAdaptationTuner):
    """Blend each perceptron's base activation toward a target, with KD recovery."""

    _target_activation_type = "Target"
    _old_activation_type = "ReLU"

    _skip_one_shot = True
    _initial_ramp_step = 0.125
    _ramp_step_growth = 1.0
    _max_stabilization_rounds = 3
    _theta_cotrain = False

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._final_metric = None
        self._finalize_cliff = None
        self._theta_cotrain_params = None
        self._theta_cotrain_stats = None

        self._configure()
        self._ramp = self._make_ramp_strategy()
        self._teacher = self._snapshot_teacher()
        self._install_blend()
        self.trainer.loss_function = self._make_kd_loss()
        self._after_install_blend()

        self._axis = self._make_axis()
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

    def _configure(self) -> None:
        """Read config, set names/params, and any adaptation_manager flags."""

    def _make_ramp_strategy(self) -> RampStrategy:
        """The ramp strategy bundling the 0→1 seams. Default = the value-domain
        proxy ramp; subclasses pick a strategy from the flags set in ``_configure``."""
        return ValueDomainProxyRamp()

    def _make_target_activation(self, perceptron) -> nn.Module:
        raise NotImplementedError

    def _blend_old_activation(self, perceptron) -> nn.Module:
        """Old side of the blend. Default: the perceptron's current base activation."""
        return perceptron.base_activation

    def _make_blend(self, old: nn.Module, target: nn.Module, rate: float) -> nn.Module:
        return self._ramp.make_blend(self, old, target, rate)

    def _make_axis(self):
        """The rate axis the tuner walks (the ramp strategy owns the choice;
        default = the value-domain ``BlendAxis``)."""
        return self._ramp.make_axis(self)

    def _after_make_target(self, perceptron, target: nn.Module) -> None:
        """Per-perceptron hook after the blend is installed (before update_activation)."""

    def _wrap_encoding_input(self, perceptron) -> None:
        """Optional encoding-layer input wrapping (e.g. ChipInputQuantizer)."""

    def _append_encoding_input_module(self, perceptron, module: nn.Module) -> None:
        """Append a wire op (e.g. an STE input quantizer) after input_activation."""
        if isinstance(perceptron.input_activation, nn.Identity):
            perceptron.input_activation = module
        else:
            perceptron.input_activation = nn.Sequential(
                perceptron.input_activation, module,
            )

    def _after_install_blend(self) -> None:
        """Strategy pre-step, the pre-install hook, the ramp forward, then the opt-in
        theta promotion (which must follow any scalar-theta calibration)."""
        self._ramp.after_install_blend_pre(self)
        self._before_ramp_forward_install()
        self._install_ramp_forward()
        self._maybe_promote_theta_cotrain()

    def _before_ramp_forward_install(self) -> None:
        """Hook between the ramp pre-step and the ramp-forward install (e.g. gain-base capture)."""

    def _maybe_promote_theta_cotrain(self) -> None:
        """Promote per-channel theta when the tuner opted in via ``_theta_cotrain``."""
        if self._theta_cotrain:
            self._promote_per_channel_theta()

    def _install_ramp_forward(self) -> None:
        fwd = self._ramp_forward()
        if fwd is not None:
            self._install_forward(fwd)

    def _promote_per_channel_theta(self) -> None:
        """Rebind each perceptron's ``activation_scale`` to a per-output-channel
        trainable Parameter so the optimiser co-trains the firing-gain theta with the
        weights through the deployed forward. Idempotent; families opt in per site."""
        # Lazy: spiking package init pulls chip_simulation, a top-level import cycle.
        from mimarsinan.spiking.theta_cotrain import (
            promote_activation_scale_per_channel,
        )

        params = promote_activation_scale_per_channel(self.model)
        self._theta_cotrain_params = list(params)
        self._theta_cotrain_stats = {"n_theta": len(params)}
        self.pipeline.reporter.report(
            f"{self.name} theta_cotrain", self._theta_cotrain_stats,
        )

    def _make_kd_loss(self):
        return self._ramp.make_kd_loss(self)

    def _kd_classification_loss(self, teacher):
        """The config-driven KD+CE blend loss (the ``blend_ramp`` SSOT) for this pipeline."""
        return kd_loss_from_config(self.pipeline.config, teacher)

    def _remove_forward(self) -> None:
        """Let the ramp strategy clean up its owned references, then unpatch."""
        self._ramp.on_remove_forward(self)
        super()._remove_forward()

    def _ramp_forward(self):
        """Cross-layer forward installed during the blend ramp (the ramp strategy
        owns the choice; default ``None`` = the value-domain ramp)."""
        return self._ramp.ramp_forward(self, self.model)

    def _finalize_forward_for(self, model) -> LazyExecutorForward | None:
        """The shared genuine-forward builder bound to ``model`` (``None`` for
        schedules with no separate genuine forward). The probe and the finalize
        install both route through this, so probe ≡ deploy by construction."""
        return None

    def _finalize_forward(self):
        """Cross-layer forward installed at finalize (``None`` keeps the class
        forward — the deployed dynamics for analytical/synchronized schedules)."""
        return self._finalize_forward_for(self.model)

    def _update_target_activations(self, model=None) -> None:
        target = self.model if model is None else model
        rebuild_activations(target, self.adaptation_manager, self.pipeline.config)

    def _before_finalize_rebuild(self, model=None) -> None:
        """Set adaptation-manager flags the activation rebuild must see (e.g. LIF sets
        ``lif_active``). ``model`` is the rebuild target."""

    def _after_finalize_rebuild(self, model=None) -> None:
        """Per-model finalize work after activations are rebuilt, before the deployed
        forward is installed (e.g. LIF applies cycle-accurate trains)."""

    def _finalize_rebuild(self, model=None) -> None:
        """The finalize-style activation rebuild on ``model``: pre-rebuild flags →
        rebuild → post-rebuild work. Shared by ``_finalize`` and the genuine probe
        (on an isolated clone), so probe ≡ deploy."""
        self._before_finalize_rebuild(model)
        self._update_target_activations(model)
        self._after_finalize_rebuild(model)

    def _finalize(self) -> None:
        """At full rate: rebuild the committed activations, then install the genuine
        cross-layer forward so every downstream step runs the deployed dynamics.
        Subclasses customize via the ordered ``_before/_after_finalize_rebuild``
        hooks."""
        self._before_finalize_rebuild()
        self._update_target_activations()
        self._after_finalize_rebuild()
        fwd = self._finalize_forward()
        self._stabilization_refinds_lr = fwd is not None
        if fwd is not None:
            self._install_forward(fwd)

    _finalize_manager_flags = ("lif_active", "ttfs_active")

    def _full_transform_eval(self):
        """The GENUINE full-transform accuracy: run the deployed finalize forward at
        rate 1.0 on an isolated clone (falling back to the value-domain measure when
        there is no separate genuine forward). Non-destructive to the live model and
        the shared adaptation_manager."""
        if self._finalize_forward_for(self.model) is None:
            return super()._full_transform_eval()

        device = self.pipeline.config["device"]
        manager = self.adaptation_manager
        flag_snapshot = {
            name: getattr(manager, name)
            for name in self._finalize_manager_flags
            if hasattr(manager, name)
        }

        def prepare(clone):
            set_blend_rate(clone, 1.0)
            self._finalize_rebuild(clone)

        try:
            return genuine_acc_on_clone(
                self.model,
                device,
                prepare=prepare,
                build_forward=lambda m: self._finalize_forward_for(m),
                evaluate=lambda fwd, m: eval_forward_over_val(
                    self.trainer, fwd, m,
                    self._budget.progress_eval_batches, device,
                ),
            )
        finally:
            for name, value in flag_snapshot.items():
                setattr(manager, name, value)

    def _snapshot_teacher(self) -> nn.Module:
        return snapshot_frozen_teacher(self.model, self.pipeline.config["device"])

    def _calibration_inputs(self, n_batches: int = 8) -> torch.Tensor:
        """A few concatenated validation batches as a calibration anchor (shared by
        the TTFS/LIF teacher-distribution matchers)."""
        device = self.pipeline.config["device"]
        batches = [
            x.to(device)
            for x, _ in iter_val_batches(self.trainer, n_batches)
        ]
        return torch.cat(batches)

    def _install_blend(self) -> None:
        for perceptron in self.model.get_perceptrons():
            old = self._blend_old_activation(perceptron)
            target = self._make_target_activation(perceptron)
            perceptron.base_activation = self._make_blend(old, target, 0.0)
            self._after_make_target(perceptron, target)
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
            self._wrap_encoding_input(perceptron)

    def _set_rate(self, rate: float) -> None:
        self._axis.set_rate(rate)

    def _get_extra_state(self):
        return self._axis.get_extra_state()

    def _set_extra_state(self, extra):
        self._axis.set_extra_state(extra)

    def _update_and_evaluate(self, rate: float):
        self._set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _safe_eval(self):
        """Best-effort validation read for the finalize-cliff diagnostic only."""
        metric = None
        with best_effort("finalize-cliff validation eval"):
            metric = float(
                self.trainer.validate_n_batches(self._budget.eval_n_batches)
            )
        return metric

    def _after_run(self):
        try:
            self._continue_to_full_rate()
            self._set_rate(1.0)
            ramp_metric = self._safe_eval()
        finally:
            self._remove_forward()
        self._finalize()
        post_finalize = self._safe_eval()
        if ramp_metric is not None and post_finalize is not None:
            self._finalize_cliff = ramp_metric - post_finalize
            self.pipeline.reporter.report(
                f"{self.name} finalize_cliff", self._finalize_cliff,
            )
        self._report_cliff_probe_consistency()
        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def _report_cliff_probe_consistency(self) -> None:
        """Drift sentinel: the last per-commit ``genuine_drop`` and the once-only
        ``finalize_cliff`` measure the same proxy↔genuine gap, so a large divergence
        means probe and deploy disagree (warn, do not assert)."""
        log = getattr(self, "_full_transform_log", None)
        if not log:
            return
        last_genuine_drop = float(log[-1]["genuine_drop"])
        finalize_cliff = self._finalize_cliff
        if finalize_cliff is None:
            return
        finalize_cliff = float(finalize_cliff)
        abs_diff = abs(last_genuine_drop - finalize_cliff)
        self.pipeline.reporter.report(f"{self.name} cliff_probe_consistency", {
            "last_genuine_drop": round(last_genuine_drop, 4),
            "finalize_cliff": round(finalize_cliff, 4),
            "abs_diff": round(abs_diff, 4),
        })
        if abs_diff > 0.1:
            warnings.warn(
                f"{self.__class__.__name__}: per-commit genuine probe drop "
                f"({last_genuine_drop:+.4f}) and finalize cliff "
                f"({finalize_cliff:+.4f}) diverge by {abs_diff:.4f}; the genuine "
                "probe may not be tracking the deployed finalize dynamics.",
                stacklevel=2,
            )

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
