"""Smooth-adaptation LIF tuner with knowledge-distillation recovery.

Swaps every Perceptron's ``base_activation`` with a blended module that
linearly interpolates between the original ReLU-like activation and a
:class:`LIFActivation`, then drives the blend ``rate`` from 0 → 1 across
cycles of :class:`SmoothAdaptationTuner`.  At each cycle the recovery
loss is::

    α · CE(student, y)  +  (1 − α) · T² · KL(soft_student ‖ soft_teacher)

with α = 0.3, T = 3 (matching ``spikingjelly-example/train.py``).  The
teacher is a frozen deep copy of the pre-LIF model; the student is the
live blended model.  A single one-shot swap is not sufficient to recover
the T-level quantization that LIF introduces, so we lean on the same
``SmartSmoothAdaptation`` loop ``ClampTuner`` uses — per-step rollback,
LR caching, per-cycle eval — to find a stable ramp.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.activations import (
    ChipInputQuantizer,
    LIFActivation,
    run_cycle_accurate,
)
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class _CycleAccurateForward:
    """Picklable callable that drives :func:`run_cycle_accurate` on a model.

    Installed by :meth:`LIFAdaptationTuner._install_cycle_accurate_forward`
    as the model's ``forward`` attribute when ``cycle_accurate_lif_forward``
    is on. Lives at module scope so :func:`torch.save` (used by the
    pipeline cache between steps) can pickle the model's ``__dict__``
    including this wrapper — a local closure cannot be pickled and
    would abort the pipeline.

    The per-cycle re-entrant call goes through the model's *class-level*
    ``forward`` (via ``type(model).forward(model, x)``) rather than a
    captured bound method. This bypasses the patched instance attribute
    (avoiding infinite recursion) without holding a separate bound-method
    reference that complicates pickle's reference graph.
    """

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)

    def _call_unpatched_forward(self, x):
        # ``type(self.model).forward(self.model, x)`` resolves to the
        # class-defined ``forward`` (e.g. ``ConvertedModelFlow.forward``)
        # — i.e. the original code path, regardless of what's bound to
        # the instance's ``forward`` attribute.
        return type(self.model).forward(self.model, x)

    def __call__(self, x):
        return run_cycle_accurate(
            self.model, x, self.T,
            forward_fn=self._call_unpatched_forward,
        )


class LIFBlendActivation(nn.Module):
    """Weighted mix of an original activation and a LIFActivation.

    ``rate = 0.0`` → pure original activation (identical to pre-swap model).
    ``rate = 1.0`` → pure LIFActivation output (final deployment target).
    ``0 < rate < 1`` → linear blend.

    Rate is updated externally by the tuner; no autograd through it.
    """

    def __init__(
        self,
        old_activation: nn.Module,
        lif_activation: LIFActivation,
        rate: float = 0.0,
    ):
        super().__init__()
        self.old_activation = old_activation
        self.lif_activation = lif_activation
        self.rate = float(rate)

    @property
    def activation_type(self) -> str:
        return "LIF" if self.rate >= 1.0 else "ReLU"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cycle-accurate mode is invisible here: ``self.lif_activation`` is
        # a :class:`LIFActivation` whose ``forward`` already branches on
        # its own ``_cycle_accurate_mode`` flag. Same signature ``(B, …)
        # → (B, …)`` in either mode, so the blend math is identical —
        # in rate-mode it interpolates rates, in cycle-accurate mode it
        # interpolates per-cycle pre-activations.
        if self.rate <= 0.0:
            return self.old_activation(x)
        if self.rate >= 1.0:
            return self.lif_activation(x)
        return (1.0 - self.rate) * self.old_activation(x) + self.rate * self.lif_activation(x)


class _KDClassificationLoss:
    """KD + CE loss compatible with ``BasicTrainer.loss_function(model, x, y)``.

    T = 3, α = 0.3 (SpikingJelly example defaults).  Teacher is frozen and
    lazy-moved onto the input batch's device on the first call.
    """

    def __init__(self, teacher: nn.Module, temperature: float = 3.0, alpha: float = 0.3):
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if next(self.teacher.parameters()).device != x.device:
                self.teacher.to(x.device)
            teacher_logits = self.teacher(x)
        student_logits = model(x)
        ce = F.cross_entropy(student_logits, y)
        T = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kd


class LIFAdaptationTuner(SmoothAdaptationTuner):
    """Smooth ramp of base_activation from pre-LIF → LIFActivation with KD recovery."""

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self.name = "LIF Adaptation"
        self._T = int(pipeline.config["simulation_steps"])
        # Training must fire under the same comparator as the chip path.
        # Default ``<=`` matches the LIF deployment defaults set in
        # DeploymentPipeline; the chip simulators read the same
        # ``thresholding_mode`` config so training and chip-side
        # comparators stay aligned.
        self._thresholding_mode = str(pipeline.config.get("thresholding_mode", "<="))
        # When the cycle-accurate knob is on, the model is forwarded with
        # (T, B, ...) spike trains threaded through the Mapper DAG so each
        # Perceptron's LIFActivation integrates the actual bursty per-cycle
        # output of upstream perceptrons (not a broadcast rate). Closes the
        # NF→SCM rate↔cycle gap by training the weights against the same
        # dynamics the chip will execute at deployment.
        self._cycle_accurate = bool(pipeline.config.get("cycle_accurate_lif_forward", False))
        self._patched_forward = False
        self._final_metric = None

        # Snapshot the teacher BEFORE any swap — it's the pre-LIF model.
        # Deep copy via CPU to avoid double-resident GPU parameters.
        device = self.pipeline.config["device"]
        self.model.to("cpu")
        self._teacher = copy.deepcopy(self.model)
        self.model.to(device)
        self._teacher.to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

        # Install the blend activation on every Perceptron (rate=0 so
        # behaviour is identical to the pre-swap model until the tuner
        # moves the rate up).
        self._install_blend()

        # Swap the trainer's loss to KD from the outset so recovery
        # gradients see soft teacher targets even at small rates.
        self.trainer.loss_function = _KDClassificationLoss(self._teacher)

        # Optional cycle-accurate forward: route student forwards through
        # the Mapper DAG cycle-accurate executor. Teacher stays in rate
        # mode (it's the pre-LIF reference) so KD targets are unchanged.
        if self._cycle_accurate:
            self._install_cycle_accurate_forward()

    def _install_cycle_accurate_forward(self) -> None:
        """Route every training-time ``model(x)`` through the T-loop driver.

        Works for any model topology — there is no model-side
        implementation requirement. The driver
        (:func:`run_cycle_accurate`) toggles each LIFActivation into
        single-step mode, encodes the input as a spike train, calls the
        model's *original* forward once per cycle, and means the
        outputs. Everything else (Linear, Conv, BN, einops, ...) stays
        on the same code path it uses in rate-mode forward.
        """
        # Track whether we patched ``forward`` so ``_after_run`` knows to
        # restore the rate-mode class forward when CA is off.
        self._patched_forward = True

        # Use a module-level picklable callable rather than a local
        # closure: the pipeline saves the model via ``torch.save``
        # between steps, which pickles ``__dict__`` — including the
        # patched ``forward`` attribute. ``_CycleAccurateForward`` is
        # picklable because it and the underlying model both live at
        # module scope (or are themselves picklable nn.Module instances).
        self.model.forward = _CycleAccurateForward(
            model=self.model,
            T=self._T,
        )

    # -------------------------------------------------------- blend install

    def _install_blend(self) -> None:
        for perceptron in self.model.get_perceptrons():
            old_base = perceptron.base_activation
            # LIFActivation references the same ``activation_scale`` nn.Parameter,
            # so scale updates remain in sync with Perceptron state.  The
            # firing comparator follows the pipeline's ``thresholding_mode``
            # so training and chip simulation use the same rule.
            lif = LIFActivation(
                T=self._T,
                activation_scale=perceptron.activation_scale,
                thresholding_mode=self._thresholding_mode,
            )
            perceptron.base_activation = LIFBlendActivation(old_base, lif, rate=0.0)
            # Rebuild the TransformedActivation wrapper so the new base
            # module participates in forward passes.
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

            # Encoding-layer perceptrons (those whose input comes from a
            # host-side ComputeOp — patch_embed, mean_reduce, etc.) receive
            # continuous values that the chip will re-encode via
            # ``to_uniform_spikes`` = ``round(T · x / ias)``.  Pre-discretise
            # the input here during training so the student sees the same
            # discretisation the chip will apply at deployment — closes the
            # ~15 pp input-boundary gap between training FP forward and the
            # spiking-sim / nevresim / Lava deployments.
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

    def _get_rates(self) -> list[float]:
        return [p.base_activation.rate for p in self.model.get_perceptrons()]

    def _set_rate(self, rate: float) -> None:
        for p in self.model.get_perceptrons():
            p.base_activation.rate = float(rate)

    # -------------------------------------------------------- state protocol

    def _get_extra_state(self):
        return self._get_rates()

    def _set_extra_state(self, extra):
        for p, r in zip(self.model.get_perceptrons(), extra):
            p.base_activation.rate = float(r)

    # -------------------------------------------------------- cycle evaluate

    def _update_and_evaluate(self, rate: float):
        self._set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    # -------------------------------------------------------- finishing touch

    def _after_run(self):
        self._continue_to_full_rate()

        # Hard-commit rate=1 so downstream code sees the pure LIFActivation
        # at deployment.
        self._set_rate(1.0)

        # Forward-dispatch handoff:
        #   * When ``cycle_accurate_lif_forward`` is on, *keep* the
        #     :class:`_CycleAccurateForward` wrapper installed on
        #     ``model.forward`` for the rest of the pipeline. Subsequent
        #     steps (Weight Quantization, Normalization Fusion, the
        #     SCM/HCM accuracy checks, downstream evaluation) then
        #     evaluate the model under exactly the same dynamics the
        #     chip simulators (HCM/SCM/SANA-FE/nevresim/Lava) execute —
        #     this is the *whole point* of the knob: NF, SCM, HCM and
        #     every chip simulator should report the same accuracy
        #     because they all run cycle-accurate LIF with the same
        #     per-cycle spike trains.
        #   * When CA is off, drop the instance attribute so the class-
        #     level ``forward`` becomes visible again — restoring
        #     rate-mode evaluation for the rest of the pipeline.
        if not self._cycle_accurate and getattr(self, "_patched_forward", False):
            try:
                del self.model.forward
            except AttributeError:
                pass
            self._patched_forward = False

        # Mark the AdaptationManager so clamp/shift/quantize decorators
        # are never reintroduced on subsequent update_activation calls
        # (LIF subsumes their effect).
        self.adaptation_manager.lif_active = True
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
