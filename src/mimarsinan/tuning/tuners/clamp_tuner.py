"""Tuner for gradual activation clamping.

Learnable activation scale (optional)
-------------------------------------
The clamp ceiling used to be fixed at construction from p99 activation
analysis. That choice was reasonable on paper but brittle in practice:
the p99 statistic is computed without considering downstream
quantization-level alignment, so the optimal ceiling can drift during
training.

When ``pipeline.config["clamp_learnable_scale"]`` is truthy, the tuner
makes each perceptron's clamp ceiling an ``nn.Parameter``. A quadratic
log-scale regulariser (see :func:`clamp_scale_regulariser`) pulls the
learned scalar toward the p99 reference so the IR-visible scale remains
bounded. At the end of the tuner, the learned scale is frozen back onto
``perceptron.set_activation_scale(...)`` so downstream IR / simulation
stages see a plain scalar — exactly as before.

Enabling the learnable scale is opt-in because downstream consumers
that may subscribe to a *fixed* ceiling during tuning must be audited
before flipping the default. The minimal refactor stops at "parameter
added, regulariser + freeze wired"; more principled quantizer
formulations (LSQ / LSQ+) are documented in ``TUNING_FUTURE_WORK.md``.
"""

import math

import torch
import torch.nn as nn

from mimarsinan.models.layers import SavedTensorDecorator
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


def clamp_scale_regulariser(
    scale_param: nn.Parameter, reference: float
) -> torch.Tensor:
    """Symmetric quadratic penalty keeping ``scale_param`` near ``reference``.

    Returns ``((scale - reference) / reference) ** 2``. This is:

    - **Zero** at ``scale == reference``.
    - **Symmetric** around the reference: ``scale = reference + delta``
      incurs the same penalty as ``scale = reference - delta`` (for
      any ``|delta| <= reference``).
    - **Relative**: scaled by ``1 / reference`` so the same relative
      drift produces the same loss regardless of the magnitude of the
      reference.

    Use as an additional loss term with a small multiplier — the
    p99 reference is a soft prior, not a hard constraint.
    """
    ref = torch.as_tensor(reference, dtype=scale_param.dtype, device=scale_param.device)
    ref = ref.clamp_min(torch.finfo(scale_param.dtype).eps)
    return ((scale_param - ref) / ref) ** 2


def freeze_learnable_scale(scale_param: nn.Parameter) -> float:
    """Detach a learnable clamp scale to a plain Python float.

    Called at the end of the tuner: downstream stages (IR assembly,
    spike simulation) consume a scalar scale, so the learned parameter
    is serialised to a float and the returned value is written back onto
    the perceptron.
    """
    return float(scale_param.detach().cpu().item())


class ClampTuner(SmoothAdaptationTuner):
    def __init__(
        self,
        pipeline,
        model,
        target_accuracy,
        lr,
        adaptation_manager,
        activation_scales,
        activation_scale_stats,
    ):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.activation_scales = [float(scale) for scale in activation_scales]
        self.activation_scale_stats = activation_scale_stats or {}
        self._final_metric = None

        self.scale_diagnostics = self._calculate_activation_scales(
            self.activation_scales,
            self.activation_scale_stats,
        )

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.saturation_diagnostics = self._probe_clamp_saturation()
        self._log_scale_diagnostics()

    def _calculate_activation_scales(self, scales, activation_scale_stats=None):
        perceptrons = list(self.model.get_perceptrons())
        if len(scales) != len(perceptrons):
            raise ValueError(
                "ClampTuner received activation scales that do not match the model "
                f"perceptron count: {len(scales)} != {len(perceptrons)}"
            )

        metadata_layers = list((activation_scale_stats or {}).get("layers", []))
        if metadata_layers and len(metadata_layers) != len(perceptrons):
            raise ValueError(
                "ClampTuner received activation-scale metadata with the wrong layer count: "
                f"{len(metadata_layers)} != {len(perceptrons)}"
            )

        diagnostics = []
        for idx, (perceptron, act_scale) in enumerate(zip(perceptrons, scales)):
            if not math.isfinite(float(act_scale)) or float(act_scale) <= 0.0:
                raise ValueError(
                    f"Invalid activation scale for perceptron {idx} ({perceptron.name}): {act_scale}"
                )

            if metadata_layers:
                layer = metadata_layers[idx]
                expected_name = layer.get("name")
                expected_scale = layer.get("scale")
                if expected_name is not None and expected_name != perceptron.name:
                    raise ValueError(
                        "Activation-scale metadata order mismatch: "
                        f"expected layer '{expected_name}', got '{perceptron.name}'"
                    )
                if expected_scale is not None and abs(float(expected_scale) - float(act_scale)) > 1e-6:
                    raise ValueError(
                        "Activation-scale metadata mismatch for "
                        f"layer '{perceptron.name}': {act_scale} != {expected_scale}"
                    )

            perceptron.set_activation_scale(float(act_scale))
            diagnostics.append(
                {
                    "index": idx,
                    "name": perceptron.name,
                    "scale": float(act_scale),
                }
            )

        return diagnostics

    def _probe_clamp_saturation(self, n_batches: int = 1):
        """Measure how often outputs hit the clamp ceiling under full clamp."""
        perceptrons = list(self.model.get_perceptrons())
        old_rate = self.adaptation_manager.clamp_rate
        hit_counts = [0 for _ in perceptrons]
        total_counts = [0 for _ in perceptrons]

        self.adaptation_manager.clamp_rate = 1.0
        for perceptron in perceptrons:
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        decorators = []
        for perceptron in perceptrons:
            # sample_to_cpu avoids pinning every perceptron's full output
            # in VRAM during the forward pass; the saturation ratio is a
            # distribution statistic that is unbiased under subsampling.
            decorator = SavedTensorDecorator(sample_to_cpu=True)
            perceptron.activation.decorate(decorator)
            decorators.append(decorator)

        try:
            self.model.eval()
            with torch.no_grad():
                for x, _ in self.trainer.iter_validation_batches(int(n_batches)):
                    x = x.to(self.pipeline.config["device"])
                    _ = self.model(x)
                    for idx, (perceptron, decorator) in enumerate(zip(perceptrons, decorators)):
                        latest = decorator.latest_output
                        if latest is None:
                            continue
                        ceiling = perceptron.activation_scale.detach().to(latest.device)
                        total_counts[idx] += int(latest.numel())
                        hit_counts[idx] += int((latest >= ceiling * 0.999).sum().item())
                        decorator.latest_output = None
        finally:
            for perceptron in perceptrons:
                perceptron.activation.pop_decorator()
            self.adaptation_manager.clamp_rate = old_rate
            for perceptron in perceptrons:
                self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
            self.trainer.val_iter = iter(self.trainer.validation_loader)

        diagnostics = []
        for idx, perceptron in enumerate(perceptrons):
            total = total_counts[idx]
            ratio = float(hit_counts[idx]) / float(total) if total else 0.0
            diagnostics.append(
                {
                    "index": idx,
                    "name": perceptron.name,
                    "saturation_ratio": ratio,
                }
            )
        return diagnostics

    def _log_scale_diagnostics(self):
        scales = [entry["scale"] for entry in self.scale_diagnostics] or [1.0]
        sorted_by_scale = sorted(self.scale_diagnostics, key=lambda entry: entry["scale"])
        most_saturated = sorted(
            self.saturation_diagnostics,
            key=lambda entry: entry["saturation_ratio"],
            reverse=True,
        )
        saturation_max = most_saturated[0]["saturation_ratio"] if most_saturated else 0.0
        saturation_mean = (
            sum(entry["saturation_ratio"] for entry in self.saturation_diagnostics)
            / max(1, len(self.saturation_diagnostics))
        )

        print(
            "[ClampTuner] "
            f"activation_scales={len(scales)}, range=[{min(scales):.4f}, {max(scales):.4f}], "
            f"top_saturated={[(entry['name'], round(entry['saturation_ratio'], 4)) for entry in most_saturated[:3]]}"
        )
        print(
            "[ClampTuner] "
            f"smallest_scales={[(entry['name'], round(entry['scale'], 4)) for entry in sorted_by_scale[:3]]}"
        )

        self.pipeline.reporter.report("Clamp saturation max", saturation_max)
        self.pipeline.reporter.report("Clamp saturation mean", saturation_mean)

        highly_saturated = [
            entry for entry in most_saturated if entry["saturation_ratio"] >= 0.95
        ]
        if highly_saturated:
            print(
                "[ClampTuner] Warning: near-fully clamped layers detected before tuning: "
                f"{[(entry['name'], round(entry['saturation_ratio'], 4)) for entry in highly_saturated[:5]]}"
            )

    def _get_extra_state(self):
        return self.adaptation_manager.clamp_rate

    def _set_extra_state(self, extra):
        self.adaptation_manager.clamp_rate = extra
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.clamp_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()

    def _freeze_learnable_scales(self):
        """Walk every perceptron's activation chain and freeze any learnable
        clamp scale back onto ``perceptron.set_activation_scale(...)``.

        When the learnable-scale path is not enabled (the default), no
        ``ClampDecorator`` will carry a ``scale_param`` and this method is
        a cheap no-op. When enabled, it ensures downstream IR / simulation
        steps see a plain scalar ceiling (they don't know about
        ``nn.Parameter``).
        """
        from mimarsinan.models.decorators import ClampDecorator

        frozen_count = 0
        for perceptron in self.model.get_perceptrons():
            activation = getattr(perceptron, "activation", None)
            if activation is None:
                continue
            decorators = getattr(activation, "decorators", None) or []
            for dec in decorators:
                inner = getattr(dec, "decorator", dec)
                if isinstance(inner, ClampDecorator) and inner.scale_param is not None:
                    scalar = freeze_learnable_scale(inner.scale_param)
                    perceptron.set_activation_scale(scalar)
                    frozen_count += 1
                    break
        if frozen_count:
            print(f"[ClampTuner] Froze {frozen_count} learnable clamp scales to scalars.")

    def _after_run(self):
        self._continue_to_full_rate()

        self.adaptation_manager.clamp_rate = 1.0
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

        self._freeze_learnable_scales()

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric
