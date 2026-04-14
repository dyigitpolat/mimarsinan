"""Tuner for gradual activation clamping."""

import math

import torch

from mimarsinan.models.layers import SavedTensorDecorator
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


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
        return self.trainer.test()

    def _after_run(self):
        self._continue_to_full_rate()

        self.adaptation_manager.clamp_rate = 1.0
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric
