"""Weight quantization tuner using NormalizationAwarePerceptronQuantization."""

import torch

from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.tuning.axes import NAPQAxis
from mimarsinan.tuning.orchestration.frontier.endpoint_recovery import run_endpoint_recovery
from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner


class NormalizationAwarePerceptronQuantizationTuner(PerceptronTransformTuner):
    """Projection at the recipe rates + the bounded P1'' endpoint recovery.

    Demoted from its accidental-controller role (theory 5g-v): the gated fixed
    ladder walks projection-only rungs (0 training steps) and the generic
    endpoint stage owns the bounded, high-water-anchored recovery.
    """

    # The previous-transform is a no-op below: the mixing path may clone
    # parameter tensors instead of deep-copying the module per step.
    _prev_transform_is_identity = True

    def __init__(
        self, pipeline, model, quantization_bits, target_accuracy, lr,
        adaptation_manager, two_scale_projection=False,
    ):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.quantization_bits = quantization_bits
        self.adaptation_manager = adaptation_manager
        # Resolved by WeightQuantizationStep (config key AND the platform's
        # on-chip bias capability); every projection this tuner applies —
        # rungs, probe replicas, endpoint reprojection — must share it.
        self.two_scale_projection = bool(two_scale_projection)
        self._axis = NAPQAxis(
            self._apply_rate, replica_apply_fn=self._apply_rate_to,
        )
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)
        self._consume_optimization_driver(
            rates=self.pipeline.config.get("wq_fast_rates", [0.5, 1.0]),
            steps_per_rate=int(
                self.pipeline.config.get("wq_fast_steps_per_rate", 0)
            ),
        )

    def _get_previous_perceptron_transform(self, rate):
        return lambda perceptron: None

    def _get_new_perceptron_transform(self, rate):
        def transform(perceptron):
            NormalizationAwarePerceptronQuantization(
                self.quantization_bits,
                self.pipeline.config["device"],
                rate,
                two_scale=self.two_scale_projection,
            ).transform(perceptron)
        return transform

    def _apply_rate(self, rate):
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        with torch.no_grad():
            self.trainer._update_and_transform_model()

    def _apply_rate_to(self, model, rate):
        """Probe-replica apply: transform ``model``'s perceptrons directly, leaving
        the live trainer wiring and the live model untouched."""
        transform = self._mixed_transform(rate)
        with torch.no_grad():
            for perceptron in model.get_perceptrons():
                transform(perceptron)

    def _update_and_evaluate(self, rate):
        self._axis.set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.eval_n_batches)

    def _post_stabilization_hook(self):
        if not getattr(self, "_fixed_ladder_policy", False):
            return
        # P1'' through the transform trainer: recovery trains the float aux
        # model and reprojects, so the shipped model stays chip-quantized.
        # [5u generalized] the WQ endpoint is the FINAL, quantized conversion
        # composition, so it carries the well-conditioned floor
        # (wq_endpoint_target_floor) scoped to itself alone — the intermediate
        # mode endpoints never lift it (one lift, bounded wall). The bit-parity
        # family's endpoint_target_floor still lifts every endpoint via config.
        cfg = self.pipeline.config
        target_floor = max(
            float(cfg.get("endpoint_target_floor", 0.0)),
            float(cfg.get("wq_endpoint_target_floor", 0.0)),
        )
        run_endpoint_recovery(
            self,
            base_steps=int(cfg.get("wq_endpoint_recovery_steps", 0)),
            target_floor=target_floor,
        )
