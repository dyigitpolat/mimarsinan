"""One-shot activation-shift tuner with recovery training.

Applies the full shift once, then recovers accuracy with LR search +
step-budgeted training. Not a smooth adaptation -- extends TunerBase directly.
"""

import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.axes import ActivationShiftAxis
from mimarsinan.tuning.orchestration.rate_tuner_seam import OneShotRateTunerSeamMixin
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import TunerBase


class ActivationShiftTuner(OneShotRateTunerSeamMixin, TunerBase):
    """Apply activation-shift semantics, then recover with step-budgeted training.

    Exposes the uniform ``RateTunerSeam`` over its one-shot controller methods so a
    driver can drive it through the same three verbs as the smooth family (E1)."""

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = DeploymentPlan.of(pipeline)
        # TTFS value-domain modes keep the half-step inside the quantize decorator
        # (shift_back); the LIF-style branch below (bias mutation + shift_rate) would
        # double-shift them at mapping-time bias compensation. The synchronized
        # floor-collapse trains the same convention as ttfs_quantized, so it must
        # take the TTFS branch (sync_collapse_verify regression: deployed -1.9pp).
        self._use_ttfs = (
            plan.spiking_mode == "ttfs" or plan.uses_ttfs_floor_ceil_convention
        )
        self._final_metric = None
        self.name = "Shift Recovery"
        self._axis = ActivationShiftAxis(self._apply_shift)
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

        # EF1: the one-shot shift tuner READS the pipeline-wide optimization-driver
        # axis to record its resolved decision (`self._optimization_driver`). It has no
        # smooth fast ladder (one-shot apply-then-recover, not a rate ramp), so the fast
        # arm is a no-op for this family — it stays the controller path regardless,
        # byte-identical — but it still consumes the axis like every other family.
        self._optimization_driver = plan.optimization_driver_for_family(
            rates=[1.0], steps_per_rate=0
        )

    def _apply_shift(self):
        config = self.pipeline.config
        transformer = PerceptronTransformer()
        if not self._use_ttfs:
            self.adaptation_manager.shift_rate = 1.0

        for perceptron in self.model.get_perceptrons():
            if not self._use_ttfs:
                shift_amount = calculate_activation_shift(
                    config["target_tq"], perceptron.activation_scale
                )
                act_scale = perceptron.activation_scale
                if torch.is_tensor(act_scale) and torch.is_tensor(shift_amount):
                    act_scale = act_scale.to(shift_amount.device)
                effective_bias_shift = shift_amount / act_scale

                self.adaptation_manager.update_activation(config, perceptron)
                transformer.apply_effective_bias_transform(
                    perceptron,
                    lambda bias, delta=effective_bias_shift: bias + delta,
                )
            else:
                self.adaptation_manager.update_activation(config, perceptron)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()

    def run(self):
        self._axis.set_rate(1.0)
        self.pipeline.reporter.report(self.name, 1.0)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        lr = self._find_lr()
        self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            self._get_target(),
            0,
            validation_n_batches=self._budget.progress_eval_batches,
            check_interval=self._budget.check_interval,
            patience=5,
            min_steps=self._budget.check_interval * 3,
            min_improvement=self._budget.accuracy_se(),
        )
        # Tuner internals never call ``trainer.test()`` — the pipeline's
        # ``PipelineStep.pipeline_metric()`` owns the single test() pass.
        self._final_metric = self.trainer.validate()
        return self._final_metric
