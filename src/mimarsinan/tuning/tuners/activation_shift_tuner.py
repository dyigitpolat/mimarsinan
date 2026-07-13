"""One-shot activation-shift tuner with recovery training."""

import torch

from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.axes import ActivationShiftAxis
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_active
from mimarsinan.tuning.orchestration.rate_tuner_seam import OneShotRateTunerSeamMixin
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import TunerBase


class ActivationShiftTuner(OneShotRateTunerSeamMixin, TunerBase):
    """Apply activation-shift semantics, then recover with step-budgeted training."""

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = DeploymentPlan.of(pipeline)
        # TTFS keeps the half-step inside the quantize decorator; the LIF-style bias-shift branch would double-shift it at mapping-time bias compensation.
        self._use_ttfs = (
            plan.spiking_mode == "ttfs" or plan.uses_ttfs_floor_ceil_convention
        )
        # [lif_exact_qat_program §6.1(1), P-L6] under the exact arm the QAT owns
        # every offset: the unflagged one-shot bias bake (a measured operating-
        # point displacement) is skipped; the ttfs branch is untouched.
        self._skip_lif_bias_bake = lif_exact_qat_active(pipeline.config)
        self._final_metric = None
        self.name = "Shift Recovery"
        self._axis = ActivationShiftAxis(self._apply_shift)
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

        self._optimization_driver = plan.optimization_driver_for_family(
            rates=[1.0], steps_per_rate=0
        )

    def _apply_shift(self):
        config = self.pipeline.config
        transformer = PerceptronTransformer()
        bake = not self._use_ttfs and not self._skip_lif_bias_bake
        if bake:
            self.adaptation_manager.shift_rate = 1.0

        for perceptron in self.model.get_perceptrons():
            if bake:
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

        self._train_recovery(self._get_target(), final_validation=False)
        self._final_metric = self.trainer.validate()
        return self._final_metric
