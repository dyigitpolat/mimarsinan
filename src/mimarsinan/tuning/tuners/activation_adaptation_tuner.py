"""Tuner for gradual activation adaptation (non-ReLU → ReLU)."""

from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.tuning.axes import ActivationAdaptationAxis
from mimarsinan.tuning.orchestration.blend_ramp import kd_loss_from_config
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.orchestration.tuning_policy import FAST_LADDER_STEPS_PER_RATE
from mimarsinan.tuning.teacher import snapshot_frozen_teacher


class ActivationAdaptationTuner(SmoothAdaptationTuner):

    _budget_multiplier = 2.0
    _skip_one_shot = True

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._kd_teacher = None

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self._axis = ActivationAdaptationAxis()
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

        self._install_kd_teacher()

        self._consume_optimization_driver(
            rates=self.pipeline.config.get(
                "activation_adaptation_fast_rates", [0.25, 0.5, 0.75, 1.0]
            ),
            steps_per_rate=int(
                self.pipeline.config.get(
                    "activation_adaptation_fast_steps_per_rate", FAST_LADDER_STEPS_PER_RATE
                )
            ),
            # [WS-A A3] floor the spanning cosine: with eta_min=0 the FINAL rung
            # (rate 1.0 — the only one needing real adaptation) trains at lr~0.
            eta_min_factor=float(
                self.pipeline.config.get(
                    "activation_adaptation_fast_lr_eta_min", 0.1
                )
            ),
        )

    def _install_kd_teacher(self):
        """Distil the GELU->ReLU morph against a frozen snapshot of the entry
        (pre-blend) model instead of plain CE. Full hard-ReLU collapses a
        GELU-pretrained backbone to chance; CE from that start is a weak signal,
        so KD carries the teacher's soft targets across the ReLU cliff. Default-off."""
        if not bool(self.pipeline.config.get("activation_adaptation_kd", False)):
            return
        self._kd_teacher = snapshot_frozen_teacher(
            self.model, self.pipeline.config["device"]
        )
        self.trainer.loss_function = kd_loss_from_config(
            self.pipeline.config, self._kd_teacher
        )

    def _set_rate(self, rate):
        self._axis.set_rate(rate)

    def _get_extra_state(self):
        base_acts = [
            (p.base_activation, p.base_activation_name)
            for p in self.model.get_perceptrons()
        ]
        return (self.adaptation_manager.activation_adaptation_rate, base_acts)

    def _set_extra_state(self, extra):
        rate, base_acts = extra
        for p, (ba, ban) in zip(self.model.get_perceptrons(), base_acts):
            p.base_activation = ba
            p.base_activation_name = ban
        self._set_rate(rate)

    def _update_and_evaluate(self, rate):
        self._set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        from mimarsinan.pipelining.pipeline_steps.activation_utils import (
            needs_relu_adaptation,
        )
        from mimarsinan.tuning.orchestration.mbh_gate import (
            finalize_on_best_deployed,
        )

        self._continue_to_full_rate()
        # [WS-A A1] a ladder that ends below rate 1.0 hard-swaps from a state
        # whose deployed read can sit far below a mid-ladder candidate; the
        # gate arbitration restores the best deployed state (inert when the
        # final state already reads best).
        finalize_on_best_deployed(self)

        for p in self.model.get_perceptrons():
            if needs_relu_adaptation(p):
                p.base_activation = make_activation("ReLU")
                p.base_activation_name = "ReLU"

        self.adaptation_manager.activation_adaptation_rate = 0.0
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self._committed_metric = self._ensure_pipeline_threshold()
        self._final_metric = self._committed_metric
        self._committed_rate = 1.0
        return self._committed_metric

    def validate(self):
        if hasattr(self, "_committed_metric") and self._committed_metric is not None:
            return self._committed_metric
        return self.trainer.validate()
