"""SmoothAdaptationTuner for a single AdaptationManager rate field."""

from __future__ import annotations

from mimarsinan.tuning.axes import ManagerRateAxis
from mimarsinan.tuning.orchestration.adaptation_manager import (
    mbh_lif_realloc_ladder_steps,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


_DECISION_SEED = 1234


class AdaptationRateTuner(SmoothAdaptationTuner):
    """Drive one ``adaptation_manager.<rate_attr>`` across all perceptrons."""

    rate_attr: str = "quantization_rate"

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._axis = ManagerRateAxis(self.rate_attr)
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)
        self._axis.set_decision_seed(_DECISION_SEED)

        self._consume_optimization_driver(
            rates=self.pipeline.config.get(
                "manager_rate_fast_rates", [0.25, 0.5, 0.75, 1.0]
            ),
            steps_per_rate=mbh_lif_realloc_ladder_steps(
                self.pipeline.config,
                self.rate_attr,
                int(self.pipeline.config.get("manager_rate_fast_steps_per_rate", 120)),
            ),
        )

    def _get_extra_state(self):
        return self._axis.get_extra_state()

    def _apply_rate(self, rate) -> None:
        self._axis.set_rate(rate)

    def _set_extra_state(self, extra):
        self._apply_rate(extra)

    def _update_and_evaluate(self, rate):
        self._apply_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        self._continue_to_full_rate()
        self._apply_rate(1.0)
        self._committed_rate = 1.0
        self._final_metric = self._ensure_pipeline_threshold()
        return self._final_metric
