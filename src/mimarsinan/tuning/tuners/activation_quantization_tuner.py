"""Tuner for gradual activation quantization."""

from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner
from mimarsinan.tuning.orchestration.adaptation_manager import (
    install_sync_entry_grid_snap,
)


class ActivationQuantizationTuner(AdaptationRateTuner):
    rate_attr = "quantization_rate"
    _budget_multiplier = 2.0

    def __init__(self, pipeline, model, target_tq, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr, adaptation_manager)
        self.target_tq = target_tq
        self._final_metric = None
        # [MBH T6] exact-endpoint QAT also trains through the deployed per-stage
        # input grid snap (no-op unless the sync_exact_qat recipe knob + synchronized).
        install_sync_entry_grid_snap(self.model, self.pipeline.config)

    def _stabilization_budget(self):
        return 4 * int(self._budget.max_training_steps)

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
