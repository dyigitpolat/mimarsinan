"""PruningTuner: structured pruning driven by the rate scheduler."""

from __future__ import annotations

from mimarsinan.transformations.pruning import apply_pruning_masks
from mimarsinan.tuning.axes import PruningAxis
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.tuners.pruning.pruning_tuner_enforce import (
    enforce_pruning_persistently,
    register_prune_buffers,
)
from mimarsinan.tuning.tuners.pruning.pruning_tuner_masks import (
    force_to_full_rate,
    get_masks,
    refresh_pruning_importance,
    register_recovery_hooks,
)


class PruningTuner(SmoothAdaptationTuner):

    _budget_multiplier = 2.0
    _skip_one_shot = True

    def __init__(
        self,
        pipeline,
        model,
        target_accuracy,
        lr,
        adaptation_manager,
        pruning_fraction,
    ):
        super().__init__(pipeline, model, target_accuracy, lr)

        self.target_accuracy = target_accuracy
        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.pruning_fraction = pruning_fraction
        self._device = pipeline.config["device"]

        self.base_row_imp = []
        self.base_col_imp = []
        self.original_weights = []
        self.original_biases = []
        self._persistent_pruned_rows: list[set] = []
        self._persistent_pruned_cols: list[set] = []

        self._axis = PruningAxis(
            self._apply_masks, recovery_hooks_fn=self._recovery_training_hooks
        )
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

    def _get_masks(self, rate):
        return get_masks(self, rate)

    def _get_extra_state(self):
        return {
            "pruned_rows": [set(s) for s in self._persistent_pruned_rows],
            "pruned_cols": [set(s) for s in self._persistent_pruned_cols],
        }

    def _set_extra_state(self, extra):
        if extra is None:
            return
        self._persistent_pruned_rows = [set(s) for s in extra["pruned_rows"]]
        self._persistent_pruned_cols = [set(s) for s in extra["pruned_cols"]]

    def _refresh_pruning_importance(self):
        refresh_pruning_importance(self)

    def _register_hooks(self, target_row_masks, target_col_masks, rate):
        return register_recovery_hooks(self, target_row_masks, target_col_masks, rate)

    def _before_cycle(self):
        self._refresh_pruning_importance()

    def _apply_masks(self, rate):
        rate = min(max(rate, 0.0), 1.0)
        perceptrons = self.model.get_perceptrons()
        target_row_masks, target_col_masks = self._get_masks(rate)
        for i, p in enumerate(perceptrons):
            apply_pruning_masks(
                p, target_row_masks[i], target_col_masks[i],
                rate, self.original_weights[i], self.original_biases[i],
            )

    def _update_and_evaluate(self, rate):
        self._axis.set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _recovery_training_hooks(self, rate):
        rate = min(max(rate, 0.0), 1.0)
        target_row_masks, target_col_masks = self._get_masks(rate)
        return self._register_hooks(target_row_masks, target_col_masks, rate)

    def _init_original_weights(self):
        perceptrons = self.model.get_perceptrons()
        self.original_weights = []
        self.original_biases = []
        for p in perceptrons:
            self.original_weights.append(p.layer.weight.data.clone())
            if p.layer.bias is not None:
                self.original_biases.append(p.layer.bias.data.clone())
            else:
                self.original_biases.append(None)

    def _force_to_full_rate(self):
        force_to_full_rate(self)

    def _after_run(self):
        if self._committed_rate < 1.0 - 1e-6:
            self._force_to_full_rate()

        self._apply_masks(1.0)

        perceptrons = self.model.get_perceptrons()
        row_masks, col_masks = self._get_masks(1.0)
        register_prune_buffers(perceptrons, row_masks, col_masks)
        enforce_pruning_persistently(perceptrons, row_masks, col_masks)

        self._final_metric = self._ensure_pipeline_threshold()
        return self._final_metric

    def run(self, max_cycles=None):
        perceptrons = self.model.get_perceptrons()

        initial_acc = self.trainer.validate()
        print(f"[PruningTuner] Initial accuracy: {initial_acc:.4f}")

        self._init_original_weights()
        self._persistent_pruned_rows = [set() for _ in range(len(perceptrons))]
        self._persistent_pruned_cols = [set() for _ in range(len(perceptrons))]

        print("[PruningTuner] Starting fractional discrete adaptation...")
        super().run()

        final_acc = self._final_metric if self._final_metric is not None else self.trainer.validate()
        print(f"[PruningTuner] Final validation accuracy: {final_acc:.4f} "
              "(authoritative test metric set by PipelineStep.pipeline_metric())")

        row_masks, col_masks = self._get_masks(1.0)
        for i, p in enumerate(perceptrons):
            pruned_rows = (~row_masks[i]).sum().item()
            pruned_cols = (~col_masks[i]).sum().item()
            print(
                f"[PruningTuner] Perceptron {i}: rows {pruned_rows}/{row_masks[i].shape[0]} pruned, "
                f"cols {pruned_cols}/{col_masks[i].shape[0]} pruned"
            )

        return final_acc
