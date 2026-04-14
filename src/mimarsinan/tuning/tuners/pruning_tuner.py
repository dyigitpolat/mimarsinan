"""PruningTuner: structured pruning using SmartSmoothAdaptation.

This tuner recomputes activation-based significance (row/column importance) at
the start of each adaptation cycle, then smoothly shrinks the targeted pruned
weights towards zero for that cycle. Unpruned weights are left free to train
and heal the network capacity loss.

Uses the base-class ``_adaptation()`` loop (with LR search, rollback, and
recovery training).  Pruning-specific forward-pre-hooks that enforce the
mask pattern during recovery training are injected via
``_recovery_training_hooks(rate)``.
"""

import torch

from mimarsinan.transformations.pruning import (
    _collect_activation_stats,
    apply_pruning_masks,
    compute_masks_from_importance,
)
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner, _RECOVERY_PATIENCE


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

    # -- Mask computation -------------------------------------------------------

    def _get_masks(self, rate):
        perceptrons = self.model.get_perceptrons()
        n_layers = len(perceptrons)
        return compute_masks_from_importance(
            perceptrons,
            rate,
            self.pruning_fraction,
            self.base_row_imp,
            self.base_col_imp,
            exempt_input_layers={0},
            exempt_output_layers={n_layers - 1} if n_layers > 0 else set(),
        )

    def _refresh_pruning_importance(self):
        """Recompute activation-based row/column importance for the current model."""
        perceptrons = self.model.get_perceptrons()
        activation_stats = _collect_activation_stats(
            self.model,
            self.trainer.validation_loader,
            self._device,
            num_batches=5,
        )
        self.base_row_imp.clear()
        self.base_col_imp.clear()
        for i, p in enumerate(perceptrons):
            w = p.layer.weight.data
            if activation_stats[i]["output_importance"] is not None:
                self.base_row_imp.append(activation_stats[i]["output_importance"].clone())
            else:
                self.base_row_imp.append(w.abs().sum(dim=1))
            if activation_stats[i]["input_importance"] is not None:
                self.base_col_imp.append(activation_stats[i]["input_importance"].clone())
            else:
                self.base_col_imp.append(w.abs().sum(dim=0))

    # -- Hook management --------------------------------------------------------

    def _register_hooks(self, target_row_masks, target_col_masks, rate):
        hooks = []
        for i, p in enumerate(self.model.get_perceptrons()):
            rm = target_row_masks[i]
            cm = target_col_masks[i]

            pruned_rows = ~rm
            pruned_cols = ~cm
            prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)

            scale = 1.0 - rate
            target_w = self.original_weights[i][prune_mask] * scale

            b_mask = None
            target_b = None
            if p.layer.bias is not None:
                b_mask = pruned_rows
                target_b = self.original_biases[i][pruned_rows] * scale

            def make_hook(layer, p_mask, t_w, b_m, t_b):
                def hook(module, inputs):
                    module.weight.data[p_mask] = t_w
                    if b_m is not None and module.bias is not None:
                        module.bias.data[b_m] = t_b
                return hook

            hooks.append(
                p.layer.register_forward_pre_hook(
                    make_hook(p.layer, prune_mask, target_w, b_mask, target_b)
                )
            )
        return hooks

    # -- Base-class protocol overrides ------------------------------------------

    def _before_cycle(self):
        self._refresh_pruning_importance()

    def _apply_masks(self, rate):
        """Apply pruning masks at *rate* to model weights (no training)."""
        rate = min(max(rate, 0.0), 1.0)
        perceptrons = self.model.get_perceptrons()
        target_row_masks, target_col_masks = self._get_masks(rate)
        for i, p in enumerate(perceptrons):
            apply_pruning_masks(
                p, target_row_masks[i], target_col_masks[i],
                rate, self.original_weights[i], self.original_biases[i],
            )

    def _update_and_evaluate(self, rate):
        """Apply pruning masks and evaluate — pure evaluation, no training."""
        self._apply_masks(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _recovery_training_hooks(self, rate):
        """Return forward-pre-hooks that enforce the pruning pattern during recovery."""
        rate = min(max(rate, 0.0), 1.0)
        target_row_masks, target_col_masks = self._get_masks(rate)
        return self._register_hooks(target_row_masks, target_col_masks, rate)

    # -- Weight snapshot --------------------------------------------------------

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

    # -- Forced completion (pruning must finish) --------------------------------

    def _force_to_full_rate(self):
        """Gradually push pruning to full rate without rollback.

        Uses 3-4 increments from the current committed rate to 1.0,
        each with a full epoch of guaranteed training before patience
        checks kick in, so sub-0.1%-per-check improvements accumulate.
        """
        current = self._committed_rate
        remaining = 1.0 - current
        n_increments = max(3, min(6, int(remaining / 0.15) + 1))

        for i in range(1, n_increments + 1):
            target = current + remaining * i / n_increments
            target = min(target, 1.0)

            self._refresh_pruning_importance()
            self._apply_masks(target)

            hooks = self._recovery_training_hooks(target)
            try:
                lr = self._find_lr()
                self.trainer.train_steps_until_target(
                    lr,
                    self._budget.max_training_steps,
                    self._get_target(),
                    0,
                    validation_n_batches=self._budget.progress_eval_batches,
                    check_interval=self._budget.check_interval,
                    patience=_RECOVERY_PATIENCE,
                    min_steps=self._budget.check_interval * 3,
                    min_improvement=self._budget.accuracy_se() / 2,
                )
            finally:
                for h in hooks:
                    h.remove()

        self._apply_masks(1.0)
        self._committed_rate = 1.0

    # -- _after_run with final recovery -----------------------------------------

    def _after_run(self):
        """Final recovery after adaptation completes.

        The base ``run()`` already calls ``_continue_to_full_rate()`` before
        this method.  If the rate is still below 1.0, we use the gradual
        ``_force_to_full_rate()`` (which includes recovery at each increment).
        Then a single final recovery pass + safety-net check.
        """
        if self._committed_rate < 1.0 - 1e-6:
            self._force_to_full_rate()

        self._apply_masks(1.0)
        return self._ensure_pipeline_threshold()

    # -- Main entry point -------------------------------------------------------

    def run(self, max_cycles=None):
        perceptrons = self.model.get_perceptrons()

        initial_acc = self.trainer.validate()
        print(f"[PruningTuner] Initial accuracy: {initial_acc:.4f}")

        self._init_original_weights()

        print("[PruningTuner] Starting fractional discrete adaptation...")
        super().run()

        final_acc = self.trainer.test()
        print(f"[PruningTuner] Final overall accuracy: {final_acc:.4f}")

        row_masks, col_masks = self._get_masks(1.0)
        for i, p in enumerate(perceptrons):
            pruned_rows = (~row_masks[i]).sum().item()
            pruned_cols = (~col_masks[i]).sum().item()
            print(
                f"[PruningTuner] Perceptron {i}: rows {pruned_rows}/{row_masks[i].shape[0]} pruned, "
                f"cols {pruned_cols}/{col_masks[i].shape[0]} pruned"
            )

        for i, p in enumerate(perceptrons):
            rm = row_masks[i]
            cm = col_masks[i]
            p.layer.register_buffer("prune_row_mask", (~rm).clone())
            p.layer.register_buffer("prune_col_mask", (~cm).clone())
            p.layer.register_buffer(
                "prune_mask",
                ((~rm).unsqueeze(1) | (~cm).unsqueeze(0)).clone(),
            )
            if p.layer.bias is not None:
                p.layer.register_buffer("prune_bias_mask", (~rm).clone())

        return final_acc
