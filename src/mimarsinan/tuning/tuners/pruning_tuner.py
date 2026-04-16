"""PruningTuner: structured pruning using SmartSmoothAdaptation.

The pruned set grows monotonically: once an index is committed as pruned
at some rate, it stays pruned at all higher rates. Activation-based
importance is still refreshed per cycle and used to PICK new indices to
add, but never reshuffles already-pruned indices. This eliminates mask
churn between cycles, which otherwise causes catastrophic fast-fails near
k_r boundaries (rate × pruning_fraction × dim ≈ integer) where a tiny rate
bump would add one new pruned row whose weight was still full-strength.

Uses the base-class ``_adaptation()`` loop (with LR search, rollback, and
recovery training).  The persistent pruned sets are snapshotted via
``_get_extra_state`` / ``_set_extra_state`` so rollback restores them.

After run() completes, persistent grad hooks and a normalization
forward-hook are registered on the pruned layers so downstream training
phases (Activation Adaptation, Clamp, Shifting, Activation/Weight
Quantization) cannot push pruned positions away from zero. Without that,
``soft_core_mapping``'s IR elimination would drop rows/cols whose actual
values had drifted — functional equivalence between torch forward and IR
simulation breaks.
"""

import torch
import torch.nn as nn

from mimarsinan.transformations.pruning import (
    _collect_activation_stats,
    apply_pruning_masks,
)
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner, _RECOVERY_PATIENCE


# Module-level hook functions (not closures) so torch.save can pickle the
# model after PruningTuner attaches them. Each hook reads its mask from a
# module buffer registered alongside the hook — buffers survive pickle and
# are re-bound on load, so the hooks keep working after a pipeline resume.

def _pruning_enforce_linear_pre_hook(module, inputs):
    """Zero pruned weight / bias entries on a Linear layer before forward.

    Reads ``prune_mask`` (2-D bool, True=pruned) and ``prune_bias_mask``
    (1-D, True=pruned rows) from the module. A no-op when buffers are
    absent, so it is safe to attach unconditionally to any layer that
    *might* carry pruning buffers.
    """
    prune_mask = getattr(module, "prune_mask", None)
    if prune_mask is not None:
        module.weight.data[prune_mask] = 0.0
    prune_bias_mask = getattr(module, "prune_bias_mask", None)
    if prune_bias_mask is not None and module.bias is not None:
        module.bias.data[prune_bias_mask] = 0.0


def _pruning_enforce_norm_pre_hook(module, inputs):
    """Zero BN ``running_mean`` and ``beta`` for pruned output rows.

    Reads ``_prune_row_mask`` (1-D bool, True=pruned). Forces fused bias
    after NormalizationFusion to be exactly zero at pruned rows so IR
    elimination preserves functional equivalence.
    """
    mask = getattr(module, "_prune_row_mask", None)
    if mask is None:
        return
    if getattr(module, "running_mean", None) is not None:
        module.running_mean.data[mask] = 0.0
    beta = getattr(module, "bias", None)
    if beta is not None and isinstance(beta, torch.nn.Parameter):
        beta.data[mask] = 0.0


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
        # Monotonic pruned sets (per perceptron). Built via _get_masks; snapshotted
        # via _get_extra_state for rollback. Never shrink within a run.
        self._persistent_pruned_rows: list[set] = []
        self._persistent_pruned_cols: list[set] = []

    # -- Mask computation (monotonic) ------------------------------------------

    def _get_masks(self, rate):
        """Build (row_masks, col_masks) at *rate* with monotonic pruned sets.

        The pruned set for each perceptron grows only: we start from the
        currently-committed set, then add the lowest-importance not-yet-pruned
        indices until we reach k = floor(rate * pruning_fraction * dim).
        Since additions are driven by fresh importance, this preserves
        adaptivity, but fixing already-pruned indices eliminates churn.
        """
        import math as _math
        perceptrons = self.model.get_perceptrons()
        n_layers = len(perceptrons)
        exempt_input_layers = {0}
        exempt_output_layers = {n_layers - 1} if n_layers > 0 else set()

        if len(self._persistent_pruned_rows) != n_layers:
            self._persistent_pruned_rows = [set() for _ in range(n_layers)]
            self._persistent_pruned_cols = [set() for _ in range(n_layers)]

        row_masks = []
        col_masks = []
        for i, p in enumerate(perceptrons):
            out_f, in_f = p.layer.weight.data.shape
            device = p.layer.weight.device

            k_r = int(_math.floor(rate * self.pruning_fraction * out_f))
            if i in exempt_output_layers:
                k_r = 0
            pruned_r = set(self._persistent_pruned_rows[i])
            if len(pruned_r) < k_r and i < len(self.base_row_imp):
                _, idx = self.base_row_imp[i].to(device).sort()
                for j in idx.tolist():
                    if len(pruned_r) >= k_r:
                        break
                    pruned_r.add(int(j))
            self._persistent_pruned_rows[i] = pruned_r
            rm = torch.ones(out_f, dtype=torch.bool, device=device)
            if pruned_r:
                rm[list(pruned_r)] = False
            row_masks.append(rm)

            k_c = int(_math.floor(rate * self.pruning_fraction * in_f))
            if i in exempt_input_layers:
                k_c = 0
            pruned_c = set(self._persistent_pruned_cols[i])
            if len(pruned_c) < k_c and i < len(self.base_col_imp):
                _, idx = self.base_col_imp[i].to(device).sort()
                for j in idx.tolist():
                    if len(pruned_c) >= k_c:
                        break
                    pruned_c.add(int(j))
            self._persistent_pruned_cols[i] = pruned_c
            cm = torch.ones(in_f, dtype=torch.bool, device=device)
            if pruned_c:
                cm[list(pruned_c)] = False
            col_masks.append(cm)

        for i in range(n_layers - 1):
            if row_masks[i].shape[0] == col_masks[i + 1].shape[0]:
                col_masks[i + 1] = col_masks[i + 1] & row_masks[i]
        return row_masks, col_masks

    # -- Rollback-aware snapshot -----------------------------------------------

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
        # Refresh importance so that ADDITIONS to the persistent pruned set
        # adapt to current activations; already-pruned indices are fixed.
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
        self._final_metric = self._ensure_pipeline_threshold()
        return self._final_metric

    # -- Main entry point -------------------------------------------------------

    def run(self, max_cycles=None):
        perceptrons = self.model.get_perceptrons()

        initial_acc = self.trainer.validate()
        print(f"[PruningTuner] Initial accuracy: {initial_acc:.4f}")

        self._init_original_weights()
        self._persistent_pruned_rows = [set() for _ in range(len(perceptrons))]
        self._persistent_pruned_cols = [set() for _ in range(len(perceptrons))]

        print("[PruningTuner] Starting fractional discrete adaptation...")
        super().run()

        final_acc = self._final_metric if self._final_metric is not None else self.trainer.test()
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

        self._enforce_pruning_persistently(perceptrons, row_masks, col_masks)

        return final_acc

    # -- Persistent enforcement through downstream training phases -------------

    def _enforce_pruning_persistently(self, perceptrons, row_masks, col_masks):
        """Lock pruned positions to zero for the lifetime of the model.

        Subsequent pipeline phases (Activation Adaptation / Clamp / Shifting /
        Activation Quantization / Weight Quantization) run their own training
        loops. Without enforcement, their optimizers push pruned weights back
        to non-zero via gradient descent — so by the time ``soft_core_mapping``
        reads weight values to build the IR, the "pruned" rows/cols are not
        actually zero. IR elimination then drops math that the trained forward
        still depends on, which shows up as a simulation accuracy drop.

        Mechanism (all picklable so ``torch.save(model)`` works and hooks
        survive a pipeline resume):
        - One final zeroing of ``weight.data`` / ``bias.data`` at pruned entries.
        - Forward-pre-hook on the Linear layer (module-level function reading
          ``prune_mask`` / ``prune_bias_mask`` buffers) re-zeros those entries
          before every forward, so even if optimizer.step() temporarily pushes
          them non-zero between steps, the next forward sees zero.
        - For BatchNorm: zero ``running_mean`` and ``beta`` at pruned output
          rows, register a forward-pre-hook (module-level, reads
          ``_prune_row_mask`` buffer) that re-zeros those on every forward.
          This ensures NormalizationFusion produces exactly zero fused bias
          at pruned rows.
        """
        for i, p in enumerate(perceptrons):
            layer = p.layer
            w = layer.weight
            pruned_rows = (~row_masks[i]).to(device=w.device)
            pruned_cols = (~col_masks[i]).to(device=w.device)
            prune_mask_2d = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)

            with torch.no_grad():
                w.data[prune_mask_2d] = 0.0
                if layer.bias is not None:
                    layer.bias.data[pruned_rows] = 0.0

            # ``prune_mask`` / ``prune_bias_mask`` buffers are already registered
            # on layer in run(); the pre-hook reads them by name.
            layer.register_forward_pre_hook(_pruning_enforce_linear_pre_hook)

            norm = getattr(p, "normalization", None)
            if norm is None or isinstance(norm, nn.Identity):
                continue

            norm.register_buffer("_prune_row_mask", pruned_rows.clone())

            if getattr(norm, "running_mean", None) is not None:
                with torch.no_grad():
                    norm.running_mean.data[pruned_rows] = 0.0
            beta = getattr(norm, "bias", None)
            if beta is not None and isinstance(beta, torch.nn.Parameter):
                with torch.no_grad():
                    beta.data[pruned_rows] = 0.0

            norm.register_forward_pre_hook(_pruning_enforce_norm_pre_hook)
