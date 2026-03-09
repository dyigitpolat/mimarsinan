"""PruningTuner: structured pruning using SmartSmoothAdaptation.

This tuner recomputes activation-based significance (row/column importance) at
the start of each adaptation cycle, then smoothly shrinks the targeted pruned
weights towards zero for that cycle. Unpruned weights are left free to train
and heal the network capacity loss.
"""
import copy
import torch

from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.pruning import (
    _collect_activation_stats,
    apply_pruning_masks,
    compute_masks_from_importance,
)
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation

class PruningTuner(PerceptronTuner):
    def __init__(
        self,
        pipeline,
        model,
        target_accuracy,
        lr,
        adaptation_manager,
        pruning_fraction,
    ):
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.target_accuracy = target_accuracy

        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.pruning_fraction = pruning_fraction

        self.device = pipeline.config['device']
        self.epochs = max(pipeline.config.get('tuner_epochs', 5), 3)

        self.base_row_imp = []
        self.base_col_imp = []

    def _get_target_decay(self):
        return 0.99

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
        """Recompute activation-based row/column importance for the current model.
        Called at the start of each adaptation cycle so the pruning set is fresh.
        """
        perceptrons = self.model.get_perceptrons()
        activation_stats = _collect_activation_stats(
            self.model,
            self.trainer.validation_loader,
            self.device,
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

            hooks.append(p.layer.register_forward_pre_hook(make_hook(p.layer, prune_mask, target_w, b_mask, target_b)))
        return hooks

    def run(self, max_cycles=None):
        perceptrons = self.model.get_perceptrons()
        n_layers = len(perceptrons)

        initial_acc = self.trainer.validate()
        print(f"[PruningTuner] Initial accuracy: {initial_acc:.4f}")

        self.original_weights = []
        self.original_biases = []
        for p in perceptrons:
            self.original_weights.append(p.layer.weight.data.clone())
            if p.layer.bias is not None:
                self.original_biases.append(p.layer.bias.data.clone())
            else:
                self.original_biases.append(None)

        def _update_and_eval(rate):
            rate = min(max(rate, 0.0), 1.0)  # Clamp overshoot bug
            target_row_masks, target_col_masks = self._get_masks(rate)
            for i, p in enumerate(perceptrons):
                apply_pruning_masks(p, target_row_masks[i], target_col_masks[i], rate, self.original_weights[i], self.original_biases[i])
            hooks = self._register_hooks(target_row_masks, target_col_masks, rate)
            self.trainer.train_one_step(self.lr / 2.0)
            for hook in hooks:
                hook.remove()
            for i, p in enumerate(perceptrons):
                apply_pruning_masks(p, target_row_masks[i], target_col_masks[i], rate, self.original_weights[i], self.original_biases[i])
            return self.trainer.validate()

        def _adaptation(rate):
            rate = min(max(rate, 0.0), 1.0)
            self.pipeline.reporter.report("Tuning Rate", rate)
            _update_and_eval(rate)
            target_row_masks, target_col_masks = self._get_masks(rate)
            hooks = self._register_hooks(target_row_masks, target_col_masks, rate)
            self.trainer.train_until_target_accuracy(self.lr, self.epochs, self.target_adjuster.get_target(), 0)
            for hook in hooks:
                hook.remove()
            _update_and_eval(rate)
            acc = self.trainer.validate()
            self.target_adjuster.update_target(acc)

        before_cycle = lambda: self._refresh_pruning_importance()
        adapter = SmartSmoothAdaptation(
            _adaptation,
            lambda: copy.deepcopy(self.model.state_dict()),
            lambda state: self.model.load_state_dict(state),
            _update_and_eval,
            [BasicInterpolation(0.0, 1.0)],
            self.target_adjuster.get_target(),
            before_cycle=before_cycle,
        )
        adapter.tolerance = 0.05

        print(f"[PruningTuner] Starting fractional discrete adaptation...")
        adapter.adapt_smoothly(max_cycles=max_cycles)

        _update_and_eval(1.0)
        
        final_acc = self.trainer.validate()
        print(f"[PruningTuner] Final overall accuracy: {final_acc:.4f}")
        
        row_masks, col_masks = self._get_masks(1.0)
        for i, p in enumerate(perceptrons):
            pruned_rows = (~row_masks[i]).sum().item()
            pruned_cols = (~col_masks[i]).sum().item()
            print(f"[PruningTuner] Perceptron {i}: rows {pruned_rows}/{row_masks[i].shape[0]} pruned, cols {pruned_cols}/{col_masks[i].shape[0]} pruned")
        
        # Register 1D pruning masks as persistent buffers for IR pruning (lossless; no 2D recovery).
        for i, p in enumerate(perceptrons):
            rm = row_masks[i]
            cm = col_masks[i]
            # True = pruned (same convention as get_initial_pruning_masks_from_model)
            p.layer.register_buffer("prune_row_mask", (~rm).clone())  # out_f
            p.layer.register_buffer("prune_col_mask", (~cm).clone())  # in_f
            # Legacy 2D/bias for any code that still reads them
            p.layer.register_buffer("prune_mask", ((~rm).unsqueeze(1) | (~cm).unsqueeze(0)).clone())
            if p.layer.bias is not None:
                p.layer.register_buffer("prune_bias_mask", (~rm).clone())

        return final_acc

