from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import (
    build_optimizer,
    build_scheduler,
)

import torch
from torch.amp.grad_scaler import GradScaler
import copy

class WeightTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, weight_transformation,
            recipe=None, *, tuning_recipe_recovery=False):
        super().__init__(
            model, device, data_provider_factory, loss_function, recipe=recipe,
            tuning_recipe_recovery=tuning_recipe_recovery,
        )
        self.weight_transformation = weight_transformation
        self.aux_model = copy.deepcopy(self.model).to(self.device)

    def _written_pairs(self):
        """``(deployed, trained)`` parameter pairs the update writes: every parameter named a weight or a bias."""
        return [(param, aux_param)
                for (name, param), (_, aux_param) in zip(self.model.named_parameters(), self.aux_model.named_parameters())
                if ('weight' in name) or ('bias' in name)]

    def _params_to_optimize(self):
        """[U-TGAUGE T1-C] Exactly the parameters the update writes to the deployed model (``_written_pairs``); any
        other trained parameter would move in the aux model and never reach the deployed one."""
        return [aux_param for _param, aux_param in self._written_pairs()]

    def _update_writes(self):
        """[U-TGAUGE3] The deployed parameters its update writes (``_written_pairs``): the gradients its transfer moves to
        the trained copies its optimizer steps."""
        return [param for param, _aux_param in self._written_pairs()]

    def _recipe_optimization_model(self):
        return self.aux_model

    def _update_and_transform_model(self):
        for param, aux_param in self._written_pairs():
            param.data[:] = self.weight_transformation(aux_param.data[:]).to(self.device)

    def _transfer_gradients_to_aux(self):
        """Moves each deployed gradient to its trained copy. [U-TGAUGE2] The deployed one is cleared: the optimizer clears
        only the trained copies, so a gradient left (aliased) on the deployed model would collect every later backward."""
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if param.requires_grad:
                aux_param.grad = param.grad
            param.grad = None

    def _backward_pass_on_loss(self, x, y, scaler):
        self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y, scaler)

        self._transfer_gradients_to_aux()
        return loss

    def _get_optimizer_and_scheduler(self, lr, epochs):
        if self.recipe is not None and epochs > 0:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe, only=self._params_to_optimize())
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=epochs)
            return optimizer, scheduler, GradScaler("cuda")

        optimizer = torch.optim.AdamW(
            self._params_to_optimize(), lr = lr, betas = (self.beta1, self.beta2))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs, eta_min = lr * 1e-3)

        return optimizer, scheduler, GradScaler("cuda")

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps: int, *, constant_lr: bool = False):
        if self._recipe_step_recovery_enabled():
            optimizer = self._build_recipe_step_optimizer(lr)
            scheduler = self._build_recipe_step_scheduler(optimizer, total_steps)
            return optimizer, scheduler, GradScaler("cuda")
        if self.recipe is not None and total_steps > 0 and not constant_lr:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe, only=self._params_to_optimize())
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=int(total_steps))
            return optimizer, scheduler, GradScaler("cuda")

        optimizer = torch.optim.AdamW(
            self._params_to_optimize(), lr=lr, betas=(self.beta1, self.beta2)
        )
        if constant_lr or total_steps <= 0:
            warmup_iters = max(1, int(total_steps * 0.05)) if total_steps > 0 else 1
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(total_steps), eta_min=lr * 1e-3
            )
        return optimizer, scheduler, GradScaler("cuda")
