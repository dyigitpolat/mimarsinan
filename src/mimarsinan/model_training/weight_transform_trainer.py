from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import (
    build_optimizer,
    build_scheduler,
)

import torch
import copy

class WeightTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, weight_transformation,
            recipe=None):
        super().__init__(model, device, data_provider_factory, loss_function, recipe=recipe)
        self.weight_transformation = weight_transformation
        self.aux_model = copy.deepcopy(self.model).to(self.device)

    def _params_to_optimize(self):
        return self.aux_model.parameters()

    def _update_and_transform_model(self):
        for (name, param), (_, aux_param) in zip(self.model.named_parameters(), self.aux_model.named_parameters()):

            if ('weight' in name) or ('bias' in name):
                param.data[:] = self.weight_transformation(aux_param.data[:]).to(self.device)

    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if param.requires_grad:
                aux_param.grad = param.grad

    def _backward_pass_on_loss(self, x, y, scaler):
        self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y, scaler)

        self._transfer_gradients_to_aux()
        return loss

    def _get_optimizer_and_scheduler(self, lr, epochs):
        if self.recipe is not None and epochs > 0:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe)
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=epochs)
            return optimizer, scheduler, torch.amp.GradScaler("cuda")

        optimizer = torch.optim.AdamW(
            self.aux_model.parameters(), lr = lr, betas = (self.beta1, self.beta2))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs, eta_min = lr * 1e-3)

        return optimizer, scheduler, torch.amp.GradScaler("cuda")

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps: int, *, constant_lr: bool = False):
        if self.recipe is not None and total_steps > 0 and not constant_lr:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe)
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=int(total_steps))
            return optimizer, scheduler, torch.amp.GradScaler("cuda")

        optimizer = torch.optim.AdamW(
            self.aux_model.parameters(), lr=lr, betas=(self.beta1, self.beta2)
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
        return optimizer, scheduler, torch.amp.GradScaler("cuda")
