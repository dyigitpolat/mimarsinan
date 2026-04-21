from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import (
    build_optimizer,
    build_scheduler,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

import copy
import torch
import torch.nn as nn
import functools


class PerceptronTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation,
            recipe=None):

        super().__init__(model, device, data_provider_factory, loss_function, recipe=recipe)
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        self.perceptron_transformation = perceptron_transformation

        # Pre-allocate one reusable "temp" Perceptron per slot. Each training step
        # copies aux params/buffers into temp, runs the (possibly non-differentiable)
        # transformation on temp in-place, then copies transformed params back into
        # main. This replaces four per-step deep-copies (two state_dict clones plus
        # per-perceptron module clones plus a final load_state_dict-with-deepcopy).
        self._perceptron_slots = []
        for name, module in self.model.named_modules():
            if isinstance(module, Perceptron):
                main_p = module
                aux_p = self._get_module_by_name(self.aux_model, name)
                temp_p = copy.deepcopy(aux_p).to(self.device)
                self._perceptron_slots.append((main_p, aux_p, temp_p))

    def _get_module_by_name(self, model, access_string):
        names = access_string.split('.')
        return functools.reduce(getattr, names, model)

    @staticmethod
    def _copy_tensors(src_named, dst_named):
        dst_map = dict(dst_named)
        for name, src_tensor in src_named:
            dst_tensor = dst_map.get(name)
            if dst_tensor is not None:
                dst_tensor.data.copy_(src_tensor.data)

    def _update_and_transform_model(self):
        for main_p, aux_p, temp_p in self._perceptron_slots:
            # Refresh temp from aux (latent weights). Include buffers so the
            # transformation sees the same state a deepcopy(aux_p) would have.
            self._copy_tensors(aux_p.named_parameters(), temp_p.named_parameters())
            self._copy_tensors(aux_p.named_buffers(), temp_p.named_buffers())

            # Apply transformation in-place on temp. Transformations only mutate
            # tensor values — no structural changes — so reusing temp is safe.
            self.perceptron_transformation(temp_p)

            # Write transformed params (only params, matching original semantics)
            # back into the main model.
            self._copy_tensors(temp_p.named_parameters(), main_p.named_parameters())


    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if aux_param.requires_grad and param.grad is not None:
                grad = param.grad.clone()
                # Replace NaN/Inf gradients with zero to prevent aux_model corruption
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                if aux_param.grad is None:
                    aux_param.grad = grad
                else:
                    aux_param.grad.copy_(grad)
            # optimizer.zero_grad() only clears aux_model grads (optimizer is on
            # aux). self.model.grad is never zeroed elsewhere, so without this
            # line backward() accumulates a running sum across every step and
            # the effective LR grows linearly — catastrophic at LR ≳ 1e-3.
            param.grad = None

    def _backward_pass_on_loss(self, x, y, scaler):
        with torch.no_grad():
            self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y, scaler)

        with torch.no_grad():
            self._transfer_gradients_to_aux()
        return loss

    def _params_to_optimize(self):
        # The optimizer steps aux_model, so gradient clipping must target its
        # parameters (gradients are transferred from self.model -> self.aux_model
        # in _backward_pass_on_loss).
        return self.aux_model.parameters()

    def _get_optimizer_and_scheduler(self, lr, epochs):
        if self.recipe is not None and epochs > 0:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe)
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=epochs)
            return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)

        optimizer = torch.optim.Adam(
            self.aux_model.parameters(), lr = lr, weight_decay=0, betas=(self.beta1, self.beta2))

        identity_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda = lambda epoch: 1)
        scheduler = identity_scheduler

        return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps: int, *, constant_lr: bool = False):
        if self.recipe is not None and total_steps > 0 and not constant_lr:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe)
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=int(total_steps))
            return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)

        optimizer = torch.optim.Adam(
            self.aux_model.parameters(), lr=lr, weight_decay=0, betas=(self.beta1, self.beta2)
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
        return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)
