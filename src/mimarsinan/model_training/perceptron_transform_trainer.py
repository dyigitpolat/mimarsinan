from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import (
    build_optimizer,
    build_scheduler,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.position_fold import copy_position_state

import copy
import torch
from torch.amp.grad_scaler import GradScaler
import functools


class PerceptronTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation,
            recipe=None):

        super().__init__(model, device, data_provider_factory, loss_function, recipe=recipe)
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        self.perceptron_transformation = perceptron_transformation

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
            if dst_tensor is None:
                continue
            if dst_tensor.shape != src_tensor.shape:
                # [U-PCH] a grid's shape follows its projection (one scale per perceptron or one per output neuron)
                dst_tensor.data = src_tensor.data.clone().to(dst_tensor.device)
            else:
                dst_tensor.data.copy_(src_tensor.data)

    def _update_and_transform_model(self):
        for main_p, aux_p, temp_p in self._perceptron_slots:
            self._copy_tensors(aux_p.named_parameters(), temp_p.named_parameters())
            self._copy_tensors(aux_p.named_buffers(), temp_p.named_buffers())

            # Transformations only mutate tensor values (no structural changes), so reusing ``temp_p`` in-place is safe.
            self.perceptron_transformation(temp_p)

            self._copy_tensors(temp_p.named_parameters(), main_p.named_parameters())
            # [U-POST2] The projection's membrane-lattice record travels with the parameters it describes.
            main_p.membrane_lattice_kind = temp_p.membrane_lattice_kind


    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if aux_param.requires_grad and param.grad is not None:
                grad = param.grad.clone()
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                if aux_param.grad is None:
                    aux_param.grad = grad
                else:
                    aux_param.grad.copy_(grad)
            # ``optimizer.zero_grad()`` only clears aux grads, so ``self.model`` grads must be cleared here or backward() accumulates across steps and inflates the effective LR.
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
        """[U-TGAUGE T1-C] Exactly the parameters ``_update_and_transform_model`` writes to the deployed model: each
        perceptron's own (through its transformation). A parameter outside every perceptron (a host op's, a host
        normalization's) is never written, so it is held out of the optimizer and the clip."""
        return list({id(p): p for _main, aux_p, _temp in self._perceptron_slots for p in aux_p.parameters()}.values())

    def _update_writes(self):
        """[U-TGAUGE3] The deployed parameters its update writes (each perceptron's, through its transformation): the
        gradients its transfer moves to the trained copies its optimizer steps."""
        return [p for main_p, _aux_p, _temp_p in self._perceptron_slots for p in main_p.parameters()]

    def _recipe_optimization_model(self):
        return self.aux_model

    def _get_optimizer_and_scheduler(self, lr, epochs):
        if self.recipe is not None and epochs > 0:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe, only=self._params_to_optimize())
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=epochs)
            return optimizer, scheduler, GradScaler("cuda", enabled=False)

        optimizer = torch.optim.Adam(
            self._params_to_optimize(), lr = lr, weight_decay=0, betas=(self.beta1, self.beta2))

        identity_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda = lambda epoch: 1)
        scheduler = identity_scheduler

        return optimizer, scheduler, GradScaler("cuda", enabled=False)

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps: int, *, constant_lr: bool = False):
        if self.recipe is not None and total_steps > 0 and not constant_lr:
            optimizer = build_optimizer(self.aux_model, lr, self.recipe, only=self._params_to_optimize())
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=int(total_steps))
            return optimizer, scheduler, GradScaler("cuda", enabled=False)

        optimizer = torch.optim.Adam(
            self._params_to_optimize(), lr=lr, weight_decay=0, betas=(self.beta1, self.beta2)
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
        return optimizer, scheduler, GradScaler("cuda", enabled=False)

    @property
    def perceptron_transformation(self):
        return self._perceptron_transformation

    @perceptron_transformation.setter
    def perceptron_transformation(self, transformation):
        """[U-PFOLD] Every transformation run on a trained copy carries a fold per position's state: the copy takes its
        master's first (a projection reads the latent weight, never the rows of the step before), and the deployed
        perceptron takes the projected rows and bias offsets it wrote, beside the parameters the update copies."""
        def carrying(temp_p):
            slot = next(((m, a) for m, a, t in self._perceptron_slots if t is temp_p), None)
            if slot is not None:
                copy_position_state(slot[1], temp_p)
            transformation(temp_p)
            if slot is not None:
                copy_position_state(temp_p, slot[0])

        self._perceptron_transformation = carrying
