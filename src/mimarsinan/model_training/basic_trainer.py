from contextlib import contextmanager

from mimarsinan.model_training.training_utilities import AccuracyTracker
from mimarsinan.data_handling.data_loader_factory import shutdown_data_loader
from mimarsinan.model_training.training_recipe import (
    TrainingRecipe,
    build_optimizer,
    build_scheduler,
)
from mimarsinan.model_training import basic_trainer_subsample
from mimarsinan.model_training import basic_trainer_steps
from mimarsinan.model_training import basic_trainer_epochs
from mimarsinan.model_training import basic_trainer_eval

import warmup_scheduler
import torch


class BasicTrainer:
    def __init__(
            self, model, device, data_loader_factory, loss_function, recipe: TrainingRecipe | None = None):
        self.model = model.to(device)
        self.device = device
        self.recipe = recipe

        self.data_loader_factory = data_loader_factory
        self.data_provider = data_loader_factory.create_data_provider()

        self.training_batch_size = self.data_provider.get_training_batch_size()
        self.validation_batch_size = self.data_provider.get_validation_batch_size()
        self.test_batch_size = self.data_provider.get_test_batch_size()

        self.train_loader = data_loader_factory.create_training_loader(
            self.training_batch_size, self.data_provider)
        self.validation_loader = data_loader_factory.create_validation_loader(
            self.validation_batch_size, self.data_provider)
        self.test_loader = data_loader_factory.create_test_loader(
            self.test_batch_size, self.data_provider)
        
        self.report_function = None
        self.loss_function = loss_function

        self.val_iter = iter(self.validation_loader)
        self.train_iter = iter(self.train_loader)

        self.beta1 = 0.9
        self.beta2 = 0.99

        self._validation_context = None

    def set_training_batch_size(self, batch_size):
        self.training_batch_size = batch_size
        self.train_loader = self.data_loader_factory.create_training_loader(
            self.training_batch_size, self.data_provider)
        self.train_iter = iter(self.train_loader)
    
    def set_validation_batch_size(self, batch_size):
        self.validation_batch_size = batch_size
        self.validation_loader = self.data_loader_factory.create_validation_loader(
            self.validation_batch_size, self.data_provider)
        self.val_iter = iter(self.validation_loader)
        self._gpu_val_cache = None

    def set_test_batch_size(self, batch_size):
        self.test_batch_size = batch_size
        self.test_loader = self.data_loader_factory.create_test_loader(
            self.test_batch_size, self.data_provider)

    def close(self):
        shutdown_data_loader(self.train_loader)
        shutdown_data_loader(self.validation_loader)
        shutdown_data_loader(self.test_loader)
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.train_iter = None
        self.val_iter = None

    def _report(self, metric_name, metric_value):
        if self.report_function is not None:
            self.report_function(metric_name, metric_value)

    def _validation_metric_name(self, base: str) -> str:
        kind = getattr(self, "_validation_context", None)
        if kind is None:
            return base
        return f"{base} ({kind})"

    @contextmanager
    def validation_context(self, kind: str | None):
        previous = getattr(self, "_validation_context", None)
        self._validation_context = str(kind) if kind else None
        try:
            yield
        finally:
            self._validation_context = previous

    def _get_optimizer_and_scheduler(self, lr, epochs):
        if self.recipe is not None and epochs > 0:
            optimizer = build_optimizer(self.model, lr, self.recipe)
            scheduler, _warmup = build_scheduler(optimizer, self.recipe, total_steps=epochs)
            return optimizer, scheduler, torch.amp.GradScaler("cuda")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr = lr, betas = (self.beta1, self.beta2), weight_decay=5e-5)

        if epochs > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max = epochs, eta_min = lr * 1e-3)
        else:
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

        return optimizer, scheduler, torch.amp.GradScaler("cuda")

    def _get_optimizer_and_scheduler_steps(self, lr, total_steps: int, *, constant_lr: bool = False):
        adam_kwargs = dict(lr=lr, betas=(self.beta1, self.beta2), weight_decay=5e-5)
        if torch.device(self.device).type == "cuda":
            adam_kwargs["fused"] = True
        optimizer = torch.optim.Adam(self.model.parameters(), **adam_kwargs)
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

    def _backward_pass_on_loss(self, x, y, scaler):
        self.model.train()
        with torch.amp.autocast("cuda"):
            loss = self.loss_function(self.model, x, y)
        scaler.scale(loss).backward()
        return loss
    
    def _params_to_optimize(self):
        return self.model.parameters()

    def _optimize(self, x, y, optimizer, scaler):
        optimizer.zero_grad()
        loss = self._backward_pass_on_loss(x, y, scaler)
        clip = getattr(self.recipe, "grad_clip_norm", 0.0) if self.recipe is not None else 0.0
        if clip and clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._params_to_optimize(), float(clip))
        scaler.step(optimizer)
        scaler.update()

        return loss

    def _train_one_epoch(self, optimizer, scheduler, scaler):
        tracker = AccuracyTracker()
        loss = None
        for (x, y) in self.train_loader:
            self.model = self.model.to(self.device)
            x, y = x.to(self.device), y.to(self.device)

            hook_handle = self.model.register_forward_hook(tracker.create_hook(y))
            loss = self._optimize(x, y, optimizer, scaler)
            hook_handle.remove()
            
            self._report("Training loss", loss.detach().item())

        scheduler.step()
        self._report("LR", optimizer.param_groups[0]["lr"])
        return tracker.get_accuracy()
    
    def test(self, max_batches: int | None = None):
        return basic_trainer_eval.test(self, max_batches)

    def test_on_subsample(self, *, max_samples: int, seed: int = 0):
        return basic_trainer_subsample.test_on_subsample(
            self, max_samples=max_samples, seed=seed
        )

    def next_validation_batch(self):
        try:
            x, y = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.validation_loader)
            x, y = next(self.val_iter)
        return x, y

    def iter_validation_batches(self, n_batches: int):
        return basic_trainer_eval.iter_validation_batches(self, int(n_batches))

    def next_training_batch(self):
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        return x, y

    def evaluate_loss_on_batch(self, batch) -> float:
        return basic_trainer_eval.evaluate_loss_on_batch(self, batch)

    def validate(self):
        return basic_trainer_eval.validate(self)

    def validate_n_batches(self, n_batches: int) -> float:
        return basic_trainer_eval.validate_n_batches(self, n_batches)

    def validate_train(self):
        return basic_trainer_eval.validate_train(self)

    def train_n_steps(self, lr, steps: int, warmup_steps: int = 0, *, constant_lr: bool = False):
        return basic_trainer_steps.train_n_steps(
            self, lr, steps, warmup_steps, constant_lr=constant_lr
        )

    def train_steps_until_target(
        self,
        lr,
        max_steps,
        target_accuracy,
        warmup_steps=0,
        *,
        validation_n_batches: int = 1,
        check_interval: int = 1,
        patience: int = 3,
        min_steps: int = 0,
        min_improvement: float = 1e-3,
    ):
        return basic_trainer_steps.train_steps_until_target(
            self,
            lr,
            max_steps,
            target_accuracy,
            warmup_steps,
            validation_n_batches=validation_n_batches,
            check_interval=check_interval,
            patience=patience,
            min_steps=min_steps,
            min_improvement=min_improvement,
        )

    def train_one_step(
        self,
        lr,
        *,
        batch=None,
        eval_batch=None,
        return_post_update_loss: bool = False,
    ):
        return basic_trainer_steps.train_one_step(
            self,
            lr,
            batch=batch,
            eval_batch=eval_batch,
            return_post_update_loss=return_post_update_loss,
        )
    
    def train_n_epochs(self, lr, epochs, warmup_epochs = 0):
        return self.train_until_target_accuracy(lr, epochs, 1.0, warmup_epochs)

    def train_validation_epochs(self, lr, n, warmup_epochs=0):
        return basic_trainer_epochs.train_validation_epochs(self, lr, n, warmup_epochs)

    def train_until_target_accuracy(self, lr, max_epochs, target_accuracy, warmup_epochs):
        return basic_trainer_epochs.train_until_target_accuracy(
            self, lr, max_epochs, target_accuracy, warmup_epochs
        )
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['train_loader']
        del state['validation_loader']
        del state['test_loader']
        del state['val_iter']
        del state['train_iter']
        del state['data_provider']

        state['model'] = state['model'].cpu()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        data_loader_factory = state['data_loader_factory']
        self.data_provider = data_loader_factory.create_data_provider()

        self.train_loader = data_loader_factory.create_training_loader(
            self.training_batch_size, self.data_provider)
        self.validation_loader = data_loader_factory.create_validation_loader(
            self.validation_batch_size, self.data_provider)
        self.test_loader = data_loader_factory.create_test_loader(
            self.test_batch_size, self.data_provider)
        
        self.val_iter = iter(self.validation_loader)
        self.train_iter = iter(self.train_loader)

        self.model = self.model.to(self.device)
