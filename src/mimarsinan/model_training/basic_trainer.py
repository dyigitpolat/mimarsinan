from mimarsinan.model_training.training_utilities import AccuracyTracker
from mimarsinan.data_handling.data_loader_factory import shutdown_data_loader
from mimarsinan.model_training.training_recipe import (
    TrainingRecipe,
    build_optimizer,
    build_scheduler,
)

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

        # Phase D2 two-tier validation:
        #   * ``validate_fast`` -> in-flight LR probes & per-cycle progress.
        #     Default to at most 16 batches (matches TuningBudget.progress_eval_batches).
        #   * ``validate_full`` -> rollback / safety-net / baseline probes.
        #     Default to the entire validation loader so callers that never
        #     configure a budget still get an honest "full-set" metric.
        # Tuners overwrite these in ``TunerBase`` via
        # ``set_fast_validation_batches`` / ``set_full_validation_batches``
        # so the two tiers align with ``progress_eval_batches`` and
        # ``eval_n_batches`` respectively.
        try:
            full_default = max(1, len(self.validation_loader))
        except TypeError:
            full_default = 1
        self._fast_n_batches = min(16, full_default)
        self._full_n_batches = full_default

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
        # Keep the two-tier defaults in sync with the new loader length.
        # Explicit overrides via ``set_fast_validation_batches`` /
        # ``set_full_validation_batches`` still win because tuners call
        # those setters after changing the validation batch size.
        try:
            full_default = max(1, len(self.validation_loader))
        except TypeError:
            full_default = 1
        self._fast_n_batches = min(self._fast_n_batches, full_default)
        self._full_n_batches = full_default

    def set_test_batch_size(self, batch_size):
        self.test_batch_size = batch_size
        self.test_loader = self.data_loader_factory.create_test_loader(
            self.test_batch_size, self.data_provider)

    def close(self):
        """Shut down DataLoader workers. Call when done with training/testing."""
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
        """Schedule over ``total_steps`` gradient updates.

        When *constant_lr* is ``True`` the learning rate stays fixed after a
        short linear warmup (5% of steps).  Otherwise cosine annealing is used.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=(self.beta1, self.beta2), weight_decay=5e-5
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

    def _backward_pass_on_loss(self, x, y, scaler):
        self.model.train()
        with torch.amp.autocast("cuda"):
            loss = self.loss_function(self.model, x, y)
        scaler.scale(loss).backward()
        return loss
    
    def _params_to_optimize(self):
        """Parameters that the optimizer actually steps.

        Subclasses that optimize a companion model (e.g.
        :class:`PerceptronTransformTrainer` optimizes ``self.aux_model``)
        override this so gradient clipping targets the correct parameters.
        """
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
    
    def _validate_on_loader(self, x, y):
        total = 0
        correct = 0
        with torch.no_grad():
            self.model = self.model.to(self.device)
            x, y = x.to(self.device), y.to(self.device)
            _, predicted = self.model(x).max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
            
        return correct / total
    
    def test(self, max_batches: int | None = None):
        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.test_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                self.model.eval()
                self.model = self.model.to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                _, predicted = self.model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())

        if total <= 0:
            return 0.0
        acc = correct / total
        self._report("Test accuracy", acc)
        return acc

    def next_validation_batch(self):
        """Return the next validation minibatch, rewinding on exhaustion."""
        try:
            x, y = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.validation_loader)
            x, y = next(self.val_iter)
        return x, y

    def iter_validation_batches(self, n_batches: int):
        """Yield ``n_batches`` validation minibatches with iterator wraparound."""
        for _ in range(int(n_batches)):
            yield self.next_validation_batch()

    def next_training_batch(self):
        """Return the next training minibatch, rewinding on exhaustion."""
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        return x, y

    def evaluate_loss_on_batch(self, batch) -> float:
        """Evaluate loss on a fixed batch without updating model weights."""
        x, y = batch
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            loss = self.loss_function(self.model, x, y)
        return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
    
    def validate(self):
        x, y = self.next_validation_batch()
        self.model.eval()
        acc = self._validate_on_loader(x.to(self.device), y.to(self.device))
        self._report("Validation accuracy", acc)
        return acc

    def validate_n_batches(self, n_batches: int) -> float:
        """Average classification accuracy over ``n_batches`` validation minibatches."""
        if n_batches <= 0:
            return 0.0
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in self.iter_validation_batches(int(n_batches)):
                x, y = x.to(self.device), y.to(self.device)
                _, predicted = self.model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
        acc = correct / total if total else 0.0
        self._report("Validation accuracy", acc)
        return acc

    def set_fast_validation_batches(self, n_batches: int) -> None:
        """Configure ``validate_fast``'s batch count (Phase D2).

        ``validate_fast`` is the explicit API for in-flight LR probes
        and per-cycle progress checks inside tuners; it must stay
        cheap (a small, fixed-size subset).  Tuners set this to their
        budget's ``progress_eval_batches``.
        """
        self._fast_n_batches = max(1, int(n_batches))

    def set_full_validation_batches(self, n_batches: int) -> None:
        """Configure ``validate_full``'s batch count (Phase D2).

        ``validate_full`` is the explicit API for rollback / safety-net
        / baseline decisions; it must reflect the entire validation
        loader (up to the cap the tuning budget computes for its
        per-cycle wall-clock ceiling).  Tuners set this to their
        budget's ``eval_n_batches``.
        """
        self._full_n_batches = max(1, int(n_batches))

    def validate_fast(self) -> float:
        """Phase D2 fast-tier validation.

        Evaluates on a small subset (``_fast_n_batches``, default 16)
        for LR probes and per-cycle progress checks.  Cheap by design:
        callers that need a statistically-strong metric (rollback,
        safety-net, baseline) must use :meth:`validate_full` instead.
        """
        return self.validate_n_batches(self._fast_n_batches)

    def validate_full(self) -> float:
        """Phase D2 full-tier validation.

        Evaluates on the full validation set (``_full_n_batches``,
        default = ``len(validation_loader)``).  This is the metric
        used for rollback decisions, safety-net recovery acceptance,
        and the baseline probe at the start of every tuner's main
        loop.  Intentionally heavier than :meth:`validate_fast` --
        callers that want an in-flight probe should use
        :meth:`validate_fast`.
        """
        return self.validate_n_batches(self._full_n_batches)
    
    def validate_train(self):
        x, y = self.next_training_batch()
        self.model.train()
        acc = self._validate_on_loader(x.to(self.device), y.to(self.device))
        self._report("Validation accuracy on train set", acc)
        return acc

    def train_n_steps(self, lr, steps: int, warmup_steps: int = 0, *, constant_lr: bool = False):
        """Run exactly ``steps`` gradient updates (plus optional LR warmup steps).

        When *constant_lr* is True the learning rate is held fixed for the
        entire run instead of following a CosineAnnealing schedule.  This is
        used by the LR range finder so that each probe reflects the sustained
        effect of a candidate LR, not its rapidly-decayed tail.
        """
        optimizer, scheduler, scaler = self._get_optimizer_and_scheduler_steps(lr, steps)
        if constant_lr:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=0
            )
        if warmup_steps > 0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=warmup_steps,
                after_scheduler=scheduler,
            )
        total = int(steps) + int(warmup_steps)
        for _ in range(total):
            try:
                x, y = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                x, y = next(self.train_iter)
            x, y = x.to(self.device), y.to(self.device)
            self._optimize(x, y, optimizer, scaler)
            scheduler.step()
            self._report("LR", optimizer.param_groups[0]["lr"])
        del optimizer, scheduler, scaler

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
        """Train until target reached, converged, or ``max_steps`` exhausted.

        Uses constant LR with linear warmup (5% of steps) for predictable
        behavior regardless of when convergence triggers.  Saves the best
        model state and restores it on exit.

        Progress is checked every ``check_interval`` steps. Training stops early
        when the target is met or when ``patience`` consecutive checks show no
        improvement of at least ``min_improvement`` (convergence). Patience-based
        stopping is suppressed until at least ``min_steps`` gradient steps have
        been executed, giving the optimizer a minimum runway before convergence
        detection kicks in.
        """
        optimizer, scheduler, scaler = self._get_optimizer_and_scheduler_steps(
            lr, max_steps, constant_lr=True
        )
        if warmup_steps > 0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer, multiplier=1.0, total_epoch=warmup_steps, after_scheduler=scheduler
            )
        total = int(max_steps) + int(warmup_steps)
        n_val = max(1, int(validation_n_batches))
        interval = max(1, int(check_interval))
        min_s = max(0, int(min_steps))
        imp_eps = float(min_improvement)

        # Use the shared clone/restore helpers so that trainers with an
        # ``aux_model`` (e.g. PerceptronTransformTrainer, where
        # ``_update_and_transform_model`` regenerates self.model from aux
        # every forward pass) round-trip BOTH models.  Without this the
        # best-state restore only fixes self.model; the next
        # ``_update_and_transform_model`` call regenerates self.model from
        # the last (possibly worse) aux_model, silently destroying the
        # best state we preserved.
        from mimarsinan.tuning.learning_rate_explorer import (
            clone_state_for_trainer,
            restore_state_for_trainer,
        )

        best_acc = 0.0
        best_state = None
        stale_checks = 0

        for step_idx in range(total):
            x, y = self.next_training_batch()
            x, y = x.to(self.device), y.to(self.device)
            self._optimize(x, y, optimizer, scaler)
            scheduler.step()
            self._report("LR", optimizer.param_groups[0]["lr"])

            if (step_idx + 1) % interval == 0 or step_idx == total - 1:
                acc = self.validate_n_batches(n_val)
                if acc >= target_accuracy:
                    best_state = clone_state_for_trainer(self)
                    for _ in range(2):
                        x, y = self.next_training_batch()
                        x, y = x.to(self.device), y.to(self.device)
                        self._optimize(x, y, optimizer, scaler)
                        scheduler.step()
                    break
                if acc > best_acc + imp_eps:
                    best_acc = acc
                    best_state = clone_state_for_trainer(self)
                    stale_checks = 0
                else:
                    stale_checks += 1
                    if step_idx + 1 >= min_s and stale_checks >= patience:
                        break

        if best_state is not None:
            restore_state_for_trainer(self, best_state)
        del optimizer, scheduler, scaler, best_state
        return self.validate_n_batches(n_val)

    def train_one_step(
        self,
        lr,
        *,
        batch=None,
        eval_batch=None,
        return_post_update_loss: bool = False,
    ):
        optimizer, _, scaler = self._get_optimizer_and_scheduler(lr, epochs=0)
        if batch is None:
            x, y = self.next_training_batch()
        else:
            x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = self._optimize(x, y, optimizer, scaler)
        if return_post_update_loss:
            probe_batch = eval_batch if eval_batch is not None else (x, y)
            return self.evaluate_loss_on_batch(probe_batch)
        return float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)
    
    def train_n_epochs(self, lr, epochs, warmup_epochs = 0):
        return self.train_until_target_accuracy(lr, epochs, 1.0, warmup_epochs)

    def train_validation_epochs(self, lr, n, warmup_epochs=0):
        """
        Run exactly ``n`` training epochs (plus optional LR warmup), validating
        after each epoch. Does **not** run :meth:`test` — for calibration and
        other paths that must avoid a full test-set pass.
        """
        optimizer, scheduler, scaler = self._get_optimizer_and_scheduler(lr, n)

        if self.recipe is not None:
            warmup_epochs = 0

        if warmup_epochs > 0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=warmup_epochs,
                after_scheduler=scheduler,
            )

        validation_accuracy = 0.0
        for _ in range(int(n) + int(warmup_epochs)):
            training_accuracy = self._train_one_epoch(optimizer, scheduler, scaler)
            self._report("Training accuracy", training_accuracy)
            validation_accuracy = self.validate()

        return validation_accuracy

    def train_until_target_accuracy(self, lr, max_epochs, target_accuracy, warmup_epochs):
        optimizer, scheduler, scaler = self._get_optimizer_and_scheduler(lr, max_epochs)

        # When a recipe is active, warmup is already built into the scheduler
        # via SequentialLR — do not double-wrap.
        if self.recipe is not None:
            warmup_epochs = 0

        if warmup_epochs > 0:
            scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=warmup_epochs, after_scheduler=scheduler)

        validation_accuracy = 0.0
        for _ in range(max_epochs + warmup_epochs):
            training_accuracy = self._train_one_epoch(optimizer, scheduler, scaler)
            self._report("Training accuracy", training_accuracy)

            validation_accuracy = self.validate()
            if validation_accuracy >= target_accuracy:
                self._train_one_epoch(optimizer, scheduler, scaler)
                self._train_one_epoch(optimizer, scheduler, scaler)
                # Re-measure after the extra epochs so the returned metric
                # reflects the actual final model weights, not the pre-extra-epoch
                # snapshot that triggered early stopping.
                validation_accuracy = self.validate()
                break

        self.test()
        return validation_accuracy
    
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