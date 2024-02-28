from mimarsinan.model_training.training_utilities import AccuracyTracker

import torch

class BasicTrainer:
    def __init__(
            self, model, device, data_loader_factory, loss_function):
        self.model = model.to(device)
        self.device = device

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

    def set_test_batch_size(self, batch_size):
        self.test_batch_size = batch_size
        self.test_loader = self.data_loader_factory.create_test_loader(
            self.test_batch_size, self.data_provider)

    def _report(self, metric_name, metric_value):
        if self.report_function is not None:
            self.report_function(metric_name, metric_value)

    def _get_optimizer_and_scheduler(self, lr):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0 = 5, T_mult = 2, eta_min = lr * 1e-3)

        return optimizer, scheduler

    def _backward_pass_on_loss(self, x, y):
        self.model.train()
        loss = self.loss_function(self.model, x, y)
        loss.backward()
        return loss
    
    def _optimize(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self._backward_pass_on_loss(x, y)
        optimizer.step()
        return loss

    def _train_one_epoch(self, optimizer, scheduler):
        tracker = AccuracyTracker()
        loss = None
        for (x, y) in self.train_loader:
            self.model = self.model.to(self.device)
            x, y = x.to(self.device), y.to(self.device)

            hook_handle = self.model.register_forward_hook(tracker.create_hook(y))
            loss = self._optimize(x, y, optimizer)
            hook_handle.remove()
            
            self._report("Training loss", loss)
        
        if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()
        else:
            scheduler.step(loss)

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
    
    def test(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for (x, y) in self.test_loader:
                self.model.eval()
                self.model = self.model.to(self.device)
                x, y = x.to(self.device), y.to(self.device)
                _, predicted = self.model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
                
        acc = correct / total
        self._report("Test accuracy", acc)
        return acc
    
    def validate(self):
        try:
            x, y = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.validation_loader)
            x, y = next(self.val_iter)

        self.model.eval()
        acc = self._validate_on_loader(x.to(self.device), y.to(self.device))
        self._report("Validation accuracy", acc)
        return acc
    
    def validate_train(self):
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)

        self.model.train()
        acc = self._validate_on_loader(x.to(self.device), y.to(self.device))
        self._report("Validation accuracy on train set", acc)
        return acc

    def train_one_step(self, lr):
        optimizer, _ = self._get_optimizer_and_scheduler(lr)
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
            
        x, y = x.to(self.device), y.to(self.device)
        self._optimize(x, y, optimizer)
    
    def train_n_epochs(self, lr, epochs):
        return self.train_until_target_accuracy(lr, epochs, 1.0)

    def train_until_target_accuracy(self, lr, max_epochs, target_accuracy):
        self._report("LR", lr)
        optimizer, scheduler = self._get_optimizer_and_scheduler(lr)

        validation_accuracy = 0
        for _ in range(max_epochs):
            training_accuracy = self._train_one_epoch(optimizer, scheduler)
            self._report("Training accuracy", training_accuracy)

            validation_accuracy = self.validate()
            if validation_accuracy >= target_accuracy: 
                self._train_one_epoch(optimizer, scheduler)
                self._train_one_epoch(optimizer, scheduler)
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