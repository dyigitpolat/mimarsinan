from mimarsinan.model_training.training_utilities import AccuracyTracker

import torch.nn as nn
import torch

class BasicTrainer:
    def __init__(
            self, model, device, train_loader, validation_loader, loss_function):
        self.model = nn.DataParallel(model).to(device)
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.report_function = None
        self.loss_function = loss_function

    def _report(self, metric_name, metric_value):
        if self.report_function is not None:
            self.report_function(metric_name, metric_value)

    def _get_optimizer_and_scheduler(self, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', patience=5, factor=0.9, min_lr=lr/100, verbose=True)

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
            x, y = x.to(self.device), y.to(self.device)

            hook_handle = self.model.register_forward_hook(tracker.create_hook(y))
            loss = self._optimize(x, y, optimizer)
            hook_handle.remove()
            
            self._report("Training loss", loss)
        
        scheduler.step(loss)
        return tracker.get_accuracy()
    
    def _validate_on_loader(self, loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for (x, y) in loader:
                x, y = x.to(self.device), y.to(self.device)

                self.model.eval()
                _, predicted = self.model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
                break
            
        return correct / total
    
    def validate(self):
        acc = self._validate_on_loader(self.validation_loader)
        self._report("Validation accuracy", acc)
        return acc
    
    def validate_train(self):
        acc = self._validate_on_loader(self.train_loader)
        self._report("Validation accuracy on train set", acc)
        return acc

    def train_one_step(self, lr):
        optimizer, _ = self._get_optimizer_and_scheduler(lr)
        for (x, y) in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self._optimize(x, y, optimizer)
            break
    
    def train_n_epochs(self, lr, epochs):
        return self.train_until_target_accuracy(lr, epochs, 1.0)

    def train_until_target_accuracy(self, lr, max_epochs, target_accuracy):
        self._report("LR", lr)
        optimizer, scheduler = self._get_optimizer_and_scheduler(lr)

        training_accuracy = 0
        for _ in range(max_epochs):
            training_accuracy = self._train_one_epoch(optimizer, scheduler)
            self._report("Training accuracy", training_accuracy)
            self.validate()
            if training_accuracy >= target_accuracy: break
        
        return training_accuracy
