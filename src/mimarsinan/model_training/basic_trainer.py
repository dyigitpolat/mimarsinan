from mimarsinan.model_training.training_utilities import AccuracyTracker

import torch

class BasicTrainer:
    def __init__(
            self, model, device, train_loader, test_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _report(self, metric_name, metric_value):
        print(f"    {metric_name}: {metric_value}")

    def _get_optimizer_and_scheduler(self, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', patience=5, factor=0.9, min_lr=lr/100, verbose=True)

        return optimizer, scheduler

    def _backward_pass_on_loss(self, loss_function, x, y):
        self.model.train()
        loss = loss_function(self.model, x, y)
        loss.backward()
        return loss
    
    def _optimize(self, loss_function, x, y, optimizer):
        optimizer.zero_grad()
        loss = self._backward_pass_on_loss(loss_function, x, y)
        optimizer.step()
        return loss

    def _train_one_epoch(self, optimizer, scheduler, loss_function):
        tracker = AccuracyTracker()
        loss = None
        for (x, y) in self.train_loader:
            hook_handle = self.model.register_forward_hook(tracker.create_hook(y))
            loss = self._optimize(loss_function, x, y, optimizer)
            hook_handle.remove()
        
        scheduler.step(loss)
        return tracker.get_accuracy()
    
    def test(self):
        total = 0
        correct = 0
        with torch.no_grad():
            for (x, y) in self.test_loader:
                self.model.eval()
                _, predicted = self.model(x).max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
        
        return correct / total
    
    def train_n_epochs(self, lr, loss_function, epochs):
        return self.train_until_target_accuracy(lr, loss_function, epochs, 1.0)

    def train_until_target_accuracy(self, lr, loss_function, max_epochs, target_accuracy):
        optimizer, scheduler = self._get_optimizer_and_scheduler(lr)

        training_accuracy = 0
        for _ in range(max_epochs):
            training_accuracy = self._train_one_epoch(optimizer, scheduler, loss_function)
            self._report("Training accuracy", training_accuracy)
            if training_accuracy >= target_accuracy: break
        
        return training_accuracy