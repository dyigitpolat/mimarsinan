from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_utilities import AccuracyTracker

import torch
import copy

class WeightTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, train_loader, test_loader, weight_transformation):
        super().__init__(model, device, train_loader, test_loader)
        self.weight_transformation = weight_transformation
        self.aux_model = None

    def _update_and_transform_model(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            param.data[:] = self.weight_transformation(aux_param.data[:]).to(self.device)

    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if param.requires_grad:
                aux_param.grad = param.grad
    
    def _backward_pass_on_loss(self, loss_function, x, y):
        self._update_and_transform_model()
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(loss_function, x, y)

        self._transfer_gradients_to_aux()
        return loss
    
    def _get_optimizer_and_scheduler(self, lr):
        optimizer = torch.optim.Adam(self.aux_model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', patience=5, factor=0.9, min_lr=lr/100, verbose=True)
        
        return optimizer, scheduler
    
    def train_until_target_accuracy(self, lr, loss_function, max_epochs, target_accuracy):
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        accuracy = super().train_until_target_accuracy(lr, loss_function, max_epochs, target_accuracy)
        return accuracy

