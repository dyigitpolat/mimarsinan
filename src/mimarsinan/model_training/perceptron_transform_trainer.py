from mimarsinan.model_training.basic_trainer import BasicTrainer

import copy
import torch
import torch.nn as nn

def _copy_param(param_to, param_from):

    with torch.no_grad():
        if len(param_to.shape) == 0:
            param_to.data = param_from.data
        else:
            param_to.data[:] = param_from.data[:]

class PerceptronTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation):
        
        super().__init__(model, device, data_provider_factory, loss_function)
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        self.perceptron_transformation = perceptron_transformation
    
    def _update_and_transform_model(self):
        for perceptron, aux_perceptron in zip(self.model.get_perceptrons(), self.aux_model.get_perceptrons()):
            # Transfer non-grad params
            for aux_param, param in zip(aux_perceptron.parameters(), perceptron.parameters()):
                if not aux_param.requires_grad:
                    _copy_param(aux_param, param)

            temp = copy.deepcopy(aux_perceptron).to(self.device)

            if not isinstance(perceptron.normalization, nn.Identity):
                temp.normalization.running_mean.data[:] = perceptron.normalization.running_mean.data[:]
                temp.normalization.running_var.data[:] = perceptron.normalization.running_var.data[:]

            self.perceptron_transformation(temp)

            # Handle non-grad params
            for aux_param, temp_param in zip(aux_perceptron.parameters(), temp.parameters()):
                if not aux_param.requires_grad:
                    _copy_param(aux_param, temp_param)

            for param, temp_param in zip(perceptron.parameters(), temp.parameters()):
                _copy_param(param, temp_param)

    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if param.requires_grad:
                aux_param.grad = param.grad

    def _backward_pass_on_loss(self, x, y):
        self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y)

        self._transfer_gradients_to_aux()
        return loss
    
    def _get_optimizer_and_scheduler(self, lr):
        optimizer = torch.optim.AdamW(self.aux_model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', patience=2, factor=0.9, min_lr=lr/100, verbose=True)
        
        return optimizer, scheduler
