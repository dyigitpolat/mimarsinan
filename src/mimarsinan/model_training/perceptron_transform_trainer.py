from mimarsinan.model_training.basic_trainer import BasicTrainer

import copy
import torch
import torch.nn as nn

class PerceptronTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation):
        
        super().__init__(model, device, data_provider_factory, loss_function)
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        self.perceptron_transformation = perceptron_transformation
    
    def _update_and_transform_model(self):
        for perceptron, aux_perceptron in zip(self.model.get_perceptrons(), self.aux_model.get_perceptrons()):
            transformed_perceptron = self.perceptron_transformation(aux_perceptron).to(self.device)
            perceptron.layer.weight.data = transformed_perceptron.layer.weight.data
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.data = transformed_perceptron.layer.bias.data

            if not isinstance(perceptron.normalization, nn.Identity):
                perceptron.normalization.weight.data = transformed_perceptron.normalization.weight.data
                perceptron.normalization.bias.data = transformed_perceptron.normalization.bias.data

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
        optimizer = torch.optim.Adam(self.aux_model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', patience=5, factor=0.9, min_lr=lr/100, verbose=True)
        
        return optimizer, scheduler

