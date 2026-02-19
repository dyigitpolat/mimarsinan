from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

import copy
import torch
import torch.nn as nn
import functools

def _copy_param(param_to, param_from):
    with torch.no_grad():
        param_to.data.copy_(param_from.data.clone())

class PerceptronTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, perceptron_transformation):
        
        super().__init__(model, device, data_provider_factory, loss_function)
        self.aux_model = copy.deepcopy(self.model).to(self.device)
        self.perceptron_transformation = perceptron_transformation

    def _get_module_by_name(self, model, access_string):
        names = access_string.split('.')
        return functools.reduce(getattr, names, model)
    
    def _update_and_transform_model(self):
        perceptron_ids = []
        for name, module in self.model.named_modules():
            if isinstance(module, Perceptron):
                perceptron_ids.append(name)

        aux_model_state_dict = copy.deepcopy(self.aux_model.state_dict())
        model_state_dict = copy.deepcopy(self.model.state_dict())

        for perceptron_id in perceptron_ids:
            perceptron = self._get_module_by_name(self.model, perceptron_id)
            aux_perceptron = self._get_module_by_name(self.aux_model, perceptron_id)

            perceptron_param_ids = []
            for name, _ in perceptron.named_parameters():
                perceptron_param_ids.append(name)

            temp = copy.deepcopy(aux_perceptron).to(self.device)
            #temp.load_state_dict(copy.deepcopy(aux_perceptron.state_dict()))

            self.perceptron_transformation(temp)

            for perceptron_param_id in perceptron_param_ids:
                model_state_dict[perceptron_id + '.' + perceptron_param_id].data.copy_(
                    temp.state_dict()[perceptron_param_id].data.clone().detach())
                
        self.model.load_state_dict(copy.deepcopy(model_state_dict))
        self.model = self.model.to(self.device)


    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if aux_param.requires_grad and param.grad is not None:
                if aux_param.grad is None:
                    aux_param.grad = param.grad.clone()
                
                aux_param.grad.copy_(param.grad.clone())

    def _backward_pass_on_loss(self, x, y, scaler):
        with torch.no_grad():
            self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y, scaler)

        with torch.no_grad():
            self._transfer_gradients_to_aux()
        return loss
    
    def _get_optimizer_and_scheduler(self, lr, epochs):
        optimizer = torch.optim.Adam(
            self.aux_model.parameters(), lr = lr, weight_decay=0, betas=(self.beta1, self.beta2))
        
        identity_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda = lambda epoch: 1)
        scheduler = identity_scheduler
        
        return optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False)
