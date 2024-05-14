from mimarsinan.model_training.basic_trainer import BasicTrainer

import torch
import copy

class WeightTransformTrainer(BasicTrainer):
    def __init__(
            self, model, device, data_provider_factory, loss_function, weight_transformation):
        super().__init__(model, device, data_provider_factory, loss_function)
        self.weight_transformation = weight_transformation
        self.aux_model = copy.deepcopy(self.model).to(self.device)

    def _update_and_transform_model(self):
        for (name, param), (_, aux_param) in zip(self.model.named_parameters(), self.aux_model.named_parameters()):
            
            if ('weight' in name) or ('bias' in name):
                param.data[:] = self.weight_transformation(aux_param.data[:]).to(self.device)

    def _transfer_gradients_to_aux(self):
        for param, aux_param in zip(self.model.parameters(), self.aux_model.parameters()):
            if param.requires_grad:
                aux_param.grad = param.grad
    
    def _backward_pass_on_loss(self, x, y, scaler):
        self._update_and_transform_model()
        self.aux_model = self.aux_model.to(self.device)
        self.aux_model.train()

        loss = super()._backward_pass_on_loss(x, y, scaler)

        self._transfer_gradients_to_aux()
        return loss
    
    def _get_optimizer_and_scheduler(self, lr, epochs):
        optimizer = torch.optim.AdamW(
            self.aux_model.parameters(), lr = lr, betas = (self.beta1, self.beta2))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = epochs, eta_min = lr * 1e-3)
        
        return optimizer, scheduler, torch.cuda.amp.GradScaler()