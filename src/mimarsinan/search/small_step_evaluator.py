from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.layers import ClampedReLU, NoisyDropout

import torch.nn as nn
class SmallStepEvaluator:
    def __init__(self, data_provider, loss, lr, device):
        self.data_provider = data_provider
        self.loss = loss
        self.lr = lr
        self.device = device
        self.trainer = BasicTrainer(
            nn.Identity(), 
            self.device, self.data_provider,
            self.loss)
        self.input_data, _ = next(iter(data_provider.get_training_loader(
            data_provider.get_training_set_size(),
        )))
        self.steps = 5

    def evaluate(self, model):
        model.set_activation(nn.LeakyReLU())
        
        self.trainer.model = model.to(self.device)

        self.trainer.train_n_epochs(self.lr, 1)
        return self.trainer.validate()
        
