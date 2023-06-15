from mimarsinan.model_evaluation.te_nas_utils import get_ntk
from mimarsinan.model_training.basic_trainer import BasicTrainer

import torch.nn as nn
class TE_NAS_Evaluator:
    def __init__(self, data_provider, loss, lr, device):
        self.data_provider = data_provider
        self.loss = loss
        self.lr = lr
        self.device = device
        self.trainer = BasicTrainer(
            nn.Identity(), 
            self.device, self.data_provider,
            self.loss)
        self.input_data, _ = next(iter(data_provider.get_validation_loader(
            data_provider.get_validation_set_size(),
        )))

    def evaluate(self, model):
        self.trainer.model = model.to(self.device)
        self.trainer.train_one_step(self.lr)
        
        ntk_score = get_ntk(
            self.input_data.to(self.device),
            model,
            self.device)[0]
        
        return 1.0 / ntk_score
