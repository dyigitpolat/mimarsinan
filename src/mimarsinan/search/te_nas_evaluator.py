from mimarsinan.model_evaluation.te_nas_utils import get_ntk
from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn
class TE_NAS_Evaluator:
    def __init__(self, data_provider_factory, loss, lr, device):
        self.data_loader_factory = DataLoaderFactory(data_provider_factory),
        data_provider = self.data_loader_factory.create_data_provider()

        self.loss = loss
        self.lr = lr
        self.device = device
        self.trainer = BasicTrainer(
            nn.Identity(), 
            self.device, self.data_provider_factory,
            self.loss)
        
        self.input_data, _ = \
            next(iter(self.data_loader_factory.create_validation_loader(
                    data_provider.get_validation_set_size(), data_provider)))

    def evaluate(self, model):
        self.trainer.model = model.to(self.device)
        self.trainer.train_one_step(self.lr)
        
        ntk_score = get_ntk(
            self.input_data.to(self.device),
            model,
            self.device)[0]

        return 1.0 / ntk_score
