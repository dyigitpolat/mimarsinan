from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.utils import *

import torch.nn as nn

import json
class SmallStepEvaluator:
    def __init__(self, data_provider, loss, lr, device, model_builder):
        self.data_provider = get_multiprocess_friendly_data_provider(data_provider)
        self.loss = loss
        self.lr = lr
        self.device = device
        self.model_builder = model_builder
        self.steps = 1

        self.cache = {}

    def _configuration_to_json(self, configuration):
        return json.dumps(configuration)
    
    def _evaluate_model(self, model):
        for perceptron in model.get_perceptrons():
            perceptron.set_activation(nn.LeakyReLU())
    
        trainer = BasicTrainer(
            model.to(self.device),
            self.device, self.data_provider,
            self.loss)

        trainer.train_n_epochs(self.lr, self.steps)
        return trainer.validate()
    
    def validate(self, configuration):
        try:
            self.model_builder.validate(configuration)
            return True
        except Exception as e:
            return False

    def evaluate(self, configuration):
        key = self._configuration_to_json(configuration)
        if key in self.cache:
            return self.cache[key]
        
        model = self.model_builder.build(configuration)
        self.cache[key] = self._evaluate_model(model)

        return self.cache[key]
    