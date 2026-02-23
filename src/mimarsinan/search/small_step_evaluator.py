from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn

import json
class SmallStepEvaluator:
    def __init__(self, data_provider_factory, loss, lr, device, model_builder):
        self.data_loader_factory = DataLoaderFactory(data_provider_factory, num_workers = 0)

        self.loss = loss
        self.lr = lr
        self.device = device
        self.model_builder = model_builder
        self.steps = 1

        self.cache = {}

    def _configuration_to_json(self, configuration):
        return json.dumps(configuration)
    
    def _evaluate_model(self, model):
        trainer = BasicTrainer(
            model.to(self.device),
            self.device, 
            self.data_loader_factory,
            self.loss)

        trainer.train_n_epochs(self.lr, self.steps)
        return trainer.validate()
    
    def validate(self, configuration):
        try:
            # Prefer a lightweight validate() if the builder provides it; otherwise,
            # fall back to a best-effort build() to catch structural errors (e.g. invalid
            # patch sizes for einops rearrange).
            if hasattr(self.model_builder, "validate"):
                self.model_builder.validate(configuration)
            else:
                _ = self.model_builder.build(configuration)
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
    