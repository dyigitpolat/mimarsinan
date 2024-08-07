from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.layers import LeakyGradReLU

import torch.nn as nn

class ModelBuildingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model_config", "model_builder"]
        promises = ["model", "adaptation_manager"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()
    
    def set_activation(self, model_config, perceptron, adaptation_manager):
        if model_config['base_activation'] == "ReLU":
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()
        elif model_config['base_activation'] == "LeakyReLU":
            perceptron.base_activation = nn.LeakyReLU()
            perceptron.activation = nn.LeakyReLU()
        elif model_config['base_activation'] == "GELU":
            perceptron.base_activation = nn.GELU()
            perceptron.activation = nn.GELU()
        else:
            print("No (or Invalid) base activation provided, using LeakyGradReLU")
            perceptron.base_activation = LeakyGradReLU()
            perceptron.activation = LeakyGradReLU()
        
        adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def process(self):
        builder = self.get_entry('model_builder')
        init_model = builder.build(self.get_entry("model_config"))

        adaptation_manager = AdaptationManager()
        for perceptron in init_model.get_perceptrons():
            self.set_activation(self.get_entry("model_config"), perceptron, adaptation_manager)

        self.add_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.add_entry("model", (init_model), "torch_model")