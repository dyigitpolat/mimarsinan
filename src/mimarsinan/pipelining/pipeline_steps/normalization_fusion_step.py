from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer



import torch.nn as nn

class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        for perceptron in model.get_perceptrons():
            perceptron.to(self.pipeline.config['device'])
            w = PerceptronTransformer().get_effective_weight(perceptron)
            b = PerceptronTransformer().get_effective_bias(perceptron)

            perceptron.layer = nn.Linear(
                perceptron.input_features, 
                perceptron.output_channels, bias=True)
            
            perceptron.layer.weight.data = w 
            perceptron.layer.bias.data = b 

            perceptron.normalization = nn.Identity()

        print(self.validate())

        self.update_entry("model", model, 'torch_model')