from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.model_training.basic_trainer import BasicTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator

import torch

class ActivationAnalysisStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["activation_scales"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        self.trainer.validate()

        activation_scales = []
        for perceptron in model.get_perceptrons():
            saved_tensor = perceptron.activation.pop_decorator()

            flat_acts = saved_tensor.latest_output.view(-1)  # flatten to 1D
            sorted_acts, _ = torch.sort(flat_acts)  # sort ascending
            cumsum_acts = torch.cumsum(sorted_acts, dim=0)  # cumulative sum
            norm_cumsum = cumsum_acts / cumsum_acts[-1]  # normalize by total sum
            threshold_idx = torch.searchsorted(norm_cumsum, 0.99)  # index of first value >= 0.99

            activation_scales.append(sorted_acts[threshold_idx].item())

        print(activation_scales)

        self.add_entry("activation_scales", activation_scales)


        