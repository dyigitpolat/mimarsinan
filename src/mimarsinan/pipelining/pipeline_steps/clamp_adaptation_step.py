from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner 

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.models.layers import SavedTensorDecorator

import torch
class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()
    
    def _calculate_activation_scales(self, adaptation_manager):

        for perceptron in self.get_entry("model").get_perceptrons():
            adaptation_manager.update_activation(self.pipeline.config, perceptron)
            perceptron.activation.decorate(SavedTensorDecorator())

        # BasicTrainer(
        #     self.get_entry("model"), 
        #     self.pipeline.config['device'], 
        #     DataLoaderFactory(self.pipeline.data_provider_factory),
        #     self.pipeline.loss).validate()

        for perceptron in self.get_entry("model").get_perceptrons():
            stats = perceptron.activation.pop_decorator()
            # activation_hist = stats.in_hist.clone().to(self.pipeline.config['device'])
            # activation_hist *= stats.in_hist_bin_edges[1:].to(self.pipeline.config['device'])
            # activation_hist[activation_hist < 0] = 0
            # hist_sum = activation_hist.sum()
            # cumulative_hist = activation_hist.cumsum(0)
            # cumulative_hist /= hist_sum

            # rate = 0.8
            
            # # find the index of the bin which first exceeds the rate
            # index = (cumulative_hist > rate).flatten().nonzero()[0]
            # perceptron.activation_scale = stats.in_hist_bin_edges[index].item()

            perceptron.set_activation_scale(1.0)

    def process(self):
        adaptation_manager = self.get_entry("adaptation_manager")

        self._calculate_activation_scales(adaptation_manager)
        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'],
            adaptation_manager = adaptation_manager)
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", self.tuner.model, 'torch_model')
        