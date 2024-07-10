from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.core_flow_tuner import CoreFlowTuner
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

import torch.nn as nn
import torch

from math import ceil

class CoreFlowTuningStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["soft_core_mapping", "model"]
        promises = ["tuned_soft_core_mapping", "scaled_simulation_length"]
        updates = []
        clears = ["soft_core_mapping"]
        super().__init__(requires, promises, updates, clears, pipeline)
        
        self.tuner = None
        self.preprocessor = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        model = self.get_entry('model')
        # scale = model.get_perceptrons()[0].scale_factor
        # scale = max(self.get_entry('activation_scales'))
        # print(model.get_perceptrons()[0].scale_factor)
        # print(max(self.get_entry('activation_scales')))
        
        self.preprocessor = nn.Sequential(
            model.get_preprocessor(),
            model.in_act)
        
        self.tuner = CoreFlowTuner(
            self.pipeline, self.get_entry('soft_core_mapping'), self.preprocessor)
        scaled_simulation_length = self.tuner.run()

        self.add_entry("scaled_simulation_length", scaled_simulation_length)
        self.add_entry("tuned_soft_core_mapping", self.tuner.mapping, 'pickle')