from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.tuning.tuners.scale_tuner import ScaleTuner 
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import torch
import torch.nn as nn

class ScaleAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        adaptation_manager = self.get_entry("adaptation_manager")
        model = self.get_entry('model')

        scale = max(self.get_entry('activation_scales'))
        model.in_act = TransformedActivation(
            base_activation = nn.Identity(),
            decorators = [
                ScaleDecorator(torch.tensor(1.0/scale)),
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0/scale)),
                QuantizeDecorator(torch.tensor(self.pipeline.config['target_tq']), torch.tensor(1.0/scale)),
            ])
        
        ActivationFunctionVisualizer(model.in_act, -3, 3, 0.001).plot(f"./in_act.png")

        for idx, perceptron in enumerate(model.get_perceptrons()):
            # adaptation_manager.scale_rate = 1.0
            # print("as", perceptron.activation_scale)
            # print("sf", perceptron.scale_factor)
            # perceptron.set_scale_factor(perceptron.activation_scale)
            # PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda b: b/scale)
            print("sf", perceptron.scale_factor)
            perceptron.set_activation_scale(1.0)
            adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.tuner = ScaleTuner(
            self.pipeline,
            model = model,
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'] * 1e-3,
            adaptation_manager = adaptation_manager,
            activation_scales=self.get_entry('activation_scales'))
        #self.tuner.run()

        #for perceptron in model.get_perceptrons():

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", self.tuner.model, 'torch_model')
        