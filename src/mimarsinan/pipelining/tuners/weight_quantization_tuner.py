from mimarsinan.pipelining.tuners.basic_tuner import BasicTuner
from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.transformations.weight_clipping import SoftTensorClipping
from mimarsinan.transformations.weight_quantization import TensorQuantization

from mimarsinan.models.layers import CQ_Activation

import torch.nn as nn
import torch

class WeightQuantizationTuner(BasicTuner):
    def __init__(self, pipeline, max_epochs, quantization_bits, target_accuracy):
        super().__init__(pipeline, max_epochs, target_accuracy)

        self.target_tq = pipeline.target_tq
        self.quantization_bits = quantization_bits

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        top_p_rate = 0.01
        return SequentialTransform([ 
            SoftTensorClipping(top_p_rate).get_clipped_weights, 
            lambda p: torch.clamp(p, -1, 1),
            TensorQuantization(self.quantization_bits).quantize])

    def _update(self, rate):
        self.model.set_activation(CQ_Activation(self.target_tq))
        self.trainer.weight_transformation = self._mixed_transform(rate)
        self.trainer.train_one_step(self.pipeline_lr / 40)

    def run(self):
        self.model.set_regularization(nn.Identity())
        return super().run()
