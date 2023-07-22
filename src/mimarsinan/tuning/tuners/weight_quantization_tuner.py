from mimarsinan.tuning.tuners.basic_tuner import BasicTuner
from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.weight_clipping import SoftTensorClipping

from mimarsinan.models.layers import CQ_Activation

import torch.nn as nn
import torch

class WeightQuantizationTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model,
                 quantization_bits, 
                 target_tq,
                 target_accuracy,
                 lr):
        
        super().__init__(pipeline, model, target_accuracy, lr)

        self.target_tq = target_tq
        self.quantization_bits = quantization_bits

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return SequentialTransform([ 
            lambda p: torch.clamp(p, -1, 1),
            SoftTensorClipping(0.1).get_clipped_weights,
            TensorQuantization(self.quantization_bits).quantize])

    def _update_and_evaluate(self, rate):
        for perceptron in self.model.get_perceptrons():
            perceptron.set_activation(
                CQ_Activation(self.target_tq, perceptron.base_threshold))
            
        self.trainer.train_one_step(self.pipeline_lr / 40)
        return self.trainer.validate()

    def run(self):
        return super().run()
