from mimarsinan.pipelining.pipeline_steps.tuners.basic_tuner import BasicTuner

from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.transformations.weight_clipping import SoftTensorClipping

from mimarsinan.models.layers import CQ_Activation_Parametric, CQ_Activation

import torch

class ActivationQuantizationTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 max_epochs, 
                 model, 
                 target_tq, 
                 target_accuracy, 
                 lr):
        
        super().__init__(
            pipeline, 
            max_epochs, 
            model, 
            target_accuracy, 
            lr)

        self.target_tq = target_tq
        self.base_activation = model.activation

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        top_p_rate = 0.01
        return SequentialTransform([ 
            SoftTensorClipping(top_p_rate).get_clipped_weights, 
            lambda p: torch.clamp(p, -1, 1) ])

    def _update(self, rate):
        self.model.set_activation(CQ_Activation_Parametric(self.target_tq, rate, self.base_activation))
        self.trainer.weight_transformation = self._mixed_transform(rate)
        self.trainer.train_n_epochs(self._find_lr() / 2, 1)

    def run(self):
        super().run()
        
        self.model.set_activation(CQ_Activation(self.target_tq))
        self.trainer.weight_transformation = self._get_new_parameter_transform()
        self.trainer.train_until_target_accuracy(self._find_lr() / 2, self.epochs, self._prev_acc)

        return self.trainer.validate()