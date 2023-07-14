from mimarsinan.tuning.tuners.basic_tuner import BasicTuner

from mimarsinan.transformations.parameter_transforms.sequential_transform import SequentialTransform
from mimarsinan.transformations.weight_clipping import SoftTensorClipping

from mimarsinan.models.layers import NoisyDropout

import torch

class NoiseTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.target_noise_amount = 2.0 / (pipeline.config['weight_bits'] ** 2)
    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x

    def _update_and_evaluate(self, rate):
        self.model.set_regularization(NoisyDropout(0.0, rate, self.target_noise_amount))
        self.trainer.train_one_step(self.lr / 2)
        return self.trainer.validate_train()

    def run(self):
        super().run()
        
        self.model.set_regularization(NoisyDropout(0.0, 1.0, self.target_noise_amount))
        self.trainer.train_until_target_accuracy(self._find_lr() / 2, self.epochs, self._prev_acc)

        return self.trainer.validate()
