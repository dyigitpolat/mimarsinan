from mimarsinan.pipelining.tuners.basic_tuner import BasicTuner
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import CQ_Activation_Parametric

class ActivationQuantizationTuner(BasicTuner):
    def __init__(self, pipeline, max_epochs, target_accuracy):
        super().__init__(pipeline, max_epochs, target_accuracy)

        self.target_tq = pipeline.target_tq

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return noisy_clip_and_decay_param

    def _update(self, rate):
        self.model.set_activation(CQ_Activation_Parametric(self.target_tq, rate))
        self.trainer.weight_transformation = self._mixed_transform(rate)
        self.trainer.train_n_epochs(self._find_lr() / 2, 1)
