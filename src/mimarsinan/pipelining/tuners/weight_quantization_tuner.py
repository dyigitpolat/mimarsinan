from mimarsinan.pipelining.tuners.basic_tuner import BasicTuner
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import CQ_Activation

class WeightQuantizationTuner(BasicTuner):
    def __init__(self, pipeline, max_epochs, quantization_bits, target_accuracy):
        super().__init__(pipeline, max_epochs, target_accuracy)

        self.target_tq = pipeline.target_tq

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return noisy_clip_decay_and_quantize_param

    def _update(self, rate):
        self.model.set_activation(CQ_Activation(self.target_tq))
        self.trainer.weight_transformation = self._mixed_transform(rate)
        self.trainer.train_one_step(self.pipeline_lr / 40)
