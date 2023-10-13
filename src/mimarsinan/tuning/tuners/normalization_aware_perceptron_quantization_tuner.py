from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization

import copy

class NormalizationAwarePerceptronQuantizationTuner(PerceptronTuner):
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
        self.rate = 0.0

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_perceptron_transform(self):
        return lambda perceptron: copy.deepcopy(perceptron)
    
    def _get_new_perceptron_transform(self):
        return NormalizationAwarePerceptronQuantization(
            self.quantization_bits, self.pipeline.config['device'], self.rate).transform

    def _update_and_evaluate(self, rate):
        self.rate = rate
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        return super().run()
