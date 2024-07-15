from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner
from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization

class NormalizationAwarePerceptronQuantizationTuner(PerceptronTransformTuner):
    def __init__(self, 
                 pipeline, 
                 model,
                 quantization_bits, 
                 target_accuracy,
                 lr,
                 adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)


        self.quantization_bits = quantization_bits
        self.adaptation_manager = adaptation_manager

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_perceptron_transform(self, rate):
        return lambda perceptron: None
    
    def _get_new_perceptron_transform(self, rate):
        def transform(perceptron):
            NormalizationAwarePerceptronQuantization(
                self.quantization_bits, 
                self.pipeline.config['device'], 
                rate
                ).transform(perceptron)
        
        return transform

    def _update_and_evaluate(self, rate):
        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        return super().run()
