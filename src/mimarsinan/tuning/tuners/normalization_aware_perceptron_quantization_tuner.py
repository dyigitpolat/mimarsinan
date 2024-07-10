from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization

class NormalizationAwarePerceptronQuantizationTuner(PerceptronTuner):
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
    
    # def _mix_params(self, prev_param, new_param, rate):
    #     new_param_ = new_param 
    #     prev_param_ = prev_param

    #     return \
    #         rate * new_param_\
    #         + (1 - rate) * prev_param_

    def _update_and_evaluate(self, rate):
        # iters = 10
        # for _ in range(iters):
        #     self.trainer.train_one_step(self._find_lr())

        # self.adaptation_manager.noise_rate = 1.0
        # for perceptron in self.model.get_perceptrons():
        #     self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        return super().run()
