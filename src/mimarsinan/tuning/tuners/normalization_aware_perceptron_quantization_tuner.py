from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.normalization_aware_perceptron_quantization import NormalizationAwarePerceptronQuantization

import torch.nn as nn
class NormalizationAwarePerceptronQuantizationTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model,
                 quantization_bits, 
                 target_tq,
                 target_accuracy,
                 lr):
        
        for perceptron in model.get_perceptrons():
            if not isinstance(perceptron.normalization, nn.Identity):
                for param in perceptron.normalization.parameters():
                    param.requires_grad = False

        super().__init__(pipeline, model, target_accuracy, lr)

        self.target_tq = target_tq
        self.quantization_bits = quantization_bits
        

        from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer
        for idx, perceptron in enumerate(self.model.get_perceptrons()):
            ActivationFunctionVisualizer(perceptron.activation, -3, 3, 0.001).plot(f"./generated/_wqt{idx}.png")

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_perceptron_transform(self):
        return lambda x: x
    
    def _get_new_perceptron_transform(self):
        return NormalizationAwarePerceptronQuantization(
            self.quantization_bits, self.pipeline.config['device']).transform

    def _update_and_evaluate(self, rate):
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        #self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        return super().run()
