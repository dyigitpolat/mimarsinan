from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, DecoratedActivation, RandomMaskAdjustmentStrategy, RateAdjustedDecorator, ClampDecorator, QuantizeDecorator

import torch

class ActivationQuantizationTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_tq, 
                 target_accuracy, 
                 lr,
                 adaptation_manager):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.target_tq = target_tq
        self.adaptation_manager = adaptation_manager
        self.base_input_activation = model.input_activation

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_perceptron_transform(self, rate):
        return lambda p: None
    
    def _get_new_perceptron_transform(self, rate):
        return lambda p: None

    def _update_and_evaluate(self, rate):
        self.model.input_activation = TransformedActivation(
            base_activation = self.base_input_activation,
            decorators = [
                ClampDecorator(torch.tensor(0.0), torch.tensor(1.0)),
                RateAdjustedDecorator(
                    rate,
                    QuantizeDecorator(torch.tensor(self.pipeline.config["target_tq"]), torch.tensor(1.0)),
                    RandomMaskAdjustmentStrategy()
                )
            ])
        
        self.adaptation_manager.quantization_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
        
        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()

        return self.trainer.validate()
