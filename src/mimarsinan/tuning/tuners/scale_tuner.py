from mimarsinan.tuning.tuners.basic_tuner import BasicTuner
import torch.nn as nn

class ScaleTuner(BasicTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        # self.base_activations = []

        # for perceptron in model.get_perceptrons():
        #     self.base_activations.append(perceptron.activation)
        self.adaptation_manager = adaptation_manager

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_parameter_transform(self):
        return lambda x: x
    
    def _get_new_parameter_transform(self):
        return lambda x: x

    def _update_and_evaluate(self, rate):
        # for perceptron, activation in zip(self.model.get_perceptrons(), self.base_activations):
        #     perceptron.set_activation(ParametricScaleActivation(activation, 1.0 / perceptron.base_threshold, rate))

        self.adaptation_manager.scale_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(perceptron)

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        super().run()

        # for perceptron, activation in zip(self.model.get_perceptrons(), self.base_activations):
        #     perceptron.set_activation(ScaleActivation(activation, 1.0 / perceptron.base_threshold))

        # self.trainer.train_until_target_accuracy(self._find_lr(), self.epochs, self._get_target())
        return self.trainer.validate()
