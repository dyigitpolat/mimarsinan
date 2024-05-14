from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

from mimarsinan.models.layers import SavedTensorDecorator

import torch

class ScaleTuner(PerceptronTuner):
    def __init__(self, 
                 pipeline, 
                 model, 
                 target_accuracy, 
                 lr,
                 adaptation_manager,
                 activation_scales):
        
        super().__init__(
            pipeline, 
            model, 
            target_accuracy, 
            lr)

        self.lr = lr
        self.adaptation_manager = adaptation_manager
        self.activation_scales = activation_scales

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _get_target_decay(self):
        return 0.99
    
    def _get_previous_perceptron_transform(self, _):
        return lambda p: None
    
    def _get_new_perceptron_transform(self, _):
        return lambda p: None

    def _update_activation_scales(self):

        for perceptron in self.model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        self.trainer.validate()

        for perceptron in self.model.get_perceptrons():
            saved_tensor = perceptron.activation.pop_decorator()
            flat_acts = saved_tensor.latest_output.view(-1)  # flatten to 1D
            sorted_acts, _ = torch.sort(flat_acts)  # sort ascending
            cumsum_acts = torch.cumsum(sorted_acts, dim=0)  # cumulative sum
            norm_cumsum = cumsum_acts / cumsum_acts[-1]  # normalize by total sum
            threshold_idx = torch.searchsorted(norm_cumsum, 0.99)  # index of first value >= 0.99

            prev_act = perceptron.activation_scale
            perceptron.set_activation_scale(prev_act * sorted_acts[threshold_idx].item())


    def _update_and_evaluate(self, rate):
        #self._update_activation_scales()

        self.adaptation_manager.scale_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()
        return self.trainer.validate()
