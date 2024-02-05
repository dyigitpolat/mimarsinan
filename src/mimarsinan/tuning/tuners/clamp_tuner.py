from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner

from mimarsinan.models.layers import SavedTensorDecorator

import torch

class ClampTuner(PerceptronTuner):
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
        self.adaptation_manager = adaptation_manager

        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

    def _get_target_decay(self):
        return 0.999
    
    def _get_previous_perceptron_transform(self, rate):
        return lambda p: None
    
    def _get_new_perceptron_transform(self, rate):
        return lambda p: None
    
    def _calculate_activation_scales(self, adaptation_manager, validator, rate):
        for perceptron in self.model.get_perceptrons():
            adaptation_manager.update_activation(self.pipeline.config, perceptron)
            perceptron.activation.decorate(SavedTensorDecorator())

        validator.validate()

        max_target_scale = 0.0
        for perceptron in self.model.get_perceptrons():
            saved_tensor_dec = perceptron.activation.pop_decorator()
            in_min = saved_tensor_dec.latest_input.min()
            in_max = saved_tensor_dec.latest_input.max()
            x = saved_tensor_dec.latest_input
            
            bins = 1000
            activation_hist = torch.histc(x.flatten(), bins=bins, min=in_min.item(), max=in_max.item())
            bin_edges = torch.linspace(in_min.item(), in_max.item(), steps=bins+1).to(self.pipeline.config['device'])

            activation_hist *= bin_edges[1:].to(self.pipeline.config['device'])
            activation_hist[activation_hist < 0] = 0
            hist_sum = activation_hist.sum()
            cumulative_hist = activation_hist.cumsum(0)
            cumulative_hist /= hist_sum

            clip_rate = 0.999
            
            # # find the index of the bin which first exceeds the rate
            index = (cumulative_hist > clip_rate).flatten().nonzero()[0].to(self.pipeline.config['device'])
            clipped_act_scale = bin_edges[index].item()

            target_act_scale = (in_max * (1.0 - rate) + rate * clipped_act_scale) 

            perceptron.set_activation_scale(target_act_scale)
            max_target_scale = max(max_target_scale, target_act_scale)

    def _update_and_evaluate(self, rate):
        self._calculate_activation_scales(self.adaptation_manager, self.trainer, rate)

        self.adaptation_manager.clamp_rate = rate
        for perceptron in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

        self.trainer.train_one_step(self._find_lr())
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()
        return self.trainer.validate()
