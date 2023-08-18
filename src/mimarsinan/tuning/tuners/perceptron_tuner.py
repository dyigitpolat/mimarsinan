from mimarsinan.tuning.tuners.basic_tuner import BasicTuner
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch
import copy

class PerceptronTuner(BasicTuner):
    def __init__(
            self, 
            pipeline,  
            model, 
            target_accuracy, 
            lr):
        super().__init__(pipeline, model, target_accuracy, lr)

        self.device = pipeline.config['device']

        # Trainer
        self.trainer = PerceptronTransformTrainer(
            self.model, 
            pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            pipeline.loss, self._mixed_transform(0.0))
        self.trainer.report_function = self.pipeline.reporter.report

    def _get_previous_perceptron_transform(self):
        raise NotImplementedError() # clip_decay_whatever

    def _get_new_perceptron_transform(self):
        raise NotImplementedError() # noisy_clip_decay_whatever
    
    def _mixed_transform(self, rate):
            print("called")
            return (
                lambda perceptron: self._mixed_perceptron_transform(perceptron, rate))
    
    def _mix_params(self, prev_param, new_param, rate):
        random_mask = torch.rand(prev_param.shape, device=prev_param.device)
        random_mask = (random_mask < rate).float()
        return \
            random_mask * new_param \
            + (1 - random_mask) * prev_param
            
    def _mixed_perceptron_transform(self, perceptron, rate):
        out_perceptron = copy.deepcopy(perceptron).to(self.device)

        prev_perceptron = self._get_previous_perceptron_transform()(perceptron)
        new_perceptron = self._get_new_perceptron_transform()(perceptron)
        
        for param, prev_param, new_param in zip(out_perceptron.parameters(), prev_perceptron.parameters(), new_perceptron.parameters()):
            param.data[:] = self._mix_params(prev_param.data, new_param.data, rate)
        
        return out_perceptron.to(self.device)