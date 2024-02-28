from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn
import torch
import copy

class PerceptronTuner:
    def __init__(
            self, 
            pipeline,  
            model, 
            target_accuracy, 
            lr):
        
        self.pipeline = pipeline

        # Targets
        self.original_target = target_accuracy
        self.target_adjuster = AdaptationTargetAdjuster(self.original_target, self._get_target_decay())

        # Model
        self.model = model
        
        # Epochs
        self.epochs = pipeline.config['tuner_epochs']

        # Adaptation
        self.pipeline_lr = lr
        self.lr = lr

        # Adaptation
        self.name = "Tuning Rate"

        self.device = pipeline.config['device']

        # Trainer
        self.trainer = PerceptronTransformTrainer(
            self.model, 
            pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            pipeline.loss, self._mixed_transform(1.0))
        self.trainer.report_function = self.pipeline.reporter.report

    
    def validate(self):
        return self._get_trainer().validate()

    def run(self):
        def evaluate_model(rate):
            self.trainer.perceptron_transformation = self._mixed_transform(rate)
            return self._update_and_evaluate(rate)

        def clone_state():
            return (
                copy.deepcopy(self._get_trainer().aux_model.state_dict()), 
                copy.deepcopy(self._get_trainer().model.state_dict()))

        def restore_state(state):
            self._get_trainer().aux_model.load_state_dict(state[0])
            self._get_trainer().model.load_state_dict(state[1])

        adapter = SmartSmoothAdaptation (
            self._adaptation,
            clone_state,
            restore_state,
            evaluate_model,
            interpolators=[BasicInterpolation(0.0, 1.0),],
            target_metric=self._get_target()
        )
        adapter.adapt_smoothly()
        
        return self._get_trainer().validate()
    
    def _get_trainer(self):
        return self.trainer
    
    def _get_model(self):
        return self.model

    def _get_target_decay(self):
        raise NotImplementedError() # 0.99

    def _update_and_evaluate(self, rate):
        raise NotImplementedError()
    
    def _get_previous_perceptron_transform(self, rate):
        raise NotImplementedError() # clip_decay_whatever

    def _get_new_perceptron_transform(self, rate):
        raise NotImplementedError() # noisy_clip_decay_whatever
    
    def _get_target(self):
        return self.target_adjuster.get_target()
    
    def _find_lr(self):
        # if (self.lr_once):
        #     return self.lr_once
        
        # def clone_state():
        #     return (
        #         copy.deepcopy(self._get_trainer().aux_model.state_dict()), 
        #         copy.deepcopy(self._get_trainer().model.state_dict()))

        # def restore_state(state):
        #     self._get_trainer().aux_model.load_state_dict(state[0])
        #     self._get_trainer().model.load_state_dict(state[1])

        # self.lr_once = LearningRateExplorer(
        #     clone_state,
        #     restore_state,
        #     self._get_trainer(),
        #     self._get_model(), 
        #     self.pipeline_lr, 
        #     self.pipeline_lr / 10000, 
        #     -0.01).find_lr_for_tuning()
        
        return self.pipeline_lr / 200

    def _adaptation(self, rate):
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        
        self._update_and_evaluate(rate)

        lr = self._find_lr()
        self._get_trainer().train_until_target_accuracy(
            lr, self.epochs, self._get_target())
        
        acc = self._get_trainer().validate()
        self.target_adjuster.update_target(acc)
    
    def _mixed_transform(self, rate):
        print("called")
        return (
            lambda perceptron: self._mixed_perceptron_transform(perceptron, rate))
    
    def _mix_params(self, prev_param, new_param, rate):
        new_param_ = new_param 
        prev_param_ = prev_param

        random_mask = torch.rand(prev_param.shape, device=prev_param.device)
        random_mask = (random_mask < rate).float()
        return \
            random_mask * new_param_\
            + (1 - random_mask) * prev_param_
            
    def _mixed_perceptron_transform(self, perceptron, rate):
        temp_prev_perceptron = copy.deepcopy(perceptron).to(self.device)

        self._get_previous_perceptron_transform(rate)(temp_prev_perceptron)
        self._get_new_perceptron_transform(rate)(perceptron)

        for param, prev_param in zip(perceptron.parameters(), temp_prev_perceptron.parameters()):
            if len(param.shape) == 0:
                param.data = self._mix_params(prev_param.data, param.data, rate)
            else:
                param.data[:] = self._mix_params(prev_param.data[:], param.data[:], rate)