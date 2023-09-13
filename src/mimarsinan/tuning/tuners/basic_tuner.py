from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch

class BasicTuner:
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

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            pipeline.loss, self._mixed_transform(0.0))
        self.trainer.report_function = self.pipeline.reporter.report
    
    def _get_trainer(self):
        return self.trainer
    
    def _get_model(self):
        return self.model

    def _get_target_decay(self):
        raise NotImplementedError() # 0.99

    def _get_previous_parameter_transform(self):
        raise NotImplementedError() # clip_decay_whatever

    def _get_new_parameter_transform(self):
        raise NotImplementedError() # noisy_clip_decay_whatever

    def _update_and_evaluate(self, rate):
        raise NotImplementedError()
    
    def _get_target(self):
        return self.target_adjuster.get_target()
    
    def _find_lr(self):
        return LearningRateExplorer(
            self._get_trainer(), 
            self._get_model(), 
            self.pipeline_lr / 20, 
            self.pipeline_lr / 1000, 
            0.01).find_lr_for_tuning()
    
    def _mixed_transform(self, rate):
            def transform(param):
                random_mask = torch.rand(param.shape, device=param.device)
                random_mask = (random_mask < rate).float()
                return \
                    random_mask * self._get_new_parameter_transform()(param) \
                    + (1 - random_mask) * self._get_previous_parameter_transform()(param)
            return transform

    def _adaptation(self, rate):
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())
        
        self._update_and_evaluate(rate)

        lr = self._find_lr()
        self._get_trainer().train_until_target_accuracy(
            lr, self.epochs, self._get_target())
        
        acc = self._get_trainer().validate()
        self.target_adjuster.update_target(acc)

    def validate(self):
        return self._get_trainer().validate()

    def run(self):
        def evaluate_model(rate):
            return self._update_and_evaluate(rate)

        def clone_state():
            return self._get_model().state_dict()

        def restore_state(state):
            self._get_model().load_state_dict(state)

        adapter = SmartSmoothAdaptation (
            self._adaptation,
            clone_state,
            restore_state,
            evaluate_model,
            interpolators=[BasicInterpolation(0.0, 1.0)]
        )
        adapter.adapt_smoothly()
        
        return self._get_trainer().validate()