from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.tolerance_calibration import (
    initial_tolerance_fn_for_pipeline_if_enabled,
)
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.model_training.basic_trainer import BasicTrainer

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
        self.trainer = BasicTrainer(
            self.model, 
            pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

    
    def validate(self):
        return self._get_trainer().validate()

    def run(self):
        def evaluate_model(rate):
            return self._update_and_evaluate(rate)

        def clone_state():
            print("cloning state")
            print(self._get_trainer().model.get_perceptrons()[0].layer.weight.data.mean())
            return copy.deepcopy(self._get_trainer().model.state_dict())

        def restore_state(state):
            print("restoring state")
            print(self._get_trainer().model.get_perceptrons()[0].layer.weight.data.mean())
            self._get_trainer().model.load_state_dict(state)

        initial_tol_fn = initial_tolerance_fn_for_pipeline_if_enabled(
            self.pipeline.config,
            clone_state=clone_state,
            restore_state=restore_state,
            evaluate_at_rate=evaluate_model,
            validate_fn=self.validate,
            train_validation_epochs=lambda lr, n, w: self._get_trainer().train_validation_epochs(
                lr, n, w
            ),
            lr_probe=self.pipeline_lr,
        )

        adapter = SmartSmoothAdaptation (
            self._adaptation,
            clone_state,
            restore_state,
            evaluate_model,
            interpolators=[BasicInterpolation(0.0, 1.0),],
            target_metric=self._get_target(),
            initial_tolerance_fn=initial_tol_fn,
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
    
    def _get_target(self):
        return self.target_adjuster.get_target()
    
    def _find_lr(self):
        return self.pipeline_lr / 200

    def _adaptation(self, rate):
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())
        
        self._update_and_evaluate(rate)
        print(self._get_trainer().model.get_perceptrons()[0].layer.weight.data.mean())
        self._get_trainer().train_until_target_accuracy(
            self.pipeline_lr, self.epochs, self._get_target(), 0)
        print(self._get_trainer().model.get_perceptrons()[0].layer.weight.data.mean())
        
        acc = self._get_trainer().validate()
        self.target_adjuster.update_target(acc)