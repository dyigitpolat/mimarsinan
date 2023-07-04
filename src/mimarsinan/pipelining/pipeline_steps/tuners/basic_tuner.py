from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

import torch

class BasicTuner:
    def __init__(
            self, 
            pipeline, 
            max_epochs, 
            model, 
            target_accuracy, 
            lr):
        
        self.pipeline = pipeline

        # Targets
        self.target_accuracy = target_accuracy * self._get_target_decay()

        # Model
        self.model = model
        
        # Epochs
        self.epochs = max_epochs

        # Adaptation
        self._prev_acc = target_accuracy
        self.pipeline_lr = lr
        self.lr = lr

        # Adaptation
        self.name = "Tuning Rate"

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.config['device'], 
            pipeline.data_provider, 
            pipeline.loss, self._mixed_transform(0.001))
        self.trainer.report_function = self.pipeline.reporter.report

    def _get_target_decay(self):
        raise NotImplementedError() # 0.99

    def _get_previous_parameter_transform(self):
        raise NotImplementedError() # clip_decay_whatever

    def _get_new_parameter_transform(self):
        raise NotImplementedError() # noisy_clip_decay_whatever

    def _update(self, rate):
        raise NotImplementedError()

    
    def _find_lr(self):
        return LearningRateExplorer(
            self.trainer, 
            self.model, 
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

        self._update(rate)

        lr = self._find_lr()
        acc = self.trainer.train_until_target_accuracy(
            lr, self.epochs, self._prev_acc)
        
        acc = self.trainer.train_n_epochs(lr / 2, 2)
        
        self._prev_acc = max(self._prev_acc * self._get_target_decay(), acc)

    def run(self):
        def evaluate_model(rate):
            self._update(rate)
            return self.trainer.validate()

        def clone_state():
            return self.model.state_dict()

        def restore_state(state):
            self.model.load_state_dict(state)

        adapter = SmartSmoothAdaptation (
            self._adaptation,
            clone_state,
            restore_state,
            evaluate_model
        )

        adapter.adapt_smoothly(interpolators=[BasicInterpolation(0.0, 1.0)])
        
        return self.trainer.validate()