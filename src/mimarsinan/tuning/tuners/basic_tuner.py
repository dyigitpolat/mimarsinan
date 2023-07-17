from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

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
        self.target_accuracy = target_accuracy * self._get_target_decay()
        self.original_target_accuracy = target_accuracy

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
            pipeline.data_provider, 
            pipeline.loss, self._mixed_transform(0.001))
        self.trainer.report_function = self.pipeline.reporter.report

    def _get_target_decay(self):
        raise NotImplementedError() # 0.99

    def _get_previous_parameter_transform(self):
        raise NotImplementedError() # clip_decay_whatever

    def _get_new_parameter_transform(self):
        raise NotImplementedError() # noisy_clip_decay_whatever

    def _update_and_evaluate(self, rate):
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
    
    def _update_target_accuracy(self, current_accuracy):
        decayed_target_metric = self.target_accuracy * self._get_target_decay()
        promoted_current_metric = current_accuracy

        if current_accuracy > self.target_accuracy:
            promoted_current_metric = max(
                current_accuracy,
                0.1 * self.original_target_accuracy + 0.9 * current_accuracy
            )

        self.target_accuracy = max(
            decayed_target_metric, 
            promoted_current_metric)
        
        print("Target accuracy: ", self.target_accuracy)

    def _adaptation(self, rate):
        self.pipeline.reporter.report(self.name, rate)

        self._update_and_evaluate(rate)

        lr = self._find_lr()
        self.trainer.train_until_target_accuracy(
            lr, self.epochs, self.target_accuracy)
        
        self.trainer.train_n_epochs(lr / 2, 2)
        
        acc = self.trainer.validate_train()
        self._update_target_accuracy(acc)

    def run(self):
        def evaluate_model(rate):
            return self._update_and_evaluate(rate)

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

        self.trainer._update_and_transform_model()
        
        return self.trainer.validate()