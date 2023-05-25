from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

class WeightQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy * 0.99

        # Model
        self.model = pipeline.model

        def mixed_transform(rate):
            def transform(param):
                random_mask = torch.rand(param.shape, device=param.device)
                random_mask = (random_mask < rate).float()
                return \
                    random_mask * clip_decay_and_quantize_param(param) \
                    + (1 - random_mask) * clip_and_decay_param(param)
            return transform
        
        self.transform = mixed_transform

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.wq_loss, mixed_transform(0.5))
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"WQ accuracy: {value}")
            pipeline.reporter.report(key, value)

        self.trainer.report_function = report
        
        # Epochs
        self.epochs = epochs
        self.cycles = pipeline.wq_cycles

        # Adaptation
        self._prev_acc = target_accuracy
        self.lr = pipeline.lr / 20
        def adaptation(q_rate):
            print(f"  Tuning model with q_rate = {q_rate}...")
            pipeline.reporter.report("Quantization Rate", q_rate)

            self.model.set_activation(CQ_Activation(self.target_tq))
            self.trainer.weight_transformation = mixed_transform(q_rate)

            self.lr = LearningRateExplorer(
                self.trainer, 
                self.model, 
                pipeline.lr / 20, 
                pipeline.lr / 1000, 
                0.01).find_lr_for_tuning()
            
            acc = self.trainer.train_until_target_accuracy(
                self.lr, self.epochs, self._prev_acc)
            
            acc = self.trainer.train_n_epochs(self.lr / 2, 2)
            
            self._prev_acc = max(self._prev_acc, acc) * 0.999
            
        self.adaptation_function = adaptation

    def run(self):
        def evaluate_model(q_rate):
            self.model.set_activation(CQ_Activation(self.target_tq))
            self.trainer.weight_transformation = self.transform(q_rate)

            self.trainer.train_n_epochs(self.lr / 2, 1)
            return self.trainer.validate_train()

        def clone_state():
            return self.model.state_dict()

        def restore_state(state):
            self.model.load_state_dict(state)

        adapter = SmartSmoothAdaptation (
            self.adaptation_function,
            clone_state,
            restore_state,
            evaluate_model
        )

        adapter.adapt_smoothly(interpolators=[BasicInterpolation(0.0, 1.0)])
        
        return self.trainer.validate_train()