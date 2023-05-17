from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.basic_smooth_adaptation import BasicSmoothAdaptation

class WeightQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy

        # Model
        self.model = pipeline.model
        self.model.set_activation(CQ_Activation(self.target_tq))

        def mixed_transform(rate):
            def transform(param):
                a = clip_decay_and_quantize_param(param)
                b = param
                return a * rate + b * (1.0 - rate)
            return transform

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

        # Automatic learning rate
        self.lr = LearningRateExplorer(
            self.trainer, 
            self.model, 
            pipeline.lr, 
            pipeline.lr / 1000, 
            0.1 / pipeline.num_classes).find_lr_for_tuning()
        
        # Epochs
        self.epochs = epochs
        self.cycles = 10

        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy

        # Adaptation
        self._prev_acc = target_accuracy
        def adaptation(q_rate):
            print(f"  Tuning model with q_rate = {q_rate}...")
            pipeline.reporter.report("Quantization Rate", q_rate)

            self.model.set_activation(CQ_Activation(self.target_tq))
            self.trainer.weight_transformation = mixed_transform(q_rate)

            acc = self.trainer.train_until_target_accuracy(
                self.lr, self.epochs, self._prev_acc)
            
            self._prev_acc = max(self._prev_acc, acc)
            
        self.adaptation_function = adaptation

    def run(self):
        self.model.set_activation(CQ_Activation(self.target_tq))

        BasicSmoothAdaptation (
            self.adaptation_function
        ).adapt_smoothly(
            interpolators=[BasicInterpolation(0.0, 1.0)], 
            cycles=self.cycles)
        
        return self.trainer.validate_train()