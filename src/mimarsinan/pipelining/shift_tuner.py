from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import ShiftedClampedReLU
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

class ShiftTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy * 0.99

        # Model
        self.model = pipeline.model

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.aq_loss, decay_param)
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"Shift accuracy: {value}")
            pipeline.reporter.report(key, value)

        self.report = report
        self.trainer.report_function = self.report

        # Epochs
        self.epochs = epochs
        self.cycles = pipeline.aq_cycles

        # Adaptation
        self._prev_acc = target_accuracy
        def adaptation(shift):
            print("  Tuning model with shift = {}...".format(shift))
            pipeline.reporter.report("shift", shift)

            self.model.set_activation(ShiftedClampedReLU(shift))
            self.trainer.weight_transformation = decay_param

            lr = LearningRateExplorer(
                self.trainer, 
                self.model, 
                pipeline.lr / 10, 
                pipeline.lr / 1000, 
                0.01).find_lr_for_tuning()

            acc = self.trainer.train_until_target_accuracy(
                lr, self.epochs, self._prev_acc)
            
            acc = self.trainer.train_n_epochs(lr / 2, 2)
            
            self._prev_acc = max(self._prev_acc, acc) * 0.999

        self.adaptation_function = adaptation
        

    def run(self):
        shift_interpolator = BasicInterpolation(
            0, 
            1.0 / (self.target_tq * 2))
        
        def evaluate_model(shift):
            self.model.set_activation(ShiftedClampedReLU(shift))
            return self.trainer.validate_train()

        def clone_state():
            return self.model.state_dict()

        def restore_state(state):
            self.model.load_state_dict(state)

        shift_adapter = SmartSmoothAdaptation (
            self.adaptation_function,
            clone_state,
            restore_state,
            evaluate_model
        )
        shift_adapter.tolerance = 0.05

        shift_adapter.adapt_smoothly(interpolators=[shift_interpolator])
        
        return self.trainer.validate_train()