from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import CQ_Activation_Soft, CQ_Activation, ShiftedActivation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

class ActivationQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy * 0.99

        # Model
        self.model = pipeline.model

        def mixed_transform(rate):
            def transform(param):
                a = clip_and_decay_param(param)
                b = param
                return a * rate + b * (1 - rate)
            return transform
        
        self.parametric_transform = mixed_transform

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.aq_loss, mixed_transform(0.01))
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"AQ accuracy: {value}")
            pipeline.reporter.report(key, value)

        self.report = report
        self.trainer.report_function = self.report

        # Epochs
        self.epochs = epochs
        self.cycles = pipeline.aq_cycles

        # Adaptation
        self._prev_acc = target_accuracy
        self.pipeline_lr = pipeline.lr
        self.lr = pipeline.lr / 10
        def adaptation(alpha, tq, q_rate):
            print(f"  CQ Tuning with alpha = {alpha}, tq = {tq}, q_rate = {q_rate}...")
            pipeline.reporter.report("alpha", alpha)

            shift_amount = (0.5 / self.target_tq) - (0.5 / tq)
            soft_cq = CQ_Activation_Soft(tq, alpha)
            self.model.set_activation(ShiftedActivation(soft_cq, shift_amount))
            self.trainer.weight_transformation = self.parametric_transform(q_rate)

            self.lr = LearningRateExplorer(
                self.trainer, 
                self.model, 
                pipeline.lr / 10, 
                pipeline.lr / 1000, 
                0.01).find_lr_for_tuning()

            acc = self.trainer.train_until_target_accuracy(
                self.lr, self.epochs, self._prev_acc)
            
            acc = self.trainer.train_n_epochs(self.lr / 2, 2)
            
            self._prev_acc = max(self._prev_acc, acc) * 0.99

        self.adaptation_function = adaptation
        

    def run(self):
        alpha_interpolator = BasicInterpolation(
            0.1, 
            20, 
            curve = lambda x: x ** 2)
        tq_interpolator = BasicInterpolation(
            self.target_tq * 2 * self.cycles, 
            self.target_tq, 
            curve = lambda x: x ** 0.2)
        q_rate_interpolator = BasicInterpolation(0,1)
        
        def evaluate_model(alpha, tq, q_rate):
            shift_amount = (0.5 / self.target_tq) - (0.5 / tq)
            soft_cq = CQ_Activation_Soft(tq, alpha)
            self.model.set_activation(ShiftedActivation(soft_cq, shift_amount))
            self.trainer.weight_transformation = self.parametric_transform(q_rate)
            
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
        adapter.tolerance = 0.01

        adapter.adapt_smoothly(interpolators=[
            alpha_interpolator, tq_interpolator, q_rate_interpolator])
        
        self.trainer.weight_transformation = clip_and_decay_param
        self.model.set_activation(CQ_Activation(self.target_tq))
        lr = LearningRateExplorer(
                self.trainer, 
                self.model, 
                self.pipeline_lr / 10, 
                self.pipeline_lr / 1000, 
                0.01).find_lr_for_tuning()
        self.trainer.train_until_target_accuracy(
            lr, self.epochs, self.target_accuracy)

        
        return self.trainer.validate_train()