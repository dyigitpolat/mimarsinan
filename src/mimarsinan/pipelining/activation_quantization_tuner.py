from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import CQ_Activation_Parametric, CQ_Activation, ShiftedActivation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation

class ActivationQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy * 0.99

        # Model
        self.model = pipeline.model

        def mixed_transform(transform_rate):
            def transform(param):
                random_mask = torch.rand(param.shape, device=param.device)
                random_mask = (random_mask < transform_rate).float()
                return \
                    random_mask * clip_and_decay_param(param) \
                    + (1 - random_mask) * param
            return transform
        
        self.parametric_transform = mixed_transform

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.data_provider,
            pipeline.loss, mixed_transform(0.001))
        self.trainer.report_function = pipeline.reporter.report

        # Epochs
        self.epochs = epochs
        self.cycles = pipeline.aq_cycles

        # Adaptation
        self._prev_acc = target_accuracy
        self.pipeline_lr = pipeline.lr
        self.lr = pipeline.lr / 10
        def adaptation(cq_rate, transform_rate):
            print(f"  CQ Tuning with rate = {cq_rate}, transform_rate = {transform_rate}...")
            pipeline.reporter.report("cq_rate", cq_rate)

            self.model.set_activation(CQ_Activation_Parametric(self.target_tq, cq_rate))
            self.trainer.weight_transformation = self.parametric_transform(transform_rate)

            self.lr = LearningRateExplorer(
                self.trainer, 
                self.model, 
                pipeline.lr / 10, 
                pipeline.lr / 1000, 
                0.01).find_lr_for_tuning()

            acc = self.trainer.train_until_target_accuracy(
                self.lr, self.epochs, self._prev_acc)
            
            acc = self.trainer.train_n_epochs(self.lr / 2, 2)
            
            self._prev_acc = max(self._prev_acc * 0.99, acc)

        self.adaptation_function = adaptation
        

    def run(self):
        cq_rate_interpolator = BasicInterpolation(0, 1)
        transform_rate_interpolator = BasicInterpolation(0,1)
        
        def evaluate_model(cq_rate, transform_rate):
            self.model.set_activation(CQ_Activation_Parametric(self.target_tq, cq_rate))
            self.trainer.weight_transformation = self.parametric_transform(transform_rate)
            
            self.trainer.train_n_epochs(self.lr / 2, 1)
            return self.trainer.validate()

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
            cq_rate_interpolator, transform_rate_interpolator])
        
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

        
        return self.trainer.validate()