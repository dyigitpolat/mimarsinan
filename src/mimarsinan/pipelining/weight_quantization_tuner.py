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
            pipeline.data_provider, 
            pipeline.loss, mixed_transform(0.001))
        self.trainer.report_function = pipeline.reporter.report
        
        # Epochs
        self.epochs = epochs
        self.cycles = pipeline.wq_cycles

        # Adaptation
        self._prev_acc = target_accuracy
        self.pipeline_lr = pipeline.lr 
        def adaptation(q_rate):
            print(f"  Tuning model with q_rate = {q_rate}...")
            pipeline.reporter.report("Quantization Rate", q_rate)

            self.model.set_activation(CQ_Activation(self.target_tq))
            self.trainer.weight_transformation = mixed_transform(q_rate)

            lr = self.find_lr()
            acc = self.trainer.train_until_target_accuracy(
                lr, self.epochs, self._prev_acc)
            
            acc = self.trainer.train_n_epochs(lr / 2, 2)
            
            self._prev_acc = max(self._prev_acc * 0.999, acc)
            
        self.adaptation_function = adaptation
    
    def find_lr(self):
        return LearningRateExplorer(
            self.trainer, 
            self.model, 
            self.pipeline_lr / 20, 
            self.pipeline_lr / 1000, 
            0.01).find_lr_for_tuning()

    def run(self):
        def evaluate_model(q_rate):
            self.model.set_activation(CQ_Activation(self.target_tq))
            self.trainer.weight_transformation = self.transform(q_rate)
            self.trainer.train_one_step(self.pipeline_lr / 40)
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

        adapter.adapt_smoothly(interpolators=[BasicInterpolation(0.0, 1.0)])
        
        return self.trainer.validate()