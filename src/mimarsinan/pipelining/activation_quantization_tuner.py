from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.models.layers import CQ_Activation
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.basic_smooth_adaptation import BasicSmoothAdaptation

class ActivationQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy

        # Model
        self.model = pipeline.model
        self.model.set_activation(CQ_Activation(self.target_tq))

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.aq_loss, clip_and_decay_param)
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"AQ accuracy: {value}")
            pipeline.reporter.report(key, value)

        self.report = report
        self.trainer.report_function = self.report

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

        # Adaptation
        self._prev_acc = target_accuracy
        def tq_adaptation(Tq):
            print("  Tuning model with soft CQ with Tq = {}...".format(Tq))
            pipeline.reporter.report("Tq", Tq)

            self.model.set_activation(CQ_Activation(Tq))
            self.trainer.weight_transformation = clip_and_decay_param

            acc = self.trainer.train_until_target_accuracy(
                self.lr, self.epochs, self._prev_acc)
            
            self._prev_acc = max(self._prev_acc, acc)

        self.adaptation_function = tq_adaptation
        

    def run(self):
        Tq_interpolator = BasicInterpolation(
            100, self.target_tq, 
            curve = lambda x: x ** 0.2)

        BasicSmoothAdaptation (
            self.adaptation_function
        ).adapt_smoothly(
            interpolators=[Tq_interpolator], 
            cycles=self.cycles)
        
        return self.trainer.validate_train()