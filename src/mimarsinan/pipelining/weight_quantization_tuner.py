from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.models.layers import CQ_Activation

class WeightQuantizationTuner:
    def __init__(self, pipeline, epochs, target_tq, target_accuracy):
        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.wq_loss, clip_decay_and_quantize_param)
        self.trainer.report_function = pipeline.reporter.report

        # Automatic learning rate
        desired_improvement = -(1.0 / self.num_classes) / 100
        self.tuned_lr = LearningRateExplorer(
            self.trainer, 
            self.model, 
            max_lr=1e-1, 
            min_lr=1e-5, 
            desired_improvement=desired_improvement).find_lr_for_tuning()
        
        # Epochs
        self.epochs = epochs

        # Targets
        self.target_tq = target_tq
        self.target_accuracy = target_accuracy

    def run(self):
        self.model.set_activation(CQ_Activation(self.target_tq))
        self.trainer.train_until_target_accuracy(
            self.tuned_lr, self.epochs, self.target_accuracy)