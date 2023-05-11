from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.models.layers import ClampedReLU

class Pretrainer:
    def __init__(self, pipeline, epochs):
        # Dependencies
        self.model = pipeline.model

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.training_dataloader, 
            pipeline.validation_dataloader, 
            pipeline.pt_loss, decay_param)
        self.trainer.report_function = pipeline.reporter.report

        # Automatic learning rate
        desired_improvement = (1.0 / pipeline.num_classes) / 10
        self.tuned_lr = LearningRateExplorer(
            self.trainer, 
            self.model, 
            max_lr=1e-1, 
            min_lr=1e-5, 
            desired_improvement=desired_improvement).find_lr_for_tuning()
        
        # Epochs
        self.epochs = epochs
        
    def run(self):
        self.model.set_activation(ClampedReLU())
        return self.trainer.train_n_epochs(self.tuned_lr, self.epochs)