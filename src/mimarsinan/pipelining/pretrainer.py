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
        
        def report(key, value):
            if key == "Training accuracy":
                print(f"Pretraining accuracy: {value}")
            pipeline.reporter.report(key, value)
        self.trainer.report_function = report
        
        self.lr = pipeline.lr
        
        # Epochs
        self.epochs = epochs
        
    def run(self):
        self.model.set_activation(ClampedReLU())
        return self.trainer.train_n_epochs(self.lr, self.epochs)