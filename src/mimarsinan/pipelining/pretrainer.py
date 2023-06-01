from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.models.layers import ClampedReLU, NoisyDropout
class Pretrainer:
    def __init__(self, pipeline, epochs):
        # Dependencies
        self.model = pipeline.model

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            pipeline.device, 
            pipeline.data_provider, 
            pipeline.loss, clip_and_decay_param)
        self.trainer.report_function = pipeline.reporter.report
        
        self.lr = pipeline.lr
        
        # Epochs
        self.epochs = epochs
        
    def run(self):
        self.model.set_activation(ClampedReLU())
        self.model.set_regularization(NoisyDropout(0.5, 0.5, 0.2))

        return self.trainer.train_n_epochs(self.lr, self.epochs)