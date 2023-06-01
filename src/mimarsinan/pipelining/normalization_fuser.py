from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.transformations.parameter_transforms.collection import *

class NormalizationFuser:
    def __init__(self, pipeline, target_accuracy):
        # Model
        self.model = pipeline.model

        # Training
        self.lr = pipeline.lr
        self.device = pipeline.device
        self.data_provider = pipeline.data_provider
        self.loss = pipeline.loss
        self.reporter = pipeline.reporter

        # Targets
        self.target_accuracy = target_accuracy * 0.99

    def run(self):
        self.model.fuse_normalization()

        # Trainer
        self.trainer = WeightTransformTrainer(
            self.model, 
            self.device, 
            self.data_provider,
            self.loss, clip_and_decay_param)
        self.trainer.report_function = self.reporter.report

        self.trainer.train_until_target_accuracy(self.lr / 40, 2, self.target_accuracy)
        return self.trainer.validate()