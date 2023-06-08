from mimarsinan.model_training.basic_trainer import BasicTrainer

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
        self.trainer = BasicTrainer(
            self.model, 
            self.device, 
            self.data_provider,
            self.loss)
        self.trainer.report_function = self.reporter.report
        return self.trainer.validate()