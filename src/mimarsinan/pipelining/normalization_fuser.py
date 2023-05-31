from mimarsinan.model_training.basic_trainer import BasicTrainer

class NormalizationFuser:
    def __init__(self, pipeline):
        # Model
        self.model = pipeline.model

        # Trainer
        self.trainer = BasicTrainer(
            self.model, 
            pipeline.device, 
            pipeline.data_provider, 
            pipeline.wq_loss)
        self.trainer.report_function = pipeline.reporter.report

    def run(self):
        self.model.fuse_normalization()

        return self.trainer.validate()