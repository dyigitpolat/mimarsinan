from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer

class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["aq_model", "aq_accuracy"]
        promises = ["nf_model", "nf_accuracy"]
        clears = ["aq_model"]
        super().__init__(requires, promises, clears, pipeline)


    def process(self):
        model = self.pipeline.cache["aq_model"]
        model.fuse_normalization()

        # Trainer
        trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider,
            self.pipeline.loss)
        trainer.report_function = self.pipeline.reporter.report

        trainer.train_until_target_accuracy(
            LearningRateExplorer(
                trainer,
                model,
                self.pipeline.config['lr'] / 2,
                self.pipeline.config['lr'] / 1000,
                0.01
            ).find_lr_for_tuning(),
            self.pipeline.config['tuner_epochs'],
            self.pipeline.cache['aq_accuracy'] * 0.99,
        )
        accuracy = trainer.validate()

        assert accuracy > self.pipeline.cache['aq_accuracy'] * 0.9, \
            "Normalization fusion step failed to retain validation accuracy."

        self.pipeline.cache.add("nf_model", model, 'torch_model')
        self.pipeline.cache.add("nf_accuracy", accuracy)

        self.pipeline.cache.remove("aq_model")