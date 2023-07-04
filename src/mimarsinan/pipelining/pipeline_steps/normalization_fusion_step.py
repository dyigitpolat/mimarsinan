from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer

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
        validation_accuracy = trainer.validate()

        assert validation_accuracy > self.pipeline.cache['aq_accuracy'] * 0.9, \
            "Normalization fusion step failed to retain validation accuracy."

        self.pipeline.cache.add("nf_model", model, 'torch_model')
        self.pipeline.cache.add("nf_accuracy", validation_accuracy)

        self.pipeline.cache.remove("aq_model")