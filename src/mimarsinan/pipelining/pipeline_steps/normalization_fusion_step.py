from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron


class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = ["fused_model"]
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        if self.trainer is not None:
            return self.trainer.test()
        return self.pipeline.get_target_metric()

    def cleanup(self):
        if self.trainer is not None:
            self.trainer.close()

    def process(self):
        model = self.get_entry("model")

        self.trainer = BasicTrainer(
            model,
            self.pipeline.config['device'],
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

        device = self.pipeline.config['device']
        for perceptron in model.get_perceptrons():
            fuse_into_perceptron(perceptron, device=device)

        print(self.validate())

        self.update_entry("model", model, 'torch_model')
        self.add_entry("fused_model", model, 'torch_model')
