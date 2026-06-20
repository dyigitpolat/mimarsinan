from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron


class NormalizationFusionStep(TrainerPipelineStep):
    REQUIRES = ("model",)
    PROMISES = ("fused_model",)
    UPDATES = ("model",)

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        if self.trainer is not None:
            return self.trainer.test()
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry("model")
        self.trainer = make_basic_trainer(self.pipeline, model)
        device = self.pipeline.config["device"]
        for perceptron in model.get_perceptrons():
            fuse_into_perceptron(perceptron, device=device)
        print(self.validate())
        self.update_entry("model", model, "torch_model")
        self.add_entry("fused_model", model, "torch_model")
