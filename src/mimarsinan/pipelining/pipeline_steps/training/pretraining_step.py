from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep


class PretrainingStep(TrainerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return not plan.weight_source

    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        self.trainer = make_basic_trainer(self.pipeline, model)
        self.trainer.train_n_epochs(
            self.pipeline.config["lr"],
            self.pipeline.config["training_epochs"],
            warmup_epochs=5,
        )
        self.update_entry("model", model, "torch_model")
