from mimarsinan.pipelining.core.engine.pipeline_helpers import safe_warmup_forward
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.tuning.orchestration.adaptation_manager_factory import create_adaptation_manager_for_model

import torch


class ModelBuildingStep(PipelineStep):
    REQUIRES = ("model_config", "model_builder")
    PROMISES = ("model", "adaptation_manager")

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def _is_supermodel(self, model):
        return hasattr(model, "get_perceptrons") and hasattr(model, "get_mapper_repr")

    def process(self):
        builder = self.get_entry('model_builder')
        init_model = builder.build(self.get_entry("model_config"))

        adaptation_manager = create_adaptation_manager_for_model(
            self.pipeline.config, init_model
        )

        # Warmup forward pass to initialize any Lazy modules (e.g. LazyBatchNorm1d),
        # so subsequent transformations / mapping that touch normalization parameters
        # won't crash if the pipeline is resumed from a later step.
        safe_warmup_forward(
            init_model,
            self.pipeline.config["input_shape"],
            self.pipeline.config["device"],
        )

        self.add_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.add_entry("model", (init_model), "torch_model")
