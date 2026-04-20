from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.adaptation_manager import AdaptationManager

import torch


class ModelBuildingStep(PipelineStep):
    # Model construction is a setup step with no test metric (``validate``
    # just returns the running target).  Opted out of the Phase-B3 floor
    # check so an early 0.0 ``pipeline_metric`` -- which would otherwise
    # predate the first trained baseline -- does not clobber
    # ``previous_metric`` when reasoning about later steps.
    skip_from_floor_check = True

    def __init__(self, pipeline):
        requires = ["model_config", "model_builder"]
        promises = ["model", "adaptation_manager"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def _is_supermodel(self, model):
        return hasattr(model, "get_perceptrons") and hasattr(model, "get_mapper_repr")

    def process(self):
        builder = self.get_entry('model_builder')
        init_model = builder.build(self.get_entry("model_config"))

        adaptation_manager = AdaptationManager()

        if self._is_supermodel(init_model):
            for perceptron in init_model.get_perceptrons():
                adaptation_manager.update_activation(self.pipeline.config, perceptron)

        # Warmup forward pass to initialize any Lazy modules (e.g. LazyBatchNorm1d),
        # so subsequent transformations / mapping that touch normalization parameters
        # won't crash if the pipeline is resumed from a later step.
        try:
            init_model.eval()
            with torch.no_grad():
                input_shape = tuple(self.pipeline.config["input_shape"])
                dummy = torch.zeros((1, *input_shape))
                _ = init_model(dummy)
        except Exception as e:
            print(f"[ModelBuildingStep] Warmup forward failed: {e}")

        self.add_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.add_entry("model", (init_model), "torch_model")
