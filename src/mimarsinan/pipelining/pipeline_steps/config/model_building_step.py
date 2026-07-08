from mimarsinan.config_schema.registry import effective_value
from mimarsinan.mapping.verification.onchip_fraction import (
    assert_onchip_majority_estimate_or_raise,
)
from mimarsinan.pipelining.core.engine.pipeline_helpers import safe_warmup_forward
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)
from mimarsinan.tuning.orchestration.adaptation_manager_factory import create_adaptation_manager_for_model


class ModelBuildingStep(PipelineStep):
    REQUIRES = ("model_config", "model_builder")
    PROMISES = ("model", "adaptation_manager")

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def _is_supermodel(self, model):
        return hasattr(model, "get_perceptrons") and hasattr(model, "get_mapper_repr")

    def process(self):
        builder = self.get_entry('model_builder')
        init_model = builder.build(self.get_entry("model_config"))

        adaptation_manager = create_adaptation_manager_for_model(
            self.pipeline.config, init_model
        )

        # Materialize Lazy modules (e.g. LazyBatchNorm1d) so a later-step resume that touches normalization parameters does not crash.
        safe_warmup_forward(
            init_model,
            self.pipeline.config["input_shape"],
            self.pipeline.config["device"],
        )

        self._run_static_onchip_majority_gate(init_model)

        self.add_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.add_entry("model", (init_model), "torch_model")

    def _run_static_onchip_majority_gate(self, model) -> None:
        """Static fail-fast twin of the SCM on-chip-majority floor gate: a
        host-majority model SPEC dies in seconds at build instead of after
        pretraining (W2 Q3; SAME floor key onchip_min_fraction, same opt-out knob)."""
        config = self.pipeline.config
        if not bool(config.get("onchip_majority_gate", True)):
            return
        num_classes = config.get("num_classes")
        if not num_classes:
            return
        estimate = assert_onchip_majority_estimate_or_raise(
            model,
            config["input_shape"],
            int(num_classes),
            encoding_placement=str(
                config.get("encoding_layer_placement", "subsume")
            ),
            min_fraction=float(effective_value(config, "onchip_min_fraction")),
        )
        print(
            f"[ModelBuildingStep] static on-chip parameter majority: "
            f"{estimate.fraction:.2%} on chip (on-chip={estimate.onchip}, "
            f"host={estimate.host}, total={estimate.total})"
        )
