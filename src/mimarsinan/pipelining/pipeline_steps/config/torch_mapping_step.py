"""TorchMappingStep -- convert a native PyTorch model to a mimarsinan PerceptronFlow."""

from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.tuning.orchestration.adaptation_manager_factory import create_adaptation_manager_for_model
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.conversion_probe import ConversionProbeError
import torch


class TorchMappingStep(TrainerPipelineStep):
    """Trace a trained torch model, convert it to a Mapper DAG (weights + activation types), and set up the AdaptationManager."""

    REQUIRES = ("model",)
    PROMISES = ("adaptation_manager",)
    UPDATES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        return plan.model_category == "torch"

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        tag = "TorchMappingStep"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        native_model = self.get_entry("model")

        with phase_profiler(tag, "convert_torch_model"):
            flow = convert_torch_model(
                native_model,
                input_shape=tuple(self.pipeline.config["input_shape"]),
                num_classes=self.pipeline.config["num_classes"],
                device=self.pipeline.config["device"],
                Tq=self.pipeline.config["target_tq"],
                encoding_layer_placement=str(
                    self.pipeline.config.get("encoding_layer_placement", "subsume")
                ),
            )

        adaptation_manager = create_adaptation_manager_for_model(
            self.pipeline.config, flow
        )

        with phase_profiler(tag, "verify_equivalence"):
            self._verify_equivalence(native_model, flow)
        native_model.cpu()
        del native_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with phase_profiler(tag, "make_basic_trainer"):
            self.trainer = make_basic_trainer(self.pipeline, flow)
        self.update_entry("model", flow, "torch_model")
        self.add_entry("adaptation_manager", adaptation_manager, "pickle")

        print(f"[{tag}] Converted native model to torch-mapped PerceptronFlow")
        print(f"  Perceptrons: {len(flow.get_perceptrons())}")
        with phase_profiler(tag, "validate"):
            val = self.validate()
        print(f"  Validation: {val}")

        if torch.cuda.is_available():
            step_peak = torch.cuda.max_memory_allocated() / (1 << 20)
            print(f"[{tag}] cuda peak (step): {step_peak:.0f} MiB")

    def _verify_equivalence(self, native_model, flow):
        """Compare native vs converted output shapes; converted-flow forward failure is fatal."""
        device = self.pipeline.config["device"]
        input_shape = tuple(self.pipeline.config["input_shape"])

        native_model.eval().to(device)
        dummy = torch.randn(2, *input_shape, device=device)

        try:
            with torch.no_grad():
                native_out = native_model(dummy).detach().cpu()
        except Exception as exc:
            print(f"[TorchMappingStep] Native forward skipped: {exc}")
            return

        native_model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        flow.eval().to(device)
        try:
            with torch.no_grad():
                converted_out = flow(dummy).detach().cpu()
        except Exception as exc:
            raise ConversionProbeError(
                "[TorchMappingStep] Converted-flow forward failed during "
                "equivalence check; the converted model is structurally broken."
            ) from exc

        if native_out.shape != converted_out.shape:
            print(
                f"[TorchMappingStep] WARNING: output shape mismatch: "
                f"native={native_out.shape} vs converted={converted_out.shape}"
            )
