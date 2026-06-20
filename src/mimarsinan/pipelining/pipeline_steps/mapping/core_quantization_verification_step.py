from __future__ import annotations

from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.mapping.export.chip_quantize import verify_ir_graph_quantized


class CoreQuantizationVerificationStep(PipelineStep):
    """Fail fast if weight_quantization=True but IR NeuralCores are not chip-quantized."""

    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        requires = ["ir_graph"]
        promises = []
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        ir_graph = self.get_entry("ir_graph")
        if not DeploymentPlan.of(self.pipeline).weight_quantization:
            cores = ir_graph.get_neural_cores()
            if cores:
                print(
                    "[CoreQuantizationVerificationStep] Skipping verification (weight_quantization=False); "
                    f"IR has {len(cores)} NeuralCores with float weights."
                )
            return

        bits = int(self.pipeline.config["weight_bits"])
        cores = ir_graph.get_neural_cores()
        if not cores:
            print("[CoreQuantizationVerificationStep] No NeuralCores found in IRGraph (nothing to verify).")
            return

        verify_ir_graph_quantized(ir_graph, bits)
        print(
            f"[CoreQuantizationVerificationStep] OK: verified {len(cores)} "
            f"NeuralCores at {bits}-bit quantization."
        )


