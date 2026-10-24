"""[N2] Operand Lowering: every product of two activations onto the declared operand core type."""

from __future__ import annotations

from mimarsinan.mapping.ir.dynamic_bank import OperandBindingError
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.support.operand_lowering import lower_contractions, operand_declaration
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.steps.pipeline_step import METRIC_CARRIED, PipelineStep


class OperandLoweringStep(PipelineStep):
    """Lowers each Contraction of the operand form (design §2.1) before any calibration, so the bank perceptrons
    take their entry grids, readout ranges and laws with every other perceptron; the rest keep G1's host op."""

    REQUIRES = ("model", "platform_constraints_resolved")
    PROMISES = ("operand_lowering_report",)
    UPDATES = ("model",)
    CLEARS = ()

    @classmethod
    def applies_to(cls, plan):
        return any(isinstance(ct, dict) and ct.get("operand") is not None for ct in plan.config.get("cores") or ())

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def process(self):
        if not packaging_contract_for(DeploymentPlan.of(self.pipeline)).is_value_domain:
            raise OperandBindingError(
                "an operand core type under a spiking contract: a payload written at run time has no window-exact "
                "spiking form, so dynamic operands run in the value domain only")
        model = self.get_entry("model")
        platform = self.get_entry("platform_constraints_resolved")
        operand_declaration(platform)
        report = lower_contractions(model.get_mapper_repr(), platform, owner=model,
                                    input_shape=tuple(self.pipeline.config["input_shape"]))
        for product in report.lowered:
            print(f"[OperandLoweringStep] {product.name}: {product.banks} bank(s), {product.cells} payload cells, "
                  f"{product.reason}; absorbed {list(product.follower_kinds) or 'none'}")
        for product in report.host:
            print(f"[OperandLoweringStep] {product.name}: host op ({product.reason})")
        self.add_entry("operand_lowering_report", {
            "lowered": [vars(p) for p in report.lowered], "host": [vars(p) for p in report.host],
        }, "basic")
        self.update_entry("model", model, "torch_model")
