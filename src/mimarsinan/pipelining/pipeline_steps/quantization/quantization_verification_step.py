import torch

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.quantization_bounds import LAYER_WEIGHT_BITS, layer_weight_bits, quantization_bounds


def integer_grid_stats(ints: torch.Tensor, q_max: int) -> dict:
    """Diagnostics of one layer's deployed integer weight grid."""
    flat = ints.flatten()
    return {
        "n_weights": int(flat.numel()),
        "q_max": int(q_max),
        "zero_frac": float((flat == 0).float().mean().item()),
        "clip_frac": float((flat.abs() >= q_max).float().mean().item()),
        "effective_levels": int(torch.unique(flat).numel()),
        "int_min": int(flat.min().item()),
        "int_max": int(flat.max().item()),
    }


def _assert_integer_lattice(scaled: torch.Tensor, what: str) -> torch.Tensor:
    rounded = torch.round(scaled)
    assert torch.allclose(scaled, rounded, atol=1e-3, rtol=1e-3), f"{what}: {scaled}"
    return rounded


def assert_effective_parameters_on_chip_grid(perceptron, q_max: int, *, law_bias: bool = False) -> None:
    """The WQ install contract, both projection modes: effective weights are
    integer on the ``parameter_scale`` grid; the effective bias is integer on
    its OWN ``bias_scale`` grid within the ±q_max register range AND on the
    weight-grid lattice (``bias * parameter_scale = r * bias_int`` — what the
    chip export emits, so both parity sides see the same semantics). A bias
    declared at the law's resolution (``law_bias``) has no grid to be on."""
    transformer = PerceptronTransformer()
    fused_w = transformer.get_effective_weight(perceptron)
    fused_b = transformer.get_effective_bias(perceptron)
    param_scale = perceptron.parameter_scale
    bias_scale = getattr(perceptron, "bias_scale", None)
    if bias_scale is None:
        bias_scale = param_scale

    _assert_integer_lattice(fused_w * param_scale, "weights off the chip grid")
    if law_bias:
        return
    _assert_integer_lattice(
        fused_b * param_scale, "bias off the weight-grid lattice"
    )
    bias_ints = _assert_integer_lattice(
        fused_b * bias_scale, "bias off its own grid"
    )
    assert float(bias_ints.abs().max()) <= q_max + 0.5, (
        f"bias outside its ±{q_max} register range: {bias_ints}"
    )


def _declared_q_max(perceptron, q_max: int) -> int:
    """The register range of ``perceptron``'s grid: its core type's width (WI-5g), else the run's ``q_max``."""
    declared = getattr(perceptron, LAYER_WEIGHT_BITS, None)
    return q_max if declared is None else quantization_bounds(int(declared))[1]


def assert_weights_within_layer_grid(perceptron, q_max: int) -> None:
    """A perceptron on a core type of its own width holds weight codes inside that width's register (WI-5g)."""
    ints = torch.round(PerceptronTransformer().get_effective_weight(perceptron) * perceptron.parameter_scale)
    assert float(ints.abs().max()) <= q_max + 0.5, (
        f"{getattr(perceptron, 'name', '?')}: weight codes up to {float(ints.abs().max())} outside the ±{q_max} "
        f"register of its {layer_weight_bits(perceptron, 0)}-bit core type")


def quantization_grid_report(perceptrons, q_max: int) -> list[dict]:
    """Per-perceptron integer-grid rows following the verification convention
    (``round(fused_w * parameter_scale)`` lies on the ±q_max grid, or that of the layer's core type, WI-5g)."""
    rows: list[dict] = []
    transformer = PerceptronTransformer()
    for index, perceptron in enumerate(perceptrons):
        fused_w = transformer.get_effective_weight(perceptron)
        ints = torch.round(fused_w * perceptron.parameter_scale)
        rows.append({
            "index": index,
            "name": str(getattr(perceptron, "name", f"perceptron_{index}")),
            "parameter_scale": float(perceptron.parameter_scale.item()),
            **integer_grid_stats(ints, _declared_q_max(perceptron, q_max)),
        })
    return rows


class QuantizationVerificationStep(TrainerPipelineStep):
    REQUIRES = ("model",)

    @classmethod
    def applies_to(cls, plan):
        return plan.weight_quantization

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    @property
    def q_max(self) -> int:
        """The register range this run quantizes onto, read WHERE IT IS USED.

        Never captured in ``__init__``: steps are constructed at pipeline
        assembly, but a hardware search stamps the winner's ``weight_bits``
        into the config afterwards. A captured value would gate the winner's
        weights against the pre-search grid (measured: a searched
        ``weight_bits=6`` asserted against the declared 5-bit ±15 range).
        """
        return (2 ** (int(self.pipeline.config["weight_bits"]) - 1)) - 1

    def process(self):
        model = self.get_entry("model")
        q_max = self.q_max
        self.trainer = make_basic_trainer(self.pipeline, model)
        for perceptron in model.get_perceptrons():
            print(perceptron.parameter_scale)
            print(perceptron.scale_factor)
            perceptron.to(self.pipeline.config["device"])
            assert_effective_parameters_on_chip_grid(
                perceptron, _declared_q_max(perceptron, q_max),
                law_bias=ParameterGrids.from_config(self.pipeline.config).law_bias)
            if getattr(perceptron, LAYER_WEIGHT_BITS, None) is not None:
                assert_weights_within_layer_grid(perceptron, _declared_q_max(perceptron, q_max))

        # The asserts above ARE the gate; surviving them is the verdict. The
        # per-layer grid diagnostics ship as one low-rate structured event.
        layers = quantization_grid_report(model.get_perceptrons(), q_max)
        bits = int(self.pipeline.config["weight_bits"])
        emit_reporter_event(
            getattr(self.pipeline, "reporter", None),
            "quantization_report",
            {"bits": bits, "q_max": q_max, "layers": layers},
        )
        self._verdict = {
            "status": "pass",
            "rule": f"fused weights integer on the ±{self.q_max} grid ({bits}-bit)",
            "detail": {"perceptrons": len(layers), "weight_bits": bits},
        }
