import torch

from mimarsinan.common.reporter import emit_reporter_event
from mimarsinan.mapping.support.bias_rows import bias_register_span
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.models.perceptron_mixer.perceptron import placed_on_cores
from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_wq_two_scale_projection
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.trainer_pipeline_step import TrainerPipelineStep
from mimarsinan.transformations.perceptron.bias_drive import bias_row_values
from mimarsinan.transformations.perceptron.neuron_scales import is_per_neuron_scale, scale_view
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import output_axes
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


def assert_effective_parameters_on_chip_grid(perceptron, q_max: int, *, law_bias: bool = False,
                                             register_bias: bool = False) -> None:
    """The WQ install contract, every projection mode: effective weights are
    integer on the ``parameter_scale`` grid; the effective bias is integer on
    the weight-grid lattice (``bias * parameter_scale = r * bias_int`` — what the
    chip export emits, so both parity sides see the same semantics) AND on its
    OWN ``bias_scale`` grid, within that grid's ±q_max register range. On the
    two-scale projection's on-chip register route (``register_bias``) the bias
    grid IS the weight lattice and the register spans the bias's own codes,
    ``[-(q_max+1) r, q_max r]`` with ``r = ceil(max|code| / q_max)`` (the span the
    export allocates). A bias declared at the law's resolution (``law_bias``)
    has no grid to be on."""
    transformer = PerceptronTransformer(positions=True)  # [U-PFOLD] each weight row and position of a fold per position
    fused_w = transformer.get_effective_weight(perceptron)
    # An always-on row driven at d holds the bias over d (d = 1.0 without a stamp): its codes are what the lattice holds.
    fused_b = bias_row_values(perceptron, transformer.get_effective_bias(perceptron))
    param_scale = perceptron.parameter_scale
    bias_scale = getattr(perceptron, "bias_scale", None)
    if bias_scale is None:
        bias_scale = param_scale

    # [U-PCH] a per-neuron grid reads each output neuron on its own scale (``scale_view``), [U-PFOLD r2] on the axis
    # the reader declares for each view.
    axes = output_axes(perceptron, transformer.positions)
    _assert_integer_lattice(fused_w * scale_view(param_scale, fused_w, axes.weight), "weights off the chip grid")
    if law_bias:
        return
    codes = _assert_integer_lattice(
        fused_b * scale_view(param_scale, fused_b, axes.bias), "bias off the weight-grid lattice"
    )
    bias_ints = _assert_integer_lattice(
        fused_b * scale_view(bias_scale, fused_b, axes.bias), "bias off its own grid"
    )
    if register_bias:
        span = bias_register_span(float(codes.abs().max()) if codes.numel() else 0.0, q_max)
        assert float(codes.min()) >= -(q_max + 1) * span - 0.5 and float(codes.max()) <= q_max * span + 0.5, (
            f"bias outside its register span ±{q_max}x{span}: {codes}"
        )
        return
    assert float(bias_ints.abs().max()) <= q_max + 0.5, (
        f"bias outside its ±{q_max} register range: {bias_ints}"
    )


def _declared_q_max(perceptron, q_max: int) -> int:
    """The register range of ``perceptron``'s grid: its core type's width (WI-5g), else the run's ``q_max``."""
    declared = getattr(perceptron, LAYER_WEIGHT_BITS, None)
    return q_max if declared is None else quantization_bounds(int(declared))[1]


def assert_weights_within_layer_grid(perceptron, q_max: int) -> None:
    """A perceptron on a core type of its own width holds weight codes inside that width's register (WI-5g)."""
    fused_w = PerceptronTransformer(positions=True).get_effective_weight(perceptron)  # [U-PFOLD] each weight row
    ints = torch.round(fused_w * scale_view(perceptron.parameter_scale, fused_w, output_axes(perceptron, True).weight))
    assert float(ints.abs().max()) <= q_max + 0.5, (
        f"{getattr(perceptron, 'name', '?')}: weight codes up to {float(ints.abs().max())} outside the ±{q_max} "
        f"register of its {layer_weight_bits(perceptron, 0)}-bit core type")


def quantization_grid_report(perceptrons, q_max: int) -> list[dict]:
    """Per-perceptron integer-grid rows following the verification convention
    (``round(fused_w * parameter_scale)`` lies on the ±q_max grid, or that of the layer's core type, WI-5g)."""
    rows: list[dict] = []
    transformer = PerceptronTransformer(positions=True)  # [U-PFOLD] the weights the program holds, each weight row
    for index, perceptron in enumerate(perceptrons):
        fused_w = transformer.get_effective_weight(perceptron)
        scale = perceptron.parameter_scale
        ints = torch.round(fused_w * scale_view(scale, fused_w, output_axes(perceptron, transformer.positions).weight))
        rows.append({
            "index": index,
            "name": str(getattr(perceptron, "name", f"perceptron_{index}")),
            # [U-PCH] a per-neuron grid reports each output neuron's scale
            "parameter_scale": scale.detach().cpu().tolist() if is_per_neuron_scale(scale) else float(scale.item()),
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
        # The run's grids: the declared rules and the two-scale projection's bias route (as the WQ step named it).
        grids = ParameterGrids.from_config(
            self.pipeline.config, register_bias=resolve_wq_two_scale_projection(self.pipeline.config))
        # [U-POST G3] The verification applies to the perceptrons the placement maps to cores, as the projection does: a
        # host-placed perceptron (a subsumed encoder) keeps its float weights and runs as a host op.
        projected = [p for p in model.get_perceptrons() if placed_on_cores(p)]
        for perceptron in projected:
            print(perceptron.parameter_scale)
            print(perceptron.scale_factor)
            perceptron.to(self.pipeline.config["device"])
            assert_effective_parameters_on_chip_grid(
                perceptron, _declared_q_max(perceptron, q_max),
                law_bias=grids.law_bias, register_bias=grids.register_bias)
            if getattr(perceptron, LAYER_WEIGHT_BITS, None) is not None:
                assert_weights_within_layer_grid(perceptron, _declared_q_max(perceptron, q_max))

        # The asserts above ARE the gate; surviving them is the verdict. The
        # per-layer grid diagnostics ship as one low-rate structured event.
        layers = quantization_grid_report(projected, q_max)
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
