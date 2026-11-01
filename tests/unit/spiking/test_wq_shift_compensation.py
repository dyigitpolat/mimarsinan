"""[U-WQB B, phase C defect D-WQ] The negative-boundary shift compensation re-derived on the projected weights.

A lifted boundary (the seam's ``(f + sigma) / kappa``, U-SC2) and a Soft Core Mapping shift both feed their consumers
``x + s`` and bake the consumer's bias once, ``b~ = b - W~ s``, on the float effective weights
(``apply_negative_shift_bias``, once per perceptron). Weight Quantization then replaces ``W~`` by ``Q(W~)`` and left the
bias carrying ``-W~ s``: every lifted entry deployed an extra ``(Q(W~) - W~) s`` (all 54 of S4's entries, 0.04 to 0.18
theta per cycle; ``RA/12_tcad_readiness/phase_c_r/d_wq/report.md`` section 3.3).

The bake now records the shift it compensated (``input_shift``, scalar or per axon, effective units) and the projection
re-derives the compensation on the weights it installs, ``b~ <- b~ - (Q(w~) - w~) s``, at every rate and every
re-projection, before the bias is coded: the deployed bias is ``b - Q(W~) s`` within the bias quantum. The NF and the
deployed program stay count-exact (atol 0) on the lifted program through WQ and Soft Core Mapping (guard).

The projection is read as the WQ step's tuner applies it, on two routes: the on-chip register (two-scale, the route
named) and the bias at the law's resolution (weights quantized, bias float). A tree that names no route (R) reads its
own projection.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation
from mimarsinan.mapping.support.negative_shift import apply_negative_shift_bias, apply_negative_value_shifts
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.tuning.orchestration.negative_boundary_seam import NegativeBoundarySeam, _entries
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner as Tuner,
)

from unit.spiking.host_gather_forms import T, Program, count_mismatches, nf_window_counts, scm_window_counts

BITS = 4
Q_MAX = 2 ** (BITS - 1) - 1
SHAPE, CLASSES = (3, 16, 16), 10
PT = PerceptronTransformer()
ROUTES = ("register", "law")


def _tuner(route: str, bits: int = BITS):
    """The WQ step's tuner on ``route`` (built without ``__init__``, as the projection's tests build it)."""
    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits = bits
    tuner.two_scale_projection = route in ("register", "rows")
    tuner.fractional_scale, tuner.float_bias = False, route == "law"
    tuner.bias_rows_floor = None
    tuner.register_bias = route == "register"
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    return tuner


def _project(route: str, perceptrons, rate: float = 1.0) -> None:
    transform = _tuner(route)._get_new_perceptron_transform(rate)
    with torch.no_grad():
        for p in perceptrons:
            transform(p)


def _tolerance(route: str, p) -> float:
    """The bias quantum of ``route``: half a weight step on the register (the code is round(b s_w)), float dust on the
    law's float bias."""
    return 0.5 / float(p.parameter_scale) + 1e-5 if route == "register" else 2e-5


# ---------------------------------------------------------------------------
# Fixtures: a lifted host op feeding conv cores (S4's form), a Soft Core Mapping per-axon shift into an fc core
# ---------------------------------------------------------------------------


def _pyramidnet():
    """S4's family at a small size (PyramidNet depth 8, alpha 12, 3x16x16): every core reads a signed host BatchNorm
    (``units_k_bn1``), at random init with randomized BatchNorm statistics."""
    from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model

    builder = BUILDERS_REGISTRY["cifar_pyramidnet"]("cpu", SHAPE, CLASSES, {})
    torch.manual_seed(0)
    model = build_model(builder, {"depth": 8, "alpha": 12}, encoding_placement="subsume").eval()
    gen = torch.Generator().manual_seed(100)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.running_mean.uniform_(-0.3, 0.3, generator=gen)
                m.running_var.uniform_(0.5, 1.5, generator=gen)
        model(torch.zeros(1, *SHAPE))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, SHAPE, CLASSES, device="cpu", encoding_layer_placement="subsume")
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(torch.tensor(1.5 + 0.2 * (i % 3)))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        lif.use_cycle_accurate_trains = True
        p.base_activation = lif
        p.activation = lif
    return flow.eval()


def _inputs(seed: int, n: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return 2.0 * torch.randn(n, *SHAPE)


def _lifted_pyramidnet():
    """The lifted program and, per lifted entry core, its bake: ``(perceptron, s = sigma / kappa, b~ + W~ s)``, read
    from the lifted op's stamps exactly as ``negative_boundary_seam._bake`` computes them."""
    flow = _pyramidnet()
    seam = NegativeBoundarySeam(calibration_x=_inputs(7, 64), simulation_steps=T, input_data_scale=2.0,
                                soma_law=DEFAULT_SOMA_LAW)
    establish_deployed_wire_gauge(flow, input_data_scale=2.0, negative_boundary=seam)
    repr_ = flow.get_mapper_repr()
    consumers = repr_.consumer_map()
    baked = []
    for op in repr_.execution_order():
        if isinstance(op, ComputeOpMapper) and op.output_value_offset is not None:
            kappa = float(torch.as_tensor(op.output_scale).float().mean())
            s = float(op.output_value_offset) / max(kappa, 1e-12)
            for p in _entries(op, consumers):
                unbaked = PT.get_effective_bias(p) + (PT.get_effective_weight(p) * s).sum(dim=-1)
                baked.append((p, torch.tensor(s), unbaked.detach().clone()))
    assert len(baked) == 3, "fixture: one lifted signed BatchNorm per core"
    return flow, baked


class _LayerNormFC(nn.Module):
    """A signed host LayerNorm read by an fc core: Soft Core Mapping's own shift, per axon."""

    def __init__(self):
        super().__init__()
        self.fc1, self.r1 = nn.Linear(10, 24), nn.ReLU()
        self.ln = nn.LayerNorm(24)
        self.fc2, self.r2 = nn.Linear(24, 5), nn.ReLU()

    def forward(self, x):
        return self.r2(self.fc2(self.ln(self.r1(self.fc1(x)))))


def _shifted_fc():
    """The fc core behind a per-axon Soft Core Mapping shift: ``(flow, perceptron, s, b~_pre)``."""
    torch.manual_seed(4)
    flow = convert_torch_model(_LayerNormFC().eval(), (10,), 5, device="cpu").eval()
    (ln,) = [n for n in flow.get_mapper_repr().execution_order()
             if isinstance(n, ComputeOpMapper) and isinstance(getattr(n, "module", None), nn.LayerNorm)]
    p = flow.get_perceptrons()[-1]
    with torch.no_grad():
        p.layer.weight.mul_(1.7)
    mins = -torch.linspace(0.3, 2.0, 24)
    mins[3] = 0.4  # one axon reads no negative value: no shift there
    s = torch.clamp(-mins, min=0.0)
    unbaked = PT.get_effective_bias(p).detach().clone()  # b~_pre: the bias before the bake
    shifted = apply_negative_value_shifts(flow, {ln: mins})
    assert list(shifted) == [ln] and p._neg_shift_baked
    baked = PT.get_effective_bias(p) + (PT.get_effective_weight(p) * s).sum(dim=-1)
    torch.testing.assert_close(baked, unbaked, atol=1e-6, rtol=0)  # fixture: the bake is b~_pre - W~ s
    return flow, p, s, unbaked


def _compensation_error(p, s, unbaked) -> torch.Tensor:
    """|deployed bias - (b - Q(W~) s)| per neuron, on the weights the projection installed."""
    expected = unbaked - (PT.get_effective_weight(p) * s).sum(dim=-1)
    return (PT.get_effective_bias(p) - expected).abs()


# ---------------------------------------------------------------------------
# The bake records the shift it compensated (every caller)
# ---------------------------------------------------------------------------


def test_the_lift_bake_records_its_shift():
    _flow, baked = _lifted_pyramidnet()
    for p, s, _ in baked:
        assert p._neg_shift_baked
        torch.testing.assert_close(p.input_shift.double(), s.double().expand_as(p.input_shift), atol=1e-7, rtol=0)


def test_the_soft_core_mapping_shift_records_its_per_axon_shift():
    _flow, p, s, _ = _shifted_fc()
    torch.testing.assert_close(p.input_shift, s.to(p.input_shift.dtype), atol=0, rtol=0)


def test_a_direct_bake_records_and_stays_once_only():
    p = Perceptron(3, 4, normalization=nn.Identity())
    apply_negative_shift_bias(p, 0.25)
    bias = p.layer.bias.detach().clone()
    apply_negative_shift_bias(p, torch.zeros(4))
    assert float(p.input_shift) == 0.25 and torch.equal(p.layer.bias.detach(), bias)
    assert "input_shift" not in p.state_dict(), "the record leaves the state_dict layout unchanged"
    assert float(copy.deepcopy(p).input_shift) == 0.25


# ---------------------------------------------------------------------------
# T-B1, T-B2: after the 4-bit projection the bias compensates Q(W~), not W~
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_a_lifted_conv_entry_compensates_the_projected_weights(route):
    """T-B1. A lifted host op (sigma, kappa) feeding a conv core, baked by ``negative_boundary_seam._bake``: after the
    4-bit projection the effective bias equals ``b~_pre - Q(w~) s`` within the bias quantum. On R the error is
    ``(Q(w~) - w~) s`` (plus the ratio grid's quantum on the register route)."""
    flow, baked = _lifted_pyramidnet()
    _project(route, list(flow.get_perceptrons()))
    for p, s, unbaked in baked:
        err = _compensation_error(p, s, unbaked)
        assert float(err.max()) <= _tolerance(route, p), (getattr(p, "name", "?"), float(err.max()))


@pytest.mark.parametrize("route", ROUTES)
def test_a_per_axon_seam_shift_compensates_the_projected_weights(route):
    """T-B2. The same check for Soft Core Mapping's shift (``apply_negative_value_shifts``), per axon."""
    flow, p, s, unbaked = _shifted_fc()
    _project(route, list(flow.get_perceptrons()))
    err = _compensation_error(p, s, unbaked)
    assert float(err.max()) <= _tolerance(route, p), float(err.max())


def test_the_compensation_follows_the_projection_at_every_rate():
    """At rate 0.5 the projection installs halfway weights; the bias compensates exactly those, then takes its own
    halfway code (the law route's float bias isolates the compensation)."""
    flow, p, s, unbaked = _shifted_fc()
    _project("law", [p], rate=0.5)
    err = _compensation_error(p, s, unbaked)
    assert float(err.max()) <= 2e-5


def test_a_perceptron_without_a_shift_keeps_its_law_bias():
    """Guard (green before and after): no record, no compensation. The law route (every T-INT and T-VAL row: a float
    bias, no pre-WQ bake) leaves the bias as trained."""
    p = Perceptron(4, 6, normalization=nn.Identity())
    with torch.no_grad():
        p.layer.weight.mul_(0.3)
        p.layer.bias.copy_(torch.tensor([0.9, -0.4, 0.05, 0.6]))
    bias = p.layer.bias.detach().clone()
    _project("law", [p])
    assert torch.equal(p.layer.bias.detach(), bias)
    assert getattr(p, "input_shift", None) is None


# ---------------------------------------------------------------------------
# T-B3: re-projections through the transform trainer do not drift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ROUTES)
def test_re_projections_with_changed_float_weights_do_not_drift(route):
    """T-B3. The transform trainer projects a copy of its float master every step (``_update_and_transform_model``):
    after k steps that change the master's weights and bias, the deployed bias equals ``b~_pre(k) - Q(w~_k) s`` with
    ``b~_pre(k) = b~_k + w~_k s`` (the master's own function), the master never absorbs a compensation, and an
    unchanged master re-projects to the same parameters."""
    flow, main, s, _ = _shifted_fc()
    aux = copy.deepcopy(main)
    temp = copy.deepcopy(aux)
    trainer = PerceptronTransformTrainer.__new__(PerceptronTransformTrainer)
    trainer._perceptron_slots = [(main, aux, temp)]
    trainer.perceptron_transformation = _tuner(route)._mixed_transform(1.0)
    gen = torch.Generator().manual_seed(11)
    for _ in range(4):
        with torch.no_grad():
            aux.layer.weight.add_(0.02 * torch.randn(aux.layer.weight.shape, generator=gen))
            aux.layer.bias.add_(0.05 * torch.randn(aux.layer.bias.shape, generator=gen))
        master_bias = aux.layer.bias.detach().clone()
        unbaked = (PT.get_effective_bias(aux) + (PT.get_effective_weight(aux) * s).sum(dim=-1)).detach().clone()
        trainer._update_and_transform_model()
        assert torch.equal(aux.layer.bias.detach(), master_bias), "the master absorbs no compensation"
        err = _compensation_error(main, s, unbaked)
        assert float(err.max()) <= _tolerance(route, main), float(err.max())
    deployed = [t.detach().clone() for t in main.parameters()]
    trainer._update_and_transform_model()
    assert all(torch.equal(a, b.detach()) for a, b in zip(deployed, main.parameters())), "no drift"


# ---------------------------------------------------------------------------
# T-P (guard): NF and the deployed program stay count-exact through WQ and Soft Core Mapping
# ---------------------------------------------------------------------------


def _route_grids(route: str) -> ParameterGrids:
    """The export's grids on ``route`` where the tree declares the register route; R declares none."""
    if route == "register" and "register_bias" in getattr(ParameterGrids, "__dataclass_fields__", {}):
        return ParameterGrids(register_bias=True)
    return ParameterGrids()


@pytest.mark.parametrize("route", ("register", "rows"))
def test_nf_and_scm_are_count_exact_on_the_lifted_4bit_program(route):
    """T-P guard (green before and after, atol 0): the lifted 4-bit LIF program through WQ (the route's projection,
    with the compensation where the tree applies it) then Soft Core Mapping's sequence (the IR at the route's bias
    delivery, ``quantize_ir_graph`` with the route's grids, the integer membrane lattice): every core's window
    counts, NF against the deployed flow, are equal."""
    flow, _ = _lifted_pyramidnet()
    perceptrons = list(flow.get_perceptrons())
    _project(route, perceptrons)
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=float(Q_MAX), firing_mode="Default", max_axons=1024, max_neurons=1024,
                   hardware_bias=route == "register").map(repr_)
    quantize_ir_graph(ir, BITS, weight_quantization=True, grids=_route_grids(route))
    arm_integer_membrane_lattice(flow)
    declare_host_kernel_evaluation(ir, "per_sample")
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=True)
    program = Program("pyramidnet_wq", SHAPE, flow, repr_, ir, hybrid, perceptrons,
                      [float(p.activation_scale) for p in perceptrons], True)
    x = _inputs(5, 8)
    _, nf = nf_window_counts(program, x, "windowed_lif")
    _, scm = scm_window_counts(program, x, "windowed_lif")
    mism = count_mismatches(nf, scm)
    assert sum(int(c.size) for c in scm.values()) > 0
    assert mism == {pi: 0 for pi in mism}


# ---------------------------------------------------------------------------
# A model baked before the record (a cache of an earlier tree) is refused at WQ entry by name
# ---------------------------------------------------------------------------


class _Pipeline:
    def __init__(self, config, entries):
        self.config = config
        self.entries = dict(entries)
        self.reporter = SimpleNamespace(report=lambda *a, **k: None)

    def get_entry(self, _step, key):
        return self.entries[key]

    def add_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value

    def update_entry(self, _step, key, value, _strategy="basic"):
        self.entries[key] = value


def _wq_entry(monkeypatch, flow) -> dict:
    from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep

    captured: dict = {}
    monkeypatch.setattr(WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
    monkeypatch.setattr(WeightQuantizationStep, "_canonicalize_starved_bias_outliers", lambda self, model: None)
    config = {"device": "cpu", "weight_quantization": True, "weight_bits": BITS, "wq_two_scale_projection": True,
              "has_bias": True}
    WeightQuantizationStep(_Pipeline(config, {"model": flow, "adaptation_manager": object()})).process()
    return captured


def test_a_bake_without_its_record_is_refused_at_weight_quantization(monkeypatch):
    """A perceptron baked (``_neg_shift_baked``) by a tree that kept no record cannot have its compensation re-derived:
    Weight Quantization refuses it by name before any budget is spent, instead of deploying the stale bake."""
    from mimarsinan.common.refusal import ToolchainRefusal

    flow, p, _s, _ = _shifted_fc()
    del p._buffers["input_shift"]
    with pytest.raises(ToolchainRefusal, match="input_shift") as info:
        _wq_entry(monkeypatch, flow)
    assert getattr(p, "name", "?") in str(info.value)


def test_a_recorded_bake_enters_weight_quantization(monkeypatch):
    flow, _p, _s, _ = _shifted_fc()
    assert _wq_entry(monkeypatch, flow)["two_scale_projection"] is True
