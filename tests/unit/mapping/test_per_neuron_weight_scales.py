"""[U-PCH] Per-neuron weight scales as a declared platform capability (the weight study's P).

A chip description that declares a threshold register per neuron (``core_value_granularity: {"threshold":
"per_neuron"}``) lets Weight Quantization project each output neuron on its own scale ``s_c = max(1, floor(q_max /
max_j |w_cj|))`` with ``theta_c = s_c`` and a membrane lattice per neuron, declared by the row's
``weight_scale_granularity`` (absent: one scale per perceptron, byte for byte). On a core with one threshold register
the row is refused by name. The mapping (the IR's thresholds, the packed hard cores, neuron splitting), the static
program check, the NF and SCM twins and the chip backends carry the per-neuron thresholds through.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph, verify_ir_graph_quantized
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.mapping.support.host_kernel_evaluation import declare_host_kernel_evaluation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.models.perceptron_mixer.membrane_lattice import record_membrane_lattice_kind
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    assert_effective_parameters_on_chip_grid,
)
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from mimarsinan.spiking.lif_utils import arm_integer_membrane_lattice
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization as NAPQ,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

from unit.spiking import host_gather_forms as H

BITS = 4
Q_MAX = 2 ** (BITS - 1) - 1  # 7
PT = PerceptronTransformer()
CAPABLE = {"core_value_granularity": {"threshold": "per_neuron"}}
PER_NEURON = {"weight_scale_granularity": "per_neuron"}


def _thresholds_module():
    from mimarsinan.mapping.support import neuron_thresholds

    return neuron_thresholds


# ---------------------------------------------------------------------------
# pc-1: codes within q_max per neuron, each neuron on its own grid
# ---------------------------------------------------------------------------


def _uneven_perceptron(seed: int = 0, out: int = 6, fan_in: int = 10) -> Perceptron:
    """Rows whose peaks differ tenfold: one grid per perceptron starves the small rows."""
    gen = torch.Generator().manual_seed(seed)
    p = Perceptron(out, fan_in, normalization=nn.Identity())
    p.set_activation_scale(1.25)
    peaks = torch.logspace(-1.2, 0, out)
    w = (torch.rand(out, fan_in, generator=gen) * 2 - 1) * peaks[:, None]
    w[torch.arange(out), torch.arange(out) % fan_in] = peaks
    with torch.no_grad():
        p.layer.weight.copy_(w * 1.25)
        p.layer.bias.copy_((torch.rand(out, generator=gen) - 0.5) * peaks * 1.25)
    return p


@pytest.mark.parametrize("two_scale,register", [(False, False), (True, True), (True, False)])
def test_codes_lie_within_q_max_on_each_neurons_own_grid(two_scale, register):
    p = _uneven_perceptron()
    w_before = PT.get_effective_weight(p).detach().clone()
    NAPQ(BITS, "cpu", rate=1.0, two_scale=two_scale, register_bias=register, per_neuron=True).transform(p)
    scale = p.parameter_scale.detach()
    assert scale.shape == (6,), "one scale per output neuron"
    peaks_w = w_before.abs().amax(dim=1)
    if two_scale:
        expected = torch.clamp(torch.floor(Q_MAX / peaks_w), min=1.0)
    else:
        b = PT.get_effective_bias(p).detach()
        expected = scale  # shared grid: sized by max(|w_c|, |b_c|) per neuron, checked below
        assert bool((scale * torch.maximum(peaks_w, b.abs()) <= Q_MAX + 1e-4).all())
    assert torch.equal(scale, expected)
    assert len(set(scale.tolist())) > 1, "the fixture's rows take different scales"
    codes = PT.get_effective_weight(p).detach() * scale[:, None]
    assert torch.allclose(codes, codes.round(), atol=1e-4)
    assert float(codes.abs().max()) <= Q_MAX + 1e-4
    if two_scale:  # each neuron's peak code reaches its own grid's top within its scale's floor and the rounding
        assert bool((codes.abs().amax(dim=1) >= Q_MAX - Q_MAX / scale - 0.5).all())
    assert_effective_parameters_on_chip_grid(p, Q_MAX, register_bias=register)


def test_a_conv_perceptron_takes_one_scale_per_output_channel():
    _, flow, perceptrons, _ = H.lif_net("pool_downsample", 0, 1.3)
    for p in perceptrons:
        w = PT.get_effective_weight(p).detach().clone()
        NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=True, per_neuron=True).transform(p)
        scale = p.parameter_scale.detach()
        assert scale.shape == (w.shape[0],)
        expected = torch.clamp(torch.floor(Q_MAX / w.abs().reshape(w.shape[0], -1).amax(dim=1)), min=1.0)
        assert torch.equal(scale, expected)
        codes = PT.get_effective_weight(p).detach() * scale.reshape(-1, *([1] * (w.dim() - 1)))
        assert torch.allclose(codes, codes.round(), atol=1e-4) and float(codes.abs().max()) <= Q_MAX + 1e-4


# ---------------------------------------------------------------------------
# pc-5: the default unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("two_scale,register", [(False, False), (True, True)])
def test_the_default_projection_is_unchanged_byte_for_byte(two_scale, register):
    a, b = _uneven_perceptron(3), _uneven_perceptron(3)
    NAPQ(BITS, "cpu", rate=1.0, two_scale=two_scale, register_bias=register).transform(a)
    NAPQ(BITS, "cpu", rate=1.0, two_scale=two_scale, register_bias=register, per_neuron=False).transform(b)
    assert a.parameter_scale.dim() == 0
    for (name, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(x, y), name


def test_an_absent_declaration_reads_one_scale_per_perceptron():
    nt = _thresholds_module()
    assert nt.per_neuron_weight_scales({}, value_domain=False) is False
    assert nt.per_neuron_weight_scales({"weight_scale_granularity": "per_perceptron"}, value_domain=False) is False
    assert nt.per_neuron_weight_scales({**CAPABLE, **PER_NEURON}, value_domain=False) is True


def test_the_ir_threshold_stays_one_register_without_the_declaration():
    program = H.build("encoding_skip", 0, 1.3, "windowed_lif", wq=True)
    for node in program.ir.get_neural_cores():
        assert np.ndim(node.threshold) == 0


def test_the_registry_declares_the_key_with_the_default_absent():
    from mimarsinan.config_schema.registry import REGISTRY

    entry = REGISTRY["weight_scale_granularity"]
    assert tuple(entry.options) == ("per_perceptron", "per_neuron")
    assert "byte for byte" in entry.empty_means


# ---------------------------------------------------------------------------
# pc-4: the refusal by name
# ---------------------------------------------------------------------------


def test_a_chip_with_one_threshold_register_per_core_refuses_the_row_by_name():
    nt = _thresholds_module()
    for chip in ({}, {"core_value_granularity": {"threshold": "per_core"}}):
        with pytest.raises(nt.PerNeuronThresholdRefusal, match="core_value_granularity"):
            nt.per_neuron_weight_scales({**chip, **PER_NEURON}, value_domain=False)


def test_a_value_domain_core_refuses_the_row_by_name():
    nt = _thresholds_module()
    with pytest.raises(nt.PerNeuronThresholdRefusal, match="value-domain"):
        nt.per_neuron_weight_scales({**CAPABLE, **PER_NEURON}, value_domain=True)


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


def _flow():
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4)).eval()
    return convert_torch_model(model, (8,), 4, device="cpu").eval()


def _step(monkeypatch, **config) -> dict:
    captured: dict = {}
    monkeypatch.setattr(WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
    monkeypatch.setattr(WeightQuantizationStep, "_canonicalize_starved_bias_outliers", lambda self, model: None)
    base = {"device": "cpu", "weight_quantization": True, "weight_bits": BITS, "wq_two_scale_projection": True,
            "has_bias": True}
    WeightQuantizationStep(_Pipeline({**base, **config}, {"model": _flow(), "adaptation_manager": object()})).process()
    return captured


def test_the_weight_quantization_step_refuses_the_row_before_any_projection(monkeypatch):
    nt = _thresholds_module()
    with pytest.raises(nt.PerNeuronThresholdRefusal, match="one threshold register per core"):
        _step(monkeypatch, **PER_NEURON)


def test_the_weight_quantization_step_projects_per_neuron_on_a_capable_chip(monkeypatch):
    assert _step(monkeypatch, **CAPABLE, **PER_NEURON)["per_neuron"] is True
    assert _step(monkeypatch, **CAPABLE).get("per_neuron", False) is False


def test_a_consumer_with_one_register_per_core_refuses_a_per_neuron_core_by_name():
    nt = _thresholds_module()
    core = SimpleNamespace(name="c0", threshold=np.array([3.0, 5.0]))
    with pytest.raises(nt.PerNeuronThresholdRefusal, match="nevresim"):
        nt.core_threshold(core, "the nevresim chip export")
    assert nt.core_threshold(SimpleNamespace(name="c1", threshold=4.0), "the nevresim chip export") == 4.0


# ---------------------------------------------------------------------------
# pc-3: per-neuron ties decided exactly
# ---------------------------------------------------------------------------


def _tie_perceptron(scales, fan_in: int = 12, seed: int = 0):
    """A Linear perceptron whose effective weights are codes over each neuron's own scale: per-cycle charges on each
    neuron's 1/s_c lattice, many of them reaching the threshold exactly."""
    gen = torch.Generator().manual_seed(seed)
    out = len(scales)
    p = Perceptron(out, fan_in, normalization=nn.Identity())
    p.set_activation_scale(1.0)
    codes = torch.randint(0, 3, (out, fan_in), generator=gen).float()
    with torch.no_grad():
        p.layer.weight.copy_(codes / torch.tensor(scales, dtype=torch.float32)[:, None])
        p.layer.bias.zero_()
    p.set_parameter_scale(torch.tensor(scales, dtype=torch.float32))
    lif = LIFActivation(T=H.T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    return p, codes.long()


def _integer_counts(codes: torch.Tensor, scales, spikes: torch.Tensor) -> torch.Tensor:
    """The chip's window counts: integer membranes, threshold s_c, '<=' comparator, subtractive reset."""
    s = torch.tensor(scales, dtype=torch.int64)
    memb = torch.zeros(spikes.shape[1], codes.shape[0], dtype=torch.int64)
    count = torch.zeros_like(memb)
    for t in range(spikes.shape[0]):
        memb += spikes[t].long() @ codes.T
        fired = memb >= s
        count += fired.long()
        memb -= fired.long() * s
    return count


def _float_counts(p, spikes: torch.Tensor) -> torch.Tensor:
    lif = p.activation
    lif.set_cycle_accurate(True)
    total = 0
    for t in range(spikes.shape[0]):
        total = total + lif(p.layer(spikes[t]))
    lif.set_cycle_accurate(False)
    return torch.round(total).long()


SCALES = [7.0, 10.0, 13.0, 29.0, 3.0, 18.0]


def test_per_neuron_ties_are_decided_exactly_in_the_measurement_plane():
    p, codes = _tie_perceptron(SCALES)
    model = SimpleNamespace(get_perceptrons=lambda: [p])
    record_membrane_lattice_kind(p)  # [U-POST2] the projection's record: every arming derives the lattice
    assert arm_integer_membrane_lattice(model) == 1
    gen = torch.Generator().manual_seed(4)
    spikes = (torch.rand(H.T, 4000, 12, generator=gen) < 0.35).float()
    reference = _integer_counts(codes, SCALES, spikes)
    with torch.no_grad(), measurement_plane():
        exact = _float_counts(p, spikes)
    assert torch.equal(exact, reference)


def test_one_lattice_for_every_neuron_does_not_decide_per_neuron_ties():
    """The guard the per-neuron lattice exists for: the free plane (float dust) and one shared lattice both miss."""
    p, codes = _tie_perceptron(SCALES)
    gen = torch.Generator().manual_seed(4)
    spikes = (torch.rand(H.T, 4000, 12, generator=gen) < 0.35).float()
    reference = _integer_counts(codes, SCALES, spikes)
    with torch.no_grad():
        free = _float_counts(p, spikes)
    p.activation.set_membrane_lattice(max(SCALES))
    with torch.no_grad(), measurement_plane():
        shared = _float_counts(p, spikes)
    assert not torch.equal(free, reference), "the fixture's ties are decided by float dust in the free plane"
    assert not torch.equal(shared, reference)


# ---------------------------------------------------------------------------
# pc-2: the NF and SCM twins count for count, the IR's per-neuron thresholds
# ---------------------------------------------------------------------------


#: The bias routes: ``(two_scale, on-chip bias register)``: the on-chip bias register, bias rows on their own
#: grid (two-scale, param-encoded), and the shared grid (bias rows on the weight grid).
ROUTES = {"register": (True, True), "rows": (True, False), "shared": (False, False)}


def _per_neuron_sequence(flow, perceptrons, *, retimed: bool, register, geometry: dict):
    two_scale, on_chip = ROUTES[register] if isinstance(register, str) else (register, register)
    repr_ = flow.get_mapper_repr()
    compute_per_source_scales(repr_, arm_wire_value_ops=True)
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    for p in perceptrons:
        NAPQ(BITS, "cpu", rate=1.0, two_scale=two_scale, register_bias=on_chip, per_neuron=True).transform(p)
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=float(Q_MAX), firing_mode="Default", hardware_bias=on_chip, **geometry).map(repr_)
    quantize_ir_graph(ir, BITS, weight_quantization=True, grids=ParameterGrids(register_bias=on_chip))
    arm_integer_membrane_lattice(flow)
    declare_host_kernel_evaluation(ir, "per_sample")
    return repr_, ir


def _program(form: str, seed: int, *, register, geometry: dict, semantics: str = "windowed_lif"):
    retimed = H.SEMANTICS[semantics][0]
    shape, flow, perceptrons, thetas = H.lif_net(form, seed, 1.3)
    if register == "rows":  # biases large enough that a neuron's bias takes several rows
        with torch.no_grad():
            for p in perceptrons:
                if p.layer.bias is not None:
                    p.layer.bias.mul_(8.0)
    repr_, ir = _per_neuron_sequence(flow, perceptrons, retimed=retimed, register=register, geometry=geometry)
    hybrid = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed)
    return H.Program(form, shape, flow, repr_, ir, hybrid, perceptrons, thetas, True)


GEOMETRIES = (dict(max_axons=256, max_neurons=256), dict(max_axons=256, max_neurons=4))


def test_the_projection_stamps_each_neurons_lattice():
    from mimarsinan.models.nn.lif_kernels import NeuronLattice
    from mimarsinan.models.perceptron_mixer.membrane_lattice import CORE_LATTICE

    p, _ = _tie_perceptron(SCALES)
    p.set_parameter_scale(torch.tensor(1.0))
    NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=True, per_neuron=True).transform(p)
    assert p.membrane_lattice_kind == CORE_LATTICE  # [U-POST2] the projection records; the arming derives
    assert arm_integer_membrane_lattice(SimpleNamespace(get_perceptrons=lambda: [p])) == 1
    lattice = p.activation.if_node.lattice_scale
    assert isinstance(lattice, NeuronLattice)
    assert torch.equal(lattice.scales, 2.0 * p.parameter_scale.detach().double())


def test_an_encoding_layers_neurons_take_their_lattice_times_t():
    from mimarsinan.models.nn.lif_kernels import NeuronLattice
    from mimarsinan.models.perceptron_mixer.membrane_lattice import membrane_lattice_theta

    from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer

    p, _ = _tie_perceptron(SCALES)  # its persisted scale: one entry per neuron
    p.is_encoding_layer = True  # [U-PCH2] an encoder whose input carries the window's 1/T entry grid
    p.append_input_wire_op(ChipInputQuantizer(T=int(p.activation.T), activation_scale=1.0))
    record_membrane_lattice_kind(p)  # [U-POST2] the record; the lattice is derived from the persisted scale
    lattice = membrane_lattice_theta(p)
    assert isinstance(lattice, NeuronLattice)
    assert torch.equal(lattice.scales, torch.tensor(SCALES, dtype=torch.float64) * H.T)
    p.is_encoding_layer = False
    record_membrane_lattice_kind(p)
    assert torch.equal(membrane_lattice_theta(p).scales, torch.tensor(SCALES, dtype=torch.float64))


def test_the_membrane_readout_reads_each_neurons_threshold():
    from mimarsinan.models.spiking.hybrid.membrane_readout import accumulate_membrane_correction

    memb = torch.tensor([[3.0, 10.0, 6.0, 9.0]], dtype=torch.float64)
    thetas = torch.tensor([2.0, 5.0, 3.0, 9.0], dtype=torch.float64)
    span = SimpleNamespace(kind="core", dst_start=0, dst_end=3, src_core=0, src_start=1, src_end=4)
    correction = torch.zeros(1, 3, dtype=torch.float64)
    accumulate_membrane_correction(correction, output_spans=[span], neuron_states=[{"memb": memb}],
                                   thresholds=[thetas], eligible_ranges=[(0, 3)], half_step_charge=0.0)
    assert torch.equal(correction, (memb[:, 1:4] / thetas[1:4]))


@pytest.mark.parametrize("rate", [0.5, 1.0])
def test_the_tuners_rate_mixing_takes_a_per_neuron_grid(rate):
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner as Tuner,
    )

    p = _uneven_perceptron()
    tuner = Tuner.__new__(Tuner)
    tuner.quantization_bits, tuner.two_scale_projection, tuner.fractional_scale = BITS, True, False
    tuner.float_bias, tuner.bias_rows_floor, tuner.register_bias, tuner.per_neuron = False, None, True, True
    tuner.pipeline = SimpleNamespace(config={"device": "cpu"})
    torch.manual_seed(0)
    with torch.no_grad():
        tuner._mixed_transform(rate)(p)
    assert p.parameter_scale.shape == (6,)
    if rate == 1.0:
        assert_effective_parameters_on_chip_grid(p, Q_MAX, register_bias=True)


@pytest.mark.parametrize("form", ["encoding_skip", "pool_downsample", "token_skip"])
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=["whole", "tiled"])
def test_nf_and_scm_count_for_count_under_per_neuron_scales(form, route, geometry):
    program = _program(form, 0, register=route, geometry=geometry)
    per_neuron = [n for n in program.ir.get_neural_cores() if np.ndim(n.threshold) == 1]
    assert per_neuron, "the cores carry per-neuron thresholds"
    assert any(len(set(np.asarray(n.threshold).tolist())) > 1 for n in per_neuron)
    if route == "rows":  # the rows take the largest neuron's count: the fixture's biases need more than one
        assert any(sum(s.is_always_on() for s in n.input_sources.flatten()) > 1 for n in per_neuron)
    mism, compared, out_diff = H.compare(program, "windowed_lif", n=24)
    assert compared > 0
    assert all(v == 0 for v in mism.values()), mism
    assert out_diff < 1e-3, out_diff  # the output decode's float rounding, the forms' own bound (count grid units)


def test_each_core_threshold_is_its_neurons_scale():
    program = _program("pool_downsample", 1, register=True, geometry=GEOMETRIES[1])
    scales = {i: p.parameter_scale.detach().numpy() for i, p in enumerate(program.perceptrons)}
    checked = 0
    for node in program.ir.get_neural_cores():
        lo, hi = node.perceptron_output_slice
        assert np.array_equal(np.asarray(node.threshold), scales[node.perceptron_index][lo:hi])
        assert float(node.parameter_scale) == 1.0
        checked += 1
    assert checked > 0
    verify_ir_graph_quantized(program.ir, BITS)


def test_bank_instances_take_their_rows_neurons_thresholds():
    """A weight bank's instances on row windows of it (``weight_row_slice``) each hold their own neurons' registers."""
    from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank

    rng = np.random.default_rng(7)
    scale = torch.tensor([5.0, 6.0, 7.0, 8.0])
    codes = rng.integers(-7, 8, size=(3, 4)).astype(np.float64)
    bank = WeightBank(id=0, core_matrix=codes / scale.numpy()[None, :], parameter_scale=scale, perceptron_index=0)
    nodes = [NeuralCore(id=i, name=f"n{i}", input_sources=np.array([IRSource(-2, j) for j in range(3)], dtype=object),
                        core_matrix=None, weight_bank_id=0, weight_row_slice=rows, perceptron_index=0,
                        perceptron_output_slice=rows, parameter_scale=scale, latency=0)
             for i, rows in enumerate([(0, 2), (2, 4)])]
    ir = IRGraph(nodes=nodes, output_sources=np.array([IRSource(n.id, j) for n in nodes for j in range(2)],
                                                      dtype=object), weight_banks={0: bank})
    quantize_ir_graph(ir, BITS, weight_quantization=True)
    assert np.array_equal(np.asarray(nodes[0].threshold), [5.0, 6.0])
    assert np.array_equal(np.asarray(nodes[1].threshold), [7.0, 8.0])
    assert np.array_equal(ir.weight_banks[0].core_matrix, codes)


def test_ir_pruning_refuses_a_per_neuron_program_by_name():
    """IR pruning's constant folds and liveness read one threshold per core: a per-neuron program is refused."""
    from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph

    nt = _thresholds_module()
    program = _program("encoding_skip", 0, register="register", geometry=GEOMETRIES[0])
    with pytest.raises(nt.PerNeuronThresholdRefusal, match="IR pruning"):
        prune_ir_graph(program.ir, simulation_steps=H.T, spiking_mode="lif")


def test_the_parity_gate_reads_zero_flips():
    from mimarsinan.pipelining.core.nf_scm_parity import assert_torch_vs_deployed_sim_parity_or_raise
    from mimarsinan.spiking.segment_forward import SegmentForwardDriver

    program = _program("pool_downsample", 2, register=True, geometry=GEOMETRIES[0])
    flow = H.deployed_flow(program, "windowed_lif")

    class _Twin(nn.Module):
        def forward(self, x):
            return SegmentForwardDriver(program.repr_, H.T, H.SEMANTICS["windowed_lif"][2]())(x)

    x = H.samples(program, 32, 3)
    labels = torch.zeros(32, dtype=torch.long)
    assert assert_torch_vs_deployed_sim_parity_or_raise(_Twin(), flow, x, min_agreement=1.0, labels=labels) == 1.0


def test_the_packed_program_counts_as_the_identity_program():
    """Hard cores holding soft cores of several thresholds per neuron, and soft cores split across hard cores."""
    program = _program("pool_downsample", 0, register=True, geometry=GEOMETRIES[1])
    packed = build_hybrid_hard_core_mapping(
        ir_graph=program.ir, cores_config=[{"max_axons": 256, "max_neurons": 7, "count": 400, "has_bias": True}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=True, allow_coalescing=False),
        retimed_level_stages=True)
    per_neuron_cores = [c for s in packed.stages if s.kind == "neural" for c in s.hard_core_mapping.cores
                        if np.ndim(c.threshold) == 1]
    assert per_neuron_cores, "the packed cores carry per-neuron thresholds"
    x = H.samples(program, 16, 5)
    identity = H.deployed_flow(program, "windowed_lif")
    deployed = SpikingHybridCoreFlow(program.shape, packed, simulation_length=H.T, thresholding_mode="<=",
                                     membrane_integer_lattice=True, **H.SEMANTICS["windowed_lif"][1]).eval()
    with torch.no_grad():
        assert torch.equal(identity(x), deployed(x))


# ---------------------------------------------------------------------------
# The hard core, neuron splitting and the static program check carry per-neuron thresholds
# ---------------------------------------------------------------------------


def _soft(core_id: int, thresholds, axons: int = 3):
    from mimarsinan.mapping.packing.softcore import SoftCore

    width = len(thresholds)
    sc = SoftCore(core_matrix=np.ones((axons, width)), axon_sources=[], id=core_id)
    sc.threshold = np.asarray(thresholds, dtype=np.float64)
    return sc


def test_a_hard_core_holds_each_resident_neurons_threshold():
    from mimarsinan.mapping.packing.softcore import HardCore

    hc = HardCore(8, 8)
    hc.add_softcore(_soft(0, [3.0, 5.0, 7.0]))
    hc.add_softcore(_soft(1, [11.0, 2.0]))
    assert np.array_equal(np.asarray(hc.threshold)[:5], [3.0, 5.0, 7.0, 11.0, 2.0])


def test_a_split_soft_core_keeps_each_fragments_thresholds():
    from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments

    parent = _soft(0, [3.0, 5.0, 7.0, 11.0, 2.0])
    parent.axon_sources = []
    first, rest = make_real_fragments(softcore=parent, first_neurons=2, remaining_neurons=3, group_id=0, memo={})
    assert np.array_equal(first.threshold, [3.0, 5.0])
    assert np.array_equal(rest.threshold, [7.0, 11.0, 2.0])


def _per_neuron_integer_ir():
    from unit.certification.program_check import programs as P
    from unit.mapping.residency.executed import vehicles as V

    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS).map(
        V.random_rep(0))
    widths: dict = {}
    for node in ir.get_neural_cores():
        widths[node.perceptron_index] = max(widths.get(node.perceptron_index, 0), node.perceptron_output_slice[1])
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        width = widths[owner.perceptron_index]
        owner.parameter_scale = torch.tensor([float(40 + 7 * (c % 5)) for c in range(width)])
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    return P, V, ir


def test_the_static_program_check_establishes_theorem_1_with_per_neuron_thresholds():
    P, V, ir = _per_neuron_integer_ir()
    assert any(np.ndim(n.threshold) == 1 for n in ir.get_neural_cores())
    program = V.build(ir, 3, None)
    report = P.run(program, ir, P.decl("streamed_lif"))
    assert report.established("theorem1"), {c: [v.text() for v in r.violations]
                                            for c, r in report.results.items() if r.violations}
    assert report.results["H1-theta"].checked > 0


def test_the_static_program_check_sees_one_neurons_threshold_moved():
    P, V, ir = _per_neuron_integer_ir()
    program = P.clone(V.build(ir, 3, None))
    core = next(c for s in P.neural(program) for c in s.hard_core_mapping.cores if np.ndim(c.threshold) == 1)
    moved = np.array(core.threshold, dtype=np.float64)
    moved[0] += 1.0
    core.threshold = moved
    report = P.run(program, ir, P.decl("streamed_lif"))
    assert any(v.rule == "threshold" for v in report.results["H1-theta"].violations)


# ---------------------------------------------------------------------------
# The scale's shape follows its projection through the tuner's state copies
# ---------------------------------------------------------------------------


def test_a_state_of_the_other_granularity_loads_into_a_perceptron():
    p = _uneven_perceptron()
    before = copy.deepcopy(p.state_dict())
    NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=True, per_neuron=True).transform(p)
    after = copy.deepcopy(p.state_dict())
    p.load_state_dict(before)
    assert p.parameter_scale.dim() == 0
    p.load_state_dict(after)
    assert p.parameter_scale.shape == (6,)


def test_the_transform_trainer_hands_the_main_model_the_projected_scales():
    from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer

    p = _uneven_perceptron()
    main, aux, temp = p, copy.deepcopy(p), copy.deepcopy(p)
    trainer = PerceptronTransformTrainer.__new__(PerceptronTransformTrainer)
    trainer._perceptron_slots = [(main, aux, temp)]
    trainer.perceptron_transformation = NAPQ(BITS, "cpu", rate=1.0, two_scale=True, register_bias=True,
                                             per_neuron=True).transform
    with torch.no_grad():
        trainer._update_and_transform_model()
    assert main.parameter_scale.shape == (6,)
    assert aux.parameter_scale.dim() == 0
    with torch.no_grad():
        trainer._update_and_transform_model()
    assert main.parameter_scale.shape == (6,)
