"""[U-PCH2] Every static-check rule that reads a per-core value reads only the values it concerns, each in the layout
its holder stores it (one rule: [U-PCH3] the layouts its holder declares, ``mapping.support.value_layouts``).

- H1-theta: a value a hard core assembles per neuron column (declared per neuron, the per-neuron threshold registers) is
  read at the placement's columns; every other per-core value is stored once per core and compared whole, so a
  default program whose perceptrons carry one activation scale per channel establishes Theorem 1 as before.
- H8 reads every register a core holds for its charge bound (the largest), H8-staircase each register's lattice.
- H10 compares a resident pass with its programming pass on every used neuron's register.
- ir-tiling reads an IR node's value per neuron: a per-column value (one register per neuron column) at the columns, a
  perceptron's per-channel value (held whole by every node of the perceptron) at the columns' channels.
- N-identity's offset rows hold one register per core and refuse a per-neuron threshold by name.
- seam-currency reads every channel of a per-channel gauge: ``test_verdicts``'s per-channel entry currency test.

The per-channel activation scales are a per-channel theta promotion's form (``set_activation_scale`` with one theta
per output channel). The programs are the static check's vehicles and the toolchain's per-channel theta vehicle
(``tests/integration/test_per_channel_theta_deployed_parity``)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.certification.program_check.report import ConditionResult, ESTABLISHED, VACUOUS
from mimarsinan.certification.program_check.unit_forms import pruned_rows
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.mapping.residency.executed import vehicles as V

UNBOUNDED = 10 ** 6


def _violations(report) -> dict:
    return {c: [v.text() for v in r.violations][:3] for c, r in report.results.items() if r.violations}


def _channel_scales(rep):
    """One activation scale per output channel on every perceptron (a per-channel theta promotion's form)."""
    for p in rep.get_perceptrons():
        p.set_activation_scale(torch.tensor([1.0 + 0.0625 * c for c in range(int(p.layer.weight.shape[0]))]))
    return rep


def _integer_ir(rep, neurons: int, *, per_neuron: bool = False):
    """``rep``'s IR at ``neurons`` columns per node (``UNBOUNDED``: the IR at unbounded geometry) on the integer
    lattice: one grid per perceptron, or one per output channel (``40 + 7 (c mod 5)``, theta_c = s_c)."""
    axons = UNBOUNDED if neurons >= UNBOUNDED else V.MAX_AXONS
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=axons, max_neurons=neurons).map(rep)
    widths: dict = {}
    for node in ir.get_neural_cores():
        widths[node.perceptron_index] = max(widths.get(node.perceptron_index, 0), node.perceptron_output_slice[1])
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        owner.parameter_scale = (torch.tensor([float(40 + 7 * (c % 5)) for c in range(widths[owner.perceptron_index])])
                                 if per_neuron else torch.tensor(0.0))
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    return ir


def _offsets(program) -> list:
    return [int(p["neuron_offset"]) for s in P.neural(program)
            for placed in s.hard_core_mapping.soft_core_placements_per_hard_core for p in placed]


# ---------------------------------------------------------------------------
# H1-theta: the threshold at the placement's columns, every other per-core value whole
# ---------------------------------------------------------------------------

CORES = {"max_axons": 512, "max_neurons": 512, "count": 2000}


def _per_channel_theta_vehicle():
    """The toolchain's per-channel theta deployment vehicle (``test_per_channel_theta_deployed_parity._build``):
    mmixcore under LIF, every eligible perceptron promoted to unequal per-channel theta, its IR put on the integer
    lattice (the vehicles' fallback grid) and packed onto 512 x 512 cores with neuron splitting."""
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
    from mimarsinan.spiking.per_channel_theta import eligible_per_channel_perceptrons
    from mimarsinan.spiking.scale_aware_boundaries import (
        calibrate_scale_aware_boundaries, propagate_boundary_input_scales)
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

    shape = (1, 28, 28)
    torch.manual_seed(0)
    model = TorchMLPMixerCore(input_shape=shape, num_classes=10, patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8,
                              fc_w_2=6).eval()
    flow = convert_torch_model(model, input_shape=shape, num_classes=10).eval()
    repr_ = flow.get_mapper_repr()
    perceptrons = list(flow.get_perceptrons())
    for p in perceptrons:
        p.base_activation = p.activation = LIFActivation(T=4, activation_scale=p.activation_scale)
    mark_encoding_layers(repr_)
    repr_.assign_perceptron_indices()
    maxima, handles = {}, []
    for i, p in enumerate(perceptrons):
        handles.append(p.activation.register_forward_hook(
            lambda _m, _i, out, idx=i: maxima.__setitem__(idx, out.detach().abs().max().item())))
    torch.manual_seed(7)
    with torch.no_grad():
        flow(torch.rand(8, *shape))
    for h in handles:
        h.remove()
    calibrate_scale_aware_boundaries(flow, [max(maxima[i], 1e-3) for i in range(len(perceptrons))],
                                     input_data_scale=1.0)
    promoted = list(eligible_per_channel_perceptrons(flow).values())
    for p in promoted:
        p.set_activation_scale(float(p.activation_scale) * torch.linspace(0.6, 1.6, p.output_channels))
    propagate_boundary_input_scales(flow, input_data_scale=1.0)
    compute_per_source_scales(repr_)
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512).map(repr_)
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        owner.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    program = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[dict(CORES)], strategy=MappingStrategy.
                                             from_permissions(allow_neuron_splitting=True, allow_coalescing=False))
    return ir, program, promoted


def test_h1_theta_establishes_theorem_1_on_the_per_channel_theta_vehicle():
    ir, program, promoted = _per_channel_theta_vehicle()
    assert promoted and any(np.ndim(n.activation_scale) >= 1 for n in ir.get_neural_cores())
    assert max(_offsets(program)) > 0, "the vehicle places residents at a neuron offset"
    decl = CheckDeclaration(semantics="streamed_lif", window=4, pass_transfer=VERBATIM, core_types=(dict(CORES),))
    report = check_deployment(program, ir_graph=ir, declaration=decl)
    h1 = report.results["H1-theta"]
    assert h1.checked > 0 and not h1.violations, _violations(report)
    assert report.established("theorem1"), _violations(report)


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_h1_theta_compares_a_per_channel_activation_scale_whole(seed):
    ir = _integer_ir(_channel_scales(V.random_rep(seed)), 4)
    program = V.build(ir, 4, None)
    assert max(_offsets(program)) > 0
    report = P.run(program, ir, P.decl("streamed_lif", count=4))
    assert report.results["H1-theta"].checked > 0 and not report.results["H1-theta"].violations, _violations(report)
    assert report.established("theorem1"), _violations(report)


def test_h1_theta_still_sees_a_per_channel_activation_scale_moved_on_one_core():
    ir = _integer_ir(_channel_scales(V.random_rep(0)), 4)
    program = P.clone(V.build(ir, 4, None))
    core = next(c for s in P.neural(program) for c in s.hard_core_mapping.cores
                if torch.as_tensor(c.activation_scale).numel() > 1)
    moved = torch.as_tensor(core.activation_scale, dtype=torch.float64).clone()
    moved[-1] += 0.5
    core.activation_scale = moved
    report = P.run(program, ir, P.decl("streamed_lif", count=4))
    assert "activation_scale" in {v.rule for v in report.results["H1-theta"].violations}


def test_the_values_read_per_column_are_the_values_a_hard_core_assembles_per_column():
    from mimarsinan.mapping.packing.softcore import HardCore, SoftCore
    from mimarsinan.mapping.platform.core_residency import ALL_SINGLETON_NAMES
    from mimarsinan.mapping.support.value_layouts import ValueLayout, declared_names, placement_value

    hard = HardCore(8, 8)
    for index, (thetas, scale) in enumerate((([3.0, 5.0, 7.0], 1.5), ([11.0, 2.0], 1.5))):
        soft = SoftCore(core_matrix=np.ones((3, len(thetas))), axon_sources=[], id=index)
        soft.threshold, soft.activation_scale = np.asarray(thetas), torch.tensor([scale, 2.0, 2.5])
        hard.add_softcore(soft)
    # [U-PCH3] the hard core's declaration is the one table: its per-neuron singletons are the threshold registers
    assert tuple(n for n in declared_names(HardCore, ValueLayout.PER_NEURON) if n in ALL_SINGLETON_NAMES) == \
        ("threshold",)
    assert np.array_equal(placement_value(hard, "threshold", 3, 5), [11.0, 2.0])
    assert torch.equal(placement_value(hard, "activation_scale", 3, 5), torch.tensor([1.5, 2.0, 2.5]))


# ---------------------------------------------------------------------------
# H8 and H8-staircase: every register a core holds
# ---------------------------------------------------------------------------


def _per_neuron_ir_with(edit):
    """The vehicles' per-neuron IR (seed 0, IR nodes of 8 columns) with one node's registers edited before the
    program is built (the program then deploys the edited IR faithfully)."""
    ir = _integer_ir(V.random_rep(0), V.MAX_NEURONS, per_neuron=True)
    node = next(n for n in ir.get_neural_cores() if np.ndim(n.threshold) == 1 and np.size(n.threshold) > 2)
    thresholds = np.array(node.threshold, dtype=np.float64)
    edit(thresholds)
    node.threshold = thresholds
    return ir, node


def test_h8_bounds_the_charge_by_the_largest_threshold_register():
    big = float(2 ** 30)
    ir, node = _per_neuron_ir_with(lambda t: t.__setitem__(len(t) - 1, big))
    report = P.run(V.build(ir, 3, None), ir, P.decl("streamed_lif"))
    assert float(np.asarray(node.threshold)[0]) < big
    assert report.counts["h8_charge_bound"] >= big


def test_h8_staircase_reads_every_threshold_registers_lattice():
    ir, _node = _per_neuron_ir_with(lambda t: t.__setitem__(len(t) - 1, t[-1] + 0.5))
    report = P.run(V.build(ir, 3, None), ir, P.decl("streamed_lif"))
    assert "off_lattice" in {v.rule for v in report.results["H8-staircase"].violations}, _violations(report)


# ---------------------------------------------------------------------------
# H10: a resident pass on every used neuron's register
# ---------------------------------------------------------------------------


def _per_neuron_resident_chain():
    """``programs.resident_chain`` with one threshold register per neuron (each bank's instances alike)."""
    from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
    from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities

    rng = np.random.default_rng(19)
    banks = {b: WeightBank(id=b, core_matrix=np.round(rng.normal(size=(5, 4)) * 9 + 10 * (b + 1)).astype(np.float64))
             for b in range(2)}
    nodes = []
    for b in range(2):
        for inst in range(3):
            srcs = np.array([IRSource(-2, (b * 3 + inst) * 4 + i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
            nodes.append(NeuralCore(id=len(nodes), name=f"b{b}_i{inst}", input_sources=srcs, core_matrix=None,
                                    weight_bank_id=b, weight_row_slice=(0, 4), perceptron_index=b,
                                    perceptron_output_column=inst, latency=0,
                                    threshold=np.array([7.0, 8.0, 9.0, 10.0]) + 5 * b,
                                    hardware_bias=np.full(4, float(b + 1) * 3)))
    ir = IRGraph(nodes=nodes, output_sources=np.array([IRSource(n.id, j) for n in nodes for j in range(4)],
                                                      dtype=object), weight_banks=banks)
    program = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 8, "max_neurons": 8, "count": 2}],
        strategy=MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True)))
    return program, ir


def test_h10_compares_every_used_neurons_register_of_a_resident_pass():
    program, ir = _per_neuron_resident_chain()
    decl = P.decl("synchronized_lif", core_types=({"max_axons": 8, "max_neurons": 8, "count": 2},))
    clean = P.run(program, ir, decl).results["H10"]
    assert clean.checked > 0 and not clean.violations, [v.text() for v in clean.violations]
    edited = P.clone(program)
    stage = next(s for s in P.neural(edited) if getattr(s, "schedule_weights_resident", False))
    core = next(c for c in stage.hard_core_mapping.cores if np.ndim(c.threshold) == 1)
    moved = np.array(core.threshold, dtype=np.float64)
    moved[0] -= 1.0  # the first register, not the largest of the core's used neurons
    core.threshold = moved
    report = P.run(edited, ir, decl)
    assert "resident_content" in {v.rule for v in report.results["H10"].violations}, _violations(report)


# ---------------------------------------------------------------------------
# ir-tiling: per neuron, each value in its holder's layout
# ---------------------------------------------------------------------------


def test_an_ir_nodes_value_is_read_in_its_holders_layout():
    # [U-PCH3] by the node's declared layout (``NeuralCore.VALUE_LAYOUTS``), never by the value's size
    from mimarsinan.mapping.ir import IRSource, NeuralCore
    from mimarsinan.mapping.support.value_layouts import ValueLayoutError, neuron_entries

    tile = NeuralCore(id=0, name="t", input_sources=np.array([IRSource(-2, 0)], dtype=object),
                      core_matrix=np.ones((1, 4)), perceptron_index=0, perceptron_output_slice=(4, 8))
    columns, channels = np.array([1, 2]), np.array([4, 5, 6, 7])  # a tile of four columns at channels 4 to 7

    def read(name, value, table=channels):
        return neuron_entries(tile, name, columns, width=4, channels=table, value=value)

    assert read("threshold", 7.0).tolist() == [7.0, 7.0]  # one value, broadcast
    assert read("threshold", np.array([3.0, 5.0, 7.0, 9.0])).tolist() == [5.0, 7.0]  # per column
    perceptron = np.arange(10.0) * 0.5  # one entry per channel of a ten-channel perceptron, held whole
    assert read("activation_scale", perceptron).tolist() == [2.5, 3.0]  # at the columns' channels
    with pytest.raises(ValueLayoutError):  # no channel identity: its columns are its channels, one entry each
        read("activation_scale", perceptron, None)
    assert read("threshold", None) is None


def _tiling_report(rep, *, per_neuron: bool = False, edit=None):
    tiled = _integer_ir(copy.deepcopy(rep), 4, per_neuron=per_neuron)
    reference = _integer_ir(copy.deepcopy(rep), UNBOUNDED, per_neuron=per_neuron)
    if edit is not None:
        edit(tiled)
    report = P.run(V.build(tiled, 4, None), tiled, P.decl("streamed_lif", count=4), unbounded_ir=reference)
    return report, tiled


def test_ir_tiling_compares_each_columns_threshold_register():
    def move(ir):
        node = next(n for n in ir.get_neural_cores() if np.ndim(n.threshold) == 1
                    and np.size(n.threshold) > 1 and float(np.asarray(n.threshold)[0]) < np.max(n.threshold))
        moved = np.array(node.threshold, dtype=np.float64)
        moved[0] += 1.0
        node.threshold = moved

    clean, _ = _tiling_report(V.random_rep(0), per_neuron=True)
    assert clean.results["ir-tiling"].status == ESTABLISHED, _violations(clean)
    report, _ = _tiling_report(V.random_rep(0), per_neuron=True, edit=move)
    assert "threshold" in {v.rule for v in report.results["ir-tiling"].violations}, _violations(report)


@pytest.mark.parametrize("seed", [0, 1])
def test_ir_tiling_reads_a_perceptrons_per_channel_scale_at_its_columns_channels(seed):
    report, tiled = _tiling_report(_channel_scales(V.random_rep(seed)))
    assert any(torch.as_tensor(n.activation_scale).numel() > n.get_output_count() for n in tiled.get_neural_cores())
    result = report.results["ir-tiling"]
    assert result.status == ESTABLISHED and result.checked > 0, _violations(report)
    assert report.established("theorem1"), _violations(report)


def test_ir_tiling_sees_one_channels_scale_moved_on_a_tile():
    def move(ir):
        node = next(n for n in ir.get_neural_cores() if int(n.perceptron_output_slice[0]) > 0
                    and torch.as_tensor(n.activation_scale).numel() > n.get_output_count())
        moved = torch.as_tensor(node.activation_scale, dtype=torch.float32).clone()
        moved[int(node.perceptron_output_slice[0])] += 0.5
        node.activation_scale = moved

    report, _ = _tiling_report(_channel_scales(V.random_rep(0)), edit=move)
    assert "neuron_scale" in {v.rule for v in report.results["ir-tiling"].violations}, _violations(report)


# ---------------------------------------------------------------------------
# N-identity's offset rows: one register per core, a per-neuron one refused by name
# ---------------------------------------------------------------------------


def test_n_identitys_offset_rows_refuse_a_per_neuron_threshold_by_name():
    from mimarsinan.mapping.support.neuron_thresholds import PerNeuronThresholdRefusal

    _program, cached, _decl = UV.pruned("linear_reader", True, 8, 3, 2)
    ir = copy.deepcopy(cached)  # the vehicle is cached across tests: its copy is edited, never the vehicle
    readers = [n for n in ir.get_neural_cores() if getattr(n, "readout_law", None) is not None
               and pruned_rows.reader_offset(n) != 0.0]
    assert readers
    counts: dict = {}
    pruned_rows.check_dropped_offset_rows(ir, ConditionResult("N-identity", VACUOUS), counts)
    assert counts["dropped_offset_readers"] > 0
    for node in readers:
        node.threshold = np.full(int(node.get_output_count()), float(node.threshold))
    with pytest.raises(PerNeuronThresholdRefusal, match="N-identity"):
        pruned_rows.check_dropped_offset_rows(ir, ConditionResult("N-identity", VACUOUS), {})
