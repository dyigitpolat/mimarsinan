"""[U-PCH3] Every per-core or per-node value a static-check rule reads has its layout declared by name by its holder,
on IR nodes as on hard cores (``VALUE_LAYOUTS``, ``mapping.support.value_layouts``), and no reader infers a layout
from a value's size.

- The declaration: every value attribute the forms registry gives a predicate reader is declared by its holder; an
  undeclared name, and a value of a size its declaration does not admit, are refused by name.
- H1-theta: a per-neuron register at a split placement's columns on both sides; an input scale (one per input row) held
  whole by a core and compared whole (the verify's m-1 and m-6).
- ir-tiling: an input scale moved at a row outside a tile's own output channels is seen (the verify's case K3) and one
  at a row none of the tile's neurons reads is not, a per-channel value whose size equals a tile's width is read at
  its channels, each side by its own channel identity, derived only where a vector is read; a malformed value is a
  violation by name.
- The hard core assembles over a resident's neuron range exactly the singletons it declares per neuron (m-4).
- Every other reader of a declared value in the check reads it by the declaration: tiles' block (a resident's bias at
  its columns), H10's region (the used neurons' registers, a malformed register file refused by name), N-identity's
  dropped columns (a node's bias per column, one held entry every column's) and H8's value levels (a law core's law
  at each placement's columns).

The programs are the static check's vehicles (``unit.mapping.residency.executed.vehicles``) on the integer lattice."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.certification.program_check.report import ESTABLISHED
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM

from unit.certification.program_check import programs as P
from unit.mapping.residency.executed import vehicles as V

UNBOUNDED = 10 ** 6


def _violations(report) -> dict:
    return {c: [v.text() for v in r.violations][:3] for c, r in report.results.items() if r.violations}


def _integer_ir(rep, neurons: int, *, per_neuron: bool = False):
    """``rep``'s IR at ``neurons`` columns per node (``UNBOUNDED``: at unbounded geometry) on the integer lattice: one
    grid per perceptron, or one per output channel (``31 + 5 (c mod 7)``)."""
    axons = UNBOUNDED if neurons >= UNBOUNDED else V.MAX_AXONS
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=axons, max_neurons=neurons).map(rep)
    widths: dict = {}
    for node in ir.get_neural_cores():
        widths[node.perceptron_index] = max(widths.get(node.perceptron_index, 0), node.perceptron_output_slice[1])
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        owner.parameter_scale = (torch.tensor([float(31 + 5 * (c % 7)) for c in range(widths[owner.perceptron_index])])
                                 if per_neuron else torch.tensor(0.0))
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    return ir


def _channel_scales(rep):
    for p in rep.get_perceptrons():
        p.set_activation_scale(torch.tensor([1.0 + 0.125 * c for c in range(int(p.layer.weight.shape[0]))]))
    return rep


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

#: Per holder element of the forms registry: the attributes a predicate reads that are no per-core or per-node value
#: (geometry, wiring, keys, structure, a host op's own fields compared whole against the same host op).
NOT_VALUES = {
    "hard_core": {"axons_per_core", "neurons_per_core", "available_axons", "available_neurons", "core_matrix",
                  "matrix_placements", "_axon_source_spans", "axon_sources", "constrained_properties",
                  "fused_component_axons"},
    "ir_node": {"id", "name", "input_sources", "core_matrix", "weight_bank_id", "weight_row_slice", "perceptron_index",
                "perceptron_output_slice", "perceptron_output_column", "perceptron_input_slice", "psum_group_id",
                "psum_role", "coalescing_group_id", "coalescing_role", "normalization_type", "activation_type",
                "layout_softcore_index", "pruned_row_mask", "pruned_col_mask", "_source_counts_cache",
                "pre_pruning_col_mask", "core_role", "lane_operand_axons", "layout_record_inserted",
                "lane_source_name", "_depth_relay", "op_type", "params", "input_shape", "output_shape",
                "_negative_shift"},
    "ir_bank": {"id", "core_matrix", "perceptron_index"},
}


def test_every_value_a_static_check_rule_reads_is_declared_by_its_holder():
    from mimarsinan.certification.program_check.forms import BACKEND, LABEL, UNREAD, registered_attributes
    from mimarsinan.mapping.ir import NeuralCore, WeightBank
    from mimarsinan.mapping.packing.softcore import HardCore

    attributes = registered_attributes()
    for element, holder in (("hard_core", HardCore), ("ir_node", NeuralCore), ("ir_bank", WeightBank)):
        read = {name for name, readers in attributes[element].items()  # a predicate or the identity builder
                if set(readers) - {UNREAD, LABEL, BACKEND}}
        declared = set(holder.VALUE_LAYOUTS)
        assert read - NOT_VALUES[element] <= declared, (element, sorted(read - NOT_VALUES[element] - declared))
        assert not declared & NOT_VALUES[element], element


def test_a_value_is_read_only_by_its_holders_declaration():
    from mimarsinan.mapping.ir import IRSource, NeuralCore
    from mimarsinan.mapping.support.value_layouts import (
        ValueLayout, ValueLayoutError, declared_layout, neuron_entries, row_entries)

    node = NeuralCore(id=0, name="n", input_sources=np.array([IRSource(-2, i) for i in range(3)], dtype=object),
                      core_matrix=np.ones((3, 4)), perceptron_index=0, perceptron_output_slice=(4, 8))
    assert declared_layout(node, "input_activation_scale") is ValueLayout.PER_INPUT_ROW
    with pytest.raises(ValueLayoutError, match="declares no layout"):
        declared_layout(node, "undeclared_value")
    cols, channels = np.array([1, 2]), np.arange(4, 8)
    with pytest.raises(ValueLayoutError, match="per input row"):
        neuron_entries(node, "input_activation_scale", cols, width=4, channels=channels, value=np.ones(3))
    with pytest.raises(ValueLayoutError, match="neither one nor"):
        neuron_entries(node, "threshold", cols, width=4, channels=channels, value=np.arange(5.0))
    assert row_entries(node, "input_activation_scale", np.array([0, 2]), value=np.array([1.0, 2.0, 3.0])).tolist() \
        == [1.0, 3.0]
    with pytest.raises(ValueLayoutError, match="one per input row"):
        row_entries(node, "input_activation_scale", np.array([0]), value=np.ones(4))


def test_a_per_channel_value_is_read_at_its_channels_whatever_its_size():
    """A tile at channels 4 to 7 (width 4) holding a 4-entry activation scale: the declared layout is per channel of
    its perceptron, so the vector cannot be its own columns' and is refused by name (a size equal to the width once
    read it per column); a per-column value is read at its columns, a per-channel one at its channels."""
    from mimarsinan.mapping.ir import IRSource, NeuralCore
    from mimarsinan.mapping.support.value_layouts import ValueLayoutError, neuron_entries

    node = NeuralCore(id=0, name="n", input_sources=np.array([IRSource(-2, 0)], dtype=object),
                      core_matrix=np.ones((1, 4)), perceptron_index=0, perceptron_output_slice=(4, 8))
    cols, channels = np.array([1, 2]), np.arange(4, 8)
    perceptron = np.arange(10.0) * 0.5
    assert neuron_entries(node, "activation_scale", cols, width=4, channels=channels, value=perceptron).tolist() \
        == [2.5, 3.0]
    assert neuron_entries(node, "threshold", cols, width=4, channels=channels, value=np.arange(4.0)).tolist() \
        == [1.0, 2.0]
    assert neuron_entries(node, "activation_scale", cols, width=4, channels=channels, value=7.0).tolist() == [7.0, 7.0]
    with pytest.raises(ValueLayoutError, match="per output channel"):
        neuron_entries(node, "activation_scale", cols, width=4, channels=channels, value=np.arange(4.0))
    relay = NeuralCore(id=1, name="r", input_sources=np.array([IRSource(0, 0)], dtype=object),
                       core_matrix=np.ones((1, 4)))  # no perceptron identity: its columns are its channels
    assert neuron_entries(relay, "activation_scale", cols, width=4, channels=None, value=np.arange(4.0)).tolist() \
        == [1.0, 2.0]


def test_each_side_reads_a_per_channel_value_by_its_own_channel_identity():
    """The verify's m-3 (W06): the tile's channel table is never the reference's. A reference node pruned to channels
    0, 2, 3 (its pre-pruning mask) holds the perceptron's vector; its columns 1 and 2 read channels 2 and 3."""
    from mimarsinan.certification.program_check.unbounded.node_values import NeuronPair
    from mimarsinan.mapping.ir import IRSource, NeuralCore

    vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tile = NeuralCore(id=0, name="t", input_sources=np.array([IRSource(-2, 0)], dtype=object),
                      core_matrix=np.ones((1, 2)), perceptron_index=0, perceptron_output_slice=(2, 4),
                      activation_scale=vector)
    reference = NeuralCore(id=1, name="u", input_sources=np.array([IRSource(-2, 0)], dtype=object),
                           core_matrix=np.ones((1, 3)), perceptron_index=0, perceptron_output_slice=(0, 4),
                           activation_scale=vector, pre_pruning_col_mask=[False, True, False, False])
    pair = NeuronPair((None, tile, np.array([0, 1]), 2), (None, reference, np.array([1, 2]), 3))
    assert pair.differs("activation_scale", vector, vector) is None
    shifted = NeuronPair((None, tile, np.array([0, 1]), 2), (None, reference, np.array([0, 1]), 3))
    assert shifted.differs("activation_scale", vector, vector) == ""


# ---------------------------------------------------------------------------
# H1-theta
# ---------------------------------------------------------------------------


def test_h1_theta_reads_a_split_placements_registers_at_its_columns_on_both_sides():
    """The verify's m-1 (K6): per-neuron thresholds, IR nodes of 8 columns on cores of 5 with neuron splitting."""
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=8).map(
        _channel_scales(V.random_rep(0)))
    widths: dict = {}
    for n in ir.get_neural_cores():
        widths[n.perceptron_index] = max(widths.get(n.perceptron_index, 0), int(n.perceptron_output_slice[1]))
    for owner in list(ir.get_neural_cores()) + list(ir.weight_banks.values()):
        owner.parameter_scale = torch.tensor([float(31 + 5 * (c % 7)) for c in range(widths[owner.perceptron_index])])
    quantize_ir_graph(ir, 8, weight_quantization=True)
    IRLatency(ir).calculate()
    cores = {"max_axons": V.MAX_AXONS, "max_neurons": 5, "count": 400}
    program = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[dict(cores)], strategy=MappingStrategy.
                                             from_permissions(allow_neuron_splitting=True, allow_coalescing=False))
    decl = CheckDeclaration(semantics="streamed_lif", window=V.T, pass_transfer=VERBATIM, core_types=(dict(cores),))

    def split(prog):
        return [(c, p) for s in P.neural(prog) for c, placed in zip(
            s.hard_core_mapping.cores, s.hard_core_mapping.soft_core_placements_per_hard_core) for p in placed
            if int((p.get("neuron_range_in_original") or (0, 0))[0] or 0) > 0 and np.ndim(c.threshold) == 1]

    assert split(program), "the vehicle splits a per-neuron node across cores"
    clean = check_deployment(program, ir_graph=ir, declaration=decl)
    assert not clean.results["H1-theta"].violations and clean.established("theorem1"), _violations(clean)
    moved = P.clone(program)
    core, placement = split(moved)[0]
    registers = np.array(core.threshold, dtype=np.float64)
    registers[int(placement["neuron_offset"])] += 1.0
    core.threshold = registers
    report = check_deployment(moved, ir_graph=ir, declaration=decl)
    assert [v.rule for v in report.results["H1-theta"].violations] == ["threshold"], _violations(report)


def _per_input_scales(ir, perceptron: int, rows: int, *, values=None) -> torch.Tensor:
    """One input scale per input row on every node of ``perceptron`` (the form seam-currency reads row by row)."""
    vector = torch.tensor(values if values is not None else [0.5 + 0.25 * (i % 5) for i in range(rows)])
    for node in ir.get_neural_cores():
        if node.perceptron_index == perceptron:
            node.input_activation_scale = vector.clone()
    return vector


def _tiled_perceptron(ir):
    """A perceptron tiled over several nodes of 4 columns, every node of it with one input count."""
    rows: dict = {}
    widths: dict = {}
    for n in ir.get_neural_cores():
        rows.setdefault(n.perceptron_index, set()).add(int(n.get_input_count()))
        widths[n.perceptron_index] = max(widths.get(n.perceptron_index, 0), int(n.perceptron_output_slice[1]))
    perceptron = next(p for p, r in rows.items() if len(r) == 1 and widths[p] > 4)
    return perceptron, next(iter(rows[perceptron]))


def test_h1_theta_sees_an_input_scale_moved_on_a_deployed_core():
    """The verify's m-6 (K2): a core stores its input scale once (one per input row of its residents, whole)."""
    ir = _integer_ir(V.random_rep(0), 4)
    perceptron, rows = _tiled_perceptron(ir)
    _per_input_scales(ir, perceptron, rows)
    program = V.build(ir, 4, None)
    decl = P.decl("streamed_lif", count=4)
    assert not P.run(program, ir, decl).results["H1-theta"].violations
    moved = P.clone(program)
    core = next(c for s in P.neural(moved) for c in s.hard_core_mapping.cores
                if torch.as_tensor(c.input_activation_scale).numel() > 1)
    vector = torch.as_tensor(core.input_activation_scale, dtype=torch.float64).clone()
    vector[-1] += 0.5
    core.input_activation_scale = vector
    report = P.run(moved, ir, decl)
    assert "input_activation_scale" in {v.rule for v in report.results["H1-theta"].violations}, _violations(report)


# ---------------------------------------------------------------------------
# ir-tiling
# ---------------------------------------------------------------------------


def _input_scale_tiling(where):
    """The verify's case K3: one input scale per input row on a tiled perceptron's nodes, tiled at 4 columns and at
    unbounded geometry; ``where`` moves one tile's entry at a row its paired neurons read, outside (or inside) the
    tile's own output channels, or nowhere; ``scalar`` moves one tile's single input scale (no per-row vectors)."""
    tiled, reference = _integer_ir(V.random_rep(0), 4), _integer_ir(V.random_rep(0), UNBOUNDED)
    perceptron, rows = _tiled_perceptron(tiled)
    for ir in (tiled, reference):
        if where != "scalar":
            _per_input_scales(ir, perceptron, rows)
    program = V.build(tiled, 4, None)
    if where == "scalar":
        node = next(n for n in tiled.get_neural_cores() if n.perceptron_index == perceptron)
        node.input_activation_scale = torch.as_tensor(node.input_activation_scale, dtype=torch.float64) + 0.5
    elif where is not None:
        node = next(n for n in tiled.get_neural_cores()
                    if n.perceptron_index == perceptron and int(n.perceptron_output_slice[0]) > 0)
        lo, hi = (int(v) for v in node.perceptron_output_slice)
        matrix = np.asarray(node.get_core_matrix(tiled))
        read = [r for r in range(rows) if np.any(matrix[r] != 0)]
        row = next(r for r in read if (lo <= r < hi) == (where == "inside"))
        vector = torch.as_tensor(node.input_activation_scale, dtype=torch.float64).clone()
        vector[row] += 0.5
        node.input_activation_scale = vector
    return P.run(program, tiled, P.decl("streamed_lif", count=4), unbounded_ir=reference)


@pytest.mark.parametrize("where", [None, "outside", "inside", "scalar"])
def test_ir_tiling_reads_an_input_scale_by_the_rows_its_neurons_read(where):
    report = _input_scale_tiling(where)
    result = report.results["ir-tiling"]
    if where is None:
        assert result.status == ESTABLISHED and report.established("theorem1"), _violations(report)
    else:
        assert any(v.rule == "neuron_scale" and "input_activation_scale" in v.text() for v in result.violations), \
            _violations(report)


def test_ir_tiling_reads_only_the_rows_a_tiles_neurons_read():
    """A row no neuron of a tile reads (its weights zero at the tile's channels on both IRs, read by the unbounded
    node's other channels) is no row of the tile's: the tile's input scale differing there is no difference (the
    same entry moved at a row its neurons read is one: the verify's case K3, the test above)."""
    tiled, reference = _integer_ir(V.random_rep(0), 4), _integer_ir(V.random_rep(0), UNBOUNDED)
    perceptron, rows = _tiled_perceptron(tiled)
    tile = next(n for n in tiled.get_neural_cores()
                if n.perceptron_index == perceptron and int(n.perceptron_output_slice[0]) > 0)
    lo, hi = (int(v) for v in tile.perceptron_output_slice)
    home = next(n for n in reference.get_neural_cores() if n.perceptron_index == perceptron)
    tile.core_matrix = np.array(tile.core_matrix, copy=True)
    home.core_matrix = np.array(home.core_matrix, copy=True)
    tile.core_matrix[0, :] = 0
    home.core_matrix[0, lo:hi] = 0
    assert np.any(home.core_matrix[0] != 0), "the unbounded node reads the row for its other channels"
    for ir in (tiled, reference):
        _per_input_scales(ir, perceptron, rows)
    program = V.build(tiled, 4, None)
    vector = torch.as_tensor(tile.input_activation_scale, dtype=torch.float64).clone()
    vector[0] += 0.5  # the tile's own entry at the row none of its neurons reads
    tile.input_activation_scale = vector
    report = P.run(program, tiled, P.decl("streamed_lif", count=4), unbounded_ir=reference)
    assert report.results["ir-tiling"].status == ESTABLISHED, _violations(report)


def test_ir_tiling_reports_a_value_held_in_no_declared_layout_by_name():
    tiled, reference = _integer_ir(V.random_rep(0), 4), _integer_ir(V.random_rep(0), UNBOUNDED)
    program = V.build(tiled, 4, None)
    node = next(n for n in reference.get_neural_cores() if n.perceptron_index is not None)  # only ir-tiling reads it
    node.threshold = np.full(int(node.get_output_count()) + 1, float(node.threshold))  # one more than its columns
    report = P.run(program, tiled, P.decl("streamed_lif", count=4), unbounded_ir=reference)
    found = [v for v in report.results["ir-tiling"].violations if v.rule == "threshold"]
    assert found and "neither one nor" in found[0].text(), _violations(report)


@pytest.mark.parametrize("channels", [False, True])
def test_ir_tiling_derives_a_channel_identity_only_where_a_vector_is_read(channels, monkeypatch):
    """The verify's m-5: a program of one value per core derives no channel table; a per-channel one does."""
    from mimarsinan.certification.program_check.unbounded import node_values

    derived = []
    real = node_values.perceptron_channels
    monkeypatch.setattr(node_values, "perceptron_channels", lambda *a: derived.append(1) or real(*a))
    rep = (lambda: _channel_scales(V.random_rep(0))) if channels else (lambda: V.random_rep(0))
    tiled, reference = _integer_ir(rep(), 4), _integer_ir(rep(), UNBOUNDED)
    report = P.run(V.build(tiled, 4, None), tiled, P.decl("streamed_lif", count=4), unbounded_ir=reference)
    assert report.results["ir-tiling"].status == ESTABLISHED, _violations(report)
    assert bool(derived) == channels


# ---------------------------------------------------------------------------
# The hard core: one table for the assembly and the readers
# ---------------------------------------------------------------------------


def test_the_hard_core_assembles_exactly_the_singletons_it_declares_per_neuron():
    """The verify's m-4: one table. A core class declaring its activation scale per neuron assembles it over each
    resident's range (no residency refusal) and every reader reads it at the placement's columns."""
    from mimarsinan.mapping.packing.softcore import HardCore, SoftCore
    from mimarsinan.mapping.support.value_layouts import ValueLayout, placement_value

    def residents(core_class, scales):
        hard = core_class(8, 8)
        for index, (thetas, scale) in enumerate(zip(([3.0, 5.0, 7.0], [11.0, 2.0]), scales)):
            soft = SoftCore(core_matrix=np.ones((3, len(thetas))), axon_sources=[], id=index)
            soft.threshold, soft.activation_scale = np.asarray(thetas), torch.tensor(scale)
            hard.add_softcore(soft)
        return hard

    shared, own = ([1.5, 2.0, 2.5], [1.5, 2.0, 2.5]), ([1.5, 2.0, 2.5], [4.0, 4.5])
    hard = residents(HardCore, shared)
    assert np.array_equal(placement_value(hard, "threshold", 3, 5), [11.0, 2.0])
    assert torch.equal(placement_value(hard, "activation_scale", 3, 5), torch.tensor([1.5, 2.0, 2.5]))  # whole
    with pytest.raises(AssertionError, match="activation_scale"):  # stored once: its residents must agree
        residents(HardCore, own)
    declared = dict(HardCore.VALUE_LAYOUTS, activation_scale=ValueLayout.PER_NEURON)
    wide = residents(type("PerNeuronScale", (HardCore,), {"VALUE_LAYOUTS": declared}), own)
    assert np.array_equal(placement_value(wide, "activation_scale", 3, 5), [4.0, 4.5])
    assert np.array_equal(placement_value(wide, "activation_scale", 0, 3), [1.5, 2.0, 2.5])


def _inputs(count: int) -> list:
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource

    return [SpikeSource(-2, i, True, False) for i in range(count)]


def test_tiles_reads_a_residents_hardware_bias_by_the_cores_declaration():
    """tiles' block read (``tiles._block``) takes a resident's bias at its own columns of the core: the bias the core
    assembles per neuron, never the core's whole bias beside the resident's columns."""
    from types import SimpleNamespace

    from mimarsinan.certification.program_check.tiles import _block
    from mimarsinan.mapping.packing.softcore import HardCore, SoftCore
    from mimarsinan.mapping.support.value_layouts import placement_value

    hard = HardCore(8, 8)
    for index, bias in enumerate(([1.0, 2.0, 3.0], [4.0, 5.0])):
        soft = SoftCore(core_matrix=np.ones((2, len(bias))), axon_sources=_inputs(2), id=index)
        soft.hardware_bias = np.asarray(bias)
        hard.add_softcore(soft)
    assert placement_value(hard, "hardware_bias", 3, 5).tolist() == [4.0, 5.0]
    assert placement_value(hard, "input_activation_scale", 3, 5) is hard.input_activation_scale
    view = SimpleNamespace(core=hard, weights=hard.get_core_matrix, used_axons=4)
    block, bias = _block(view, 3, 5)
    assert block.shape == (4, 2) and bias.tolist() == [4.0, 5.0]


def _two_residents(thresholds=([3.0, 5.0, 7.0], [11.0, 2.0])):
    from mimarsinan.mapping.packing.softcore import HardCore, SoftCore

    hard = HardCore(8, 8)
    for index, thetas in enumerate(thresholds):
        soft = SoftCore(core_matrix=np.ones((3, len(thetas))), axon_sources=_inputs(3), id=index)
        soft.threshold = np.asarray(thetas)
        hard.add_softcore(soft)
    return hard


def test_h10_reads_a_cores_registers_by_its_declaration():
    """H10's region (``residency._region``) reads the used neurons' registers by the core's declaration: each used
    neuron's register, and a register file of another length than the core's neurons is refused by name, never cut
    to the used neurons."""
    from mimarsinan.certification.program_check.residency import _region
    from mimarsinan.mapping.support.value_layouts import ValueLayoutError

    hard = _two_residents()
    _grid, bias, key = _region(hard, 6, 5)
    assert bias is None and key == (3.0, 5.0, 7.0, 11.0, 2.0)
    hard.threshold = np.arange(9.0)  # one more register than the core's eight neurons
    with pytest.raises(ValueLayoutError, match="threshold"):
        _region(hard, 6, 5)


def _masked_producer(bias):
    """The static check's masked IR (``test_declaration``'s form): node 0's column 0 dropped by compaction with zero
    weights, its hardware bias ``bias``."""
    ir = P.integer_ir(V.random_rep(0))
    producer = ir.get_neural_cores()[0]
    rows, cols = producer.get_input_count(), producer.get_output_count()
    matrix = np.array(producer.core_matrix, dtype=np.float64)
    matrix[:, 0] = 0.0
    producer.core_matrix = matrix
    producer.pruned_row_mask = [False] * rows
    producer.pruned_col_mask = [c == 0 for c in range(cols)]
    producer.hardware_bias = bias(cols) if callable(bias) else bias
    return ir


def test_n_identity_reads_a_nodes_hardware_bias_by_its_declaration():
    """N-identity's dropped columns (``network.check_identity_network``) read a node's hardware bias per column by the
    node's declaration: one held entry is every column's, one per column at the column, any other length refused by
    name (a one-entry bias once raised an index error, a longer one was read by position)."""
    from mimarsinan.certification.program_check.network import check_identity_network
    from mimarsinan.certification.program_check.report import VACUOUS, ConditionResult
    from mimarsinan.mapping.support.value_layouts import ValueLayoutError

    def rules(bias) -> set:
        result = ConditionResult("N-identity", VACUOUS)
        check_identity_network(_masked_producer(bias), result)
        return {v.rule for v in result.violations}

    for dead in (np.float64(0.0), lambda cols: np.zeros(cols), lambda cols: np.r_[0.0, np.ones(cols - 1)]):
        assert rules(dead) == set()
    for live in (np.float64(2.0), lambda cols: np.r_[2.0, np.zeros(cols - 1)]):
        assert rules(live) == {"dropped_read_column"}
    with pytest.raises(ValueLayoutError, match="hardware_bias"):
        rules(lambda cols: np.zeros(cols + 1))


def test_h8_reads_each_placements_law_by_the_cores_declaration(monkeypatch):
    """The value domain's row levels (``check._value_levels``) read a law core's law at each placement's columns
    through the core's declaration, as tiles, H10 and the laws condition read theirs."""
    from mimarsinan.certification.program_check import check as C
    from mimarsinan.mapping.support.value_layouts import placement_value

    calls = []

    def recorded(holder, name, lo, hi, **kwargs):
        calls.append((name, int(lo), int(hi)))
        return placement_value(holder, name, lo, hi, **kwargs)

    monkeypatch.setattr(C, "placement_value", recorded)
    program, ir = P.value_program()
    report = P.run(program, ir, P.decl("value_integer", accumulation={"value_census": "float64"}))
    assert report.results["H8"].checked > 0
    assert calls and {name for name, _lo, _hi in calls} == {"readout_law"} and all(lo < hi for _n, lo, hi in calls)
