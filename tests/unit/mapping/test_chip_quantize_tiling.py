"""[E1] The mapping-time fallback lattice is one grid per perceptron, so core geometry never changes the integer network.

A core with no trained ``parameter_scale`` (``parameter_scale`` 0: a relay, or a harness vehicle that zeroes it, as the
W15 random-init harness and the fuzz vehicles do; a pipeline perceptron carries 1.0 or a trained value) takes the grid
scale ``floor(q_max / max |W|)``. Taken per IR core, a perceptron the declared geometry tiles quantizes each
tile on its own grid, so its integer weights and threshold differ from the same perceptron's at unbounded geometry: a
cost-only decision changes the function (MEASURED, ``RA/11_wave2/e1/diag/lattice_probe.txt`` and ``lattice_outputs.txt``:
under the per-core grid 322 of 395 tiled base IRs of the fuzz generator, seeds 0 to 599, fail ``ir-tiling`` by
``fan_in`` and ``threshold``, and 153 of the 395 identity programs differ from the identity program at unbounded
geometry under streamed LIF). The trained
lattice is one grid per perceptron (NAPQ's ``parameter_scale``), and so is the fallback: the largest magnitude over
every core and bank of the perceptron."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir.dynamic_bank import DynamicBankBinding
from mimarsinan.mapping.ir.weight_bank import WeightBank
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.support.grid_declarations import ParameterGrids

from unit.mapping.residency.executed import vehicles as V


def _ir(rep, axons, neurons, coalescing: bool = False, grids: ParameterGrids = ParameterGrids()):
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=axons, max_neurons=neurons,
                   allow_coalescing=coalescing).map(rep)
    for node in ir.get_neural_cores():
        node.parameter_scale = torch.tensor(0.0)
    for bank in ir.weight_banks.values():
        bank.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir, 8, weight_quantization=True, grids=grids)
    return ir


def _by_neuron(ir):
    """{(perceptron, column, channel): (integer fan-in column, threshold)}; the fan-in as the multiset of weights (the
    two IRs name their sources by other node ids)."""
    out = {}
    for node in ir.get_neural_cores():
        matrix = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
        start = int(node.perceptron_output_slice[0])
        column = -1 if node.perceptron_output_column is None else int(node.perceptron_output_column)
        for k in range(matrix.shape[1]):
            out[(node.perceptron_index, column, start + k)] = (tuple(sorted(matrix[:, k].tolist())),
                                                              float(node.threshold))
    return out


#: Fuzz vehicles the vehicle geometry (25 x 8) tiles at the IR level: output tiles, and coalesced wide fan-ins.
TILED = [("base", 0), ("base", 3), ("base", 6), ("base", 8), ("coalesce", 0), ("coalesce", 1), ("coalesce", 3),
         ("coalesce", 5)]


@pytest.mark.parametrize("family, seed", TILED)
def test_an_untrained_perceptron_quantizes_alike_at_every_geometry(family, seed):
    rep_of = V.wide_rep if family == "coalesce" else V.random_rep
    tiled = _ir(rep_of(seed), V.MAX_AXONS, V.MAX_NEURONS, coalescing=family == "coalesce")
    unbounded = _ir(rep_of(seed), None, None)
    assert len(tiled.get_neural_cores()) > len(unbounded.get_neural_cores())
    assert _by_neuron(tiled) == _by_neuron(unbounded)


def test_a_trained_scale_still_sets_its_perceptrons_grid():
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS,
                   max_neurons=V.MAX_NEURONS).map(V.random_rep(0))
    for node in ir.get_neural_cores():
        node.parameter_scale = torch.tensor(5.0)
    quantize_ir_graph(ir, 8, weight_quantization=True)
    assert {float(n.threshold) for n in ir.get_neural_cores()} == {5.0}


# [integration2/e1] The merge with the lanes unit's declared grids (``ParameterGrids.fractional_scale``: a value-domain
# law carries the scale in its multiplier) and N2's dynamic operand banks (a bound bank's matrix is a payload of
# integer codes written at run time, never quantized). The fallback grid is one per perceptron under either scale rule,
# and only weights the quantizer quantizes set a perceptron's grid.


@pytest.mark.parametrize("family, seed", TILED)
def test_an_untrained_perceptron_quantizes_alike_at_every_geometry_under_a_fractional_scale(family, seed):
    rep_of = V.wide_rep if family == "coalesce" else V.random_rep
    grids = ParameterGrids(fractional_scale=True)
    tiled = _ir(rep_of(seed), V.MAX_AXONS, V.MAX_NEURONS, coalescing=family == "coalesce", grids=grids)
    unbounded = _ir(rep_of(seed), None, None, grids=grids)
    assert len(tiled.get_neural_cores()) > len(unbounded.get_neural_cores())
    assert _by_neuron(tiled) == _by_neuron(unbounded)


def test_a_payload_bank_sets_no_perceptrons_grid():
    """A bank bound to a payload shares a perceptron with an untrained owned core: its placeholder matrix (N2 writes
    NaN; any value it holds is not a weight) leaves the core's grid as it is without the bank."""
    def quantized(with_payload: bool):
        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=None, max_neurons=None).map(V.random_rep(0))
        for node in ir.get_neural_cores():
            node.parameter_scale = torch.tensor(0.0)
        core = next(n for n in ir.get_neural_cores() if not n.has_weight_bank())
        if with_payload:
            bank_id = max(ir.weight_banks, default=-1) + 1
            ir.weight_banks[bank_id] = WeightBank(id=bank_id, core_matrix=np.full((4, 4), 1.0e3),
                                                  parameter_scale=torch.tensor(0.0),
                                                  perceptron_index=core.perceptron_index)
            ir.dynamic_bank_bindings[bank_id] = DynamicBankBinding(
                bank_id=bank_id, source_node_id=-2, payload_shape=(4, 4), transpose=False, batch_index=(),
                payload_bits=8)
        quantize_ir_graph(ir, 8, weight_quantization=True)
        return float(core.threshold), np.asarray(core.core_matrix).tolist()

    assert quantized(with_payload=True) == quantized(with_payload=False)
