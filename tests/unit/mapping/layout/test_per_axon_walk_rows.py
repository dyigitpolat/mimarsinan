"""[U-AX, SDC-D1] The layout walk's physical rows under per-axon weight sign: the build's rule, on the build's grid.

The build splits each mixed-sign row of a soft core into a W+ and a W- row and keeps an all-zero or single-sign row as
one (``sign_split.physical_row_counts``, which ``compaction.deployed_row_counts`` reads), on the integer codes the
quantizer installs. A walk that holds the network's weights and is given the grid the build reads (its
``sign_rows`` census, ``sign_row_census``) counts each core's rows by that rule on that grid; a walk without it, or one
holding a core whose weights it cannot read, keeps two rows per input and states them as an upper bound by name
(``physical_rows``); a per-synapse walk states logical rows. The two candidate planes that hold the network's weights,
the scaling ladder and the joint search, give the walk their platform's ``weight_bits``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_source_view_ops import total_size
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.softcore import deployed_core_extent, deployed_row_counts
from mimarsinan.mapping.packing.softcore.sign_split import sign_split_rows
from mimarsinan.mapping.platform.packaging_contract import SPIKING_PACKAGING
from mimarsinan.mapping.platform.platform_constraints import PER_AXON, resolve_platform_mapping_params
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.converter import convert_torch_model

CORES = [{"max_axons": 256, "max_neurons": 256, "count": 4, "has_bias": False, "weight_sign_granularity": PER_AXON}]
BITS = 4


class _Convs(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0, self.c1 = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1, bias=False)
        self.c2, self.relu = nn.Conv2d(8, 8, 3, padding=1), nn.ReLU()
        self.flat, self.fc = nn.Flatten(), nn.Linear(8 * 16, 10)

    def forward(self, x):
        return self.fc(self.flat(self.relu(self.c2(self.relu(self.c1(self.relu(self.c0(x))))))))


def _conv_rep():
    """Bank-backed convolutions (c1 bias-free: an all-zero always-on row per instance) and an owned head."""
    torch.manual_seed(0)
    model = _Convs()
    with torch.no_grad():
        model.c2.weight[:, :2] = model.c2.weight[:, :2].abs()  # single-sign rows
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model.eval(), (3, 4, 4), 10, packaging=SPIKING_PACKAGING, strict=False).eval()
    rep = flow.get_mapper_repr()
    rep.assign_perceptron_indices()
    return rep


def _census(rows, weight_bits):
    """The walk's census (``sign_row_census``); before the unit there is none, and the walk plans two rows per input."""
    try:
        from mimarsinan.mapping.layout.ir_mapping.sign_rows import sign_row_census
    except ImportError:
        return {}
    return {"sign_rows": sign_row_census(rows, weight_bits)}


def _walk(rep, *, cores=CORES, weight_bits=BITS, rows=None):
    pm = resolve_platform_mapping_params(cores)
    rows = pm.rows_per_input if rows is None else rows
    layout = LayoutIRMapping(max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
                             hardware_bias=pm.hardware_bias, rows_per_input=rows, **_census(rows, weight_bits))
    specs = layout.collect_layout_softcores(rep)
    return layout, specs


def _built(rep, *, cores=CORES):
    """The build's IR: ``IRMapping`` at the pool's limits, on the quantizer's grid for weights with no trained scale."""
    pm = resolve_platform_mapping_params(cores)
    with contextlib.redirect_stdout(io.StringIO()):
        ir = IRMapping(q_max=7.0, firing_mode="Default", max_axons=pm.effective_max_axons,
                       max_neurons=pm.effective_max_neurons, hardware_bias=pm.hardware_bias).map(rep)
    rep.clear_ir_caches()
    built = copy.deepcopy(ir)
    for bank in built.weight_banks.values():
        bank.parameter_scale = torch.tensor(0.0)
    for node in built.nodes:
        if isinstance(node, NeuralCore):
            node.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(built, BITS, weight_quantization=True)
    return built


def _deployed(rep):
    """Per layout softcore (the walk's order), the rows its core deploys on the build's grid."""
    built = _built(rep)
    rows = {int(c.layout_softcore_index): deployed_core_extent(c, PER_AXON, graph=built)[0]
            for c in built.get_neural_cores()}
    return [rows[i] for i in range(len(rows))]


class TestTheDeployedRows:
    def test_every_spec_is_its_cores_deployed_rows(self):
        rep = _conv_rep()
        _layout, specs = _walk(rep)
        built = _built(rep)
        cores = built.get_neural_cores()
        assert len(cores) == len(specs)
        for core in cores:
            spec = specs[int(core.layout_softcore_index)]
            assert (int(spec.input_count), int(spec.output_count)) == deployed_core_extent(core, PER_AXON, graph=built)

    def test_a_zero_or_single_sign_row_takes_one_row_and_a_mixed_row_two(self):
        """A hand-set 4 -> 3 perceptron with a bias: rows mixed, positive, zero, negative, and a mixed bias row."""
        perceptron = Perceptron(3, 4, normalization=nn.Identity(), base_activation_name="ReLU")
        with torch.no_grad():
            perceptron.layer.weight.copy_(torch.tensor([[1.0, 2.0, 0.0, -1.0], [-1.0, 1.0, 0.0, -2.0],
                                                         [2.0, 3.0, 0.0, -3.0]]))
            perceptron.layer.bias.copy_(torch.tensor([1.0, -1.0, 2.0]))
        rep = ModelRepresentation(PerceptronMapper(InputMapper((4,)), perceptron))
        _layout, specs = _walk(rep)
        assert [int(s.input_count) for s in specs] == [2 + 1 + 1 + 1 + 2]

    def test_the_walk_reads_the_grid_not_the_float_weights(self):
        """A row mixed in float whose only negative weight rounds to zero on the 4-bit grid takes one row."""
        perceptron = Perceptron(3, 2, normalization=nn.Identity(), base_activation_name="ReLU")
        with torch.no_grad():
            perceptron.layer.weight.copy_(torch.tensor([[1.0, 1.0], [-1e-3, 1.0], [1.0, -1.0]]))
            perceptron.layer.bias.zero_()
        rep = ModelRepresentation(PerceptronMapper(InputMapper((2,)), perceptron))
        _layout, specs = _walk(rep)
        assert [int(s.input_count) for s in specs] == [1 + 2 + 1]

    def test_the_walk_states_deployed_rows(self):
        layout, _specs = _walk(_conv_rep())
        assert (layout.physical_rows.basis, layout.physical_rows.reason) == ("deployed", None)


class TestTheUpperBound:
    def test_a_walk_without_the_grid_keeps_two_rows_per_input_and_says_so(self):
        rep = _conv_rep()
        bound, bound_specs = _walk(rep, weight_bits=None)
        logical, logical_specs = _walk(rep, weight_bits=None, rows=1)
        assert [int(s.input_count) for s in bound_specs] == [2 * int(s.input_count) for s in logical_specs]
        assert bound.physical_rows.basis == "upper_bound"
        assert "weight grid" in bound.physical_rows.reason
        assert logical.physical_rows.basis == "logical"

    def test_a_core_whose_weights_the_walk_cannot_read_bounds_the_walk_by_name(self):
        class _Shape:
            shape = (3, 5)

        class _ShapeOnly(Mapper):
            def _map_to_ir(self, ir_mapping):
                sources = self.require_source_mapper().map_to_ir(ir_mapping)
                assert total_size(sources) == 5
                return ir_mapping.add_neural_core(input_sources=sources, weights=_Shape(), name="shape_only")

        layout, specs = _walk(ModelRepresentation(_ShapeOnly(InputMapper((5,)))))
        assert [int(s.input_count) for s in specs] == [10]
        assert layout.physical_rows.basis == "upper_bound"
        assert "shape_only" in layout.physical_rows.reason


class TestThePerSynapseWalk:
    PLAIN = [{k: v for k, v in c.items() if k != "weight_sign_granularity"} for c in CORES]

    def test_a_per_synapse_census_is_unchanged_by_the_declared_grid(self):
        """Guard: the ladder's walk on a per-synapse pool reads no weights, declared grid or not."""
        import mimarsinan.deployment_record.introspection.scaling_ladder as ladder

        net = ladder.LadderNetwork(name="convs", mapper_repr=_conv_rep(), resolution=(3, 4, 4))
        assert ladder.shape_census(net, {"cores": self.PLAIN, "weight_bits": BITS}).softcores == ladder.shape_census(
            net, {"cores": self.PLAIN}).softcores

    def test_a_per_synapse_walk_states_logical_rows(self):
        with_grid, specs = _walk(_conv_rep(), cores=self.PLAIN)
        assert with_grid.physical_rows.basis == "logical"
        assert specs == _walk(_conv_rep(), cores=self.PLAIN, weight_bits=None)[1]


class TestTheOneRule:
    @pytest.mark.parametrize("seed", range(4))
    def test_the_split_takes_the_rows_the_rule_counts(self, seed):
        from mimarsinan.mapping.packing.softcore.sign_split import physical_row_counts

        matrix = np.random.default_rng(seed).integers(-2, 3, size=(40, 3)).astype(np.float64)
        matrix[3] = 0.0
        origin, _signs = sign_split_rows(matrix)
        assert np.array_equal(np.bincount(origin, minlength=40), physical_row_counts(matrix))
        core = NeuralCore(id=0, name="c", input_sources=np.array([None] * 40, dtype=object), core_matrix=matrix,
                          threshold=1.0, latency=0)
        assert np.array_equal(deployed_row_counts(core, PER_AXON), physical_row_counts(matrix))


class TestTheCandidatePlanes:
    """The two walks that hold the network's weights take the grid from their platform's ``weight_bits``."""

    def test_the_joint_search_walk_reads_the_deployed_rows(self):
        from mimarsinan.search.problems.joint.problem import JointArchHwProblem

        rep = _conv_rep()

        class _Model:
            def get_mapper_repr(self):
                return rep

        problem = object.__new__(JointArchHwProblem)
        walk = problem._walk_candidate(_Model(), {"cores": CORES, "weight_bits": BITS})
        assert [int(s.input_count) for s in walk.softcores] == _deployed(rep)
        bound = problem._walk_candidate(_Model(), {"cores": CORES})
        assert [int(s.input_count) for s in bound.softcores] == [int(s.input_count) for s in _walk(
            rep, weight_bits=None)[1]]

    def test_the_ladder_census_and_row_state_their_rows_on_a_per_axon_pool_only(self):
        import mimarsinan.deployment_record.introspection.scaling_ladder as ladder

        rep = _conv_rep()
        net = ladder.LadderNetwork(name="convs", mapper_repr=rep, resolution=(3, 4, 4))
        deployed = ladder.shape_census(net, {"cores": CORES, "weight_bits": BITS})
        assert [int(s.input_count) for s in deployed.softcores] == _deployed(rep)
        assert deployed.physical_rows == {"basis": "deployed", "reason": None}
        bound = ladder.shape_census(net, {"cores": CORES})
        assert bound.physical_rows["basis"] == "upper_bound"
        plain = [{k: v for k, v in c.items() if k != "weight_sign_granularity"} for c in CORES]
        assert ladder.shape_census(net, {"cores": plain, "weight_bits": BITS}).physical_rows is None
