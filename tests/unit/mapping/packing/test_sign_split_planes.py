"""CL-SG on the shape planes: the layout walk plans two rows per input (the worst case, no weights), the scheduled
builder's specs and the capacity bound read the exact deployed rows, so the planner and the packer coincide."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_at_deployed_extent
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs
from mimarsinan.mapping.packing.hybrid_segment import _validate_coalescing_budget
from mimarsinan.mapping.packing.softcore import deployed_core_extent
from mimarsinan.mapping.platform.platform_constraints import (
    PER_AXON,
    PER_SYNAPSE,
    resolve_platform_mapping_params,
)
from mimarsinan.mapping.verification.capacity import estimate_cores_needed
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.search.problems.joint.problem import JointArchHwProblem


def _mlp_repr(widths=(24, 16, 10)):
    mapper = InputMapper((widths[0],))
    for fan_in, width in zip(widths[:-1], widths[1:]):
        mapper = PerceptronMapper(mapper, Perceptron(width, fan_in, normalization=nn.Identity(),
                                                     base_activation_name="ReLU"))
    return ModelRepresentation(mapper)


def _src(n, *, bias=True):
    sources = [IRSource(node_id=-2, index=i) for i in range(n)]
    if bias:
        sources.append(IRSource(node_id=-3, index=0))
    return np.array(sources, dtype=object)


def _core(node_id, matrix, *, bias=True):
    rows = matrix.shape[0] - (1 if bias else 0)
    return NeuralCore(id=node_id, name=f"c{node_id}", input_sources=_src(rows, bias=bias), core_matrix=matrix,
                      threshold=1.0, latency=0)


class TestTheLayoutWalk:
    def test_a_per_axon_walk_records_two_rows_per_input_and_a_logical_wire_census(self):
        params = resolve_platform_mapping_params([{"max_axons": 64, "max_neurons": 64, "has_bias": False,
                                                   "weight_sign_granularity": PER_AXON}])
        walks = {}
        for rows in (1, params.rows_per_input):
            layout = LayoutIRMapping(max_axons=params.effective_max_axons, max_neurons=params.effective_max_neurons,
                                     hardware_bias=False, collect_wire_census=True, rows_per_input=rows)
            walks[rows] = (layout.collect_layout_softcores(_mlp_repr()), layout)
        single, double = walks[1][0], walks[2][0]
        assert [s.input_count * 2 for s in single] == [s.input_count for s in double]
        assert [s.output_count for s in single] == [s.output_count for s in double]
        assert walks[1][1]._node_wire_census == walks[2][1]._node_wire_census
        assert max(s.input_count for s in double) <= 64

    def test_the_search_candidate_walk_plans_the_per_axon_rows(self):
        """The candidate plane (the layout hook's walk, which the joint search and the cost harness price) of a
        per-axon chip records twice the rows of a per-synapse chip with the same logical limit (31 inputs)."""
        class _Model:
            def get_mapper_repr(self):
                return _mlp_repr()

        problem = object.__new__(JointArchHwProblem)
        per_axon = {"cores": [{"max_axons": 64, "max_neurons": 32, "count": 8, "has_bias": False,
                               "weight_sign_granularity": PER_AXON}]}
        per_synapse = {"cores": [{"max_axons": 32, "max_neurons": 32, "count": 8, "has_bias": False}]}
        doubled = problem._walk_candidate(_Model(), per_axon).softcores
        single = problem._walk_candidate(_Model(), per_synapse).softcores
        assert [s.input_count for s in doubled] == [2 * s.input_count for s in single]
        assert [s.output_count for s in doubled] == [s.output_count for s in single]


def _conv_repr():
    """A bank-backed convolution (one weight bank, an instance per output position) and a classifier."""
    from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
    from mimarsinan.mapping.mappers.structural import EinopsRearrangeMapper

    conv = Conv2DPerceptronMapper(InputMapper((1, 6, 6)), in_channels=1, out_channels=3, kernel_size=3, stride=1,
                                  padding=0, bias=True, use_batchnorm=False, base_activation_name="ReLU")
    flat = EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
    return ModelRepresentation(PerceptronMapper(flat, Perceptron(4, 48, normalization=nn.Identity(),
                                                                 base_activation_name="ReLU")))


class TestNoBankResidencyUnderPerAxonSign:
    """A split instance is no longer its bank's region (its rows depend on the split), so under per-axon sign no
    plane plans bank residency: the layout walk and the scheduled specs carry no bank id, as the soft cores do."""

    def test_the_per_axon_walk_records_no_bank(self):
        plain = LayoutIRMapping(max_axons=63, max_neurons=64).collect_layout_softcores(_conv_repr())
        split = LayoutIRMapping(max_axons=63, max_neurons=64, rows_per_input=2).collect_layout_softcores(_conv_repr())
        assert any(s.bank_id is not None for s in plain)
        assert all(s.bank_id is None for s in split)

    def test_the_per_axon_scheduled_specs_carry_no_bank(self):
        from mimarsinan.mapping.ir_mapping_class import IRMapping

        ir = IRMapping(q_max=7.0, firing_mode="Default", max_axons=63, max_neurons=64).map(_conv_repr())
        for bank in ir.weight_banks.values():
            bank.core_matrix = np.round(np.asarray(bank.core_matrix) * 20)
        for core in ir.get_neural_cores():
            if core.core_matrix is not None:
                core.core_matrix = np.round(np.asarray(core.core_matrix) * 20)
        cores = list(ir.get_neural_cores())
        plain, _, _ = _segment_specs(cores, ir_graph=ir)
        split, spec_to_core, _ = _segment_specs(cores, ir_graph=ir, sign_granularity=PER_AXON)
        assert any(s.bank_id is not None for s in plain)
        assert all(s.bank_id is None for s in split)
        for spec in split:
            assert (spec.input_count, spec.output_count) == deployed_core_extent(
                spec_to_core[id(spec)], PER_AXON, graph=ir)


class TestDeployedSpecs:
    def test_a_spec_takes_the_deployed_rows_under_per_axon_only(self):
        w = np.array([[1, -1], [2, 2], [-1, 3]], dtype=np.float64)
        core = _core(0, w)
        spec = LayoutSoftCoreSpec(input_count=6, output_count=2, residency_class_id=0, latency_tag=0)
        assert spec_at_deployed_extent(spec, core, PER_SYNAPSE) is spec
        assert spec_at_deployed_extent(spec, core, PER_AXON).input_count == 5
        assert deployed_core_extent(core, PER_AXON) == (5, 2)

    def test_the_scheduled_specs_are_the_runtime_extent(self):
        rng = np.random.default_rng(0)
        cores = [_core(i, rng.integers(-3, 4, size=(9, 4)).astype(np.float64)) for i in range(3)]
        specs, spec_to_core, _ = _segment_specs(cores, ir_graph=None, sign_granularity=PER_AXON)
        for spec in specs:
            assert (spec.input_count, spec.output_count) == deployed_core_extent(spec_to_core[id(spec)], PER_AXON)
        plain, _, _ = _segment_specs(cores, ir_graph=None)
        assert [s.input_count for s in plain] == [9, 9, 9]


class TestCapacityBound:
    def test_the_bound_counts_deployed_rows_against_the_rows_of_a_full_tile(self):
        """16-row per-axon cores (7 logical inputs + bias, a full tile of mixed rows takes all 16 rows): eight
        8-row single-sign cores take 8 rows each, eight 8-row all-mixed cores 16 each; a full tile is one core."""
        single = [_core(i, np.ones((8, 2))) for i in range(8)]
        mixed = [_core(10 + i, np.tile([[1.0, -1.0]], (8, 1))) for i in range(8)]
        cores = [{"max_axons": 16, "max_neurons": 64, "count": 64, "has_bias": False,
                  "weight_sign_granularity": PER_AXON}]
        assert resolve_platform_mapping_params(cores).row_capacity == 16
        out = np.array([IRSource(node_id=0, index=0)], dtype=object)
        est_single = estimate_cores_needed(IRGraph(nodes=single, output_sources=out), {"cores": cores})
        est_mixed = estimate_cores_needed(IRGraph(nodes=mixed, output_sources=out), {"cores": cores})
        assert (est_single.cores_needed, est_mixed.cores_needed) == (math.ceil(64 / 16), math.ceil(128 / 16))
        one_core = [dict(cores[0], count=1)]
        scheduled = estimate_cores_needed(IRGraph(nodes=mixed, output_sources=out), {"cores": one_core},
                                          allow_scheduling=True)
        assert scheduled.feasible, "a full per-axon tile is one core, so a one-core chip runs it in passes"

    def test_a_per_synapse_platform_keeps_its_bound(self):
        mixed = [_core(10 + i, np.tile([[1.0, -1.0]], (8, 1))) for i in range(8)]
        plain = [{"max_axons": 16, "max_neurons": 64, "count": 64, "has_bias": False}]
        out = np.array([IRSource(node_id=0, index=0)], dtype=object)
        assert estimate_cores_needed(IRGraph(nodes=mixed, output_sources=out), {"cores": plain}).cores_needed == (
            math.ceil(64 / 15))


class TestCoalescingBudget:
    def test_the_budget_counts_the_physical_weight_rows(self):
        """A 7-input core whose rows are all mixed needs 14 weight rows: two 8-row cores under per-axon sign."""
        w = np.vstack([np.tile([[1.0, -1.0]], (7, 1)), [[1.0, 1.0]]])
        core = _core(0, w)
        per_axon = [{"max_axons": 8, "max_neurons": 8, "count": 1, "weight_sign_granularity": PER_AXON}]
        with pytest.raises(RuntimeError, match="14 weight axons"):
            _validate_coalescing_budget([core], per_axon, allow_neuron_splitting=False)
        _validate_coalescing_budget([core], [dict(per_axon[0], count=2)], allow_neuron_splitting=False)
        _validate_coalescing_budget([core], [dict(per_axon[0], weight_sign_granularity=PER_SYNAPSE)],
                                    allow_neuron_splitting=False)


def _lif_mlp_repr(widths=(300, 100, 10), timesteps=4):
    """The fidelity review's ODIN-planes vehicle: a 300-100-10 LIF MLP behind a 300-wide encoding layer."""
    import torch

    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

    def lif(width, fan_in):
        p = Perceptron(width, fan_in, normalization=nn.Identity())
        p.base_activation = LIFActivation(T=timesteps, activation_scale=torch.tensor(1.0))
        p.activation = p.base_activation
        with torch.no_grad():
            p.layer.weight.copy_((torch.rand(width, fan_in) - 0.5) * 2.0)
            p.layer.bias.fill_(0.3)
        return p

    torch.manual_seed(0)
    encoder = lif(widths[0], widths[0])
    encoder.is_encoding_layer = True
    encoder.use_cycle_accurate_trains = True
    node, fan_in = PerceptronMapper(InputMapper((widths[0],)), encoder), widths[0]
    for width in widths[1:]:
        node = PerceptronMapper(node, lif(width, fan_in))
        fan_in = width
    rep = ModelRepresentation(node)
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


class TestEveryLayoutPlaneCountsPhysicalRows:
    """[MJ1] The GUI hardware-verify route, the wizard's hardware suggester and the scaling ladder walk the layout at
    the per-axon rows (``rows_per_input``) the search candidate and the runtime plan, so on a per-axon pool each plane
    packs the candidate walk's rows: the review's probe network takes 4 ODIN cores on every plane (3 before).
    [pack integration] The layout packer keeps the 602-row spec whole on one fused core of three ODIN cores (W04 class
    1), so the stats count 2 allocated crossbars (a fused core once) and the packer's snapshots 4 physical cores."""

    PLATFORM = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": False,
                           "weight_sign_granularity": PER_AXON}],
                "allow_coalescing": True, "allow_neuron_splitting": False}

    def _candidate(self):
        from mimarsinan.mapping.layout.layout_packer import pack_layout
        from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
        from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
        from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats

        pm = resolve_platform_mapping_params(self.PLATFORM["cores"], allow_coalescing=True)
        walk = LayoutIRMapping(max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
                               allow_coalescing=True, hardware_bias=pm.hardware_bias, rows_per_input=pm.rows_per_input)
        specs = walk.collect_layout_softcores(_lif_mlp_repr())
        kwargs = ChipCapabilities.from_platform_constraints(self.PLATFORM).layout_kwargs()
        stats, error = compute_mapping_stats(list(specs), core_types_from_declaration(self.PLATFORM["cores"]),
                                             **kwargs)
        assert stats.feasible, error
        pack = pack_layout(softcores=list(specs), core_types=core_types_from_declaration(self.PLATFORM["cores"]),
                           allow_coalescing=True)
        tiles = sum(snap.tiles for snap in pack.used_core_snapshots or ())
        return sorted(int(s.input_count) for s in specs), int(stats.total_cores), tiles

    def test_the_candidate_walk_takes_four_cores(self):
        rows, cores, tiles = self._candidate()
        assert (rows, cores, tiles) == ([202, 602], 2, 4)  # 2 allocated crossbars, 4 physical ODIN cores

    def test_the_scaling_ladder_packs_the_candidate_rows(self):
        from mimarsinan.deployment_record.introspection.scaling_ladder import packed_tiles, shape_census
        from mimarsinan.deployment_record.introspection.scaling_ladder_types import LadderNetwork

        rows, _cores, tiles = self._candidate()
        census = shape_census(LadderNetwork(name="odin_planes", mapper_repr=_lif_mlp_repr(), resolution=(300,)),
                              self.PLATFORM)
        assert sorted(int(s.input_count) for s in census.softcores) == rows
        assert all(s.bank_id is None for s in census.softcores)
        assert census.as_mapped_cells == sum(int(s.input_count) * int(s.output_count) for s in census.softcores)
        assert packed_tiles(census, self.PLATFORM) == tiles  # the ladder counts physical cores (W04 fix round B1)

    def test_the_wizard_request_carries_the_rows_per_input(self, monkeypatch):
        from mimarsinan.mapping.verification.layout_mapping_service import LayoutMappingService
        from mimarsinan.mapping.verification.layout_request import LayoutMappingRequest

        rows, _cores, _tiles = self._candidate()
        body = {"model_type": "probe", "input_shape": [300], "num_classes": 10,
                "core_types": self.PLATFORM["cores"], "allow_coalescing": True}
        request = LayoutMappingRequest.from_wizard_body(body)
        assert request.rows_per_input == 2
        scalar = LayoutMappingRequest.from_wizard_body({k: v for k, v in body.items() if k != "core_types"},
                                                       tiling_max_axons=127, tiling_max_neurons=256,
                                                       rows_per_input=2)
        assert scalar.rows_per_input == 2 and scalar.verification_key() != LayoutMappingRequest.from_wizard_body(
            {k: v for k, v in body.items() if k != "core_types"}, tiling_max_axons=127,
            tiling_max_neurons=256).verification_key()
        service = LayoutMappingService()
        monkeypatch.setattr(service, "_build_model_repr", lambda request: _lif_mlp_repr())
        assert sorted(int(s.input_count) for s in service.get_verification(request).softcores) == rows

    def test_the_hardware_verify_route_counts_the_candidate_cores(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from mimarsinan.gui.server import routes_layout
        from mimarsinan.mapping.verification import layout_mapping_service as service_module
        from mimarsinan.mapping.verification.verifier.mapping_verifier_hw import clear_verify_hardware_config_memo

        _rows, cores, _tiles = self._candidate()
        service = service_module.LayoutMappingService()
        monkeypatch.setattr(service, "_build_model_repr", lambda request: _lif_mlp_repr())
        monkeypatch.setattr(service_module, "DEFAULT_LAYOUT_MAPPING_SERVICE", service)
        app = FastAPI()
        routes_layout.register_routes(app)
        clear_verify_hardware_config_memo()
        body = {"model_repr_json": {"model_type": "probe", "input_shape": [300], "num_classes": 10},
                "core_types": self.PLATFORM["cores"], "allow_coalescing": True}
        response = TestClient(app).post("/api/hw_config_verify", json=body).json()
        assert response["stats"]["total_cores"] == cores, response

    def test_the_hardware_suggester_walks_the_candidate_rows(self, monkeypatch):
        from mimarsinan.mapping.verification.suggester import hw_config_suggester_scheduled as suggester

        rows, _cores, _tiles = self._candidate()
        seen = {}

        def capture(softcores, **kwargs):
            seen["rows"] = sorted(int(s.input_count) for s in softcores)
            return "suggestion"

        monkeypatch.setattr(suggester, "suggest_hardware_config", capture)
        pm = resolve_platform_mapping_params(self.PLATFORM["cores"], allow_coalescing=True)
        suggester.suggest_hardware_config_for_model(
            _lif_mlp_repr(), max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
            allow_coalescing=True, hardware_bias=pm.hardware_bias, rows_per_input=pm.rows_per_input)
        assert seen["rows"] == rows
