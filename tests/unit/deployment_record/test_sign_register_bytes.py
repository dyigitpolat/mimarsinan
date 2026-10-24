"""[MJ2] The per-row sign register of a per-axon core counts in ``params_bytes`` (plan section 21 item 11, conductor
choice; ``odin_sign/design.md``: one bit per used row). One rule (``payload_sizes.core_payload_bytes``) is read by
every payload site on both planes: the record's core records, the candidate's programming census, the period census
of the candidate, built and sealed planes, and the scaling ladder's write-once payload. A per-synapse core carries no
register, so its bytes are unchanged.

Vehicles: t0_61's on-chip network (256 -> 128 -> 256 -> 10 behind its host encoder, on 64 ODIN cores of 256 x 256
per-axon with coalescing, at 4 bits; the candidate plane prices 68,490 B of cells, the review's t0_61 cost rows), and
a chain that runs in passes on one per-axon core (every plane rewrites it each period)."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.deployment_record.build.payload_sizes import (
    core_payload_bytes,
    params_bytes,
    sign_register_bytes,
    sign_register_rows,
)
from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    candidate_programming_census,
    mapping_program_passes,
    pass_program_bytes,
    record_program_passes,
)
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses, period_census
from mimarsinan.deployment_record.schema import SegmentCoreRecord
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs, build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.platform.platform_constraints import (
    PER_AXON,
    PER_SYNAPSE,
    carries_sign_register,
    resolve_platform_mapping_params,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.deployment_record.g3_fixtures import record_with

BITS = 4
ODIN = {"max_axons": 256, "max_neurons": 256, "count": 64, "has_bias": False, "weight_sign_granularity": PER_AXON}
#: t0_61's resolved platform (the ODIN bundle with the template's overrides).
T0_61 = {"cores": [ODIN], "allow_coalescing": True, "allow_neuron_splitting": False, "weight_bits": BITS}
_SPIKING = SimpleNamespace(retimed=False, is_cycle=False, is_cascade=False, latency_group_count=None,
                           chip_latency=None)


class TestTheRule:
    @pytest.mark.parametrize("rows", range(0, 41))
    def test_one_bit_per_used_row_rounded_up_per_core(self, rows):
        assert sign_register_bytes(rows) == math.ceil(rows / 8)
        assert core_payload_bytes(3 * rows + 1, BITS, rows) == params_bytes(3 * rows + 1, BITS) + math.ceil(rows / 8)

    def test_only_a_per_axon_core_signs_its_rows(self):
        assert sign_register_rows(PER_AXON, 17) == 17
        assert sign_register_rows(PER_SYNAPSE, 17) == 0
        assert sign_register_rows(None, 17) == 0
        assert core_payload_bytes(100, BITS) == params_bytes(100, BITS)
        with pytest.raises(ValueError):
            sign_register_bytes(-1)

    def test_a_shape_only_core_carries_the_register_of_its_declared_type(self):
        per_synapse = dict(ODIN, weight_sign_granularity=PER_SYNAPSE)
        assert carries_sign_register(256, 256, [ODIN])
        assert carries_sign_register(768, 256, [ODIN]), "a fused core of per-axon components"
        assert not carries_sign_register(256, 256, [per_synapse])
        assert not carries_sign_register(256, 256, [])
        other = {"max_axons": 64, "max_neurons": 32, "count": 4}
        assert not carries_sign_register(64, 32, [ODIN, other])
        assert carries_sign_register(256, 256, [ODIN, other])
        # Two declared types of one geometry and opposite granularity: a shape-only plane cannot tell which the packer
        # takes, so it counts the register (the per-axon fold of a fused core, ``hard_core_mapping``).
        assert carries_sign_register(256, 256, [per_synapse, ODIN])

    def test_the_period_census_rule_adds_the_register_of_each_written_core(self):
        rows = (PassRow(segment_index=0, resident=False, core_cells=(300, 40), marks=frozenset(), core_extents=None,
                        compute_steps=None, core_sign_rows=(30, 9)),
                PassRow(segment_index=0, resident=False, core_cells=(50,), marks=frozenset(), core_extents=None,
                        compute_steps=None))
        census = period_census(ProgramPasses(rows=rows), weight_bits=BITS, capacity={(256, 256): 2}).census
        assert [p.payload_bytes for p in census.passes] == [150 + 4 + 20 + 2, 25]


def _t0_61_repr():
    mapper = InputMapper((256,))
    for fan_in, width in ((256, 128), (128, 256), (256, 10)):
        mapper = PerceptronMapper(mapper, Perceptron(width, fan_in, normalization=nn.Identity(),
                                                     base_activation_name="ReLU"))
    rep = ModelRepresentation(mapper)
    rep.assign_perceptron_indices()
    return rep


def _t0_61_mapping(cores=(ODIN,)):
    """t0_61's on-chip IR at the per-axon tiler (127 logical inputs, coalescing), integer weights on the 4-bit grid."""
    platform = dict(T0_61, cores=[dict(c) for c in cores])
    pm = resolve_platform_mapping_params(platform["cores"], allow_coalescing=True)
    ir = IRMapping(q_max=7.0, firing_mode="Default", max_axons=pm.effective_max_axons,
                   max_neurons=pm.effective_max_neurons, allow_coalescing=True,
                   hardware_bias=pm.hardware_bias).map(_t0_61_repr())
    rng = np.random.default_rng(61)
    for core in ir.get_neural_cores():
        core.core_matrix = rng.integers(-7, 8, size=np.asarray(core.core_matrix).shape).astype(np.float64)
        core.threshold = 16.0
    return build_hybrid_mapping_for_pipeline(ir, platform, pipeline_config={})


def _record_cores(mapping):
    schedule = schedule_record_from_mapping(mapping, weight_bits=BITS, params_reloaded=0)
    return schedule, [c for s in schedule.segments() for c in s.cores]


class TestTheRecordPlane:
    def test_every_per_axon_core_record_counts_its_sign_register(self):
        schedule, cores = _record_cores(_t0_61_mapping())
        assert len(cores) == 3 and sum(c.axons_used for c in cores) == 1285
        for core in cores:
            assert core.sign_register_rows == core.axons_used
            assert core.params_bytes == params_bytes(core.cells_used, BITS) + math.ceil(core.axons_used / 8)
            data = core.to_dict()
            assert data["sign_register_rows"] == core.axons_used
            assert SegmentCoreRecord.from_dict(data) == core
        cells_bytes = sum(params_bytes(c.cells_used, BITS) for c in cores)
        assert cells_bytes == 68_485
        assert sum(s.params_bytes for s in schedule.segments()) == cells_bytes + 163

    def test_a_per_synapse_record_keeps_its_bytes_and_seals_no_register(self):
        """The base's t0_61 bundle (128 x 256 per-synapse cores): the same IR tiling, no register, no new key."""
        base = {"max_axons": 128, "max_neurons": 256, "count": 64, "has_bias": False}
        schedule, cores = _record_cores(_t0_61_mapping(cores=(base,)))
        for core in cores:
            assert core.sign_register_rows is None
            assert core.params_bytes == params_bytes(core.cells_used, BITS)
            assert "sign_register_rows" not in core.to_dict()
        sealed = [c for s in schedule.to_dict()["stages"] if s["stage_kind"] == "segment" for c in s["cores"]]
        assert sealed and not any("sign_register_rows" in c for c in sealed), "a parent's asdict keeps no unset key"

    def test_the_schedule_seals_the_register_and_round_trips(self):
        from mimarsinan.deployment_record.schema.schedule import ScheduleRecord

        schedule, cores = _record_cores(_t0_61_mapping())
        data = schedule.to_dict()
        sealed = [c for s in data["stages"] if s["stage_kind"] == "segment" for c in s["cores"]]
        assert [c["sign_register_rows"] for c in sealed] == [c.axons_used for c in cores]
        assert ScheduleRecord.from_dict(data) == schedule


class TestTheCandidatePlane:
    def _candidate(self):
        pm = resolve_platform_mapping_params(T0_61["cores"], allow_coalescing=True)
        walk = LayoutIRMapping(max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
                               allow_coalescing=True, hardware_bias=pm.hardware_bias, rows_per_input=pm.rows_per_input)
        specs = walk.collect_layout_softcores(_t0_61_repr())
        noc = collect_noc_fragments(softcores=specs, core_types=core_types_from_declaration(T0_61["cores"]),
                                    census=None, declared_cores=T0_61["cores"],
                                    **ChipCapabilities.from_platform_constraints(T0_61).layout_kwargs())
        return specs, noc

    def test_the_candidate_census_prices_the_register_of_every_core_it_programs(self):
        _specs, noc = self._candidate()
        census = candidate_programming_census(noc, weight_bits=BITS)
        rows = [r for p in noc.pass_programs for r in p.core_sign_rows]
        assert len(rows) == len(census.reprogrammed_cells) and all(r > 0 for r in rows)
        cells_bytes = sum(params_bytes(c, BITS) for c in census.reprogrammed_cells)
        assert cells_bytes == 68_490  # t0_61's candidate cost rows before the register
        assert census.reprogrammed_bytes == cells_bytes + sum(math.ceil(r / 8) for r in rows)
        assert census.reprogrammed_bytes == sum(b for p in noc.pass_programs for b in pass_program_bytes(p, BITS))

    def test_both_planes_sign_the_rows_they_program(self):
        """The candidate signs the rows its walk plans (two per input, the worst case without weights); the record
        signs the rows the split placed. They differ by the rows the deployment left unsplit (one single-sign row
        of the 643 logical rows here), each on its own plane's cores."""
        specs, noc = self._candidate()
        _schedule, cores = _record_cores(_t0_61_mapping())
        candidate_rows = sum(r for p in noc.pass_programs for r in p.core_sign_rows)
        assert candidate_rows == sum(int(s.input_count) for s in specs) == 1286
        assert candidate_rows - sum(c.sign_register_rows for c in cores) == 1

    def test_a_per_synapse_candidate_signs_no_row(self):
        base = [{"max_axons": 128, "max_neurons": 256, "count": 64, "has_bias": False}]
        pm = resolve_platform_mapping_params(base, allow_coalescing=True)
        specs = LayoutIRMapping(max_axons=pm.effective_max_axons, max_neurons=pm.effective_max_neurons,
                                allow_coalescing=True, hardware_bias=False).collect_layout_softcores(_t0_61_repr())
        noc = collect_noc_fragments(softcores=specs, core_types=core_types_from_declaration(base), census=None,
                                    declared_cores=base,
                                    **ChipCapabilities.from_platform_constraints(dict(T0_61, cores=base)).layout_kwargs())
        census = candidate_programming_census(noc, weight_bits=BITS)
        assert all(r == 0 for p in noc.pass_programs for r in p.core_sign_rows)
        assert census.reprogrammed_bytes == sum(params_bytes(c, BITS) for c in census.reprogrammed_cells)


def _chain_ir():
    """A three-core chain whose split rows take a 256-row core each: on one per-axon core it runs in passes."""
    rng = np.random.default_rng(3)
    nodes, prev = [], -2
    for i, (a, b) in enumerate(((100, 60), (60, 90), (90, 12))):
        w = rng.integers(-7, 8, size=(a + 1, b)).astype(np.float64)
        w[1, :] = np.abs(w[1, :]) + 1.0  # one single-sign row: it takes one row and one sign bit
        src = np.array([IRSource(node_id=prev, index=j) for j in range(a)] + [IRSource(node_id=-3, index=0)],
                       dtype=object)
        nodes.append(NeuralCore(id=i, name=f"l{i}", input_sources=src, core_matrix=w, threshold=16.0, latency=i))
        prev = i
    return IRGraph(nodes=nodes, output_sources=np.array([IRSource(node_id=prev, index=j) for j in range(12)],
                                                        dtype=object))


class TestThePeriodCensusPlanes:
    """The chain on one per-axon core: every pass rewrites, and the candidate (planned at the deployed rows), built
    and sealed planes price the same payload, the register of each written core included."""

    CORE = {"max_axons": 256, "max_neurons": 256, "count": 1, "has_bias": False, "weight_sign_granularity": PER_AXON}

    def _planes(self):
        ir = _chain_ir()
        mapping = build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[dict(self.CORE)],
                                                 strategy=MappingStrategy.resolve(ChipCapabilities(
                                                     allow_scheduling=True)))
        specs, _, _ = _segment_specs(list(ir.get_neural_cores()), ir_graph=ir, sign_granularity=PER_AXON)
        noc = collect_noc_fragments(softcores=specs, core_types=core_types_from_declaration([self.CORE]),
                                    census=None, allow_scheduling=True, allow_neuron_splitting=False,
                                    allow_coalescing=False, max_schedule_passes=8, declared_cores=[self.CORE])
        schedule = schedule_record_from_mapping(mapping, weight_bits=BITS, params_reloaded=0, timesteps=4,
                                                pass_transfer="collapse")
        record = record_with(schedule, cores=[dict(self.CORE)])
        return specs, noc, mapping, schedule, record

    def test_every_plane_rewrites_the_cells_and_the_register(self):
        specs, noc, mapping, schedule, record = self._planes()
        capacity = {(256, 256): 1}
        planes = {
            "candidate": candidate_program_passes(specs, noc, semantics=_SPIKING, timesteps=4),
            "built": mapping_program_passes(mapping),
            "sealed": record_program_passes(record),
        }
        payloads = {name: [p.payload_bytes for p in period_census(passes, weight_bits=BITS,
                                                                    capacity=capacity).census.passes]
                    for name, passes in planes.items()}
        sealed = [s.params_bytes for s in schedule.segments()]
        assert len(sealed) == 3 and all(s.programming == "reprogram" for s in schedule.segments())
        assert payloads == {"candidate": sealed, "built": sealed, "sealed": sealed}
        cores = [c for s in schedule.segments() for c in s.cores]
        assert all(c.sign_register_rows == c.axons_used for c in cores)
        assert sum(sealed) == sum(params_bytes(c.cells_used, BITS) + math.ceil(c.axons_used / 8) for c in cores)
        assert candidate_programming_census(noc, weight_bits=BITS).reprogrammed_bytes == sum(sealed)


class TestTheLadder:
    def test_the_write_once_payload_reads_the_one_rule(self):
        """The ladder's write-once variant prices every tile of the program once (``pass_program_bytes``)."""
        from mimarsinan.deployment_record.introspection import scaling_ladder

        source = inspect.getsource(scaling_ladder.price_program)
        assert "pass_program_bytes" in source and "params_bytes(" not in source
        assert "declared_cores" in inspect.getsource(scaling_ladder)
