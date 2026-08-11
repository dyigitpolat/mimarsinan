"""from_mapping converters mirror the live HybridHardCoreMapping exactly (W4 stage 2)."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.deployment_record.build.from_mapping import (
    placement_record_from_mapping,
    schedule_record_from_mapping,
    utilization_record_from_mapping,
)
from mimarsinan.deployment_record.schema import ComputeOpRecord, SegmentRecord
from mimarsinan.mapping.crossbar_utilization import (
    CoreOccupancy,
    CrossbarUtilizationReport,
)
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.packing.hybrid_types import HybridStage, SegmentIOSlice
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.verification.layout_verification_hybrid import (
    stats_dict_from_hybrid_mapping,
)
from mimarsinan.mapping.weight_programming import weight_programming_report

WEIGHT_BITS = 8


def _banked_token_graph(n_tokens=7, in_features=4, out_features=4):
    rows = in_features + 1
    rng = np.random.default_rng(7)
    bank = WeightBank(
        id=0,
        core_matrix=rng.normal(size=(rows, out_features)).astype(np.float32),
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = np.array(
            [IRSource(-2, tok * in_features + i) for i in range(in_features)]
            + [IRSource(-3, 0)],
            dtype=object,
        )
        nodes.append(NeuralCore(
            id=tok, name=f"b0_col{tok}", input_sources=srcs,
            core_matrix=None, weight_bank_id=0,
            weight_row_slice=(0, out_features), latency=0,
            perceptron_index=0, perceptron_output_column=tok,
            perceptron_output_slice=(0, out_features),
        ))
    out = np.array(
        [IRSource(n.id, j) for n in nodes for j in range(out_features)],
        dtype=object,
    )
    return IRGraph(nodes=nodes, output_sources=out, weight_banks={0: bank})


def _scheduled_hybrid(n_tokens=7, count=2, policy="bank_clustered"):
    strategy = MappingStrategy.resolve(ChipCapabilities(
        allow_scheduling=True, schedule_policy=policy,
    ))
    return build_hybrid_hard_core_mapping(
        ir_graph=_banked_token_graph(n_tokens),
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": count}],
        strategy=strategy,
    )


def _unscheduled_hybrid():
    return build_hybrid_hard_core_mapping(
        ir_graph=_banked_token_graph(3),
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 8}],
    )


class TestScheduleFragment:
    def test_bank_clustered_pass_and_sync_census(self):
        hybrid = _scheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=20,
        )
        # 7 banked instances over a 2-core resident pool: 4 passes, 3 syncs
        # (the layout-stats census, mirrored exactly).
        assert schedule.pass_count == 4
        assert schedule.sync_count == 3
        assert schedule.pass_count == stats_dict_from_hybrid_mapping(
            hybrid
        )["schedule_pass_count"]
        assert schedule.reprogram_passes == 1
        assert schedule.reuse_passes == 3
        assert schedule.params_reloaded == 20
        assert schedule.compute_op_count == 0
        segments = schedule.segments()
        assert [s.pass_index for s in segments] == [0, 1, 2, 3]
        assert [s.pass_reason for s in segments] == (
            ["initial"] + ["capacity_overflow"] * 3
        )
        assert all(s.bank_ids == (0,) for s in segments)

    def test_resident_segments_carry_zero_programming(self):
        hybrid = _scheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        first, *rest = schedule.segments()
        assert first.programming == "reprogram"
        assert first.params_programmed > 0
        assert first.params_bytes > 0
        for segment in rest:
            assert segment.programming == "resident"
            assert segment.params_programmed == 0
            assert segment.params_unique == 0
            assert segment.params_bytes == 0
            assert all(core.params_bytes == 0 for core in segment.cores)

    def test_programming_totals_reconcile_with_the_report(self):
        hybrid = _scheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        report = weight_programming_report(hybrid)
        assert sum(
            s.params_programmed for s in schedule.segments()
        ) == report.params_programmed

    def test_core_geometry_occupancy_and_span_counts_mirror_the_cores(self):
        hybrid = _scheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        neural_stages = [s for s in hybrid.stages if s.kind == "neural"]
        for segment, stage in zip(schedule.segments(), neural_stages):
            hcm = stage.hard_core_mapping
            assert len(segment.cores) == len(hcm.cores)
            for record, core in zip(segment.cores, hcm.cores):
                occupancy = CoreOccupancy.from_hard_core(core)
                assert record.axons == occupancy.axons_physical
                assert record.neurons == occupancy.neurons_physical
                assert record.axons_used == occupancy.axons_used
                assert record.neurons_used == occupancy.neurons_used
                assert record.cells_used == occupancy.cells_used
                assert record.connectivity_entries == len(
                    core.get_axon_source_spans()
                )
            assert segment.connectivity_entries == sum(
                c.connectivity_entries for c in segment.cores
            )

    def test_params_bytes_are_exact_for_the_declared_width(self):
        hybrid = _scheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        first = schedule.segments()[0]
        for core in first.cores:
            assert core.params_bytes == -((core.cells_used * WEIGHT_BITS) // -8)

    def test_unscheduled_program_reads_zero_passes_like_the_layout_stats(self):
        hybrid = _unscheduled_hybrid()
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        stats = stats_dict_from_hybrid_mapping(hybrid)
        assert schedule.pass_count == stats["schedule_pass_count"] == 0
        assert schedule.sync_count == stats["schedule_sync_count"] == 0
        for position, segment in enumerate(schedule.segments()):
            assert segment.segment_index == position
            assert segment.pass_index == 0
            assert segment.pass_reason == "initial"
            assert segment.programming == "reprogram"

    def test_compute_stage_mirrors_name_type_and_output_width(self):
        hybrid = _scheduled_hybrid()
        op = ComputeOp(
            id=99, name="pool_0",
            input_sources=np.array([], dtype=object),
            op_type="max_pool2d",
        )
        hybrid.stages.insert(0, HybridStage(
            kind="compute", name="pool_0", compute_op=op,
            output_map=[
                SegmentIOSlice(node_id=99, offset=0, size=3),
                SegmentIOSlice(node_id=99, offset=3, size=2),
            ],
        ))
        schedule = schedule_record_from_mapping(
            hybrid, weight_bits=WEIGHT_BITS, params_reloaded=0,
        )
        first = schedule.stages[0]
        assert isinstance(first, ComputeOpRecord)
        assert first.stage_index == 0
        assert first.name == "pool_0"
        assert first.op_type == "max_pool2d"
        assert first.output_width == 5
        assert first.wall_s_total is None  # timed only at stage 3
        assert schedule.compute_op_count == 1
        # The pass census is untouched by interleaved compute stages.
        assert schedule.pass_count == 4
        # Segment stage indices track program order past the compute stage.
        assert [
            s.stage_index for s in schedule.stages
            if isinstance(s, SegmentRecord)
        ] == [1, 2, 3, 4]

    def test_undeclared_weight_bits_fail_loud(self):
        with pytest.raises(ValueError, match="weight width"):
            schedule_record_from_mapping(
                _unscheduled_hybrid(), weight_bits=None, params_reloaded=0,
            )


class TestPlacementFragment:
    def test_every_placement_key_is_mirrored(self):
        hybrid = _scheduled_hybrid()
        placement = placement_record_from_mapping(hybrid)
        expected = []
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        for stage in neural:
            hcm = stage.hard_core_mapping
            for core_idx, placements in enumerate(
                hcm.soft_core_placements_per_hard_core
            ):
                for p in placements:
                    expected.append((stage, core_idx, p))
        assert len(placement.softcores) == len(expected)
        for record, (stage, core_idx, p) in zip(placement.softcores, expected):
            assert record.ir_node_id == p["ir_node_id"]
            assert record.segment_index == (stage.schedule_segment_index or 0)
            assert record.pass_index == (stage.schedule_pass_index or 0)
            assert record.hard_core_index == core_idx
            assert record.axon_offset == p["axon_offset"]
            assert record.neuron_offset == p["neuron_offset"]
            assert record.axons == p["axons"]
            assert record.neurons == p["neurons"]
            assert record.perceptron_index == p.get("perceptron_index")
            assert record.weight_bank_id == p.get("weight_bank_id")
            assert record.bank_axon_range == p.get("bank_axon_range")
            assert record.bank_neuron_range == p.get("bank_neuron_range")
            assert record.split_group_id == p.get("split_group_id")
            assert record.split_fragment_index == p.get("split_fragment_index")
            assert record.coalescing_group_id == p.get("coalescing_group_id")

    def test_bank_records_carry_geometry_and_sharing_degree(self):
        hybrid = _scheduled_hybrid()
        placement = placement_record_from_mapping(hybrid)
        (bank,) = placement.banks
        assert bank.bank_id == 0
        assert (bank.rows, bank.cols) == (5, 4)
        assert bank.params == 20
        assert bank.placement_count == sum(
            1 for record in placement.softcores if record.weight_bank_id == 0
        )
        assert bank.placement_count > 0

    def test_floorplan_and_tiles_stay_empty_at_this_stage(self):
        placement = placement_record_from_mapping(_scheduled_hybrid())
        assert placement.floorplan is None
        assert placement.tiles == ()


class TestUtilizationFragment:
    def test_mirrors_equal_the_source_reports(self):
        hybrid = _scheduled_hybrid()
        report = CrossbarUtilizationReport.from_hybrid_mapping(
            hybrid, weight_bits=WEIGHT_BITS,
        )
        record = utilization_record_from_mapping(
            hybrid, crossbar_report=report, relay_cores_inserted=2,
        )
        assert record.crossbar.to_dict() == report.to_dict()
        assert record.layout.to_dict() == stats_dict_from_hybrid_mapping(hybrid)
        assert record.relay_cores_inserted == 2
