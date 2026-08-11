"""Converters: a live ``HybridHardCoreMapping`` → typed deployment-record fragments.

Every converter is a pure read that REUSES the existing measurement code
(``weight_programming_report``, ``CoreOccupancy``, the layout stats engine,
the span SSOT) — it mirrors, never re-derives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.deployment_record.build.payload_sizes import (
    core_connectivity_entries,
    params_bytes,
    require_weight_bits,
)
from mimarsinan.deployment_record.schema import (
    BankRecord,
    ComputeOpRecord,
    CrossbarUtilizationRecord,
    LayoutStatsRecord,
    PlacementRecord,
    ScheduleRecord,
    SegmentCoreRecord,
    SegmentRecord,
    SoftcorePlacementRecord,
    UtilizationRecord,
)
from mimarsinan.mapping.crossbar_utilization import (
    CoreOccupancy,
    CrossbarUtilizationReport,
)
from mimarsinan.mapping.verification.layout_verification_hybrid import (
    stats_dict_from_hybrid_mapping,
)
from mimarsinan.mapping.weight_programming import weight_programming_report


class _SingleStageProgram:
    """One-stage view so per-segment programming calls the report verbatim."""

    def __init__(self, stage: Any) -> None:
        self.stages = [stage]


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _opt_pair(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    first, second = value
    return (int(first), int(second))


def _iter_neural_stages(hybrid_mapping: Any) -> List[Tuple[int, int, int, Any]]:
    """``(stage_index, segment_index, pass_index, stage)`` in program order.

    Unscheduled programs (no ``schedule_segment_index``) fall back to the
    neural position as the segment index, pass 0.
    """
    out: List[Tuple[int, int, int, Any]] = []
    neural_position = 0
    for stage_index, stage in enumerate(hybrid_mapping.stages):
        if stage.kind != "neural":
            continue
        if stage.hard_core_mapping is None:
            raise ValueError(
                f"neural stage {stage_index} ({stage.name!r}) carries no "
                f"hard_core_mapping; the program is not deployable"
            )
        seg = stage.schedule_segment_index
        segment_index = int(seg) if seg is not None else neural_position
        pass_index = int(stage.schedule_pass_index or 0)
        out.append((stage_index, segment_index, pass_index, stage))
        neural_position += 1
    return out


def _pass_census(hybrid_mapping: Any) -> Tuple[int, int]:
    """``(pass_count, sync_count)`` — the exact ``LayoutPlan.from_hybrid_mapping``
    census, so the seal's pass-count cross-check holds by construction."""
    schedule_pass_present = False
    max_pass_by_segment: Dict[int, int] = {}
    for stage in hybrid_mapping.stages:
        if stage.kind != "neural" or stage.hard_core_mapping is None:
            continue
        sched_idx = stage.schedule_pass_index
        seg_idx = int(stage.schedule_segment_index or 0)
        if sched_idx is not None:
            schedule_pass_present = True
            max_pass_by_segment[seg_idx] = max(
                max_pass_by_segment.get(seg_idx, 0), int(sched_idx) + 1
            )
    if not schedule_pass_present:
        return 0, 0
    pass_count = sum(max_pass_by_segment.values())
    return pass_count, max(0, pass_count - len(max_pass_by_segment))


def _core_record(
    core_index: int, core: Any, *, weight_bits: int, resident: bool
) -> SegmentCoreRecord:
    occupancy = CoreOccupancy.from_hard_core(core)
    latency = getattr(core, "latency", None)
    return SegmentCoreRecord(
        core_index=int(core_index),
        axons=int(occupancy.axons_physical),
        neurons=int(occupancy.neurons_physical),
        axons_used=int(occupancy.axons_used),
        neurons_used=int(occupancy.neurons_used),
        cells_used=int(occupancy.cells_used),
        # A resident pass sends no weight payload (schema §2.1: 0 when resident).
        params_bytes=0 if resident else params_bytes(occupancy.cells_used, weight_bits),
        connectivity_entries=core_connectivity_entries(core),
        static_delay_levels=None if latency is None else int(latency),
    )


def _segment_record(
    stage_index: int,
    segment_index: int,
    pass_index: int,
    stage: Any,
    *,
    weight_bits: int,
) -> SegmentRecord:
    segment = stage.hard_core_mapping
    resident = bool(getattr(stage, "schedule_weights_resident", False))
    programming = weight_programming_report(_SingleStageProgram(stage))
    cores = tuple(
        _core_record(index, core, weight_bits=weight_bits, resident=resident)
        for index, core in enumerate(segment.cores)
    )
    bank_ids = sorted({
        int(placement["weight_bank_id"])
        for placements in segment.soft_core_placements_per_hard_core
        for placement in placements
        if placement.get("weight_bank_id") is not None
    })
    return SegmentRecord(
        stage_index=int(stage_index),
        segment_index=int(segment_index),
        pass_index=int(pass_index),
        pass_reason="initial" if pass_index == 0 else "capacity_overflow",
        programming="resident" if resident else "reprogram",
        bank_ids=tuple(bank_ids),
        cores=cores,
        params_programmed=int(programming.params_programmed),
        params_unique=int(programming.params_unique),
        params_bytes=sum(core.params_bytes for core in cores),
        connectivity_entries=sum(core.connectivity_entries for core in cores),
        # The dependence-level currency: distinct latency tiers in the segment.
        static_latency_levels=len({
            int(core.latency)
            for core in segment.cores
            if getattr(core, "latency", None) is not None
        }),
    )


def _compute_op_record(stage_index: int, stage: Any) -> ComputeOpRecord:
    op = stage.compute_op
    if op is None:
        raise ValueError(
            f"compute stage {stage_index} ({stage.name!r}) carries no ComputeOp"
        )
    return ComputeOpRecord(
        stage_index=int(stage_index),
        name=str(stage.name),
        op_type=str(op.op_type),
        output_width=sum(int(io_slice.size) for io_slice in stage.output_map),
        wall_s_total=None,
    )


def schedule_record_from_mapping(
    hybrid_mapping: Any, *, weight_bits: Any, params_reloaded: int
) -> ScheduleRecord:
    """Mirror ``HybridHardCoreMapping.stages`` 1:1 into the schedule fragment.

    ``params_reloaded`` is the IR-level plan figure from
    ``weight_reuse_plan_from_graph`` (provenance "planned@scm").
    """
    bits = require_weight_bits(weight_bits)
    neural = {
        stage_index: (segment_index, pass_index, stage)
        for stage_index, segment_index, pass_index, stage
        in _iter_neural_stages(hybrid_mapping)
    }
    stages_out: List[Any] = []
    for stage_index, stage in enumerate(hybrid_mapping.stages):
        if stage.kind == "compute":
            stages_out.append(_compute_op_record(stage_index, stage))
        elif stage.kind == "neural":
            segment_index, pass_index, _ = neural[stage_index]
            stages_out.append(_segment_record(
                stage_index, segment_index, pass_index, stage, weight_bits=bits,
            ))
        else:
            raise ValueError(
                f"stage {stage_index} has unknown kind {stage.kind!r}"
            )
    segments = [s for s in stages_out if isinstance(s, SegmentRecord)]
    pass_count, sync_count = _pass_census(hybrid_mapping)
    return ScheduleRecord(
        stages=tuple(stages_out),
        pass_count=pass_count,
        sync_count=sync_count,
        reprogram_passes=sum(1 for s in segments if s.programming == "reprogram"),
        reuse_passes=sum(1 for s in segments if s.programming == "resident"),
        params_reloaded=int(params_reloaded),
        compute_op_count=len(stages_out) - len(segments),
    )


def placement_record_from_mapping(hybrid_mapping: Any) -> PlacementRecord:
    """A pure read of every ``soft_core_placements_per_hard_core`` entry + banks.

    Floorplan/tiles stay ``None``/empty at this stage (SANA-FE joins later).
    """
    softcores: List[SoftcorePlacementRecord] = []
    bank_matrices: Dict[int, Any] = {}
    bank_placement_counts: Dict[int, int] = {}
    for _, segment_index, pass_index, stage in _iter_neural_stages(hybrid_mapping):
        segment = stage.hard_core_mapping
        for bank_id, matrix in (getattr(segment, "weight_banks", {}) or {}).items():
            bank_matrices.setdefault(int(bank_id), matrix)
        placements_per_core = segment.soft_core_placements_per_hard_core
        for hard_core_index, placements in enumerate(placements_per_core):
            for placement in placements:
                bank_id = _opt_int(placement.get("weight_bank_id"))
                if bank_id is not None:
                    bank_placement_counts[bank_id] = (
                        bank_placement_counts.get(bank_id, 0) + 1
                    )
                softcores.append(SoftcorePlacementRecord(
                    ir_node_id=int(placement["ir_node_id"]),
                    segment_index=int(segment_index),
                    pass_index=int(pass_index),
                    hard_core_index=int(hard_core_index),
                    axon_offset=int(placement["axon_offset"]),
                    neuron_offset=int(placement["neuron_offset"]),
                    axons=int(placement["axons"]),
                    neurons=int(placement["neurons"]),
                    perceptron_index=_opt_int(placement.get("perceptron_index")),
                    weight_bank_id=bank_id,
                    bank_axon_range=_opt_pair(placement.get("bank_axon_range")),
                    bank_neuron_range=_opt_pair(placement.get("bank_neuron_range")),
                    split_group_id=_opt_int(placement.get("split_group_id")),
                    split_fragment_index=_opt_int(
                        placement.get("split_fragment_index")
                    ),
                    coalescing_group_id=_opt_int(
                        placement.get("coalescing_group_id")
                    ),
                ))
    banks = tuple(
        BankRecord(
            bank_id=int(bank_id),
            rows=int(matrix.shape[0]),
            cols=int(matrix.shape[1]),
            params=int(matrix.size),
            placement_count=bank_placement_counts.get(int(bank_id), 0),
        )
        for bank_id, matrix in sorted(bank_matrices.items())
    )
    return PlacementRecord(
        softcores=tuple(softcores), banks=banks, floorplan=None, tiles=(),
    )


def utilization_record_from_mapping(
    hybrid_mapping: Any,
    *,
    crossbar_report: CrossbarUtilizationReport,
    relay_cores_inserted: int,
) -> UtilizationRecord:
    """Typed mirrors of the two existing reports + the threaded relay count.

    ``crossbar_report`` is the report the HCM step already computed; the layout
    stats come from ``stats_dict_from_hybrid_mapping`` (the wizard/GUI engine),
    never a recomputation with different formulas.
    """
    stats_dict = stats_dict_from_hybrid_mapping(hybrid_mapping)
    if stats_dict is None:
        raise ValueError(
            "layout stats unavailable: the hybrid mapping has no packed stages"
        )
    return UtilizationRecord(
        crossbar=CrossbarUtilizationRecord.from_dict(crossbar_report.to_dict()),
        layout=LayoutStatsRecord.from_dict(stats_dict),
        relay_cores_inserted=int(relay_cores_inserted),
    )
