"""Converters: a live ``HybridHardCoreMapping`` → typed deployment-record fragments.

Every converter is a pure read that REUSES the existing measurement code
(``weight_programming_report``, ``CoreOccupancy``, the layout stats engine,
the span SSOT) — it mirrors, never re-derives. The per-segment programming
census lives in ``program_census``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from mimarsinan.deployment_record.build.pass_carry import carry_record_from_mapping, wavefront_carry
from mimarsinan.deployment_record.build.payload_sizes import require_weight_bits
from mimarsinan.deployment_record.build.program_census import (
    residency_plan_record, segment_record,
)
from mimarsinan.deployment_record.schema import (
    BankRecord,
    ComputeOpRecord,
    ComputePartitionRecord,
    CrossbarUtilizationRecord,
    LayoutStatsRecord,
    PlacementRecord,
    ScheduleRecord,
    SegmentRecord,
    SoftcorePlacementRecord,
    UtilizationRecord,
)
from mimarsinan.deployment_record.schema.wavefront import WavefrontRecord
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.support.wavefront_stages import wavefront_head
from mimarsinan.mapping.verification.layout_verification_hybrid import (
    stats_dict_from_hybrid_mapping,
)


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
    hybrid_mapping: Any, *, weight_bits: Any, params_reloaded: int,
    timesteps: Any = None, pass_transfer: Any = None, value_noc_wire_hops: Optional[int] = None,
    operand_chip: bool = False,
) -> ScheduleRecord:
    """Mirror ``HybridHardCoreMapping.stages`` 1:1 into the schedule fragment.

    ``params_reloaded`` is the IR-level plan figure from
    ``weight_reuse_plan_from_graph`` (provenance "planned@scm");
    ``value_noc_wire_hops`` the value-domain wire census the run seals
    (``value_domain.record_value_noc_wire_hops``), None for none. [U-C C-6] On a chip with an operand core type
    (``operand_chip``) or a program with bound banks, every core seals its operand-only mark.
    """
    bits = require_weight_bits(weight_bits)
    neural = {
        stage_index: (segment_index, pass_index, stage)
        for stage_index, segment_index, pass_index, stage
        in _iter_neural_stages(hybrid_mapping)
    }
    stages_out: List[Any] = []
    # the program's bound banks: [U-C C-6] the operand-only marks, [E-6] the operand regions' cells (where any bind)
    banks = dict(getattr(hybrid_mapping, "dynamic_bank_bindings", None) or {})
    for stage_index, stage in enumerate(hybrid_mapping.stages):
        if stage.kind == "compute":
            stages_out.append(_compute_op_record(stage_index, stage))
        elif stage.kind == "neural":
            segment_index, pass_index, _ = neural[stage_index]
            stages_out.append(segment_record(
                stage_index, segment_index, pass_index, stage, weight_bits=bits,
                banks=banks if operand_chip or banks else None,
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
        carry=carry_record_from_mapping(hybrid_mapping, timesteps, pass_transfer),
        residency_plan=residency_plan_record(hybrid_mapping),
        value_noc_wire_hops=None if value_noc_wire_hops is None else int(value_noc_wire_hops),
        wavefronts=_wavefronts(hybrid_mapping, timesteps, pass_transfer),
    )


def _wavefronts(hybrid_mapping: Any, timesteps: Any, transfer: Any) -> Optional[Tuple[WavefrontRecord, ...]]:
    """[U-F4] Every wavefront group of the program, read off its first pass, with the carry it holds itself
    (``wavefront_carry``); None for a program without one."""
    carry = wavefront_carry(hybrid_mapping.stages, timesteps, transfer)
    rows = tuple(WavefrontRecord(segment_index=int(stage.schedule_segment_index),
                                 first_pass=int(stage.schedule_pass_index), passes=int(head.passes),
                                 levels=tuple(head.levels), layout_cores=int(head.layout_cores),
                                 **carry.get(index, {}))
                 for index, stage in enumerate(hybrid_mapping.stages) if (head := wavefront_head(stage)) is not None)
    return rows or None


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
    partition: Optional[ComputePartitionRecord] = None,
    declared_core_types: Optional[Any] = None,
) -> UtilizationRecord:
    """Typed mirrors of the two existing reports + the threaded relay count.

    ``crossbar_report`` is the report the HCM step already computed; the layout
    stats come from ``stats_dict_from_hybrid_mapping`` (the wizard/GUI engine,
    with the DECLARED chip's core types so chip-relative figures never divide
    by the allocation [R1]), never a recomputation with different formulas.
    """
    stats_dict = stats_dict_from_hybrid_mapping(
        hybrid_mapping, core_types=declared_core_types,
    )
    if stats_dict is None:
        raise ValueError("layout stats unavailable: no packed stages")
    return UtilizationRecord(
        crossbar=CrossbarUtilizationRecord.from_dict(crossbar_report.to_dict()),
        layout=LayoutStatsRecord.from_dict(stats_dict),
        relay_cores_inserted=int(relay_cores_inserted),
        partition=partition,
    )
