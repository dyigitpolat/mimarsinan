"""How each view answers each payload — pure reads, no new measurement.

The rollups are functions of ROWS, so a candidate's shape-only softcores and a
sealed record's placements are aggregated by ONE implementation: layer identity
is ``perceptron_index`` and bank identity is ``bank_id`` on both sides. The
core NAME is a label here, never a key — splitting it on separator conventions
(``_tile_``, ``_psum_pos_``, …) was how the old rollup guessed at layer identity
the mapper already knew.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.deployment_record.introspection.payloads import (
    BankRow,
    LayerRollupRow,
    PlacementRow,
    SegmentPassRow,
    SoftcoreRow,
)
from mimarsinan.deployment_record.schema.placement import (
    BankRecord,
    SoftcorePlacementRecord,
)
from mimarsinan.deployment_record.schema.schedule import ScheduleRecord
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def softcore_rows_from_specs(
    specs: Sequence[LayoutSoftCoreSpec],
) -> Tuple[SoftcoreRow, ...]:
    """One row per emitted soft core, both identities carried."""
    return tuple(
        SoftcoreRow(
            index=index,
            name=spec.name,
            input_count=int(spec.input_count),
            output_count=int(spec.output_count),
            area=int(spec.area),
            residency_class_id=int(spec.residency_class_id),
            residency_basis=str(spec.residency_basis),
            latency_tag=(None if spec.latency_tag is None else int(spec.latency_tag)),
            segment_id=(None if spec.segment_id is None else int(spec.segment_id)),
            bank_id=(None if spec.bank_id is None else int(spec.bank_id)),
            perceptron_index=(
                None if spec.perceptron_index is None else int(spec.perceptron_index)
            ),
        )
        for index, spec in enumerate(specs)
    )


def softcore_rows_from_placements(
    placements: Sequence[SoftcorePlacementRecord],
) -> Tuple[SoftcoreRow, ...]:
    """One row per placed IR node; split fragments of a node fold into its extent."""
    by_node: Dict[int, Dict[str, Any]] = {}
    for placement in placements:
        entry = by_node.setdefault(
            int(placement.ir_node_id),
            {
                "axons": 0, "neurons": 0,
                "segment_id": int(placement.segment_index),
                "perceptron_index": placement.perceptron_index,
                "bank_id": placement.weight_bank_id,
            },
        )
        entry["axons"] = max(entry["axons"], int(placement.axons))
        entry["neurons"] = max(entry["neurons"], int(placement.neurons))
    return tuple(
        SoftcoreRow(
            index=index,
            name=None,
            input_count=entry["axons"],
            output_count=entry["neurons"],
            area=entry["axons"] * entry["neurons"],
            residency_class_id=None,
            residency_basis=None,
            latency_tag=None,
            segment_id=entry["segment_id"],
            bank_id=(None if entry["bank_id"] is None else int(entry["bank_id"])),
            perceptron_index=(
                None if entry["perceptron_index"] is None
                else int(entry["perceptron_index"])
            ),
        )
        for index, entry in enumerate(by_node[node] for node in sorted(by_node))
    )


def _layer_label(row: SoftcoreRow) -> str:
    """A display label. Identity is ``perceptron_index``; this is only shown."""
    if row.perceptron_index is not None:
        return f"perceptron_{int(row.perceptron_index)}"
    return row.name or "unattributed"


def rollup_rows(rows: Sequence[SoftcoreRow]) -> Tuple[LayerRollupRow, ...]:
    """Aggregate cores by REAL layer identity.

    Cores with no perceptron identity (relays, psum accumulators) have no layer
    to belong to, so they group by name — the stated fallback, not a heuristic
    applied to everything.
    """
    grouped: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for row in rows:
        label = _layer_label(row)
        key = (row.perceptron_index, label)
        entry = grouped.setdefault(
            key,
            {
                "softcore_count": 0, "total_area": 0,
                "max_input_count": 0, "max_output_count": 0,
                "residency_classes": set(), "latency_tags": set(),
                "segments": set(), "banks": set(),
            },
        )
        entry["softcore_count"] += 1
        entry["total_area"] += int(row.area)
        entry["max_input_count"] = max(entry["max_input_count"], row.input_count)
        entry["max_output_count"] = max(entry["max_output_count"], row.output_count)
        for field_name, value in (
            ("residency_classes", row.residency_class_id),
            ("latency_tags", row.latency_tag),
            ("segments", row.segment_id),
            ("banks", row.bank_id),
        ):
            if value is not None:
                entry[field_name].add(int(value))

    built = [
        LayerRollupRow(
            perceptron_index=key[0],
            layer=key[1],
            softcore_count=entry["softcore_count"],
            total_area=entry["total_area"],
            max_input_count=entry["max_input_count"],
            max_output_count=entry["max_output_count"],
            residency_class_count=len(entry["residency_classes"]),
            latency_tag_count=len(entry["latency_tags"]),
            segment_count=len(entry["segments"]),
            bank_ids=tuple(sorted(entry["banks"])),
        )
        for key, entry in grouped.items()
    ]
    # Identified layers first, in identity order; the unattributed tail by label.
    built.sort(
        key=lambda r: (
            r.perceptron_index is None, r.perceptron_index or 0, r.layer,
        )
    )
    return tuple(built)


def bank_rows(
    rows: Sequence[SoftcoreRow],
    bank_records: Optional[Sequence[BankRecord]] = None,
) -> Tuple[Tuple[BankRow, ...], int]:
    """Per-bank sharing degree, plus how many cores own their weights."""
    sizes = {int(b.bank_id): b for b in (bank_records or ())}
    grouped: Dict[int, List[SoftcoreRow]] = {}
    unbanked = 0
    for row in rows:
        if row.bank_id is None:
            unbanked += 1
            continue
        grouped.setdefault(int(row.bank_id), []).append(row)

    built: List[BankRow] = []
    for bank_id in sorted(grouped):
        members = grouped[bank_id]
        identities = {m.perceptron_index for m in members if m.perceptron_index is not None}
        record = sizes.get(bank_id)
        built.append(
            BankRow(
                bank_id=bank_id,
                softcore_count=len(members),
                perceptron_index=(identities.pop() if len(identities) == 1 else None),
                max_input_count=max(m.input_count for m in members),
                max_output_count=max(m.output_count for m in members),
                rows=(None if record is None else int(record.rows)),
                cols=(None if record is None else int(record.cols)),
                params=(None if record is None else int(record.params)),
                segment_ids=tuple(sorted(
                    {m.segment_id for m in members if m.segment_id is not None}
                )),
            )
        )
    return tuple(built), unbanked


def placement_rows(
    placements: Sequence[SoftcorePlacementRecord],
) -> Tuple[PlacementRow, ...]:
    """The softcore→hard-core assignment, verbatim from the sealed record."""
    return tuple(
        PlacementRow(
            ir_node_id=int(p.ir_node_id),
            segment_index=int(p.segment_index),
            pass_index=int(p.pass_index),
            hard_core_index=int(p.hard_core_index),
            axon_offset=int(p.axon_offset),
            neuron_offset=int(p.neuron_offset),
            axons=int(p.axons),
            neurons=int(p.neurons),
            perceptron_index=p.perceptron_index,
            weight_bank_id=p.weight_bank_id,
        )
        for p in placements
    )


def candidate_segment_rows(rows: Sequence[SoftcoreRow]) -> Tuple[SegmentPassRow, ...]:
    """Per-segment census a shape-only view holds (its pass split is a total)."""
    counts: Dict[int, int] = {}
    for row in rows:
        segment = 0 if row.segment_id is None else int(row.segment_id)
        counts[segment] = counts.get(segment, 0) + 1
    return tuple(
        SegmentPassRow(segment_index=segment, passes=None, softcore_count=counts[segment])
        for segment in sorted(counts)
    )


def record_segment_rows(schedule: ScheduleRecord) -> Tuple[SegmentPassRow, ...]:
    """Per-segment pass structure and its programming class, as deployed."""
    grouped: Dict[int, List[Any]] = {}
    for segment in schedule.segments():
        grouped.setdefault(int(segment.segment_index), []).append(segment)
    return tuple(
        SegmentPassRow(
            segment_index=index,
            passes=len(grouped[index]),
            softcore_count=None,
            programming=(
                "resident"
                if any(s.programming == "resident" for s in grouped[index])
                else "reprogram"
            ),
        )
        for index in sorted(grouped)
    )
