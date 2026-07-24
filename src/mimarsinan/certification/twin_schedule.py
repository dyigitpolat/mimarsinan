"""[cert-plan W2] twin-certificate failure diagnostic: per-node schedules.

Not a gate predicate — packed programs can preserve the effective per-node
schedule while their core.latency tables differ (in-core chain pipelining),
so equality of these tables is neither necessary nor sufficient. On a twin
certificate FAILURE this table pair localizes where the two programs'
timetables disagree."""

from __future__ import annotations

from mimarsinan.mapping.latency.chip import ChipLatency


def _node_schedule(mapping) -> dict[int, list[tuple[int, int]]]:
    """{ir_node_id: [(neural_stage_ordinal, core_latency), ...]}."""
    out: dict[int, list[tuple[int, int]]] = {}
    ordinal = 0
    for stage in getattr(mapping, "stages", []):
        if getattr(stage, "kind", None) != "neural":
            continue
        hcm = stage.hard_core_mapping
        if hcm is None:
            continue
        if any(c.latency is None for c in hcm.cores):
            ChipLatency(hcm).calculate()
        placements = getattr(hcm, "soft_core_placements_per_hard_core", []) or []
        for core_idx, plist in enumerate(placements):
            latency = hcm.cores[core_idx].latency
            if latency is None:
                continue
            for placement in plist:
                node_id = placement.get("ir_node_id")
                if node_id is None:
                    continue
                out.setdefault(int(node_id), []).append((ordinal, int(latency)))
        ordinal += 1
    return out


def twin_schedule_diagnostic(identity_mapping, packed_mapping) -> str:
    """Human-readable per-node schedule comparison for twin-cert triage."""
    ident = _node_schedule(identity_mapping)
    packed = _node_schedule(packed_mapping)
    if not ident and not packed:
        return "twin-schedule: no neural stages / no placement metadata on either program"
    lines = ["twin-schedule per-node (stage, latency) tables:"]
    divergent = 0
    for node_id in sorted(set(ident) | set(packed)):
        a = ident.get(node_id)
        b = packed.get(node_id)
        if a == b:
            continue
        divergent += 1
        if divergent <= 24:
            lines.append(f"  node {node_id}: identity={a} packed={b}")
    if divergent == 0:
        lines.append("  identical for all "
                     f"{len(set(ident) | set(packed))} nodes")
    else:
        lines.append(f"  ({divergent} node(s) with differing tables — table "
                     "inequality is diagnostic, not itself a defect)")
    return "\n".join(lines)
