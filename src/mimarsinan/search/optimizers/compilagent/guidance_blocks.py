"""Guidance block formatting for GuidedToolset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

from compilagent.session.multi_objective import (
    metric_summary as _metric_summary,
    pareto_front as _pareto_front,
)

if TYPE_CHECKING:
    from mimarsinan.search.results import ObjectiveSpec

    from .backend import MimarsinanLayoutBackend
    from .sink import MultiObjectiveSink


def augment_inspect_workload(raw: str, state: Any) -> str:
    if state.baseline_injected:
        return raw
    baseline = baseline_footprint(state.backend)
    if baseline is None:
        return raw
    suffix = format_baseline_block(baseline, state.objectives)
    state.baseline_injected = True
    state.sink.emit_guidance(text=suffix, target_tool="inspect_workload")
    return f"{raw}\n\n[BASELINE FOOTPRINT]\n{suffix}"


def augment_run_result(raw: str, state: Any) -> str:
    rows = objective_rows(state.sink)
    if not rows:
        return raw
    suffix = format_guidance_block(rows, state)
    state.sink.emit_guidance(text=suffix, target_tool="run_candidates")
    return f"{raw}\n\n[GUIDANCE]\n{suffix}"


def baseline_footprint(backend: "MimarsinanLayoutBackend") -> Optional[dict[str, Any]]:
    try:
        payload = backend.get_candidate_payload("baseline")
    except KeyError:
        return None
    softcores = payload.get("softcores") or []
    per_layer = payload.get("per_layer") or []
    layout_stats = payload.get("layout_stats") or {}
    return {
        "softcore_count": len(softcores),
        "layer_count": len(per_layer),
        "max_input_count": max((sc.get("input_count", 0) for sc in softcores), default=0),
        "max_output_count": max((sc.get("output_count", 0) for sc in softcores), default=0),
        "total_softcore_area": sum(int(sc.get("area", 0)) for sc in softcores),
        "fragmentation_pct": layout_stats.get("fragmentation_pct"),
        "mapped_params_pct": layout_stats.get("mapped_params_pct"),
        "neural_segment_count": layout_stats.get("neural_segment_count"),
        "threshold_group_count": layout_stats.get("threshold_group_count"),
    }


def format_baseline_block(
    baseline: dict[str, Any], objectives: Sequence["ObjectiveSpec"],
) -> str:
    lines = [
        "The default-config baseline already compiled with these dimensions:",
        f"  - softcores emitted: {baseline['softcore_count']}",
        f"  - distinct layers : {baseline['layer_count']}",
        f"  - max softcore input  count: {baseline['max_input_count']}",
        f"  - max softcore output count: {baseline['max_output_count']}",
        f"  - total softcore area     : {baseline['total_softcore_area']}",
    ]
    fp = baseline.get("fragmentation_pct")
    if fp is not None:
        lines.append(f"  - baseline fragmentation_pct: {float(fp):.2f}%")
    mp = baseline.get("mapped_params_pct")
    if mp is not None:
        lines.append(f"  - baseline param_utilization_pct: {float(mp):.2f}%")
    sc = baseline["softcore_count"]
    lines.append("")
    lines.append(
        "Sizing guidance: in single-pass packing the chip needs at least "
        f"{sc} cores; each core's `max_axons` must accommodate the largest "
        f"softcore's input count ({baseline['max_input_count']}) and "
        f"`max_neurons` must accommodate the largest output count "
        f"({baseline['max_output_count']}). Push the chip down to that "
        "footprint to drive `param_utilization_pct` up."
    )
    lines.append("")
    lines.append("Active objectives (all equally weighted):")
    for s in objectives:
        lines.append(f"  - {s.name} ({s.goal})")
    return "\n".join(lines)


def objective_rows(sink: "MultiObjectiveSink") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in sink.records():
        if rec.rejected or not rec.objectives:
            continue
        rows.append({
            "candidate_id": rec.candidate_id,
            "objectives": dict(rec.objective_metadata),
        })
    return rows


def format_guidance_block(
    rows: list[dict[str, Any]], state: Any,
) -> str:
    front = _pareto_front(rows)
    summary = _metric_summary(rows)

    lines: list[str] = []
    lines.append(
        f"Live multi-objective state — {len(rows)} candidate(s) scored, "
        f"{len(front)} on the Pareto front."
    )

    lines.append("")
    lines.append("Pareto front (non-dominated):")
    if not front:
        lines.append("  (none yet — only one candidate has scored or all are dominated)")
    else:
        for i, member in enumerate(front, 1):
            cid = str(member.get("candidate_id") or "?")[:12]
            objs = member.get("objectives") or {}
            obj_str = ", ".join(
                f"{k}={format_value(v.get('value') if isinstance(v, dict) else v)}"
                for k, v in objs.items()
            )
            lines.append(f"  {i}. {cid}: {obj_str}")

    lines.append("")
    lines.append("Per-metric leaders:")
    if not summary:
        lines.append("  (no scored objectives yet)")
    else:
        for metric, info in summary.items():
            best = info.get("best", {})
            worst = info.get("worst", {})
            unit = info.get("unit") or ""
            lines.append(
                f"  - {metric} ({info.get('goal')}): "
                f"best={format_value(best.get('value'))}{unit} "
                f"(cand {str(best.get('candidate_id') or '?')[:12]}), "
                f"worst={format_value(worst.get('value'))}{unit}, "
                f"median={format_value(info.get('median'))}{unit}"
            )

    baseline = baseline_footprint(state.backend) or {}
    baseline_obj = (
        state.backend.get_candidate_payload("baseline").get("hw_objectives", {})
        if baseline_payload_safe(state.backend)
        else {}
    )
    under_explored = under_explored_axes(summary, baseline_obj, state)
    if under_explored:
        lines.append("")
        lines.append("Under-explored axes (small or zero movement from baseline):")
        for axis in under_explored:
            lines.append(f"  - {axis}")

    suggestions = suggestions_for(rows, summary, baseline, state)
    if suggestions:
        lines.append("")
        lines.append("Suggested next directions:")
        for sug in suggestions:
            lines.append(f"  - {sug}")

    lines.append("")
    lines.append(
        "Use `pareto_front`, `metric_summary`, `query_top_candidates` "
        "and `compare_candidates` for the full multi-objective picture."
    )
    return "\n".join(lines)


def baseline_payload_safe(backend: "MimarsinanLayoutBackend") -> bool:
    try:
        backend.get_candidate_payload("baseline")
        return True
    except KeyError:
        return False


def under_explored_axes(
    summary: dict[str, dict[str, Any]],
    baseline_obj: dict[str, float],
    state: Any,
) -> list[str]:
    flagged: list[str] = []
    for metric, info in summary.items():
        baseline_val = baseline_obj.get(metric)
        best_val = info.get("best", {}).get("value")
        worst_val = info.get("worst", {}).get("value")
        if best_val is None:
            continue
        if best_val == worst_val:
            flagged.append(
                f"{metric}: every candidate scored {format_value(best_val)} — "
                f"vary the levers that influence this metric"
            )
            continue
        if baseline_val is not None:
            try:
                bval = float(baseline_val)
                if bval == 0:
                    delta = abs(float(best_val))
                else:
                    delta = abs(float(best_val) - bval) / abs(bval)
                if delta < 0.10:
                    flagged.append(
                        f"{metric}: best={format_value(best_val)} is within 10% "
                        f"of baseline={format_value(bval)}"
                    )
            except (TypeError, ValueError):
                pass
    return flagged


def arch_lever_names(state: Any, limit: int = 3) -> list[str]:
    """Arch lever (model_config) key names observed across ran candidates —
    workload-derived, never a hard-coded knob vocabulary."""
    names: set[str] = set()
    for rec in state.sink.records():
        cfg = rec.configuration or {}
        model_config = (cfg.get("deployment_parameters") or {}).get("model_config") or {}
        names.update(str(key) for key in model_config)
    return sorted(names)[: int(limit)]


def suggestions_for(
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
    state: Any,
) -> list[str]:
    out: list[str] = []
    if not rows:
        return out

    util = summary.get("param_utilization_pct")
    if util and util["best"]["value"] < 10.0:
        sc = baseline.get("softcore_count") or "?"
        out.append(
            "Hardware utilization is still single-digit %. Shrink the chip "
            "aggressively: lower `0.count` toward the baseline softcore "
            f"count ({sc}) and shrink `0.max_axons`/`0.max_neurons` to the "
            "largest single softcore dimension."
        )

    acc = summary.get("estimated_accuracy")
    if acc:
        spread = acc["best"]["value"] - acc["worst"]["value"]
        if spread < 0.05:
            levers = arch_lever_names(state)
            lever_hint = (
                ", ".join(f"`{name}`" for name in levers)
                if levers
                else "the model's arch levers"
            )
            out.append(
                "Accuracy is barely moving across candidates. Try a wider "
                f"spread on {lever_hint} to learn the "
                "accuracy vs cost trade-off."
            )

    counts_seen = set()
    axons_seen = set()
    for rec in state.sink.records():
        cfg = rec.configuration or {}
        cores = cfg.get("platform_constraints", {}).get("cores", [])
        if cores:
            counts_seen.add(cores[0].get("count"))
            axons_seen.add(cores[0].get("max_axons"))
    if len(counts_seen) <= 1:
        out.append(
            "Every candidate so far uses the same `0.count`. Sweep at "
            "least 3 distinct values across the [small, medium, large] "
            "range to map the count-vs-fragmentation curve."
        )
    if len(axons_seen) <= 1:
        out.append(
            "Every candidate so far uses the same `0.max_axons`. Try "
            "halving and doubling to bracket the chip-size sweet spot."
        )
    return out


def format_value(v: Any) -> str:
    if v is None:
        return "?"
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.0f}"
        return f"{v:.3f}"
    return str(v)
