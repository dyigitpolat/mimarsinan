"""[W08] The output-stage line printed beside a priced point: lane work, host boundary bytes, host round trips and
the chip-only energy, labelled executed or estimated (the facts are ``mapping.lanes.output_stage``'s)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from mimarsinan.deployment_record.cost.absolute.context import AbsolutePricing

#: The term the energy metric prices beside its headline for a lane-block point (``energy.ENERGY_CHIP_ONLY``).
CHIP_ONLY_TERM = "energy_chip_only_mj"
PLACED_LABEL = "estimated: requirement placement on the toolchain IR, lanes not executed"


def chip_only_term(pricing: Optional[AbsolutePricing]):
    if pricing is None:
        return None
    return next((term for term in pricing.terms if term.name == CHIP_ONLY_TERM), None)


def chip_only_text(pricing: Optional[AbsolutePricing]) -> Optional[str]:
    """The chip-only energy as the output-stage line prints it, its unpriced components named (never a silent
    partial sum)."""
    term = chip_only_term(pricing)
    if term is None or term.band is None:
        return None
    band = term.band
    text = f"chip-only energy {band.nominal:.6g} mJ [{band.low:.6g}, {band.high:.6g}]"
    marker = "; unpriced: "
    if marker in term.source:
        text += f" (unpriced: {term.source.rsplit(marker, 1)[1]})"
    return text


def _work(lane_ops: Mapping[str, Any]) -> str:
    work = dict(lane_ops)
    detail = ", ".join(f"{kind} {int(n):,}" for kind, n in sorted(work.items())) or "none"
    return f"lane ops {sum(int(n) for n in work.values()):,} ({detail})"


def _census(census: Mapping[str, Any]) -> str:
    return (f"{_work(census['lane_ops'])}; host ops {census['host_ops']}; "
            f"host round trips {census['host_round_trips']} (dependency barriers {census['host_barriers']}); "
            f"host boundary bytes {int(census['host_bytes']):,}")


def output_stage_note(stage: Mapping[str, Any], pricing: Optional[AbsolutePricing] = None) -> str:
    """One line: the deployed output stage, its chip-only energy, and (lanes not executed) the placement."""
    classes = "".join(stage["classes"]) or "none"
    block = f" via lane block {stage['lane_block']}" if stage.get("lane_block") else ""
    # A candidate's stage carries its lane plan's work only (no deployed census: ``objectives.context``).
    census = _census(stage["deployed"]) if stage.get("deployed") is not None else _work(stage.get("lane_ops") or {})
    parts = [f"output stage ({stage['label']}): lanes {classes}{block}; {census}"]
    chip = chip_only_text(pricing)
    if chip is not None:
        parts.append(chip)
    if stage.get("placed") is not None:
        parts.append(f"{PLACED_LABEL}: {_census(stage['placed'])}")
    parts.extend(stage.get("notes", ()))
    return "; ".join(parts)
