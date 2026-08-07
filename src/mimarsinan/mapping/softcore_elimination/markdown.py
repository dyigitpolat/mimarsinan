"""[W6b] Markdown renderer: a run emits the paper table as a drop-in file."""

from __future__ import annotations

import os

from mimarsinan.mapping.softcore_elimination.report import (
    SoftcoreEliminationReport,
)
from mimarsinan.mapping.softcore_elimination.types import (
    SOFTCORE_ELIMINATION_TABLE_FILENAME,
    GroupElimination,
    StorageElimination,
)

_HEADER = (
    "| softcore group | instances | a×n | rows elim/inst | cols elim/inst "
    "| cells | surviving | **% cells eliminated** |"
)
_RULE = "|---|---:|---:|---:|---:|---:|---:|---:|"


def _pct(fraction: float) -> str:
    return f"{100.0 * fraction:.1f}%"


def _row(group: GroupElimination, *, bold: bool = False) -> str:
    def cell(text: str) -> str:
        return f"**{text}**" if bold else text

    dims = cell(group.dimensions) if group.axons is not None else ""
    return (
        f"| {cell(group.group)} | {cell(f'{group.instances:,}')} "
        f"| {dims} "
        f"| {cell(f'{group.rows_eliminated_per_instance:.1f}')} "
        f"| {cell(f'{group.cols_eliminated_per_instance:.1f}')} "
        f"| {cell(f'{group.cells:,}')} | {cell(f'{group.surviving:,}')} "
        f"| **{_pct(group.eliminated_fraction)}** |"
    )


def _storage_line(storage: StorageElimination) -> str:
    line = (
        f"**Physical weight storage** (shared banks counted once, W3c "
        f"intersection rule): banks {storage.bank_cells_before:,} → "
        f"{storage.bank_cells_after:,} cells = "
        f"**{_pct(storage.bank_eliminated_fraction)} eliminated**"
    )
    if storage.owned_matrices:
        line += (
            f"; unshared owned matrices {storage.owned_cells_before:,} → "
            f"{storage.owned_cells_after:,}; distinct total "
            f"{storage.cells_before:,} → {storage.cells_after:,} = "
            f"{_pct(storage.eliminated_fraction)}"
        )
    return line + "."


def _arm_table(report: SoftcoreEliminationReport) -> list[str]:
    """Per-arm cell elimination — the C1 evidence, per softcore group."""
    arms = list(report.arms)
    if len(arms) < 2:
        return []
    header = "| softcore group | " + " | ".join(arms) + " | Δ(closure−masked) | Δ(final−closure) |"
    rule = "|---" + "|---:" * (len(arms) + 2) + "|"
    lines = [
        "",
        "## Per-arm cell elimination in mapped softcores (C1)",
        "",
        "Same denominator, three propagation arms: the increment from `masked`",
        "to `closure` is seed-group coupling, the increment from `closure` to",
        "the deployed arm is emergent propagation.",
        "",
        header,
        rule,
    ]
    by_arm = {arm: {g.group: g for g in report.views[arm].groups} for arm in arms}
    names = [g.group for g in report.view.groups]
    for name in names + [report.total.group]:
        fractions = []
        for arm in arms:
            group = by_arm[arm].get(name)
            if group is None:
                group = report.views[arm].total
            fractions.append(group.eliminated_fraction)
        d_closure = fractions[1] - fractions[0] if len(fractions) > 1 else 0.0
        d_final = fractions[-1] - fractions[1] if len(fractions) > 2 else 0.0
        cells = " | ".join(_pct(f) for f in fractions)
        lines.append(
            f"| {name} | {cells} | {_pct(d_closure)} | {_pct(d_final)} |"
        )
    return lines


def render_softcore_elimination_markdown(
    report: SoftcoreEliminationReport, *, title: str | None = None
) -> str:
    """The paper table for one run: as-mapped rows + the physical-storage line."""
    total = report.total
    lines = [
        title or "# Weight-cell elimination in MAPPED SOFTCORES",
        "",
        "A cell is eliminated when its row OR its column is eliminated, so "
        "surviving = (R−r)·(C−c)",
        f"per instance. Arm: `{report.deployed_arm}`; geometry: "
        f"`{report.geometry}`.",
        "",
        _HEADER,
        _RULE,
    ]
    lines.extend(_row(g) for g in report.view.groups)
    lines.append(_row(total, bold=True))
    lines.extend(["", _storage_line(report.storage_total)])
    lines.append(
        f"**MAC sites in mapped softcores**: {total.cells:,} → "
        f"{total.surviving:,}"
        + (
            f" (**{total.cells / total.surviving:.1f}× fewer**)."
            if total.surviving else "."
        )
    )
    lines.extend(_arm_table(report))
    return "\n".join(lines) + "\n"


def write_softcore_elimination_markdown(
    report: SoftcoreEliminationReport,
    run_directory: str,
    *,
    title: str | None = None,
) -> str:
    """Write the rendered table into the run directory; returns the path."""
    os.makedirs(run_directory, exist_ok=True)
    path = os.path.join(run_directory, SOFTCORE_ELIMINATION_TABLE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_softcore_elimination_markdown(report, title=title))
    return path
