#!/usr/bin/env python
"""Aggregate `fidelity.json` across run directories — the correlation, per axis.

Report-first by design: it summarises how the search's predictions correlated with
what the runs measured, and gates nothing. Numeric tolerances are a decision to make
ON this evidence, not before it.

    python scripts/fidelity_summary.py generated_files/*/
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mimarsinan.deployment_record.fidelity import (  # noqa: E402
    FIDELITY_FILENAME,
    FidelityReport,
    load_fidelity_report,
)


def find_reports(roots: Sequence[str]) -> List[FidelityReport]:
    """Every fidelity report under the given roots, run dir or parent."""
    reports: List[FidelityReport] = []
    for root in roots:
        for dirpath, _dirnames, filenames in os.walk(root):
            if FIDELITY_FILENAME in filenames:
                reports.append(
                    load_fidelity_report(os.path.join(dirpath, FIDELITY_FILENAME))
                )
    return reports


def _fmt(value: Optional[float], width: int = 9) -> str:
    return " " * width if value is None else f"{value:>{width}.3g}"


def summarise(reports: Sequence[FidelityReport]) -> str:
    """Per-axis correlation across runs: coverage, in-band rate, error spread."""
    if not reports:
        return "no fidelity reports found"

    errors: Dict[str, List[float]] = {}
    banded: Dict[str, List[bool]] = {}
    compared: Dict[str, int] = {}
    for report in reports:
        for axis in report.axes:
            if axis.predicted is None or axis.measured is None:
                continue
            compared[axis.key] = compared.get(axis.key, 0) + 1
            if axis.relative_error is not None:
                errors.setdefault(axis.key, []).append(axis.relative_error)
            if axis.in_band is not None:
                banded.setdefault(axis.key, []).append(axis.in_band)

    lines = [
        f"fidelity over {len(reports)} run(s)",
        "",
        f"{'axis':<28}{'compared':>9}{'in band':>9}{'med err':>10}"
        f"{'p90 |err|':>10}{'max |err|':>10}",
        "-" * 76,
    ]
    for key in sorted(compared):
        verdicts = banded.get(key, [])
        rate = (
            f"{sum(verdicts)}/{len(verdicts)}" if verdicts else "-"
        )
        errs = errors.get(key, [])
        absolute = sorted(abs(e) for e in errs)
        median = statistics.median(errs) if errs else None
        p90 = absolute[int(0.9 * (len(absolute) - 1))] if absolute else None
        worst = absolute[-1] if absolute else None
        lines.append(
            f"{key:<28}{compared[key]:>9}{rate:>9}"
            f"{_fmt(median, 10)}{_fmt(p90, 10)}{_fmt(worst, 10)}"
        )
    term_errors: Dict[str, List[float]] = {}
    term_banded: Dict[str, List[bool]] = {}
    term_compared: Dict[str, int] = {}
    for report in reports:
        for term in report.terms:
            if term.predicted is None or term.measured is None:
                continue
            term_compared[term.name] = term_compared.get(term.name, 0) + 1
            if term.relative_error is not None:
                term_errors.setdefault(term.name, []).append(term.relative_error)
            if term.in_band is not None:
                term_banded.setdefault(term.name, []).append(term.in_band)
    if term_compared:
        lines += [
            "",
            "[H3] per-TERM decomposition (candidate-priced vs record-plane)",
            f"{'term':<28}{'compared':>9}{'in band':>9}{'med err':>10}"
            f"{'p90 |err|':>10}{'max |err|':>10}",
            "-" * 76,
        ]
        for name in sorted(term_compared):
            verdicts = term_banded.get(name, [])
            rate = f"{sum(verdicts)}/{len(verdicts)}" if verdicts else "-"
            errs = term_errors.get(name, [])
            absolute = sorted(abs(e) for e in errs)
            median = statistics.median(errs) if errs else None
            p90 = absolute[int(0.9 * (len(absolute) - 1))] if absolute else None
            worst = absolute[-1] if absolute else None
            lines.append(
                f"{name:<28}{term_compared[name]:>9}{rate:>9}"
                f"{_fmt(median, 10)}{_fmt(p90, 10)}{_fmt(worst, 10)}"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="run directories (or parents)")
    args = parser.parse_args(argv)
    print(summarise(find_reports(args.roots)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
